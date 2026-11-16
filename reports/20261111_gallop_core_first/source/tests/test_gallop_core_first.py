"""Core promotion cannot bypass safety; final certification stays strict."""
from dataclasses import replace
import math
import unittest

import torch

from isaac_task.gallop_core import GallopSettings, StagedGallopCurriculum, phase_targets, pair_cycle_penalties
from tests.test_gallop_staged import StagedSuccessTests, settings


class CoreFirstTests(unittest.TestCase):
    def ready(self,index=0):
        return StagedSuccessTests().ready(index,staged_core_first_enabled=True)

    def quality_failure(self,s):
        s.stability_sums['front_width_good'][:]=0
        s.guard_sums['front_high'][:]=250  #25%, fails strict10% lift gate.
        s.sums['heading2'][:]=1000*math.radians(6)**2
        s.sums['roll2'][:]=1000*math.radians(6)**2
        s.sums['pitch2'][:]=1000*math.radians(10)**2
        s.flight.totals['stable_landings'][:]=13
        s.flight.totals['aborted_landings'][:]=7

    def test_early_core_progress_is_distinct_from_strict_and_final_success(self):
        s=self.ready();self.quality_failure(s);s.complete_episodes([0])
        r=s.last_episode
        self.assertTrue(r['success']);self.assertFalse(r['staged_strict_success'])
        self.assertFalse(r['staged_final_success']);self.assertFalse(r['stable'])
        self.assertTrue(r['staged_core']);self.assertTrue(r['staged_details'])
        self.assertEqual(s.curriculum.successes,1)

    def test_width_and_lift_are_not_intermediate_gates(self):
        for index in range(8):
            with self.subTest(index=index):
                s=self.ready(index)
                s.stability_sums['front_width_good'][:]=0;s.guard_sums['front_high'][:]=300
                s.complete_episodes([0]);self.assertTrue(s.last_episode['success'])
                self.assertFalse(s.last_episode['staged_strict_success'])

    def test_final_stage_reinstates_all_original_conditions(self):
        for failed in (None,'width','height','landing','yaw','pose'):
            with self.subTest(failed=failed):
                s=self.ready(8)
                if failed=='width':s.stability_sums['front_width_good'][:]=70
                if failed=='height':s.guard_sums['front_high'][:]=101
                if failed=='landing':s.flight.totals['aborted_landings'][:]=10
                if failed=='yaw':s.sums['heading2'][:]=1000*math.radians(4)**2
                if failed=='pose':s.sums['pitch2'][:]=1000*math.radians(9)**2
                s.complete_episodes([0]);r=s.last_episode
                self.assertEqual(bool(r['success']),failed is None)
                self.assertEqual(r['success'],r['staged_strict_success'])
                self.assertEqual(r['success'],r['staged_final_success'])

    def test_core_cannot_bypass_missing_gait_or_safety(self):
        for failed in ('contact','flight','rom','fore_load','fore_motion','fore_cycle',
                       'raw_clip','actual','terminated','short','tracking','distance','landing'):
            with self.subTest(failed=failed):
                s=self.ready();self.quality_failure(s)
                if failed=='contact':s.sums['match'][:]=0
                if failed=='flight':s.flight_cycles[:]=0
                if failed=='rom':s.rom_sum[:]=0
                if failed=='fore_load':s.total_impulse[:,:2]=0
                if failed=='fore_motion':s.fore_rom_sum[:]=0
                if failed=='fore_cycle':s.fore_participation[:]=0
                if failed=='raw_clip':s.sums['clip'][:]=101
                if failed=='actual':s.guard_sums['actual'][:]=1
                if failed=='terminated':s.env.reset_terminated[:]=True
                if failed=='short':s.sums['points'][:]=450
                if failed=='tracking':s.sums['vx2'][:]=1000*.3**2
                if failed=='distance':s.leap.qualified_events[:]=0
                if failed=='landing':s.flight.totals['stable_landings'][:]=11;s.flight.totals['aborted_landings'][:]=9
                s.complete_episodes([0]);self.assertFalse(s.last_episode['success'])

    def test_requirements_use_each_episodes_latched_stage(self):
        s=self.ready();s.profile.latch(torch.tensor([1]),4);s.profile.latch(torch.tensor([2]),8)
        r=s.profile.promotion_requirements()
        torch.testing.assert_close(r['landing_fraction'],torch.tensor([.6,.65,.8]))
        torch.testing.assert_close(r['yaw_limit'],torch.deg2rad(torch.tensor([8.,5.,3.])))
        self.assertEqual(r['details_required'].tolist(),[False,False,True])

    def test_progress_quality_tightens_before_final_stage(self):
        for stage,landings,aborts,expected in ((0,13,7,True),(4,13,7,True),(6,13,7,False),(7,14,6,False)):
            s=self.ready(stage);s.flight.totals['stable_landings'][:]=landings;s.flight.totals['aborted_landings'][:]=aborts
            s.complete_episodes([0]);self.assertEqual(bool(s.last_episode['success']),expected)

    def test_new_policy_requires_opt_in(self):
        self.assertFalse(settings().staged_core_first_enabled)
        with self.assertRaises(ValueError):replace(GallopSettings(),staged_core_first_enabled=True)
        with self.assertRaises(ValueError):settings(staged_core_first_enabled=1)

    def test_final_quality_failures_hold_but_core_failures_regress(self):
        c=StagedGallopCurriculum(settings(staged_core_first_enabled=True,
            min_episodes=4,decision_steps=10,staged_min_dwell_seconds=0.))
        c.configure_envs(4);c.restore(8)
        for step in (10,20):
            for env in range(4):c.record(False,c.level,{'staged_regression':True},env,step)
            e=c.decide(step);self.assertEqual(e['reason'],'refine_quality');self.assertEqual(c.index,8)
            self.assertEqual(e['core_safety_success_rate'],1.)
        for step in (30,40):
            for env in range(4):c.record(False,c.level,{'staged_regression':False},env,step)
            c.decide(step)
        self.assertEqual(c.index,7)

    def test_late_stage_reward_uses_the_active_reference(self):
        s=self.ready(8);s.phase[:]=.7
        s.command[:,0]=3.;s.used_mix.fill_(1.);s.used_stage.fill_(3)
        s.robot.data.root_lin_vel_b[:,0]=3.
        t=phase_targets(s.phase,s.command[:,0],s.cfg,s.strength,profile=s.profile)
        self.assertFalse(t['stance'].any())
        self.assertTrue(phase_targets(s.phase,s.command[:,0],s.cfg,1.)['stance'].any())
        s.sensor.data.net_forces_w.zero_()
        s.robot.data.body_pos_w[:,:,2]=t['foot_height']+s.cfg.foot_radius
        s.leap.samples=s.env.cfg.decimation
        s.update_measurements()
        self.assertFalse(s.expected.any())
        self.assertTrue((s.components['flight']>0).all())

    def test_pair_stance_tolerance_follows_episode_duty(self):
        s=self.ready(8)
        # Front duty difference.25 fits old.30 allowance but exceeds new.21.
        stance=torch.tensor([[.45,.20,.28,.28]]).expand(3,4)
        args=(torch.ones(3,4,2),torch.ones(3,4),stance,torch.ones(3))
        old,_=pair_cycle_penalties(*args,s.cfg)
        new,_=pair_cycle_penalties(*args,s.profile)
        self.assertTrue((old['stance']==0).all());self.assertTrue((new['stance']>0).all())


if __name__=='__main__':unittest.main()
