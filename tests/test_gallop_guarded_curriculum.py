"""Guarded speed/leap progression, recovery credit and strict stability gates."""
from dataclasses import replace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,SuccessSpeedCurriculum,stabilization_leap_target,phase_targets
from tests.test_gallop_flight import Trajectory
from tests.test_gallop_guarded import fixture
from tests.test_gallop_pair_env import state_class


def settings():return GallopSettings.load(ROOT/'configs/gallop_guarded_3mps_leap.json')


def state():
    _,_,env,data,sensor=fixture()
    result=state_class()(env,settings(),None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
    env.gallop=result
    return result,env,data,sensor


class GuardedCurriculumTests(unittest.TestCase):
    def window(self,c,step,success):
        for _ in range(128):c.record(success,c.level,{'guarded_limits':success,'stabilization_landing':success})
        return c.decide(step)

    def test_no_time_only_promotion_then_joint_speed_distance_progression_and_regression(self):
        cfg=settings();c=SuccessSpeedCurriculum(cfg)
        c.decide(100000);self.assertEqual(c.cap,2.5)
        self.window(c,101000,False);self.assertEqual(c.cap,2.5)
        step=102000
        for level in range(1,6):
            before=c.cap
            self.window(c,step,True);self.assertAlmostEqual(c.cap,before)
            event=self.window(c,step+1000,True)
            self.assertAlmostEqual(c.cap,2.5+.1*level)
            self.assertAlmostEqual(event['leap_target'],.14+.016*level,places=6)
            self.assertAlmostEqual(c.highest_confirmed_speed,before)
            step+=2000
        self.assertAlmostEqual(c.cap,3.)
        self.assertAlmostEqual(c.highest_confirmed_speed,2.9)
        self.window(c,step,True);self.window(c,step+1000,True)
        self.assertAlmostEqual(c.cap,3.);self.assertAlmostEqual(c.highest_confirmed_speed,3.)
        self.window(c,step+2000,False)
        self.assertAlmostEqual(c.cap,2.9)
        c.record(True,c.level-1,{})
        self.assertEqual(c.count,0);self.assertEqual(c.discarded,1)

    def test_targets_latch_on_individual_reset_and_play_freezes_saved_cap(self):
        s,_,_,_=state();s.curriculum.cap=2.8;s.curriculum.level=1
        s.leap.active[1]=True;s.leap.current_distance[1]=.1
        s.reset(torch.tensor([0]))
        torch.testing.assert_close(s.episode_speed,torch.tensor([2.8,2.5,2.5]))
        torch.testing.assert_close(s.leap.distance_target,torch.tensor([.188,.14,.14]))
        self.assertTrue(s.leap.active[1]);self.assertAlmostEqual(s.leap.current_distance[1].item(),.1)
        s.set_evaluation_speed(3.,2.9)
        torch.testing.assert_close(s.leap.distance_target,torch.full((3,),.22))
        self.assertFalse(s.curriculum_enabled);self.assertAlmostEqual(s.curriculum.highest_confirmed_speed,2.9)

    def test_planned_air_window_covers_distance_without_higher_feet_or_joint_rates(self):
        cfg=settings();old=GallopSettings.load(ROOT/'configs/gallop_guarded_2p5.json')
        speed=torch.linspace(2.5,3.,6)
        period=phase_targets(torch.zeros_like(speed),speed,cfg)['period']
        air=(1-max(t+d for t,d in zip(cfg.touchdown,cfg.duty)))*period
        self.assertTrue((speed*air>=stabilization_leap_target(speed,cfg)).all())
        self.assertAlmostEqual(air[-1].item(),.076,places=6)
        for key in ('action_scale','guarded_sim_limit_factor','guarded_target_margin','guarded_hip_rate',
                    'guarded_leg_rate','guarded_front_height','guarded_hind_height','guarded_actual_tolerance',
                    'guarded_max_intervention','stabilization_landing_fraction','stabilization_width_fraction'):
            self.assertEqual(getattr(cfg,key),getattr(old,key))

    def test_real_episode_success_rejects_bad_landing_width_or_filter_dependence(self):
        for failed_gate in (None,'stabilization_landing','stabilization_width','guarded_limits'):
            s,env,_,_=state();s.age[:]=1500;s.first[:]=False;s.startup.stage[:]=3;s.startup.times[:,3]=20
            env.reset_terminated.zero_();env.reset_time_outs[:]=True
            s.sums['points'][:]=1000;s.sums['match'][:]=1000;s.sums['command'][:]=2500;s.sums['clip'][:]=50
            s.sums['roll2'][:]=1;s.sums['pitch2'][:]=1
            s.cycles[:]=20;s.flight_cycles[:]=20;s.rom_sum[:]=20;s.fore_rom_sum[:]=20;s.fore_participation[:]=20
            s.total_impulse[:]=25;s.events[:]=20;s.good_events[:]=20;s.sequences[:]=20
            s.flight.totals['stable_landings'][:]=20;s.flight.totals['aborted_landings'][:]=2
            s.leap.events[:]=20;s.leap.qualified_events[:]=16
            s.guard_sums['intervention'][:]=200
            for name in ('front','hind'):
                s.stability_sums[name+'_support'][:]=100;s.stability_sums[name+'_width_good'][:]=100
            if failed_gate=='stabilization_landing':s.flight.totals['aborted_landings'][:]=10
            if failed_gate=='stabilization_width':s.stability_sums['front_width_good'][:]=70
            if failed_gate=='guarded_limits':s.guard_sums['intervention'][:]=500
            s.complete_episodes([0]);row=s.last_episode
            self.assertEqual(row['success'],float(failed_gate is None),failed_gate)
            if failed_gate:self.assertEqual(row[failed_gate],0.)
            self.assertEqual(s.curriculum.successes,int(failed_gate is None))


class GuardedLongLeapTests(unittest.TestCase):
    def flight(self,air_samples,**landing):
        t=Trajectory();t.s=replace(settings(),metric_warmup_seconds=0.)
        t.leap.cfg=t.s;t.leap.distance_target.fill_(.22);t.flight.cfg=t.s
        t.sample(phase=.7,vx=3.)
        t.sample(ground=False,vz=.4,vx=3.)
        for _ in range(air_samples-1):t.sample(ground=False,vz=-.3,vx=3.)
        t.sample(vz=-.3,vx=3.,**landing)
        return t

    def test_longer_distance_has_threshold_bonus_but_waits_for_recovery(self):
        short=self.flight(14);long=self.flight(16)
        self.assertEqual(short.leap.qualified_events.item(),0.)
        self.assertEqual(long.leap.qualified_events.item(),1.)
        self.assertEqual(long.earned['distance'].item(),0.)
        self.assertLessEqual(short.flight.recovery_bank.item(),.5)
        for _ in range(16):short.sample(phase=.1,vx=3.);long.sample(phase=.1,vx=3.)
        self.assertGreater(long.earned['distance'].item(),2*short.earned['distance'].item())
        self.assertEqual(long.flight.totals['stable_landings'].item(),1.)

    def test_unstable_long_flights_never_earn_distance_or_landing_credit(self):
        for bad in ({'slide':.8},{'force':500.},{'roll':.4}):
            t=self.flight(16,**bad)
            self.assertEqual(t.leap.qualified_events.item(),1.)
            for _ in range(20):t.sample(phase=.1,vx=3.)
            self.assertEqual(t.earned['distance'].item(),0.,bad)
            self.assertEqual(t.earned['landing'].item(),0.,bad)


if __name__=='__main__':unittest.main()
