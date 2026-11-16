"""Dense stability scores, student-only scaling and reset-latched curriculum."""
import ast
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings,HeadingMotionTracker,LeapDistanceTracker,SuccessSpeedCurriculum,
    stabilized_student_actions,stabilization_leap_target,target_excess_penalty,blend_startup_actions)
from tests.test_gallop_heading_motion import Sample,quat
from tests.test_gallop_posture import posture_state
from tests.test_gallop_pair_env import state_class


def settings(**changes):return replace(GallopSettings.load(ROOT/'configs/gallop_stabilization.json'),**changes)


def sample():
    x=Sample();x.cfg=settings();x.tracker=HeadingMotionTracker(x.root,x.cfg)
    return x


def state():
    _,env,data,sensor=posture_state()
    env.max_episode_length=1500
    result=state_class()(env,settings(),None,None,False,False,walking_policy=lambda obs:torch.zeros(3,12))
    return result,env,data,sensor


class StabilizationRewardTests(unittest.TestCase):
    def observe(self,x,q=None,rate=0.,support=True):
        return x.tracker.observe(quat() if q is None else q,x.root,x.feet,torch.full((1,4),support,dtype=torch.bool),
            torch.tensor([[2.2,0.,0.]]),torch.tensor([2.2]),.02,angular_velocity=torch.tensor([[0.,0.,rate]]))

    def test_constant_lean_no_longer_earns_full_quiet_reward(self):
        level=sample();lean=sample();self.observe(level);self.observe(lean,quat(roll=.194))
        good,_=self.observe(level);bad,diag=self.observe(lean,quat(roll=.194))
        self.assertEqual(good['roll_pitch_quiet'].item(),1.)
        self.assertLess(bad['roll_pitch_quiet'].item(),.2)
        self.assertGreater(diag['pose_quality'].item(),0.)
        self.assertEqual(diag['roll_pitch_rate'].sum().item(),0.)

    def test_widening_collapsed_front_track_gives_monotonic_nonzero_feedback(self):
        x=sample();scores=[]
        for width in (.042,.08,.14,.22,.2698):
            x.feet[:,:2,1]=torch.tensor([width/2,-width/2])
            terms,_=self.observe(x);scores.append(terms['front_width'].item())
            self.assertEqual(terms['hind_width'].item(),1.)
        self.assertGreater(scores[0],.05);self.assertEqual(scores,sorted(scores));self.assertEqual(scores[-1],1.)
        air,_=self.observe(x,support=False);self.assertAlmostEqual(air['front_width'].item(),.3,places=6)

    def test_yaw_damping_distinguishes_rotation_even_at_zero_heading_error(self):
        x=sample();values=[]
        for rate in (0.,.3,.6):
            terms,_=self.observe(x,rate=rate);values.append(terms['yaw_damping'].item())
            self.assertEqual(terms['yaw_heading'].item(),0.)
        self.assertEqual(values[0],0.);self.assertGreater(values[0],values[1]);self.assertGreater(values[1],values[2])

    def test_calf_scaling_preserves_latent_action_and_teacher_endpoint(self):
        s=settings();student=torch.ones(4,12);calves=torch.tensor([8,9,10,11]);mix=torch.tensor([0.,.25,.5,1.])
        scaled=stabilized_student_actions(student,calves,mix,s)
        torch.testing.assert_close(scaled[:,8],torch.tensor([1.,.9625,.925,.85]))
        torch.testing.assert_close(scaled[:,:8],student[:,:8]);self.assertTrue((student==1).all())
        walking=torch.full_like(student,.2)
        result=blend_startup_actions(scaled,walking,torch.tensor([1,2,2,3]),mix)
        torch.testing.assert_close(result[0],walking[0]);self.assertAlmostEqual(result[-1,8].item(),.85,places=6)

    def test_excess_penalty_grows_for_commands_with_the_same_clipped_target(self):
        s=settings();calves=torch.tensor([8,9,10,11]);bounded=torch.zeros(3,12);raw=bounded.clone()
        raw[:,8]=torch.tensor([0.,.1,1.])
        cost=target_excess_penalty(raw,bounded,calves,s)
        self.assertEqual(cost[0].item(),0.);self.assertGreater(cost[2].item(),cost[1].item());self.assertGreater(cost[1].item(),0.)
        thigh=raw.clone();thigh[:,0]=thigh[:,8];thigh[:,8]=0
        torch.testing.assert_close(cost,target_excess_penalty(thigh,bounded,calves,s)*2)

    def test_profile_rejects_incompatible_or_unsafe_target_contracts(self):
        for changes in ({'heading_motion_enabled':False},{'startup_enabled':False},{'flight_shaping_enabled':False},
            {'stabilization_calf_multiplier':1.1},{'stabilization_min_leap_distance':.3},
            {'stabilization_width_min':.4},{'stabilization_yaw_rate_scale':0.}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):settings(**changes)


class StabilizationCurriculumTests(unittest.TestCase):
    def test_speed_and_distance_require_repeated_success_and_can_regress(self):
        c=SuccessSpeedCurriculum(settings())
        c.decide(100000);self.assertEqual(c.cap,2.2)
        step=101000
        for window in range(2):
            for _ in range(128):c.record(True,c.level,{'stabilization_width':True})
            c.decide(step);step+=1000
            if window==0:self.assertEqual(c.cap,2.2)
        self.assertAlmostEqual(c.cap,2.4);self.assertAlmostEqual(c.highest_confirmed_speed,2.2)
        self.assertEqual(c.level,1);c.record(True,0,{});self.assertEqual(c.count,0);self.assertEqual(c.discarded,1)
        for _ in range(128):c.record(False,c.level,{})
        c.decide(step);self.assertAlmostEqual(c.cap,2.2);self.assertEqual(c.level,2)
        targets=stabilization_leap_target(torch.tensor([2.2,2.4,2.6,2.8]),settings())
        torch.testing.assert_close(targets,torch.tensor([.14,.15333333,.16666667,.18]))

    def test_goal_changes_only_on_local_reset_and_replay_uses_saved_cap(self):
        s,env,_,_=state();s.curriculum.cap=2.6;s.curriculum.level=1
        self.assertTrue((s.startup.goal_speed==2.2).all());self.assertTrue((s.leap.distance_target==.14).all())
        s.reset(torch.tensor([0]));self.assertAlmostEqual(s.episode_speed[0].item(),2.6,places=5)
        self.assertAlmostEqual(s.leap.distance_target[0].item(),.16666667,places=6)
        self.assertAlmostEqual(s.startup.goal_speed[1].item(),2.2,places=5)
        self.assertEqual(s.level.tolist(),[1,0,0])
        s.set_evaluation_speed(2.8,2.6);self.assertTrue((s.leap.distance_target==.18).all())
        self.assertAlmostEqual(s.curriculum.highest_confirmed_speed,2.6)
        self.assertFalse(s.curriculum_enabled)
        s.age[0]=1
        with self.assertRaises(ValueError):s.set_evaluation_speed(2.2)

    def test_same_flight_can_qualify_at_initial_distance_but_not_final_distance(self):
        cfg=replace(settings(),metric_warmup_seconds=0.)
        pos=torch.tensor([[0.,0.,.35],[0.,0.,.35]])
        leap=LeapDistanceTracker(pos,cfg);leap.distance_target[:]=torch.tensor([.14,.18])
        force=torch.ones(2,4)*100;q=quat().repeat(2,1);allowed=torch.ones(2,dtype=torch.bool)
        leap.observe(pos,q,force,.005,allowed);leap.observe(pos,q,force*0,.005,allowed)
        for i in range(1,15):
            pos[:,0]=.16*i/14;leap.observe(pos,q,force if i==14 else force*0,.005,allowed)
        self.assertEqual(leap.events.tolist(),[1.,1.]);self.assertEqual(leap.qualified_events.tolist(),[1.,0.])


class StabilizationWiringTests(unittest.TestCase):
    def test_actual_reward_updates_support_stats_and_does_not_double_count(self):
        s,env,data,sensor=state();s.startup.stage[:]=3;s.startup.times[:,3]=1.1
        data.root_quat_w=quat(roll=.194).repeat(3,1);data.root_ang_vel_b[:,2]=.5
        sensor.data.net_forces_w[:,:,2]=100
        for step in range(2):
            env.common_step_counter=step+1;s.begin_step()
            for _ in range(4):s.observe_leap_physics(.005)
            s.action_excess[:]=1.;s.update_measurements();s.update_measurements()
        self.assertTrue((s.components['roll_pitch_quiet']<.2).all())
        self.assertTrue((s.components['yaw_damping']<0).all());self.assertTrue((s.components['target_excess']<0).all())
        self.assertEqual(s.stability_sums['front_support'].tolist(),[2.,2.,2.])
        self.assertIn('stabilization_leap_target',s.snapshot())
        s.reset(torch.tensor([0]));self.assertEqual(s.stability_sums['front_support'].tolist(),[0.,2.,2.])

    def test_strict_episode_gates_reject_narrow_support_pose_bias_and_empty_measurements(self):
        s,env,_,_=state();s.age[:]=1500;s.first[:]=False;s.startup.stage[:]=3;s.startup.times[:,3]=20
        s.sums['points'][:]=1000;s.sums['match'][:]=1000;s.sums['command'][:]=2200;s.sums['clip'][:]=50
        s.sums['roll2'][:]=1;s.sums['pitch2'][:]=1;s.sums['roll2'][2]=40
        s.cycles[:]=20;s.flight_cycles[:]=20;s.rom_sum[:]=20;s.fore_rom_sum[:]=20;s.fore_participation[:]=20
        s.total_impulse[:]=25;s.events[:]=20;s.good_events[:]=20;s.sequences[:]=20
        s.flight.totals['stable_landings'][:]=20;s.flight.totals['aborted_landings'][:]=2
        s.leap.events[:]=20;s.leap.qualified_events[:]=16
        for name in ('front','hind'):
            s.stability_sums[name+'_support'][:]=100;s.stability_sums[name+'_width_good'][:]=100
        s.stability_sums['front_width_good'][1]=0
        rows=[]
        for i in range(3):s.complete_episodes([i]);rows.append(s.last_episode.copy())
        self.assertEqual([r['success'] for r in rows],[1.,0.,0.])
        self.assertEqual(rows[1]['stabilization_width'],0.);self.assertEqual(rows[2]['stabilization_pose'],0.)
        from scripts.analysis.collect_gallop import episode_stats
        collected=episode_stats(rows)
        self.assertIn('stabilization_width',collected['criterion_pass_rates'])
        self.assertIn('stabilization_width',collected['failure_combinations'])
        s.sums['points'][0]=0;s.complete_episodes([0])
        self.assertEqual(s.last_episode['tracking'],0.);self.assertEqual(s.last_episode['stable'],0.)

    def test_action_term_preserves_sampled_student_and_measures_preclip_excess(self):
        source=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='BoundedJointPositionAction')
        class Base:
            def process_actions(self,a):self._processed_actions=a*.5
        ns=dict(torch=torch,JointPositionAction=Base,blend_startup_actions=blend_startup_actions,
            stabilized_student_actions=stabilized_student_actions,target_excess_penalty=target_excess_penalty)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'stable_action','exec'),ns)
        term=ns['BoundedJointPositionAction'].__new__(ns['BoundedJointPositionAction']);sampled=torch.ones(2,12)*4
        st=NS(cfg=settings(),startup=NS(stage=torch.tensor([1,3]),mix=torch.tensor([0.,1.])),
            calf_ids=torch.tensor([8,9,10,11]),walking_policy=lambda obs:torch.full((2,12),.2),
            executed_actions=torch.zeros(2,12),joint_target_excess=torch.zeros(2,12),action_excess=torch.zeros(2))
        term._env=NS(gallop=st,obs_buf={'policy':torch.zeros(2,49)},action_manager=NS(action=sampled.clone()))
        term._asset=NS(data=NS(soft_joint_pos_limits=torch.tensor([-1.,1.]).expand(2,12,2)))
        term._joint_ids=slice(None);term.process_actions(sampled)
        self.assertTrue((sampled==4).all());self.assertTrue((st.executed_actions[0]==.2).all())
        self.assertAlmostEqual(st.executed_actions[1,8].item(),3.4,places=5)
        self.assertGreater(st.action_excess[1].item(),0.);self.assertTrue((term._processed_actions<=1).all())


if __name__=='__main__':unittest.main()
