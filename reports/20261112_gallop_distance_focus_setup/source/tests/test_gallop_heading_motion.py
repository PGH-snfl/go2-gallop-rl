"""Heading, temporal stability, independent foot widths and world progress."""
import ast
from dataclasses import replace
import math
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,HeadingMotionTracker,heading_features
from tests.test_gallop_posture import posture_state
from tests.test_gallop_pair_env import state_class


def settings(**changes):
    return replace(GallopSettings.load(ROOT/'configs/gallop_2p8mps_heading_motion.json'),**changes)


def quat(roll=0.,pitch=0.,yaw=0.):
    cr,sr=math.cos(roll/2),math.sin(roll/2);cp,sp=math.cos(pitch/2),math.sin(pitch/2);cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    return torch.tensor([[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]])


class Sample:
    def __init__(self,**changes):
        self.cfg=settings(**changes);self.root=torch.tensor([[0.,0.,.35]])
        self.tracker=HeadingMotionTracker(self.root,self.cfg)
        f,h=self.cfg.front_stance_width/2,self.cfg.hind_stance_width/2
        self.feet=torch.tensor([[[.2,f,.02],[.2,-f,.02],[-.2,h,.02],[-.2,-h,.02]]])

    def step(self,q=None,velocity=(2.8,0.,0.),dt=.02,speed=2.8,contact=True):
        return self.tracker.observe(quat() if q is None else q,self.root,self.feet,
            torch.full((1,4),contact,dtype=torch.bool),torch.tensor([velocity]),torch.tensor([speed]),dt)


class HeadingMotionTests(unittest.TestCase):
    def test_absolute_yaw_zero_and_wraparound(self):
        x=Sample();zero,_=x.step();self.assertEqual(zero['yaw_heading'].item(),0.)
        for angle in (.1,-.1,2*math.pi+.1,-2*math.pi-.1):
            terms,values=x.step(quat(yaw=angle))
            self.assertAlmostEqual(terms['yaw_heading'].item(),-(.1/.35)**2,places=5)
            self.assertAlmostEqual(abs(values['yaw_error'].item()),.1,places=5)
        torch.testing.assert_close(heading_features(quat(),0.),torch.zeros(1,2))
        self.assertLess(heading_features(quat(yaw=.2),0.)[0,0].item(),0.)

    def test_quiet_reward_measures_motion_not_static_roll_pitch(self):
        x=Sample();lean=quat(roll=.2,pitch=-.15)
        first,_=x.step(lean);self.assertEqual(first['roll_pitch_quiet'].item(),0.)
        quiet,_=x.step(lean);self.assertEqual(quiet['roll_pitch_quiet'].item(),1.)
        changing,_=x.step(quat(roll=.3,pitch=-.15))
        self.assertLess(changing['roll_pitch_quiet'].item(),quiet['roll_pitch_quiet'].item())
        self.assertNotIn('upright',changing);self.assertNotIn('stance_center',changing)

    def test_angle_rate_is_timestep_normalized_and_resets_do_not_create_impulses(self):
        scores=[]
        for dt in (.01,.02):
            x=Sample();x.step();terms,values=x.step(quat(roll=dt),dt=dt)
            scores.append(terms['roll_pitch_quiet'].clone())
            self.assertAlmostEqual(values['roll_pitch_rate'][0,0].item(),1.,places=5)
            x.tracker.reset(torch.tensor([0]));terms,_=x.step(quat(roll=.5))
            self.assertEqual(terms['roll_pitch_quiet'].item(),0.)
            with self.assertRaises(ValueError):x.step(dt=0)
        torch.testing.assert_close(scores[0],scores[1])

    def test_front_and_rear_width_scores_are_independent_positive_and_below_shoulders(self):
        x=Sample(hind_shoulder_width=.32,hind_stance_width=.304)
        ideal,_=x.step();self.assertEqual(ideal['front_width'].item(),1.);self.assertEqual(ideal['hind_width'].item(),1.)
        self.assertLess(x.cfg.front_stance_width,x.cfg.front_shoulder_width)
        self.assertLess(x.cfg.hind_stance_width,x.cfg.hind_shoulder_width)
        shoulder=Sample();shoulder.feet[:,:2,1]*=shoulder.cfg.front_shoulder_width/shoulder.cfg.front_stance_width
        shoulder_terms,_=shoulder.step()
        self.assertLess(shoulder_terms['front_width'].item(),1.)
        x.feet[:,0,1]+=.1
        changed,_=x.step()
        self.assertGreaterEqual(changed['front_width'].item(),0.)
        self.assertLess(changed['front_width'].item(),ideal['front_width'].item())
        self.assertEqual(changed['hind_width'].item(),1.)
        x.feet[:,:2,1]*=-1;crossed,_=x.step()
        self.assertLess(crossed['front_width'].item(),1e-6)

    def test_pair_translation_is_not_penalized_and_swing_does_not_disable_width_score(self):
        x=Sample();before,_=x.step();x.feet[:,:,1]+=.2;x.feet[:,:,0]+=.4;x.feet[:,:,2]+=.1
        after,_=x.step(contact=False)
        self.assertEqual(before['front_width'].item(),after['front_width'].item())
        self.assertEqual(before['hind_width'].item(),after['hind_width'].item())
        self.assertNotIn('stance_center',after)
        x.feet[:,0,1]+=.1;wide,_=x.step(contact=False)
        self.assertLess(wide['front_width'].item(),after['front_width'].item())

    def test_actual_target_direction_rewards_progress_and_rejects_stationary_backward_sideways_and_excess_speed(self):
        x=Sample();ideal,_=x.step();self.assertAlmostEqual(ideal['forward_progress'].item(),1.,places=6)
        for velocity in [(0.,0.,0.),(-2.8,0.,0.),(0.,2.8,0.),(2.8,1.,0.),(5.6,0.,0.)]:
            terms,_=x.step(velocity=velocity)
            self.assertLess(terms['forward_progress'].item(),.01)
        terms,_=x.step(speed=0.);self.assertEqual(terms['forward_progress'].item(),0.)
        # This is world progress, not inferred from the robot's facing direction.
        terms,_=x.step(q=quat(yaw=math.pi/2),velocity=(0.,2.8,0.))
        self.assertEqual(terms['forward_progress'].item(),0.)

    def test_target_yaw_rotates_motion_goal_and_width_frame(self):
        x=Sample(target_yaw=math.pi/2)
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        x.feet=(x.feet-x.root[:,None])@rotation.T+x.root[:,None]
        terms,values=x.step(quat(yaw=math.pi/2),velocity=(0.,2.8,0.))
        self.assertAlmostEqual(terms['yaw_heading'].item(),0.,places=6)
        self.assertAlmostEqual(terms['forward_progress'].item(),1.,places=6)
        self.assertAlmostEqual(terms['front_width'].item(),1.,places=6)

    def test_invalid_weight_width_reference_or_old_penalty_configuration_is_rejected(self):
        for change in ({'yaw_error_scale':0},{'roll_pitch_quiet_weight':-1},{'target_yaw':float('nan')},
                       {'upright_weight':1},{'stance_center_weight':1},{'stance_shoulder_ratio':1.1},{'front_stance_width':.3}):
            with self.subTest(change=change),self.assertRaises(ValueError):settings(**change)


class HeadingMotionWiringTests(unittest.TestCase):
    def test_all_stage_metrics_snapshot_cache_and_local_history_reset(self):
        _,env,data,sensor=posture_state()
        cfg=settings();data.root_lin_vel_w[:]=torch.tensor([2.8,0.,0.])
        data.root_quat_w=quat(roll=.2,yaw=.2).repeat(3,1)
        state=state_class()(env,cfg,None,None,False,False,walking_policy=lambda obs:torch.zeros(3,12))
        state.startup.stage[:]=torch.tensor([0,1,2])
        sensor.data.net_forces_w[:,:,2]=0
        for i in range(2):
            env.common_step_counter=i+1;state.begin_step()
            for _ in range(4):state.observe_leap_physics(.005)
            state.update_measurements();state.update_measurements()
        self.assertTrue((state.components['yaw_heading']<0).all())
        self.assertTrue((state.components['roll_pitch_quiet']==1).all())
        self.assertTrue((state.components['attitude']==0).all())
        self.assertEqual(state.motion.sums['points'].tolist(),[2.,2.,2.])
        self.assertEqual(state.sums['points'].sum().item(),0.)
        state.complete_episodes([0]);self.assertAlmostEqual(state.last_episode['posture_yaw_rmse'],.2,places=5)
        self.assertEqual(state.last_episode['posture_rate_samples'],1.)
        self.assertIn('posture_roll_pitch_rate',state.snapshot())
        self.assertIn('Gallop/last_posture_forward_score',state.metrics())
        state.reset(torch.tensor([0]));self.assertEqual(state.motion.seeded.tolist(),[False,True,True])

    def test_configure_has_heading_observation_zero_yaw_reset_and_positive_independent_rewards(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_gallop')
        cfg=NS(scene=NS(terrain=NS(),contact_forces=NS()),curriculum=NS(),commands=NS(base_velocity=NS(ranges=NS(),limit_ranges=NS())),
            actions=NS(JointPositionAction=NS()),sim=NS(dt=.005),events=NS(reset_base=NS(params={'pose_range':{'yaw':(-3.14,3.14)}}),reset_robot_joints=NS(params={})),
            observations=NS(policy=NS(),critic=NS()),terminations=NS(),rewards=NS(**{k:NS(params={}) for k in ('track_lin_vel_xy','base_linear_velocity','base_angular_velocity','action_rate')}))
        ns=dict(StartupVelocityCommand=object,BoundedJointPositionAction=object,ObservationTermCfg=NS,RewardTermCfg=NS,
            TerminationTermCfg=NS,gallop_reward=lambda *a:None,phase_observation=lambda *a:None,
            heading_observation=lambda *a:None,height_termination=lambda *a:None)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'configure_heading','exec'),ns)
        ns['configure_gallop'](cfg,settings())
        self.assertEqual(cfg.events.reset_base.params['pose_range']['yaw'],(0.,0.))
        self.assertEqual(cfg.observations.policy.gallop_heading.params,{'target_yaw':0.})
        for key in ('yaw_heading','roll_pitch_quiet','front_width','hind_width','forward_progress'):
            self.assertGreater(getattr(cfg.rewards,'gallop_'+key).weight,0.)
        self.assertFalse(hasattr(cfg.rewards,'gallop_upright'));self.assertFalse(hasattr(cfg.rewards,'gallop_stance_center'))
        self.assertEqual(cfg.rewards.gallop_attitude.weight,0.)


if __name__=='__main__':unittest.main()
