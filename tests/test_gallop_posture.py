"""Upright and foot-track objectives, reward wiring and all-stage statistics."""
import ast
from dataclasses import replace
import math
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,posture_terms,phase_targets
from tests.test_gallop_leap_env import make_state
from tests.test_gallop_pair_env import state_class


def settings(**changes):
    return replace(GallopSettings.load(ROOT/'configs/gallop_2p8mps_posture.json'),**changes)


def sample():
    q=torch.tensor([[1.,0.,0.,0.]])
    root=torch.tensor([[0.,0.,.35]])
    feet=torch.tensor([[[.2,.15,.02],[.2,-.15,.02],[-.2,.15,.02],[-.2,-.15,.02]]])
    return q,root,feet,torch.ones(1,4,dtype=torch.bool)


class PostureTermsTests(unittest.TestCase):
    def test_roll_pitch_signs_quaternion_sign_and_upside_down_are_penalized(self):
        q,root,feet,c=sample();s=settings()
        level,_=posture_terms(q,root,feet,c,s)
        self.assertEqual(level['upright'].item(),0.)
        for axis in (1,2):
            for angle in (.02,.1,.2,.4,math.pi):
                for sign in (-1,1):
                    quat=torch.zeros_like(q);quat[:,0]=math.cos(angle/2);quat[:,axis]=sign*math.sin(angle/2)
                    terms,diag=posture_terms(quat,root,feet,c,s)
                    expected=min(4.,max(0.,angle-s.upright_tolerance)**2/s.upright_scale**2)
                    self.assertAlmostEqual(terms['upright'].item(),-expected,places=5)
                    self.assertAlmostEqual(diag['tilt'].item(),angle,places=5)
                    neg,_=posture_terms(-quat,root,feet,c,s)
                    torch.testing.assert_close(terms['upright'],neg['upright'])

    def test_width_reward_prefers_body_width_and_rejects_narrow_wide_and_crossed_pairs(self):
        q,root,feet,c=sample();s=settings()
        good,_=posture_terms(q,root,feet,c,s)
        self.assertEqual(good['stance_width'].item(),1.)
        for width in (.05,.18,.45,.6,-.3):
            altered=feet.clone();altered[:,:,1]=torch.tensor([width/2,-width/2,width/2,-width/2])
            terms,diag=posture_terms(q,root,altered,c,s)
            self.assertLess(terms['stance_width'].item(),good['stance_width'].item())
            torch.testing.assert_close(diag['width'],torch.full((1,2),width))
        # A correct rear track does not hide a collapsed front track.
        altered=feet.clone();altered[:,:2,1]=0
        terms,_=posture_terms(q,root,altered,c,s)
        self.assertLess(terms['stance_width'].item(),0.)

    def test_center_penalty_rejects_both_feet_moving_to_the_same_side(self):
        q,root,feet,c=sample();feet[:,:,1]+=.1
        terms,diag=posture_terms(q,root,feet,c,settings())
        self.assertAlmostEqual(terms['stance_width'].item(),1.,places=6)
        self.assertLess(terms['stance_center'].item(),0.)
        torch.testing.assert_close(diag['center'],torch.full((1,2),.1))

    def test_yaw_and_world_translation_do_not_change_costs_or_width(self):
        q,root,feet,c=sample();s=settings()
        baseline,diag=posture_terms(q,root,feet,c,s)
        for yaw in (-2.5,-.6,.8,2.8):
            rotation=torch.tensor([[math.cos(yaw),-math.sin(yaw),0.],[math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])
            offset=torch.tensor([[10.,-20.,0.]])
            quat=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
            terms,moved=posture_terms(quat,root@rotation.T+offset,feet@rotation.T+offset[:,None],c,s)
            for key in baseline:torch.testing.assert_close(terms[key],baseline[key],atol=1e-5,rtol=0)
            torch.testing.assert_close(moved['width'],diag['width'],atol=2e-6,rtol=0)

    def test_swing_preserves_forward_and_vertical_motion_and_relaxes_only_lateral_deadband(self):
        q,root,feet,c=sample();s=settings()
        support,_=posture_terms(q,root,feet,c,s)
        feet[:,:,0]+=torch.tensor([.3,-.2,-.1,.4]);feet[:,:,2]+=torch.tensor([.15,.05,.1,.08])
        swing,_=posture_terms(q,root,feet,~c,s)
        for key in support:torch.testing.assert_close(support[key],swing[key])
        feet[:,:,1]*=2
        wide,_=posture_terms(q,root,feet,~c,s)
        self.assertLess(wide['stance_width'].item(),0.)

    def test_invalid_config_cannot_disable_constraint_or_reintroduce_pitch_target(self):
        for changes in ({'upright_weight':-1},{'upright_scale':0},{'front_stance_width':0},
                        {'stance_width_scale':float('nan')},{'stance_width_tolerance':.4},
                        {'stance_swing_tolerance_multiplier':.5},{'pitch_amplitude':.08}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):settings(**changes)
        self.assertTrue((phase_targets(torch.arange(10)/10,torch.full((10,),2.8),settings())['pitch']==0).all())
        old=GallopSettings.load(ROOT/'configs/gallop_2p8mps_walk_transition.json')
        self.assertFalse(old.posture_enabled);self.assertEqual(old.pitch_amplitude,.08)


def posture_state(dt=.02):
    _,env,data,sensor=make_state(dt)
    data.root_lin_vel_b=torch.zeros(3,3);data.root_lin_vel_w=torch.zeros(3,3)
    data.body_lin_vel_w=torch.zeros(3,4,3)
    data.body_pos_w=sample()[2].repeat(3,1,1)
    data.body_pos_w[:,:,1]+=.07
    data.root_quat_w=torch.tensor([[math.cos(.1),math.sin(.1),0.,0.]]).repeat(3,1)
    env.scene['robot'].root_physx_view=NS(get_masses=lambda:torch.full((3,1),15.))
    env.cfg.sim=NS(gravity=(0.,0.,-9.81))
    command=torch.zeros(3,3);env.command_manager.get_command=lambda name:command
    env.reset_terminated[:]=False
    state=state_class()(env,settings(),None,None,True,False,walking_policy=lambda obs:torch.zeros(3,12))
    return state,env,data,sensor


class PostureWiringTests(unittest.TestCase):
    def test_production_terms_and_statistics_remain_active_before_gallop_and_during_air(self):
        state,env,data,sensor=posture_state()
        state.startup.stage[:]=torch.tensor([0,1,2])
        sensor.data.net_forces_w[:,:,2]=0.  # Airborne, not a reason to turn terms off.
        env.common_step_counter=1;state.begin_step()
        for _ in range(4):state.observe_leap_physics(.005)
        state.update_measurements()
        self.assertTrue((state.components['upright']<0).all())
        self.assertTrue((state.components['stance_center']<0).all())
        self.assertEqual(state.sums['points'].sum().item(),0.)
        # The reward callback can be called repeatedly in a timestep: no double counting.
        state.update_measurements();self.assertEqual(state.posture_sums['points'].tolist(),[1.,1.,1.])
        state.complete_episodes([0])
        self.assertAlmostEqual(state.last_episode['posture_seconds'],.02,places=6)
        self.assertAlmostEqual(state.last_episode['posture_tilt_rms'],.2,places=6)
        self.assertAlmostEqual(state.last_episode['posture_front_width'],.3,places=6)
        self.assertGreater(state.last_episode['posture_front_center_rms'],.06)
        self.assertIn('posture_width',state.snapshot());self.assertIn('Gallop/last_posture_tilt_rms',state.metrics())
        state.reset(torch.tensor([0]))
        self.assertEqual(state.posture_sums['points'].tolist(),[0.,1.,1.])
        self.assertEqual(state.posture_values['tilt'][0].item(),0.)

    def test_reward_rate_integrates_the_same_cost_at_10ms_and_20ms(self):
        totals=[]
        for dt in (.01,.02):
            state,env,_,_=posture_state(dt);earned=torch.zeros(3)
            for i in range(round(.2/dt)):
                env.common_step_counter=i+1;state.begin_step()
                for _ in range(env.cfg.decimation):state.observe_leap_physics(.005)
                state.update_measurements()
                earned+=state.components['upright']*settings().upright_weight*dt
            totals.append(earned)
        torch.testing.assert_close(totals[0],totals[1],atol=1e-6,rtol=0)

    def test_configure_registers_the_three_terms_with_the_intended_signs(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_gallop')
        command=NS(ranges=NS(),limit_ranges=NS())
        cfg=NS(scene=NS(terrain=NS(),contact_forces=NS()),curriculum=NS(),commands=NS(base_velocity=command),
            actions=NS(JointPositionAction=NS()),sim=NS(dt=.005),events=NS(reset_base=NS(params={}),reset_robot_joints=NS(params={})),
            observations=NS(policy=NS(),critic=NS()),terminations=NS(),rewards=NS(**{key:NS(params={}) for key in
                ('track_lin_vel_xy','base_linear_velocity','base_angular_velocity','action_rate')}))
        ns=dict(StartupVelocityCommand=object,BoundedJointPositionAction=object,ObservationTermCfg=NS,RewardTermCfg=NS,
            TerminationTermCfg=NS,gallop_reward=lambda *a:None,phase_observation=lambda *a:None,height_termination=lambda *a:None)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'configure_posture','exec'),ns)
        ns['configure_gallop'](cfg,settings())
        for key,weight in [('upright',6.),('stance_width',2.),('stance_center',1.)]:
            term=getattr(cfg.rewards,'gallop_'+key)
            self.assertEqual(term.weight,weight);self.assertEqual(term.params,{'key':key})


if __name__=='__main__':unittest.main()
