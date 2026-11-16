"""Stationary/walking/Gallop handover, executed actions and model compatibility."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch
from tensordict import TensorDict

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings, WalkingToGallop, FrozenWalkingPolicy,
    blend_startup_actions, phase_targets)
from isaac_task.gallop_runtime import checkpoint_source_directory
from tests.test_gallop_leap_env import make_state
from tests.test_gallop_pair_env import state_class


def settings():return GallopSettings.load(ROOT/'configs/gallop_2p8mps_walk_transition.json')


def advance(controller,seconds,velocity=None,tilt=0.,active=None):
    n=len(controller.stage)
    for _ in range(round(seconds/.02)):
        v=torch.zeros(n,3);v[:,0]=controller.command if velocity is None else velocity
        rpy=torch.zeros(n,3);rpy[:,0]=tilt
        controller.advance(.02,v,torch.zeros(n,3),rpy,torch.full((n,),.35),
                           torch.ones(n,dtype=torch.bool) if active is None else active)


class StartupTests(unittest.TestCase):
    def test_command_and_action_weight_ramps_are_continuous_and_finish_at_2p8(self):
        c=WalkingToGallop(torch.zeros(1),settings());commands=[];mix=[];stages=[]
        for _ in range(400):
            commands.append(c.command.item());mix.append(c.mix.item());stages.append(c.stage.item())
            advance(c,.02)
        self.assertEqual(set(stages),{0,1,2,3})
        self.assertEqual(c.stage.item(),3);self.assertAlmostEqual(c.command.item(),2.8,places=5)
        delta=torch.diff(torch.tensor(commands))
        self.assertGreaterEqual(delta.min().item(),-1e-6);self.assertLess(delta.max().item(),.021)
        self.assertLess(torch.diff(torch.tensor(mix)).abs().max().item(),.011)
        self.assertGreaterEqual(c.times[0,0].item(),.5)
        self.assertGreaterEqual(c.times[0,1].item(),3.)
        self.assertGreaterEqual(c.times[0,2].item(),3.-1e-5)
        t=phase_targets(torch.zeros(1),torch.tensor([2.8]),settings())
        self.assertAlmostEqual(t['period'].item(),.33,places=5)
        self.assertGreaterEqual(2.8*.2*t['period'].item(),settings().leap_min_distance)

    def test_bad_posture_or_missing_forward_tracking_prevents_handover(self):
        c=WalkingToGallop(torch.zeros(1),settings())
        advance(c,2.,tilt=.4);self.assertEqual(c.stage.item(),0)
        advance(c,.6);self.assertEqual(c.stage.item(),1)
        advance(c,5.,velocity=0.);self.assertEqual(c.stage.item(),1)
        advance(c,.6,velocity=1.);self.assertEqual(c.stage.item(),2)
        before_mix=c.mix.clone();before_command=c.command.clone()
        advance(c,2.,tilt=.4);torch.testing.assert_close(c.mix,before_mix)
        torch.testing.assert_close(c.command,before_command)
        advance(c,3.1);self.assertEqual(c.stage.item(),3)

    def test_reset_is_local_and_inactive_newborn_environments_do_not_advance(self):
        c=WalkingToGallop(torch.zeros(2),settings())
        advance(c,8.,active=torch.tensor([True,False]))
        self.assertEqual(c.stage.tolist(),[3,0]);self.assertEqual(c.times[1].sum().item(),0.)
        c.reset(torch.tensor([0]));self.assertEqual(c.command.tolist(),[0.,0.])
        self.assertEqual(c.times.sum().item(),0.)

    def test_action_mixture_has_exact_teacher_student_and_stationary_endpoints(self):
        student=torch.ones(4,12);teacher=torch.full((4,12),-.2)
        stage=torch.tensor([0,1,2,3]);mix=torch.tensor([0.,0.,.5,1.])
        actual=blend_startup_actions(student,teacher,stage,mix)
        torch.testing.assert_close(actual[:,0],torch.tensor([0.,-.2,.4,1.]))
        self.assertTrue((student==1).all())

    def test_walking_teacher_matches_original_rsl_actor_and_previous_action_units(self):
        from rsl_rl.runners import OnPolicyRunner
        from scripts.analysis.collect_unitree_baseline import load_yaml
        checkpoint=ROOT/'runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt'
        class Probe:
            num_envs=4;num_actions=12
            def get_observations(self):return TensorDict(dict(policy=torch.zeros(4,45),critic=torch.zeros(4,60)),batch_size=[4])
        cfg=load_yaml(checkpoint.parent/'params/agent.yaml')
        runner=OnPolicyRunner(Probe(),cfg,log_dir=None,device='cpu')
        runner.load(str(checkpoint),load_optimizer=False,map_location='cpu')
        teacher=FrozenWalkingPolicy(checkpoint,'cpu')
        obs=torch.randn(4,47)*.1;baseline=obs[:,:45].clone();baseline[:,33:45]*=2
        with torch.no_grad():expected=runner.alg.policy.act_inference(TensorDict(dict(policy=baseline),batch_size=[4]))*.5
        torch.testing.assert_close(teacher(obs),expected,rtol=0,atol=1e-7)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))

    def test_missing_or_modified_walking_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'source').mkdir();manifest={}
            for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]:
                f=p/'source'/name;f.write_text('# source');manifest[key]=hashlib.sha256(f.read_bytes()).hexdigest()
            teacher=p/'source/walking_teacher.pt';teacher.write_bytes(b'teacher')
            manifest['walking_teacher']={'sha256':hashlib.sha256(teacher.read_bytes()).hexdigest()}
            checkpoint_source_directory(p/'model_final.pt',manifest)
            teacher.write_bytes(b'changed')
            with self.assertRaises(ValueError):checkpoint_source_directory(p/'model_final.pt',manifest)

    def test_invalid_schedule_or_target_speed_is_rejected(self):
        for values in [{'startup_blend_seconds':0},{'startup_walk_min_seconds':1.},
                       {'startup_episode_seconds':5.},{'speed_band':.1},{'startup_walk_speed':3.}]:
            with self.assertRaises(ValueError):replace(settings(),**values)


class StartupWiringTests(unittest.TestCase):
    def test_action_manager_observes_executed_actions_and_preserves_sampled_student_actions(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoundedJointPositionAction')
        class Base:
            def process_actions(self,actions):self._processed_actions=actions*.5
        ns=dict(torch=torch,JointPositionAction=Base,blend_startup_actions=blend_startup_actions)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'bounded_actions','exec'),ns)
        term=ns['BoundedJointPositionAction'].__new__(ns['BoundedJointPositionAction'])
        student=torch.ones(4,12)
        startup=NS(stage=torch.tensor([0,1,2,3]),mix=torch.tensor([0.,0.,.5,1.]))
        state=NS(startup=startup,walking_policy=lambda obs:torch.full((4,12),.2),executed_actions=torch.zeros(4,12),cfg=NS(stabilization_enabled=False))
        term._env=NS(gallop=state,obs_buf={'policy':torch.zeros(4,47)},action_manager=NS(action=student.clone()))
        term._asset=NS(data=NS(soft_joint_pos_limits=torch.tensor([-10.,10.]).expand(4,12,2)))
        term._joint_ids=slice(None)
        term.process_actions(student)
        torch.testing.assert_close(term._env.action_manager.action[:,0],torch.tensor([0.,.2,.6,1.]))
        torch.testing.assert_close(term._processed_actions,term._env.action_manager.action*.5)
        self.assertTrue((student==1).all())

    def test_command_update_occurs_before_next_observation_and_reset_command_is_zero(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StartupVelocityCommand')
        class Base:
            def _resample_command(self,ids):self.vel_command_b[ids]=9.;self.is_standing_env[ids]=True
        ns=dict(torch=torch,math=__import__('math'),UniformVelocityCommand=Base,
                euler_xyz_from_quat=lambda q:(torch.zeros(len(q)),)*3)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'startup_command','exec'),ns)
        term=ns['StartupVelocityCommand'].__new__(ns['StartupVelocityCommand'])
        c=WalkingToGallop(torch.zeros(1),settings())
        data=NS(root_lin_vel_b=torch.zeros(1,3),root_ang_vel_b=torch.zeros(1,3),
                root_pos_w=torch.tensor([[0.,0.,.35]]),root_quat_w=torch.tensor([[1.,0.,0.,0.]]))
        term._env=NS(gallop=NS(startup=c,robot=NS(data=data),age=torch.zeros(1)),step_dt=.02,scene=NS(env_origins=torch.zeros(1,3)))
        term.vel_command_b=torch.ones(1,3);term.is_standing_env=torch.ones(1,dtype=torch.bool)
        term._resample_command(torch.tensor([0]));self.assertEqual(term.vel_command_b.sum().item(),0.)
        term._update_command();self.assertEqual(c.times.sum().item(),0.)
        term._env.gallop.age[:]=1
        for _ in range(350):
            data.root_lin_vel_b[:,0]=c.command
            term._update_command()
            torch.testing.assert_close(term.vel_command_b[:,0],c.command)
        self.assertEqual(c.stage.item(),3)

    def test_gallop_metrics_and_leaps_exclude_walking_but_balance_is_active(self):
        _,env,data,sensor=make_state()
        data.root_lin_vel_w=data.root_lin_vel_b.clone();data.body_lin_vel_w=torch.zeros(3,4,3)
        env.scene['robot'].root_physx_view=NS(get_masses=lambda:torch.full((3,1),15.))
        env.cfg.sim=NS(gravity=(0.,0.,-9.81))
        command=torch.tensor([[1.,0.,0.]]).repeat(3,1)
        env.command_manager.get_command=lambda name:command
        state=state_class()(env,settings(),None,None,False,False,walking_policy=lambda obs:torch.zeros(3,12))
        state.startup.stage[:]=1
        for step in range(1,151):
            if step==101:
                state.startup.stage[:]=3;state.startup.times[:,3]=1.1;command[:,0]=2.8
            env.common_step_counter=step;state.begin_step()
            target=phase_targets(state.phase,command[:,0],settings())
            data.joint_pos=torch.stack((torch.zeros_like(target['thigh']),target['thigh'],target['calf']),dim=2).reshape(3,12)
            sensor.data.net_forces_w[:,:,2]=target['stance'].float()*50
            for substep in range(4):state.observe_leap_physics(.005)
            state.update_measurements()
            if step==100:
                self.assertEqual(state.cycles.sum().item(),0.)
                self.assertEqual(state.sums['points'].sum().item(),0.)
                self.assertEqual(state.leap.events.sum().item(),0.)
                self.assertTrue((state.components['flight_support']==0).all())
        self.assertTrue((state.cycles>0).all())
        self.assertTrue((state.sums['points']>0).all())
        self.assertIn('executed_action',state.snapshot())
        state.reset(torch.tensor([0]));self.assertEqual(state.startup.stage[0].item(),0)
        self.assertEqual(state.startup.stage[1].item(),3)


if __name__=='__main__':unittest.main()
