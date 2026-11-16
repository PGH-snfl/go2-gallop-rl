"""Production reward/sampler wiring under synthetic tensors, not real dynamics."""
import ast
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings
from tests.test_gallop_pair_env import Scene,state_class


def make_state(control_dt=.02):
    n=3
    data=SimpleNamespace(joint_pos=torch.zeros(n,12),default_joint_pos=torch.zeros(n,12),
        soft_joint_pos_limits=torch.tensor([-10.,10.]).expand(n,12,2),
        root_lin_vel_b=torch.tensor([[3.,0.,0.]]).expand(n,3),root_ang_vel_b=torch.zeros(n,3),
        root_quat_w=torch.tensor([[1.,0.,0.,0.]]).expand(n,4),root_pos_w=torch.tensor([[0.,0.,.35]]).expand(n,3).clone(),
        body_pos_w=torch.zeros(n,4,3))
    robot=SimpleNamespace(data=data,find_bodies=lambda *a,**k:(list(range(4)),None),find_joints=lambda *a,**k:(list(range(12)),None))
    sensor=SimpleNamespace(body_names=[leg+'_foot' for leg in ('FL','FR','RL','RR')],data=SimpleNamespace(net_forces_w=torch.zeros(n,4,3)))
    scene=Scene(robot=robot,contact_forces=sensor);scene.env_origins=torch.zeros(n,3)
    env=SimpleNamespace(scene=scene,num_envs=n,device='cpu',step_dt=control_dt,max_episode_length=1000,common_step_counter=0,
        cfg=SimpleNamespace(decimation=round(control_dt/.005)),
        reset_terminated=torch.tensor([False,False,True]),reset_time_outs=torch.zeros(n,dtype=torch.bool),
        command_manager=SimpleNamespace(get_command=lambda name:torch.tensor([[3.,0.,0.]]).expand(n,3)))
    settings=replace(GallopSettings.load(ROOT/'configs/gallop_3mps_leap.json'),metric_warmup_seconds=0.)
    return state_class()(env,settings,None,None,training_targets=False,curriculum_enabled=False),env,data,sensor


class GallopLeapEnvironmentTests(unittest.TestCase):
    def test_actual_reward_term_is_decimation_independent_and_blocks_forelimb_failure_and_falls(self):
        for control_dt in (.01,.02):
            state,env,data,sensor=make_state(control_dt)
            earned=torch.zeros(3)
            for step in range(16//env.cfg.decimation):
                env.common_step_counter=step+1;state.begin_step()
                # Isolate the leap term: eligibility comes from a completed
                # forelimb cycle in production, not this synthetic trajectory.
                state.last_fore_ok[:]=torch.tensor([True,False,True])
                for substep in range(env.cfg.decimation):
                    index=step*env.cfg.decimation+substep+1
                    airborne=5<=index<13
                    data.root_pos_w[:,0]=.015*index
                    data.body_pos_w[:,:,2]=.06 if airborne else .022
                    sensor.data.net_forces_w[:,:,2]=0. if airborne else 100.
                    state.observe_leap_physics(.005)
                state.update_measurements()
                earned+=state.components['leap_distance']*control_dt
            torch.testing.assert_close(earned,torch.tensor([1.,0.,0.]),atol=1e-5,rtol=0)
            state.complete_episodes([0])
            self.assertEqual(state.last_episode['leap_events'],1.)
            self.assertEqual(state.last_episode['leap_physics_samples'],16.)
            self.assertAlmostEqual(state.last_episode['leap_distance_mean'],.12,places=5)
            self.assertIn('leap_last_distance',state.snapshot())
            state.reset(torch.tensor([0]))
            self.assertEqual(state.leap.events[0].item(),0.)
            self.assertEqual(state.leap_reward_sum[0].item(),0.)

    def test_missing_physics_sampling_fails_instead_of_silently_zeroing_reward(self):
        state,env,_,_=make_state();env.common_step_counter=1;state.begin_step()
        with self.assertRaisesRegex(RuntimeError,'physics samples'):state.update_measurements()

    def test_scene_hook_samples_after_updates_and_only_the_physics_loop(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GallopEnv')
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_update_scene_for_leap')
        ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),'scene_hook','exec'),ns)
        calls=[]
        env=SimpleNamespace(_scene_update_without_leap=lambda dt:calls.append('scene'),
                            gallop=SimpleNamespace(observe_leap_physics=lambda dt:calls.append('sample')),
                            _leap_substeps_remaining=4)
        ns['_update_scene_for_leap'](env,0.)
        for _ in range(5):ns['_update_scene_for_leap'](env,.005)
        self.assertEqual(calls,['scene']+['scene','sample']*4+['scene'])
        self.assertEqual(env._leap_substeps_remaining,0)

    def test_step_exception_disables_sampler(self):
        class Base:
            def step(self,action):raise RuntimeError('physics failed')
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GallopEnv')
        ns={'ManagerBasedRLEnv':Base};exec(compile(ast.Module(body=[cls],type_ignores=[]),'env_step','exec'),ns)
        env=ns['GallopEnv'].__new__(ns['GallopEnv'])
        env.gallop=SimpleNamespace(begin_step=lambda:None,leap_enabled=True)
        env.cfg=SimpleNamespace(decimation=4)
        with self.assertRaisesRegex(RuntimeError,'physics failed'):env.step(None)
        self.assertEqual(env._leap_substeps_remaining,0)


if __name__=='__main__':unittest.main()
