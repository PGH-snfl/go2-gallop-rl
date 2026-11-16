"""Fixed-speed rollback, physical limits, rate feasibility and low foot lift."""
import ast
from dataclasses import replace
import logging
from types import SimpleNamespace as NS,MethodType
from typing import Sequence
import unittest

import torch

from gpd.paths import ROOT
from gpd.portable import external_path, resolve_go2_asset
from isaac_task.gallop_core import (GallopSettings,GuardedJointController,guarded_physics_limits,
    guarded_foot_height_cost,stabilization_leap_target,SuccessSpeedCurriculum,
    stabilized_student_actions,blend_startup_actions,target_excess_penalty)
from tests.test_gallop_stabilization import state as old_state
from tests.test_gallop_pair_env import state_class


def settings(**kw):return replace(GallopSettings.load(ROOT/'configs/gallop_guarded_2p5.json'),**kw)


def production_functions():
    tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
    nodes=[n for n in tree.body if getattr(n,'name','') in ('BoundedJointPositionAction','apply_guarded_sim_limits','guarded_limit_termination')]
    class Base:
        def process_actions(self,action):self._processed_actions=self._asset.data.default_joint_pos+.5*action
    namespace=dict(torch=torch,JointPositionAction=Base,guarded_physics_limits=guarded_physics_limits,
        stabilized_student_actions=stabilized_student_actions,blend_startup_actions=blend_startup_actions,target_excess_penalty=target_excess_penalty)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'guarded_production','exec'),namespace)
    return namespace


def fixture():
    _,env,data,sensor=old_state();data.default_joint_pos=torch.zeros(3,12)
    data.soft_joint_pos_limits=torch.tensor([-1.,1.]).expand(3,12,2).clone()
    state=state_class()(env,settings(),None,None,False,False,walking_policy=lambda obs:torch.ones(3,12)*.2)
    env.gallop=state;env.obs_buf={'policy':torch.zeros(3,49)};env.action_manager=NS(action=torch.zeros(3,12))
    cls=production_functions()['BoundedJointPositionAction'];term=cls.__new__(cls)
    term._env=env;term._asset=NS(data=data);term._joint_ids=slice(None)
    return term,state,env,data,sensor


class GuardedCoreTests(unittest.TestCase):
    def make(self):
        default=torch.zeros(2,12);limits=torch.tensor([-1.,2.]).expand(2,12,2).clone()
        return GuardedJointController(default,limits,torch.arange(4),torch.arange(8,12),settings())

    def test_extreme_alternating_commands_always_meet_joint_and_rate_bounds(self):
        guard=self.make();previous=guard.default.clone()
        for step in range(500):
            request=torch.full((2,12),1e4 if step%2 else -1e4)
            result=guard.process(request,.02)
            self.assertTrue((result>=guard.lower-1e-7).all() and (result<=guard.upper+1e-7).all())
            self.assertTrue(((result-previous).abs()<=guard.rate*.02+1e-7).all())
            self.assertTrue(torch.isfinite(guard.cost).all());self.assertGreater(guard.cost.min().item(),0.)
            self.assertTrue((request.abs()==1e4).all());previous=result.clone()

    def test_projection_cost_grows_while_executed_target_stays_same(self):
        costs=[]
        for magnitude in (1.,10.,100.):
            guard=self.make();q=guard.process(torch.full((2,12),magnitude),.02)
            torch.testing.assert_close(q[0,:4],torch.ones(4)*.12)
            costs.append(guard.cost[0].item())
        self.assertEqual(costs,sorted(costs));self.assertGreater(costs[2],costs[1])

    def test_actual_excess_is_measured_once_per_step_without_changing_actual_q(self):
        guard=self.make();q=torch.zeros(2,12);q[0,0]=-1.1;before=q.clone()
        for step in range(3):
            guard.observe_actual(q,.02,step);guard.observe_actual(q,.02,step)
        self.assertAlmostEqual(guard.violation_run[0].item(),.06,places=6)
        self.assertGreater(guard.actual_cost[0].item(),0.);self.assertEqual(guard.actual_cost[1].item(),0.)
        torch.testing.assert_close(q,before)
        guard.reset(torch.tensor([0]));self.assertEqual(guard.violation_run[0].item(),0.)
        torch.testing.assert_close(guard.previous[0],guard.default[0])

    def test_fixed_speed_has_no_division_by_zero_and_no_promotion(self):
        cfg=settings();target=stabilization_leap_target(torch.tensor([1.,2.5,3.]),cfg)
        torch.testing.assert_close(target,torch.full((3,),.14))
        curriculum=SuccessSpeedCurriculum(cfg)
        for step in (1000,2000,3000):
            for _ in range(128):curriculum.record(True,curriculum.level,{})
            curriculum.decide(step);self.assertEqual(curriculum.cap,2.5)

    def test_independent_front_hind_height_cost_and_air_clearance(self):
        cfg=settings();height=torch.tensor([[.06,.06,.06,.06],[.20,.20,.06,.06],[.06,.06,.14,.14]])
        front,hind,high=guarded_foot_height_cost(height,cfg)
        self.assertEqual(front[0].item(),0.);self.assertEqual(hind[0].item(),0.)
        self.assertGreater(front[1].item(),0.);self.assertEqual(hind[1].item(),0.)
        self.assertEqual(front[2].item(),0.);self.assertGreater(hind[2].item(),0.)
        self.assertEqual(high[0].sum().item(),0.)

    def test_invalid_profiles_fail(self):
        for kw in ({'bounded_actions_enabled':True},{'max_speed':2.4},{'speed_band':.1},{'guarded_target_margin':.5},
                   {'guarded_sim_limit_factor':1.1},{'guarded_leg_rate':31.},{'swing_height_fast':.2}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):settings(**kw)


class GuardedWiringTests(unittest.TestCase):
    def test_physical_writer_uses_actual_isaac_api_and_does_not_shrink_twice(self):
        path=external_path('IsaacLab')/'source/isaaclab/isaaclab/assets/articulation/articulation.py'
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Articulation')
        method=next(n for n in cls.body if getattr(n,'name','')=='write_joint_position_limit_to_sim')
        namespace=dict(torch=torch,Sequence=Sequence,logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'isaac_api','exec'),namespace)
        limits=torch.tensor([-1.,2.]).expand(2,12,2).clone();data=NS(joint_pos_limits=limits.clone(),default_joint_pos=torch.zeros(2,12),soft_joint_pos_limits=limits.clone())
        backend=NS(limits=limits.clone(),calls=0)
        def setter(value,indices):backend.limits=value.clone();backend.calls+=1
        backend.set_dof_limits=setter;backend.get_dof_limits=lambda:backend.limits
        robot=NS(data=data,_data=data,cfg=NS(soft_joint_pos_limit_factor=1.),root_physx_view=backend,_ALL_INDICES=torch.arange(2))
        robot.write_joint_position_limit_to_sim=MethodType(namespace['write_joint_position_limit_to_sim'],robot)
        original=production_functions()['apply_guarded_sim_limits'](robot,settings())
        torch.testing.assert_close(original,limits)
        expected=guarded_physics_limits(limits,.9)
        torch.testing.assert_close(backend.limits,expected);torch.testing.assert_close(data.soft_joint_pos_limits,expected)
        self.assertEqual(backend.calls,1)

    def test_real_action_path_keeps_ppo_sample_and_executes_filtered_delta(self):
        term,state,env,data,_=fixture();state.startup.stage[:]=3;state.startup.mix[:]=1.
        sample=torch.ones(3,12)*100;before=sample.clone();term.process_actions(sample)
        torch.testing.assert_close(sample,before)
        torch.testing.assert_close(term._processed_actions,state.guard.target)
        torch.testing.assert_close(state.executed_actions*.5+data.default_joint_pos,state.guard.target)
        torch.testing.assert_close(env.action_manager.action,state.executed_actions)
        self.assertGreater(state.action_clip.min().item(),0.);self.assertGreater(state.guard.cost.min().item(),0.)
        self.assertTrue((state.guard.target_rate.abs()<=state.guard.rate+1e-5).all())

    def test_reward_is_active_before_gallop_and_termination_monitors_actual(self):
        term,state,env,data,sensor=fixture();state.startup.stage[:]=1;state.startup.mix[:]=0.
        data.body_pos_w[:,:2,2]=.30;data.body_pos_w[:,2:,2]=.20;sensor.data.net_forces_w[:,:,2]=100
        for step in range(1,4):
            env.common_step_counter=step;state.begin_step();term.process_actions(torch.zeros(3,12))
            data.joint_pos[0,0]=-1.1
            for _ in range(4):state.observe_leap_physics(.005)
            terminated=production_functions()['guarded_limit_termination'](env)
            state.update_measurements()
        self.assertTrue(terminated[0]);self.assertFalse(terminated[1])
        self.assertLess(state.components['guarded_front_height'].max().item(),0.)
        self.assertLess(state.components['guarded_hind_height'].max().item(),0.)
        self.assertIn('guarded_target',state.snapshot())
        state.age[:]=1500;state.sums['points'][:]=1000;state.first[:]=False
        state.guard_sums['actual'][:]=1.;state.complete_episodes([0])
        self.assertEqual(state.last_episode['guarded_limits'],0.);self.assertEqual(state.last_episode['stable'],0.)
        state.reset(torch.tensor([0]));self.assertEqual(state.guard_sums['actual'][0].item(),0.)


if __name__=='__main__':unittest.main()
