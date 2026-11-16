"""Range, PPO latent integrity, startup endpoints and physical saturation gates."""
import ast
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import tempfile
import json
from pathlib import Path

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings,bounded_joint_targets,bounded_startup_targets,
    bounded_limit_cost,stabilized_student_actions,blend_startup_actions,target_excess_penalty)
from isaac_task.gallop_adaptation import legacy_student_targets,legacy_distillation_targets,inverse_bounded_targets,imitation_loss,measured_gait
from tests.test_gallop_stabilization import state as legacy_state
from tests.test_gallop_pair_env import state_class


def settings(**kw):return replace(GallopSettings.load(ROOT/'configs/gallop_bounded.json'),**kw)


def action_class():
    tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoundedJointPositionAction')
    class Base:
        def process_actions(self,a):self._raw_actions=a.clone();self._processed_actions=a*.5
    namespace=dict(torch=torch,JointPositionAction=Base,bounded_startup_targets=bounded_startup_targets,
        bounded_limit_cost=bounded_limit_cost,stabilized_student_actions=stabilized_student_actions,
        blend_startup_actions=blend_startup_actions,target_excess_penalty=target_excess_penalty)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'bounded_action','exec'),namespace)
    return namespace['BoundedJointPositionAction']


def fixture():
    _,env,data,sensor=legacy_state()
    data.default_joint_pos=torch.zeros(3,12)
    data.soft_joint_pos_limits=torch.tensor([-1.,2.]).expand(3,12,2).clone()
    data.computed_torque=torch.zeros(3,12);data.applied_torque=torch.zeros(3,12)
    s=state_class()(env,settings(),None,None,False,False,walking_policy=lambda obs:torch.ones(3,12)*.2)
    env.gallop=s;env.obs_buf={'policy':torch.zeros(3,49)};env.action_manager=NS(action=torch.zeros(3,12))
    cls=action_class();term=cls.__new__(cls);term._env=env;term._asset=NS(data=data);term._joint_ids=slice(None)
    return term,s,env,data,sensor


class BoundedMathTests(unittest.TestCase):
    def test_extreme_latents_finite_and_inside_asymmetric_per_joint_ranges(self):
        limits=torch.stack((torch.linspace(-3,-1,12),torch.linspace(-.5,2,12)),dim=-1)
        x=torch.linspace(-1e6,1e6,1200).reshape(100,12)
        q=bounded_joint_targets(x,limits)
        self.assertTrue(torch.isfinite(q).all());self.assertTrue((q>=limits[:,0]-1e-6).all());self.assertTrue((q<=limits[:,1]+1e-6).all())
        torch.testing.assert_close(bounded_joint_targets(torch.zeros(1,12),limits),limits.mean(-1)[None])

    def test_hold_walk_blend_and_gallop_remain_within_limits(self):
        limits=torch.tensor([-2.,1.]).expand(4,12,2);default=torch.zeros(4,12)
        latent=torch.ones(4,12)*20;walking=torch.ones(4,12)*-100
        stage=torch.tensor([0,1,2,3]);mix=torch.tensor([0.,0.,.3,1.])
        q=bounded_startup_targets(latent,walking,default,limits,stage,mix,settings())
        torch.testing.assert_close(q[:,0],torch.tensor([0.,-2.,-1.1,1.]))
        self.assertTrue((q>=-2).all() and (q<=1).all())

    def test_inverse_preserves_old_executed_targets_including_boundaries(self):
        limits=torch.tensor([-2.,1.]).expand(12,2)
        q=torch.linspace(-2,1,12)[None]
        reconstructed=bounded_joint_targets(inverse_bounded_targets(q,limits),limits)
        self.assertLessEqual(float((reconstructed-q).abs().max()),.00151)

    def test_old_scaling_is_part_of_teacher_target_not_new_latent(self):
        x=torch.ones(1,12)*2;limits=torch.tensor([-10.,10.]).expand(12,2);calves=torch.arange(8,12)
        old=legacy_student_targets(x,torch.zeros(12),limits,calves,torch.ones(1),settings())
        torch.testing.assert_close(old[0,:8],torch.ones(8));torch.testing.assert_close(old[0,8:],torch.ones(4)*.85)
        new=bounded_joint_targets(torch.ones(1,12),limits)
        torch.testing.assert_close(new[0,:4],new[0,8:])

    def test_blend_distillation_matches_actual_old_clipped_target_when_representable(self):
        x=torch.ones(3,12)*2;walking=torch.ones_like(x)*-.4;limits=torch.tensor([-1.,1.]).expand(12,2)
        default=torch.zeros(12);mix=torch.tensor([.25,.5,1.]);stage=torch.tensor([2,2,3]);calves=torch.arange(8,12)
        target=legacy_distillation_targets(x,walking,default,limits,calves,stage,mix,settings())
        new=(1-mix[:,None])*(walking*.5)+mix[:,None]*target
        old=blend_startup_actions(stabilized_student_actions(x,calves,mix,settings()),walking,stage,mix)*.5
        torch.testing.assert_close(new,old.clamp(-1,1))

    def test_near_limit_cost_detects_saturation_without_clipping_and_weights_calves(self):
        limits=torch.tensor([-1.,1.]).expand(12,2);calves=torch.arange(8,12)
        center=torch.zeros(1,12);edge=center.clone();edge[0,8]=1.
        cost,frac=bounded_limit_cost(edge,limits,calves,.05)
        self.assertGreater(cost.item(),0.);self.assertAlmostEqual(frac.item(),1/12)
        self.assertEqual(bounded_limit_cost(center,limits,calves,.05)[0].item(),0.)
        thigh=center.clone();thigh[0,0]=1.
        self.assertAlmostEqual(cost.item(),2*bounded_limit_cost(thigh,limits,calves,.05)[0].item())

    def test_imitation_loss_has_finite_nonzero_gradient_at_boundary_targets_without_update(self):
        actor=torch.nn.Linear(49,12);before={k:v.clone() for k,v in actor.state_dict().items()}
        x=torch.ones(16,49);target=torch.ones(16,12);limits=torch.tensor([-1.,1.]).expand(12,2)
        loss=imitation_loss(actor,x,target,limits);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(actor.weight.grad.norm()),0.)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in actor.state_dict().items()))

    def test_invalid_settings_rejected(self):
        for changes in ({'stabilization_enabled':False},{'bounded_limit_margin':.5},{'bounded_latent_std':0.},
                        {'bounded_max_limit_fraction':2.},{'bounded_target_rate_scale':0.}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):settings(**changes)

    def test_preservation_measurement_requires_real_flight_and_forelimb_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'episodes.jsonl'
            good=dict(gallop_metric_seconds=20,pattern=1,flight=1,motion=1,front_load=1,front_motion=1,front_cycle=1)
            rows=[good,dict(good,flight=0),dict(good,front_load=0),dict(good,gallop_metric_seconds=0)]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            measured=measured_gait(p)
            self.assertEqual(measured['measured_episodes'],3);self.assertAlmostEqual(measured['gait_fraction'],1/3)


class BoundedWiringTests(unittest.TestCase):
    def test_real_ppo_transition_keeps_latent_samples_and_log_probabilities(self):
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict
        from scripts.analysis.collect_unitree_baseline import load_yaml
        class Probe:
            num_envs=3;num_actions=12
            def get_observations(self):return TensorDict(dict(policy=torch.zeros(3,49),critic=torch.zeros(3,64)),batch_size=[3])
        runner=OnPolicyRunner(Probe(),load_yaml(ROOT/'runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/params/agent.yaml'),log_dir=None,device='cpu')
        term,s,_,_,_=fixture();s.startup.stage[:]=3;s.startup.mix[:]=1.
        obs=Probe().get_observations()
        with torch.no_grad():
            latent=runner.alg.act(obs);before=latent.clone();logp=runner.alg.transition.actions_log_prob.clone()
            term.process_actions(latent)
            torch.testing.assert_close(runner.alg.transition.actions,before)
            torch.testing.assert_close(runner.alg.transition.actions_log_prob,logp)
            torch.testing.assert_close(runner.alg.policy.get_actions_log_prob(latent),logp)
        self.assertFalse(runner.alg.optimizer.state)

    def test_sampled_latent_unchanged_and_previous_action_represents_executed_target(self):
        term,s,env,data,_=fixture();s.startup.stage[:]=torch.tensor([0,1,3]);s.startup.mix[:]=torch.tensor([0.,0.,1.])
        latent=torch.ones(3,12)*4;before=latent.clone();term.process_actions(latent)
        torch.testing.assert_close(latent,before);torch.testing.assert_close(term._raw_actions,before)
        torch.testing.assert_close(term._processed_actions[0],data.default_joint_pos[0])
        torch.testing.assert_close(term._processed_actions[1],torch.ones(12)*.1)
        torch.testing.assert_close(s.executed_actions*.5+data.default_joint_pos,term._processed_actions)
        torch.testing.assert_close(env.action_manager.action,s.executed_actions)
        self.assertEqual(s.action_clip.sum().item(),0.);self.assertGreater(s.bounded_near_fraction[2].item(),.99)
        self.assertEqual(s.bounded_target_rate.abs().sum().item(),0.)
        term.process_actions(latent*0)
        self.assertGreater(s.bounded_rate_cost[2].item(),0.)

    def test_legacy_rollout_retains_preclip_observation_and_reset_clears_rate(self):
        term,s,_,data,_=fixture();s.bounded_teacher_rollout=True;s.startup.stage[:]=3;s.startup.mix[:]=1.
        latent=torch.ones(3,12)*10;term.process_actions(latent)
        expected=latent.clone();expected[:,s.calf_ids]*=.85
        torch.testing.assert_close(s.executed_actions,expected);self.assertGreater(s.action_clip.min().item(),0.)
        torch.testing.assert_close(s.bounded_target,term._processed_actions)
        s.bounded_previous_valid[:]=True;s.bounded_target_rate[:]=5;s.reset(torch.tensor([0]))
        self.assertFalse(s.bounded_previous_valid[0]);self.assertTrue(s.bounded_previous_valid[1])
        self.assertEqual(s.bounded_target_rate[0].sum().item(),0.)
        torch.testing.assert_close(s.bounded_target[0],data.default_joint_pos[0])

    def test_reward_and_measured_success_gate_do_not_treat_zero_clip_as_success(self):
        term,s,env,data,sensor=fixture();s.startup.stage[:]=3;s.startup.mix[:]=1.;s.startup.times[:,3]=1.1
        sensor.data.net_forces_w[:,:,2]=100
        for step in range(2):
            env.common_step_counter=step+1;s.begin_step();term.process_actions(torch.ones(3,12)*20)
            for _ in range(4):s.observe_leap_physics(.005)
            s.update_measurements()
        self.assertGreater(s.bounded_sums['near'].min().item(),0.)
        self.assertLess(s.components['bounded_near_limit'].max().item(),0.)
        self.assertIn('bounded_target',s.snapshot())
        s.age[:]=1500;s.first[:]=False;s.sums['points'][:]=1000;s.sums['clip'][:]=0;s.bounded_sums['near'][:]=500
        s.complete_episodes([0]);self.assertEqual(s.last_episode['bounded_limits'],0.)
        self.assertEqual(s.last_episode['stable'],0.);self.assertEqual(s.last_episode['success'],0.)
        self.assertIn('bounded_tracking_rmse',s.last_episode)

    def test_collector_includes_new_gate(self):
        from scripts.analysis.collect_gallop import episode_stats,CRITERIA
        row={k:1. for k in CRITERIA};row.update(success=0.,bounded_limits=0.)
        from scripts.analysis.collect_gallop import MEASURES
        row.update({k:0. for k in MEASURES})
        result=episode_stats([row]);self.assertEqual(result['criterion_pass_rates']['bounded_limits'],0.)
        self.assertIn('bounded_limits',next(iter(result['failure_combinations'])))


if __name__=='__main__':unittest.main()
