"""Managed Gaussian exploration and velocity diagnostics, without PPO updates."""
from dataclasses import replace
from types import SimpleNamespace as NS
import math
import unittest
import hashlib
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO

import torch
from tensordict import TensorDict

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,CycleSpeedTracker,StagedGallopCurriculum
from isaac_task.staged_adaptation import StagedSpeedAdaptation,gaussian_kl
from isaac_task.logging import logged_runner
from isaac_task.gallop_runtime import checkpoint_source_directory
from tests.test_gallop_core import runner
from tests.test_gallop_staged import StagedSuccessTests


def settings(**kw):
    return replace(GallopSettings.load(ROOT/'configs/gallop_staged_speed_adaptation.json'),**kw)


class ManagedExplorationTests(unittest.TestCase):
    def make(self,**kw):
        r=runner(49,64)
        with torch.no_grad():r.alg.policy.std.copy_(torch.linspace(.22,.37,12))
        # The production PPO update/optimizer is never executed by these tests.
        r.alg.update=lambda:(r.alg.storage.clear() or {'noop':1.})
        return r,StagedSpeedAdaptation(r,settings(**kw))

    def test_annealing_retains_parent_joint_ratios_and_never_increases_noise(self):
        r,s=self.make();initial=s.initial.clone()
        for index,mean in [(0,s.initial_mean),(250,(s.initial_mean+.20)/2),(500,.20),(1000,.20)]:
            s.apply(index);actual=r.alg.policy.std.detach()
            self.assertAlmostEqual(float(actual.mean()),mean,places=6)
            torch.testing.assert_close(actual/actual.mean(),initial/initial.mean())
            self.assertTrue((actual<=initial+1e-7).all())
        self.assertFalse(r.alg.policy.std.requires_grad);self.assertEqual(r.alg.entropy_coef,0.)
        self.assertTrue(all(p.requires_grad for p in r.alg.policy.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in r.alg.policy.critic.parameters()))

    def test_noise_changes_only_between_rollouts_and_likelihoods_match_samples(self):
        r,s=self.make(staged_noise_ramp_iterations=2)
        obs=TensorDict(dict(policy=torch.zeros(3,49),critic=torch.zeros(3,64)),batch_size=[3])
        with torch.inference_mode():
            for i in range(12):
                action=r.alg.act(obs)
                expected=s.initial*(s.std/s.initial_mean)
                torch.testing.assert_close(r.alg.transition.action_sigma,expected.expand(3,-1))
                torch.testing.assert_close(r.alg.transition.actions,action)
                torch.testing.assert_close(r.alg.transition.actions_log_prob,r.alg.policy.get_actions_log_prob(action))
                self.assertEqual(s.rollout_index,i//4)
                if i%4==0:first=r.alg.policy.std.clone()
                else:torch.testing.assert_close(r.alg.policy.std,first,rtol=0,atol=0)
        self.assertAlmostEqual(s.std,.20,places=6)

    def test_diagnostics_preserve_rng_policy_buffer_and_original_update_result(self):
        r,s=self.make()
        obs=TensorDict(dict(policy=torch.ones(3,49)*.1,critic=torch.ones(3,64)*.1),batch_size=[3])
        with torch.inference_mode():
            for _ in range(4):
                r.alg.act(obs);r.alg.process_env_step(obs,torch.zeros(3),torch.zeros(3,dtype=torch.bool),{})
        before={k:v.clone() for k,v in r.alg.policy.state_dict().items()}
        mu=r.alg.storage.mu.clone();sigma=r.alg.storage.sigma.clone()
        actions=r.alg.storage.actions.clone();logp=r.alg.storage.actions_log_prob.clone()
        rng=torch.random.get_rng_state().clone();result=r.alg.update()
        self.assertEqual(result,{'noop':1.});self.assertEqual(r.alg.storage.step,0)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(v,r.alg.policy.state_dict()[k]) for k,v in before.items()))
        for actual,expected in [(r.alg.storage.mu,mu),(r.alg.storage.sigma,sigma),(r.alg.storage.actions,actions),(r.alg.storage.actions_log_prob,logp)]:
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertLess(s.diagnostics['rollout_kl_pre_update'],1e-6)
        self.assertLess(s.diagnostics['rollout_kl_post_update'],1e-6)
        self.assertEqual(s.diagnostics['rollout_kl_samples'],12.)

    def test_true_gaussian_kl_and_invalid_scales(self):
        mean=torch.zeros(2,3);std=torch.ones_like(mean)
        torch.testing.assert_close(gaussian_kl(mean,std,mean+1,std),torch.ones(2)*1.5)
        torch.testing.assert_close(gaussian_kl(mean,std,mean,std),torch.zeros(2))
        with self.assertRaises(ValueError):gaussian_kl(mean,std*0,mean,std)

    def test_sampled_rollout_supports_backward_without_updating_parameters(self):
        r,s=self.make(staged_noise_ramp_iterations=2)
        obs=TensorDict(dict(policy=torch.ones(3,49)*.1,critic=torch.ones(3,64)*.1),batch_size=[3])
        with torch.inference_mode():
            for _ in range(4):
                r.alg.act(obs);r.alg.process_env_step(obs,torch.zeros(3),torch.zeros(3,dtype=torch.bool),{})
        batch=r.alg.storage.observations.flatten(0,1)
        old={k:v.clone() for k,v in r.alg.policy.state_dict().items()}
        r.alg.policy.act(batch)
        loss=-r.alg.policy.get_actions_log_prob(r.alg.storage.actions.flatten(0,1)).mean()+r.alg.policy.evaluate(batch).square().mean()
        loss.backward()  # Numerical compatibility only: no PPO/update/optimizer step.
        self.assertTrue(any(p.grad is not None for p in r.alg.policy.actor.parameters()))
        self.assertTrue(any(p.grad is not None for p in r.alg.policy.critic.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in r.alg.policy.parameters() if p.grad is not None))
        self.assertIsNone(r.alg.policy.std.grad)
        self.assertTrue(all(torch.equal(v,r.alg.policy.state_dict()[k]) for k,v in old.items()))

    def test_saved_low_variance_is_not_reexpanded(self):
        r=runner(49,64)
        with torch.no_grad():r.alg.policy.std.fill_(.1)
        r.alg.update=lambda:None
        s=StagedSpeedAdaptation(r,settings());s.apply(500)
        self.assertAlmostEqual(s.std,.1,places=6)

    def test_config_opt_in_and_invalid_schedule(self):
        self.assertFalse(GallopSettings().staged_speed_adaptation_enabled)
        for kw in ({'staged_core_first_enabled':False},{'staged_noise_target_std':0.},
                   {'staged_noise_target_std':float('nan')},{'staged_noise_ramp_iterations':False}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):settings(**kw)

    def test_kl_metrics_reach_tensorboard_and_progress_without_extra_console_rows(self):
        scalars=[];events=[]
        class Base:
            def __init__(self):
                self.cfg={};self.writer=NS(add_scalar=lambda *x:scalars.append(x))
                self.tot_timesteps=12;self.tot_time=1.
            def log(self,locs):print('Learning iteration 0/2')
        cls=logged_runner(Base,lambda event,**kw:events.append((event,kw)),
            metrics=lambda:{'StagedAdapt/rollout_kl_post_update':.01},compact_episode_log=True)
        r=cls();buffer=StringIO()
        with redirect_stdout(buffer):
            r.log(dict(ep_infos=[],it=0,start_iter=0,tot_iter=2,collection_time=.8,learn_time=.2,
                rewbuffer=[1.],loss_dict={'value_function':0.}))
        self.assertEqual(buffer.getvalue(),'Learning iteration 0/2\n')
        self.assertIn(('StagedAdapt/rollout_kl_post_update',.01,0),scalars)
        self.assertEqual(events[-1][1]['diagnostics']['StagedAdapt/rollout_kl_post_update'],.01)

    def test_new_checkpoint_requires_its_bundled_adaptation_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();manifest={}
            for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256'),('staged_adaptation.py','staged_adaptation_source_sha256')]:
                p=source/name;p.write_bytes(name.encode());manifest[key]=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(checkpoint_source_directory(root/'model.pt',manifest),source)
            (source/'staged_adaptation.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):checkpoint_source_directory(root/'model.pt',manifest)


class CycleVelocityTests(unittest.TestCase):
    def test_partial_cycles_do_not_count_and_variation_is_distinct_from_bias(self):
        t=CycleSpeedTracker(torch.zeros(2));yes=torch.ones(2,dtype=torch.bool)
        t.observe(torch.tensor([10.,10.]),torch.tensor([3.,3.]),yes)
        t.finish_cycles(yes);self.assertEqual(t.values['cycles'].sum().item(),0.)
        t.reset(torch.arange(2));t.finish_cycles(yes)
        for vx in (2.,4.):t.observe(torch.tensor([vx,vx]),torch.tensor([3.,3.]),yes)
        t.finish_cycles(yes);d=t.diagnostics()
        torch.testing.assert_close(d['gallop_mean_vx'],torch.tensor([3.,3.]))
        torch.testing.assert_close(d['gallop_mean_vx_error'],torch.zeros(2))
        torch.testing.assert_close(d['gallop_vx_std'],torch.ones(2))
        torch.testing.assert_close(d['gallop_cycle_mean_vx_rmse'],torch.zeros(2))
        self.assertEqual(d['gallop_speed_cycles'].tolist(),[1.,1.])

    def test_unmeasured_reset_and_mixed_environment_cycles_are_excluded(self):
        t=CycleSpeedTracker(torch.zeros(2));yes=torch.ones(2,dtype=torch.bool);t.finish_cycles(yes)
        t.observe(torch.tensor([2.,2.]),torch.tensor([3.,3.]),torch.tensor([True,False]));t.finish_cycles(yes)
        self.assertEqual(t.diagnostics()['gallop_speed_cycles'].tolist(),[1.,0.])
        t.reset(torch.tensor([0]));self.assertEqual(t.diagnostics()['gallop_speed_samples'].tolist(),[0.,0.])

    def test_environment_exports_and_resets_diagnostics_without_changing_gates(self):
        s=StagedSuccessTests().ready(1,staged_core_first_enabled=True,staged_speed_adaptation_enabled=True)
        t=s.speed_diagnostics;yes=torch.ones(3,dtype=torch.bool);t.finish_cycles(yes)
        t.observe(torch.ones(3)*2.6,torch.ones(3)*2.75,yes);t.finish_cycles(yes)
        s.complete_episodes([0]);r=s.last_episode
        self.assertTrue(r['success']);self.assertAlmostEqual(r['gallop_mean_vx_error'],-.15,places=5)
        self.assertIn('StagedAdapt/gallop_cycle_mean_vx_rmse',s.metrics())
        self.assertIn('gallop_speed_samples',s.snapshot())
        s.reset(torch.tensor([0]));self.assertEqual(t.diagnostics()['gallop_speed_samples'][0].item(),0.)

    def test_longer_adaptation_does_not_promote_by_time(self):
        c=StagedGallopCurriculum(settings());c.configure_envs(256);c.restore(1)
        self.assertFalse(c.record(True,0,{},0,8999))
        self.assertTrue(c.record(True,0,{},0,9000))
        e=c.decide(9000);self.assertEqual(e['reason'],'waiting_for_cohort');self.assertEqual(c.index,1)


if __name__=='__main__':unittest.main()
