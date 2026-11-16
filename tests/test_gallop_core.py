"""Gallop curriculum, forelimb participation and warm-start invariants. No learning."""
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaac_task.gallop_core import GallopSettings,SuccessSpeedCurriculum,phase_targets,forelimb_success,gait_pattern_success,transplant_forward_policy,restore_gallop_policy
from isaac_task.logging import logged_runner


class FakeEnv:
    num_envs=3;num_actions=12
    def __init__(self,actor,critic):self.actor=actor;self.critic=critic
    def get_observations(self):return TensorDict({'policy':torch.zeros(3,self.actor),'critic':torch.zeros(3,self.critic)},batch_size=[3])


def runner(a,c,runner_class=OnPolicyRunner):
    cfg=dict(num_steps_per_env=4,save_interval=100,obs_groups={'policy':['policy'],'critic':['critic']},
             policy=dict(class_name='ActorCritic',actor_hidden_dims=[8],critic_hidden_dims=[8],actor_obs_normalization=False,critic_obs_normalization=False),
             algorithm=dict(class_name='PPO'))
    return runner_class(FakeEnv(a,c),cfg,log_dir=None,device='cpu')


class GallopCoreTests(unittest.TestCase):
    def test_time_without_success_never_increases_speed(self):
        c=SuccessSpeedCurriculum(GallopSettings())
        c.decide(1000000)
        self.assertEqual(c.cap,1.)
        self.assertEqual(c.events[-1]['reason'],'waiting_for_episodes')

    def test_two_success_windows_promote_and_failure_regresses(self):
        s=replace(GallopSettings(),min_episodes=4,decision_steps=10)
        c=SuccessSpeedCurriculum(s)
        for step in [10,20]:
            for _ in range(4):c.record(True,0,{})
            c.decide(step)
            self.assertEqual(c.cap,1. if step==10 else 1.25)
        for _ in range(4):c.record(True,0,{})  # Old-level evidence must not promote.
        c.decide(30);self.assertEqual(c.cap,1.25);self.assertEqual(c.discarded,4)
        for _ in range(4):c.record(False,1,{'front_load':False})
        c.decide(40);self.assertEqual(c.cap,1.);self.assertEqual(c.events[-1]['reason'],'regress')

    def test_partial_windows_are_retained(self):
        c=SuccessSpeedCurriculum(replace(GallopSettings(),min_episodes=4,decision_steps=10))
        c.record(True,0,{});c.decide(10);self.assertEqual(c.count,1)
        for _ in range(3):c.record(True,0,{})
        c.decide(20);self.assertEqual(c.events[-1]['success_rate'],1.);self.assertEqual(c.cap,1.)

    def test_wrong_individual_order_is_allowed_in_relaxed_profile(self):
        args=[torch.tensor([.7]),torch.tensor([0.]),torch.tensor([12.]),torch.tensor([0.])]
        self.assertTrue(gait_pattern_success(*args,GallopSettings()).item())
        self.assertFalse(gait_pattern_success(*args,replace(GallopSettings(),require_touchdown_order=True)).item())

    def test_hind_only_front_drag_and_one_front_leg_cannot_pass(self):
        load=torch.tensor([[.2,.2,.3,.3],[0.,0.,.5,.5],[.2,.2,.3,.3],[.4,0.,.3,.3],[0.,0.,.5,.5]])
        rom=torch.tensor([[.9,.8],[0.,0.],[0.,0.],[1.,0.],[1.,1.]])
        cycles=torch.tensor([[.9,.9],[0.,0.],[0.,0.],[1.,0.],[1.,1.]])
        flags=forelimb_success(load,rom,cycles,GallopSettings())
        self.assertEqual((flags[0]&flags[1]&flags[2]).tolist(),[True,False,False,False,False])

    def test_targets_include_flight_and_larger_joint_excursions(self):
        phase=torch.linspace(0,1,2001)[:-1]
        t=phase_targets(phase,torch.full_like(phase,3.),GallopSettings())
        self.assertGreater(float((~t['stance'].any(dim=1)).float().mean()),.1)
        self.assertGreater(float(t['thigh'][:,0].max()-t['thigh'][:,0].min()),1.49)
        self.assertGreater(float(t['calf'][:,0].max()-t['calf'][:,0].min()),.79)

    def test_migration_preserves_physical_actions_critic_and_noise(self):
        old=runner(45,60);new=runner(47,62)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'parent.pt';torch.save(dict(model_state_dict=old.alg.policy.state_dict(),iter=1999),path)
            result=transplant_forward_policy(new,path,.25,.5)
        x=torch.randn(3,45);critic=torch.randn(3,60)
        old_obs=TensorDict({'policy':x,'critic':critic},batch_size=[3])
        a=x.clone();c=critic.clone();a[:,-12:]*=.5;c[:,-12:]*=.5
        new_obs=TensorDict({'policy':torch.cat([a,torch.randn(3,2)],1),'critic':torch.cat([c,torch.randn(3,2)],1)},batch_size=[3])
        with torch.inference_mode():
            torch.testing.assert_close(.25*old.alg.policy.act_inference(old_obs),.5*new.alg.policy.act_inference(new_obs),atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(old.alg.policy.evaluate(old_obs),new.alg.policy.evaluate(new_obs),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(.25*old.alg.policy.std,.5*new.alg.policy.std)
        self.assertTrue(result['optimizer_reset']);self.assertFalse(new.alg.optimizer.state)
        self.assertEqual(torch.count_nonzero(new.alg.policy.state_dict()['actor.0.weight'][:,-2:]).item(),0)

    def test_two_mps_start_keeps_reference_curve_and_curriculum_floor(self):
        old=GallopSettings();new=replace(old,start_speed=2.,reference_start_speed=1.,min_episodes=1)
        phase=torch.tensor([.2]);speed=torch.tensor([2.])
        a=phase_targets(phase,speed,old);b=phase_targets(phase,speed,new)
        for key in a:torch.testing.assert_close(a[key],b[key])
        self.assertAlmostEqual(b['period'].item(),.37,places=6)
        c=SuccessSpeedCurriculum(new)
        c.record(False,0,{});c.decide(1000)
        self.assertEqual(c.bounds(),(1.8,2.));self.assertEqual(c.events[-1]['reason'],'hold')

    def test_gallop_finetune_keeps_all_weights_and_starts_fresh_count(self):
        old=runner(47,62);new=runner(47,62)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'parent.pt';torch.save(dict(model_state_dict=old.alg.policy.state_dict(),iter=5999),path)
            info=restore_gallop_policy(new,path)
        for key,value in old.alg.policy.state_dict().items():torch.testing.assert_close(value,new.alg.policy.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(new.alg.optimizer.state);self.assertEqual(new.current_learning_iteration,0)
        self.assertTrue(info['actor_critic_restored']);self.assertTrue(info['noise_restored'])
        self.assertEqual(info['parent_stage_iterations'],6000)

    def test_compact_console_keeps_all_episode_tensorboard_scalars(self):
        model=runner(47,62,logged_runner(OnPolicyRunner,lambda *a,**k:None,compact_episode_log=True))
        with torch.inference_mode():model.alg.policy.act(model.env.get_observations())
        calls=[];model.writer=SimpleNamespace(add_scalar=lambda *args:calls.append(args));model.logger_type='tensorboard'
        locs=dict(it=0,start_iter=0,tot_iter=1,num_learning_iterations=1,collection_time=.2,learn_time=.1,
                  ep_infos=[{'Episode_Reward/gallop_pair_joint':torch.tensor(-.1),'Episode_Termination/time_out':torch.tensor(1.)}],
                  loss_dict={'value_function':.2},rewbuffer=[5.],lenbuffer=[1000.])
        output=StringIO()
        with redirect_stdout(output):model.log(locs)
        self.assertNotIn('Episode_Reward/',output.getvalue());self.assertNotIn('Episode_Termination/',output.getvalue())
        for label in ('Learning iteration','Mean reward:','Mean episode length:','Time elapsed:','ETA:'):self.assertIn(label,output.getvalue())
        self.assertIn('Episode_Reward/gallop_pair_joint',[call[0] for call in calls])
        self.assertIn('Episode_Termination/time_out',[call[0] for call in calls])


if __name__=='__main__':unittest.main()
