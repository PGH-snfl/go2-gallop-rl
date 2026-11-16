"""Verify checkpoint warm starts without any PPO update or optimizer step."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from isaac_task.finetune import restore_finetune
from isaac_task.forward_only import configure_forward_only


class FakeEnv:
    num_envs=2
    num_actions=12

    def get_observations(self):
        return TensorDict({'policy':torch.zeros(2,45),'critic':torch.zeros(2,60)},batch_size=[2])


def runner():
    cfg=dict(num_steps_per_env=2,save_interval=100,obs_groups={'policy':['policy'],'critic':['critic']},
             policy=dict(class_name='ActorCritic',actor_hidden_dims=[8],critic_hidden_dims=[8],
                         actor_obs_normalization=False,critic_obs_normalization=False),
             algorithm=dict(class_name='PPO',learning_rate=.001))
    return OnPolicyRunner(FakeEnv(),cfg,log_dir=None,device='cpu')


class ForwardFineTuneTests(unittest.TestCase):
    def test_actor_critic_adam_and_adaptive_lr_are_restored(self):
        parent=runner()
        for group in parent.alg.optimizer.param_groups:
            group['lr']=3e-5
            for p in group['params']:
                parent.alg.optimizer.state[p]=dict(step=torch.tensor(42.),exp_avg=torch.full_like(p,.01),exp_avg_sq=torch.full_like(p,.02))
        # Fixture optimizer moments are populated as data; no learning is run.
        checkpoint=dict(model_state_dict=parent.alg.policy.state_dict(),optimizer_state_dict=parent.alg.optimizer.state_dict(),iter=1999,infos=None)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(checkpoint,path)
            child=runner();result=restore_finetune(child,path)
        self.assertEqual(child.current_learning_iteration,0)
        self.assertEqual(result['parent_stage_iterations'],2000)
        self.assertEqual(child.alg.learning_rate,3e-5)
        for key,value in checkpoint['model_state_dict'].items():
            torch.testing.assert_close(child.alg.policy.state_dict()[key],value,rtol=0,atol=0)
        restored=child.alg.optimizer.state_dict()
        self.assertEqual(restored['param_groups'],checkpoint['optimizer_state_dict']['param_groups'])
        for key,state in checkpoint['optimizer_state_dict']['state'].items():
            for name,value in state.items():
                torch.testing.assert_close(restored['state'][key][name],value,rtol=0,atol=0)

    def test_empty_optimizer_is_not_silently_treated_as_trained(self):
        parent=runner()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(model_state_dict=parent.alg.policy.state_dict(),optimizer_state_dict=parent.alg.optimizer.state_dict(),iter=0,infos=None),path)
            with self.assertRaisesRegex(ValueError,'trained checkpoint'):
                restore_finetune(runner(),path)

    def test_positive_range_preserves_no_stand_side_or_yaw(self):
        cfg=NS(commands=NS(base_velocity=NS(ranges=NS(),limit_ranges=NS())),curriculum=NS(lin_vel_cmd_levels=object()))
        result=configure_forward_only(cfg,(.2,1.))
        self.assertEqual(result['lin_vel_x'],[.2,1.])
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_y,(0.,0.))
        self.assertEqual(cfg.commands.base_velocity.limit_ranges.ang_vel_z,(0.,0.))
        self.assertEqual(cfg.commands.base_velocity.rel_standing_envs,0.)
        self.assertIsNone(cfg.curriculum.lin_vel_cmd_levels)
        for values in [(0.,1.),(1.,.2),(.2,float('nan'))]:
            with self.assertRaises(ValueError):configure_forward_only(cfg,values)


if __name__=='__main__':unittest.main()
