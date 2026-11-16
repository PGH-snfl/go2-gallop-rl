"""Real RSL checkpoints can be saved before learn() without PPO updates."""
from pathlib import Path
import tempfile
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT
from isaac_task.logging import logged_runner
from scripts.analysis.collect_unitree_baseline import load_yaml


class Probe:
    num_envs=2;num_actions=12
    def get_observations(self):
        return TensorDict(dict(policy=torch.zeros(2,49),critic=torch.zeros(2,64)),batch_size=[2])


class PreLearningCheckpointTests(unittest.TestCase):
    def runner(self,directory,events,default_logger=False):
        cfg=load_yaml(ROOT/'runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/params/agent.yaml')
        if default_logger:cfg.pop('logger',None)
        return logged_runner(OnPolicyRunner,lambda event,**fields:events.append((event,fields)))(
            Probe(),cfg,log_dir=directory,device='cpu')

    def test_real_failed_adaptation_checkpoint_saves_before_learn(self):
        source=ROOT/'runs/isaac/train/unitree_go2_gallop/20261103_100059_9500d2/model_adapted.pt'
        with tempfile.TemporaryDirectory() as directory:
            events=[];runner=self.runner(directory,events)
            if source.exists():runner.load(str(source),load_optimizer=False,map_location='cpu')
            before={k:v.clone() for k,v in runner.alg.policy.state_dict().items()}
            path=Path(directory)/'model_adapted.pt'
            runner.save(str(path))
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(all(torch.equal(v,saved['model_state_dict'][k]) for k,v in before.items()))
            self.assertFalse(saved['optimizer_state_dict']['state'])
            self.assertEqual(saved['iter'],0);self.assertIsNone(runner.writer)
            self.assertEqual(events[-1],('CHECKPOINT_SAVED',dict(path=str(path))))

    def test_prelearning_save_preserves_lazy_tensorboard_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=self.runner(directory,[],default_logger=True)
            runner.save(str(Path(directory)/'before.pt'))
            self.assertEqual(runner.logger_type,'tensorboard');self.assertIsNone(runner.writer)
            runner._prepare_logging_writer();writer=runner.writer
            self.assertIsNotNone(writer)
            runner.save(str(Path(directory)/'after.pt'));runner._prepare_logging_writer()
            self.assertIs(runner.writer,writer);writer.flush();writer.close()
            self.assertFalse(runner.alg.optimizer.state)

    def test_remote_writer_is_prepared_once_without_network(self):
        class Writer:
            def __init__(self):self.saves=[]
            def save_model(self,path,iteration):self.saves.append(path)
        class Base:
            def __init__(self,kind,disabled=False):
                self.cfg={'logger':kind};self.writer=None;self.disable_logs=disabled;self.preparations=0
            def _prepare_logging_writer(self):self.preparations+=1;self.writer=Writer()
            def save(self,path):
                if self.logger_type in ('neptune','wandb') and not self.disable_logs:self.writer.save_model(path,0)
        for kind in ('WandB','Neptune'):
            runner=logged_runner(Base,lambda *a,**k:None)(kind)
            runner.save('before.pt');runner.save('after.pt')
            self.assertEqual(runner.preparations,1);self.assertEqual(len(runner.writer.saves),2)
        runner=logged_runner(Base,lambda *a,**k:None)('wandb',True);runner.save('rank1.pt')
        self.assertEqual(runner.preparations,0)


if __name__=='__main__':unittest.main()
