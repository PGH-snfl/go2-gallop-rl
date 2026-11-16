"""Fair stage windows and bounded exploration without PPO optimizer updates."""
from dataclasses import replace
import unittest

import torch
from tensordict import TensorDict

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,RelearnCurriculum,RelearnExploration
from isaac_task.gallop_runtime import map_relearn_progress
from tests.test_gallop_core import runner
from tests.test_gallop_pair_env import state_class
from tests.test_gallop_guarded import fixture


def settings(**kwargs):
    return replace(GallopSettings.load(ROOT/'configs/gallop_relearn_recovery.json'),**kwargs)


class RecoveryCurriculumTests(unittest.TestCase):
    def test_rejects_invalid_noise_and_refinement_endpoints(self):
        for change in ({'relearn_noise_std_start':0.},{'relearn_noise_std_end':False},
                       {'relearn_noise_std_end':.2},{'relearn_initial_request_tolerance':.04},
                       {'relearn_final_request_tolerance':.01}):
            with self.subTest(change=change),self.assertRaises(ValueError):settings(**change)

    def test_every_old_checkpoint_stage_maps_to_identical_goal(self):
        old=RelearnCurriculum(GallopSettings.load(ROOT/'configs/gallop_relearn.json'))
        new=RelearnCurriculum(settings())
        self.assertEqual(len(old.plan),24);self.assertEqual(len(new.plan),31)
        for i,goal in enumerate(old.plan):
            result=map_relearn_progress(dict(level=i),dict(relearn_design=dict(plan=old.plan)),new.plan)
            self.assertEqual(result['goal'],goal)
        mapped=map_relearn_progress(dict(level=1),dict(relearn_design=dict(plan=old.plan)),new.plan)
        self.assertEqual(mapped['level'],2)
        for before,after in zip(new.plan,new.plan[1:]):
            self.assertLessEqual(after['request_tolerance'],before['request_tolerance'])
            self.assertLessEqual(after['request_seconds'],before['request_seconds'])
        self.assertEqual(new.plan[-1],old.plan[-1])
        with self.assertRaises(ValueError):map_relearn_progress(dict(level=1),{},new.plan)

    def test_late_successes_count_once_and_prevent_premature_regression(self):
        c=RelearnCurriculum(settings(min_episodes=4,decision_steps=10));c.configure_envs(4);c.restore(2)
        c.record(False,0,{},env_id=0)
        for _ in range(20):c.record(False,0,{},env_id=0)
        event=c.decide(10)
        self.assertEqual(event['reason'],'waiting_for_cohort');self.assertEqual(c.index,2)
        self.assertEqual(c.count,1);self.assertEqual(c.repeated_env_episodes,20)
        for env_id in (1,2,3):c.record(True,0,{},env_id=env_id)
        event=c.decide(20)
        self.assertEqual(event['success_rate'],.75);self.assertEqual(event['reason'],'hold')
        self.assertEqual(event['episodes'],4);self.assertEqual(len(c.cohort_seen),0)

    def test_complete_cohorts_accumulate_with_few_envs_and_require_two_pass_windows(self):
        c=RelearnCurriculum(settings(min_episodes=4,decision_steps=10));c.configure_envs(2)
        for step in (10,20,30,40):
            for env_id in (0,1):c.record(True,c.level,{},env_id=env_id)
            c.decide(step)
            self.assertEqual(c.index,1 if step==40 else 0)
        previous_generation=c.level-1
        c.record(True,previous_generation,{},env_id=0)
        self.assertEqual(c.count,0);self.assertEqual(c.discarded,1)

    def test_real_state_registers_envs_and_records_early_failures(self):
        _,_,env,_,_=fixture()
        state=state_class()(env,settings(),None,None,True,True,walking_policy=lambda obs:torch.ones(3,12)*.2)
        self.assertEqual(state.curriculum.cohort_size,3)
        state.age[:]=1;state.request_monitor.failed[:]=True
        state.complete_episodes(torch.tensor([0,1,2]))
        self.assertEqual(state.curriculum.count,3)
        self.assertEqual(state.curriculum.cohort_seen,{0,1,2})


class RecoveryExplorationTests(unittest.TestCase):
    def exercise(self,logarithmic):
        r=runner(49,64);p=r.alg.policy
        if logarithmic:
            del p.std;p.noise_std_type='log';p.log_std=torch.nn.Parameter(torch.zeros(12))
        c=RelearnCurriculum(settings());c.configure_envs(3);c.restore(2)
        events=[];before={k:v.clone() for k,v in p.state_dict().items() if not k.endswith('std')}
        managed=RelearnExploration(r,c,lambda event,**fields:events.append(fields))
        self.assertFalse(managed.parameter.requires_grad);self.assertEqual(r.alg.entropy_coef,0.)
        self.assertTrue(all(v.requires_grad for v in p.actor.parameters()))
        self.assertTrue(all(v.requires_grad for v in p.critic.parameters()))
        obs=TensorDict(dict(policy=torch.randn(3,49),critic=torch.randn(3,64)),batch_size=[3])
        with torch.inference_mode():
            action=r.alg.act(obs)
            torch.testing.assert_close(p.action_std,torch.full((3,12),.12))
            torch.testing.assert_close(r.alg.transition.actions,action)
            torch.testing.assert_close(r.alg.transition.actions_log_prob,p.get_actions_log_prob(action))
            # A stage change inside collection must not change the likelihood
            # used for the rest of this rollout or its subsequent PPO update.
            c.index=next(i for i,g in enumerate(c.plan) if g['name']=='walk_limits');c.level+=1
            for _ in range(r.num_steps_per_env-1):
                r.alg.act(obs);torch.testing.assert_close(p.action_std,torch.full((3,12),.12))
            r.alg.act(obs);torch.testing.assert_close(p.action_std,torch.full((3,12),.08))
        self.assertEqual(len(events),2);self.assertFalse(r.alg.optimizer.state)
        for key,value in before.items():torch.testing.assert_close(p.state_dict()[key],value)
        values=[]
        for index in range(len(c.plan)):
            c.index=index;managed.apply();values.append(managed.std)
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertAlmostEqual(values[-1],.06)

    def test_scalar_std_and_original_gaussian_samples(self):self.exercise(False)
    def test_log_std_and_original_gaussian_samples(self):self.exercise(True)


if __name__=='__main__':unittest.main()
