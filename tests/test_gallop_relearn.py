"""Walking migration, constraint failures and reset-latched staged learning."""
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
from isaac_task.gallop_core import GallopSettings,RelearnCurriculum,RequestLimitMonitor,transplant_forward_policy
from isaac_task.gallop_runtime import relearn_checkpoint_progress
from tests.test_gallop_core import runner
from tests.test_gallop_guarded import fixture,production_functions
from tests.test_gallop_pair_env import state_class


def settings(**kw):return replace(GallopSettings.load(ROOT/'configs/gallop_relearn.json'),**kw)


def state():
    _,_,env,data,sensor=fixture()
    s=state_class()(env,settings(),None,None,True,True,walking_policy=lambda obs:torch.ones(3,12)*.2)
    env.gallop=s
    return s,env,data,sensor


class RelearnCurriculumTests(unittest.TestCase):
    def window(self,c,step,ok):
        for _ in range(c.cfg.min_episodes):c.record(ok,c.level,{})
        return c.decide(step)

    def test_no_time_promotion_same_speed_constraint_progress_and_stale_generation(self):
        c=RelearnCurriculum(settings(min_episodes=2,decision_steps=10))
        c.decide(1000);self.assertEqual(c.index,0)
        self.window(c,1010,True);self.assertEqual(c.index,0)
        self.window(c,1020,True);self.assertEqual(c.index,1);self.assertEqual(c.cap,1.)
        self.assertLess(c.goal['request_tolerance'],c.plan[0]['request_tolerance'])
        first_generation=c.level
        self.window(c,1030,False);self.assertEqual(c.index,0);self.assertGreater(c.level,first_generation)
        c.record(True,first_generation,{})
        self.assertEqual(c.count,0);self.assertEqual(c.discarded,1)
        self.assertIsNone(c.highest_confirmed_speed)

    def test_complete_plan_confirms_only_full_gallop_speeds_and_caps_at_three(self):
        c=RelearnCurriculum(settings(min_episodes=2,decision_steps=10));step=10
        while c.index<len(c.plan)-1:
            before=c.goal.copy()
            for _ in range(2):self.window(c,step,True);step+=10
            if before['gait']<1:self.assertIsNone(c.highest_confirmed_speed)
            else:self.assertAlmostEqual(c.highest_confirmed_speed,before['speed'])
        self.assertAlmostEqual(c.cap,3.);self.assertAlmostEqual(c.highest_confirmed_speed,2.9)
        for _ in range(2):self.window(c,step,True);step+=10
        self.assertEqual(c.events[-1]['reason'],'at_limit');self.assertAlmostEqual(c.highest_confirmed_speed,3.)
        self.window(c,step,False);self.assertAlmostEqual(c.cap,2.9)

    def test_invalid_profile_fails_and_final_request_limit_is_single_step(self):
        for change in ({'guarded_enabled':False},{'relearn_gallop_start_speed':3.},
                       {'relearn_final_request_tolerance':.2},{'relearn_initial_request_seconds':0.}):
            with self.subTest(change=change),self.assertRaises(ValueError):settings(**change)
        goal=RelearnCurriculum(settings()).plan[-1]
        self.assertEqual(goal['request_seconds'],.02);self.assertEqual(goal['request_fraction'],0.)
        self.assertEqual(goal['request_tolerance'],1e-6)


class RequestFailureTests(unittest.TestCase):
    def test_startup_exemption_consecutive_duration_sticky_failure_and_local_reset(self):
        monitor=RequestLimitMonitor(torch.zeros(2));request=torch.ones(2,12)*2;original=request.clone()
        lo=torch.ones(2,12)*-1;hi=torch.ones(2,12);tol=torch.ones(2)*.1;seconds=torch.ones(2)*.06
        for _ in range(5):monitor.observe(request,lo,hi,tol,seconds,torch.zeros(2,dtype=torch.bool),.02)
        self.assertFalse(monitor.failed.any());self.assertEqual(monitor.seconds.sum(),0.)
        active=torch.ones(2,dtype=torch.bool)
        for _ in range(2):monitor.observe(request,lo,hi,tol,seconds,active,.02)
        self.assertFalse(monitor.failed.any())
        monitor.observe(request,lo,hi,tol,seconds,active,.02);self.assertTrue(monitor.failed.all())
        monitor.observe(request*0,lo,hi,tol,seconds,active,.02);self.assertTrue(monitor.failed.all())
        monitor.reset(torch.tensor([0]));self.assertEqual(monitor.failed.tolist(),[False,True])
        torch.testing.assert_close(request,original)

    def test_final_stage_rejects_first_outside_request_but_not_fast_in_range_request(self):
        monitor=RequestLimitMonitor(torch.zeros(1));lo=torch.ones(1,12)*-1;hi=torch.ones(1,12)
        tol=torch.tensor([1e-6]);seconds=torch.tensor([.02]);active=torch.ones(1,dtype=torch.bool)
        for request in (torch.ones(1,12)*.9,torch.ones(1,12)*-.9):
            monitor.observe(request,lo,hi,tol,seconds,active,.02);self.assertFalse(monitor.failed.item())
        request=torch.zeros(1,12);request[0,7]=1.01
        monitor.observe(request,lo,hi,tol,seconds,active,.02);self.assertTrue(monitor.failed.item())


class RelearnEnvironmentTests(unittest.TestCase):
    def test_stage_zero_executes_student_walking_and_preserves_calf_and_ppo_sample(self):
        s,env,data,_=state();s.startup.stage[:]=3
        cls=production_functions()['BoundedJointPositionAction'];term=cls.__new__(cls)
        term._env=env;term._asset=NS(data=data);term._joint_ids=slice(None)
        sample=torch.ones(3,12)*.6;before=sample.clone();s.begin_step();term.process_actions(sample)
        torch.testing.assert_close(s.guard.requested,torch.ones(3,12)*.3)
        torch.testing.assert_close(sample,before)
        self.assertTrue(((s.guard.target-s.guard.default).abs()<=s.guard.rate*.02+1e-6).all())
        s.walking_policy=lambda obs:torch.ones(3,12)*100
        s.startup.stage[:]=1
        for _ in range(4):s.begin_step();term.process_actions(sample)
        self.assertFalse(s.request_monitor.failed.any())
        s.startup.stage[:]=3
        for _ in range(3):s.begin_step();term.process_actions(sample*100)
        self.assertTrue(s.request_monitor.failed.all())

    def test_local_reset_latches_goals_preserves_other_flight_and_replay_freezes(self):
        s,_,_,_=state();index=next(i for i,g in enumerate(s.curriculum.plan) if g['gait']==.25)
        s.curriculum.restore(index);s.curriculum.level=2
        s.leap.active[1]=True;s.leap.current_distance[1]=.1
        s.reset(torch.tensor([0]))
        torch.testing.assert_close(s.relearn_gait,torch.tensor([.25,0.,0.]))
        torch.testing.assert_close(s.episode_speed,torch.tensor([1.5,1.,1.]))
        torch.testing.assert_close(s.leap.distance_target,torch.tensor([.115,.08,.08]))
        self.assertTrue(s.leap.active[1]);self.assertAlmostEqual(s.leap.current_distance[1].item(),.1)
        last=len(s.curriculum.plan)-1;s.restore_relearn_progress(last,2.9)
        self.assertFalse(s.curriculum_enabled);self.assertTrue((s.relearn_gait==1).all())
        torch.testing.assert_close(s.leap.distance_target,torch.full((3,),.22))
        s.restore_relearn_progress(index,freeze=False);self.assertTrue(s.curriculum_enabled)
        s.age[0]=1
        with self.assertRaises(ValueError):s.restore_relearn_progress(0)

    def test_phase_observation_zero_for_walking_and_scaled_for_partial_gallop(self):
        s,env,_,_=state();s.startup.stage[:]=3;s.phase[:]=.25;s.relearn_gait[:]=torch.tensor([0.,.5,1.])
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        node=next(n for n in tree.body if getattr(n,'name','')=='phase_observation')
        namespace=dict(torch=torch,math=__import__('math'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'phase_observation','exec'),namespace)
        torch.testing.assert_close(namespace['phase_observation'](env).norm(dim=1),s.relearn_gait)

    def test_walking_progress_is_separate_from_final_success_and_early_failures_count(self):
        s,env,_,_=state();s.age[:]=1500;s.first[:]=False;s.startup.stage[:]=3;s.startup.times[:,3]=20
        env.reset_terminated.zero_();env.reset_time_outs[:]=True
        s.sums['points'][:]=1000;s.sums['command'][:]=1000;s.sums['roll2'][:]=1;s.sums['pitch2'][:]=1
        s.guard_sums['intervention'][:]=500
        s.total_impulse[:]=25
        s.complete_episodes([0])
        self.assertEqual(s.last_episode['relearn_stage_success'],1.)
        self.assertEqual(s.last_episode['success'],0.);self.assertEqual(s.last_episode['mature'],0.)
        self.assertEqual(s.curriculum.successes,1)
        s.age[1]=20;s.first[1]=True;env.reset_terminated[1]=True;s.request_monitor.failed[1]=True
        s.complete_episodes([1])
        self.assertEqual(s.last_episode['eligible'],1.);self.assertEqual(s.last_episode['relearn_request_terminated'],1.)
        self.assertEqual(s.last_episode['relearn_stage_success'],0.);self.assertEqual(s.curriculum.count,2)

    def test_full_gallop_requires_original_gates_and_zero_requested_overflow(self):
        s,env,_,_=state();s.restore_relearn_progress(len(s.curriculum.plan)-1,freeze=False)
        s.age[:]=1500;s.first[:]=False;s.startup.stage[:]=3;s.startup.times[:,3]=20
        env.reset_terminated.zero_();env.reset_time_outs[:]=True
        s.sums['points'][:]=1000;s.sums['match'][:]=1000;s.sums['command'][:]=3000
        s.sums['roll2'][:]=1;s.sums['pitch2'][:]=1
        s.cycles[:]=20;s.flight_cycles[:]=20;s.rom_sum[:]=20;s.fore_rom_sum[:]=20;s.fore_participation[:]=20
        s.total_impulse[:]=25;s.events[:]=20;s.good_events[:]=20;s.sequences[:]=20
        s.flight.totals['stable_landings'][:]=20;s.leap.events[:]=20;s.leap.qualified_events[:]=20
        for name in ('front','hind'):
            s.stability_sums[name+'_support'][:]=100;s.stability_sums[name+'_width_good'][:]=100
        s.complete_episodes([0]);self.assertEqual(s.last_episode['success'],1.);self.assertEqual(s.last_episode['relearn_stage_success'],1.)
        s.relearn_request_sum[1]=1
        s.complete_episodes([1]);self.assertEqual(s.last_episode['success'],0.);self.assertEqual(s.last_episode['relearn_stage_success'],0.)
        s.stability_sums['front_width_good'][2]=70
        s.complete_episodes([2]);self.assertEqual(s.last_episode['success'],0.);self.assertEqual(s.last_episode['relearn_stage_success'],0.)


class RelearnMigrationTests(unittest.TestCase):
    def test_real_runner_checkpoint_callback_saves_matching_stage_and_lineage(self):
        tree=ast.parse((ROOT/'scripts/training/gallop_baseline.py').read_text())
        main=next(n for n in tree.body if getattr(n,'name','')=='main')
        callback=next(n for n in main.body if getattr(n,'name','')=='runner_emit')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);checkpoint=out/'model_99.pt';checkpoint.write_bytes(b'weights')
            curriculum=RelearnCurriculum(settings());curriculum.restore(4)
            manifest=dict(migration=dict(parent_cumulative_iterations=4000))
            ns=dict(relearn=True,staged=False,adaptation=None,Path=Path,json=json,out=out,manifest=manifest,env=NS(gallop=NS(curriculum=curriculum)),
                runner=NS(current_learning_iteration=99),sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(),log=NS(emit=lambda *a,**k:None))
            exec(compile(ast.Module(body=[callback],type_ignores=[]),'runner_emit','exec'),ns)
            ns['runner_emit']('CHECKPOINT_SAVED',path=str(checkpoint))
            progress=relearn_checkpoint_progress(checkpoint,json.loads((out/'manifest.json').read_text()))
            self.assertEqual(progress['level'],4);self.assertEqual(progress['cumulative_iterations'],4100)

    def test_49_64_migration_preserves_physical_actions_values_and_noise(self):
        old=runner(45,60);new=runner(49,64)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'walk.pt';torch.save(dict(model_state_dict=old.alg.policy.state_dict(),iter=1999),path)
            info=transplant_forward_policy(new,path,.25,.5)
        actor=torch.randn(3,45);critic=torch.randn(3,60);a=actor.clone();c=critic.clone();a[:,-12:]*=.5;c[:,-12:]*=.5
        before=TensorDict(dict(policy=actor,critic=critic),batch_size=[3])
        after=TensorDict(dict(policy=torch.cat((a,torch.randn(3,4)),1),critic=torch.cat((c,torch.randn(3,4)),1)),batch_size=[3])
        with torch.inference_mode():
            torch.testing.assert_close(.25*old.alg.policy.act_inference(before),.5*new.alg.policy.act_inference(after))
            torch.testing.assert_close(old.alg.policy.evaluate(before),new.alg.policy.evaluate(after))
        torch.testing.assert_close(.25*old.alg.policy.std,.5*new.alg.policy.std)
        self.assertEqual(info['actor_observations'],[45,49]);self.assertFalse(new.alg.optimizer.state)
        self.assertEqual(torch.count_nonzero(new.alg.policy.state_dict()['actor.0.weight'][:,-4:]),0)

    def test_checkpoint_specific_progress_and_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'model_99.pt';p.write_bytes(b'checkpoint')
            record=dict(level=2,speed_cap=1.,cumulative_iterations=4100,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            manifest=dict(trained_relearn_level=12,relearn_checkpoints={p.name:record})
            self.assertEqual(relearn_checkpoint_progress(p,manifest)['level'],2)
            self.assertEqual(relearn_checkpoint_progress(p,manifest)['cumulative_iterations'],4100)
            with self.assertRaises(ValueError):relearn_checkpoint_progress(p,{})
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):relearn_checkpoint_progress(p,manifest)


if __name__=='__main__':unittest.main()
