"""Stage transitions, per-episode flight physics and checkpoint identity."""
from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings,StagedGallopCurriculum,StagedEpisodeProfile,
    staged_gallop_plan,phase_targets,flight_phase_targets,LeapDistanceTracker)
from isaac_task.gallop_runtime import staged_checkpoint_progress
from tests.test_gallop_guarded import fixture
from tests.test_gallop_pair_env import state_class


def settings(**kw):return replace(GallopSettings.load(ROOT/'configs/gallop_staged_80cm.json'),**kw)


class StagedCurriculumTests(unittest.TestCase):
    def controller(self,n=4,**kwargs):
        c=StagedGallopCurriculum(settings(min_episodes=4,decision_steps=10,staged_min_dwell_seconds=0.,**kwargs))
        c.configure_envs(n);return c

    def window(self,c,step,success=True):
        for env_id in range(c.cohort_size):c.record(success,c.level,{'flight':success},env_id,step)
        return c.decide(step)

    def test_all_nine_stages_require_two_complete_pass_windows_and_stop_at80(self):
        c=self.controller();c.decide(10);self.assertEqual(c.index,0)
        step=20
        for index in range(9):
            self.window(c,step);self.assertEqual(c.index,index)
            self.window(c,step+10);self.assertEqual(c.index,min(index+1,8))
            self.assertEqual(c.highest_confirmed_stage,index);step+=20
        self.assertEqual(c.goal['leap'],.8);self.assertEqual(c.cap,3.)

    def test_fast_repeated_failures_cannot_exclude_late_successes(self):
        c=self.controller();c.restore(4)
        for _ in range(20):c.record(False,0,{},env_id=0,step=10)
        e=c.decide(10);self.assertEqual(e['reason'],'waiting_for_cohort');self.assertEqual(c.count,1)
        for env_id in (1,2,3):c.record(True,0,{},env_id,20)
        e=c.decide(20);self.assertEqual(e['success_rate'],.75);self.assertEqual(c.index,4)
        self.assertEqual(c.repeated_env_episodes,19)

    def test_two_low_windows_regress_one_stage_and_old_generations_do_not_count(self):
        c=self.controller();c.restore(5)
        self.window(c,10,False);self.assertEqual(c.index,5)
        self.window(c,20,False);self.assertEqual(c.index,4)
        self.assertFalse(c.record(True,0,{},0,30));self.assertEqual(c.count,0)
        self.assertEqual(c.discarded,1)

    def test_small_env_complete_cohorts_accumulate_without_repeated_env_bias(self):
        c=self.controller(n=2)
        for step in (10,20,30,40):
            self.window(c,step);self.assertEqual(c.index,1 if step==40 else 0)

    def test_stage_adaptation_excludes_early_episodes_without_time_only_promotion(self):
        c=StagedGallopCurriculum(settings(min_episodes=2,decision_steps=10));c.configure_envs(2)
        self.assertFalse(c.record(True,0,{},0,2999));self.assertEqual(c.count,0)
        c.decide(100000);self.assertEqual(c.index,0)
        self.assertEqual(c.adaptation_discarded,1)

    def test_invalid_staged_combinations_are_rejected(self):
        for kw in ({'staged_max_vx_rmse':float('nan')},{'startup_pose_handover':False},
                   {'max_speed':2.5},{'staged_regress_windows':False},{'fast_period':.6}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):settings(**kw)


class StagedProfileTests(unittest.TestCase):
    def test_first_stage_preserves_original_references_bit_exactly(self):
        s=settings();old=GallopSettings.load(ROOT/'configs/gallop_pilot_18cm.json')
        phase=torch.linspace(0,.99,100);v=torch.linspace(0,2.5,100)
        profile=StagedEpisodeProfile(phase,s,staged_gallop_plan())
        new=phase_targets(phase,v,s,profile=profile);before=phase_targets(phase,v,old)
        for key in new:torch.testing.assert_close(new[key],before[key],rtol=0,atol=0)

    def test_per_env_phase_and_flight_thresholds_match_each_stage(self):
        s=settings();plan=staged_gallop_plan();like=torch.zeros(9)
        profile=StagedEpisodeProfile(like,s,plan)
        for i in range(9):profile.latch(torch.tensor([i]),i)
        speed=torch.tensor([g['speed'] for g in plan]);target=phase_targets(like,speed,s,profile=profile)
        air=(1-profile.shared_air_start)*target['period']
        torch.testing.assert_close(air,torch.tensor([g['air_seconds'] for g in plan]),atol=1e-7,rtol=0)
        self.assertTrue((air>=profile.flight_min_seconds).all())
        self.assertTrue((speed*air>=torch.tensor([g['leap'] for g in plan])).all())
        stages=flight_phase_targets(torch.full_like(like,.7),target['period'],profile)[3]
        self.assertEqual(stages[0].item(),1);self.assertEqual(stages[-1].item(),2)

    def test_physics_qualification_uses_own_stage_air_threshold(self):
        s=settings();pos=torch.tensor([[0.,0.,.35],[0.,0.,.35]])
        profile=StagedEpisodeProfile(torch.zeros(2),s,staged_gallop_plan());profile.latch(torch.tensor([1]),8)
        tracker=LeapDistanceTracker(pos,s);tracker.cfg=profile;tracker.distance_target[:]=.18
        quat=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1);force=torch.ones(2,4)*50;eligible=torch.ones(2,dtype=torch.bool)
        tracker.elapsed[:]=2.;tracker.observe(pos,quat,force,.005,eligible)
        tracker.observe(pos,quat,torch.zeros_like(force),.005,eligible)
        for _ in range(20):pos[:,0]+=.015;tracker.observe(pos,quat,torch.zeros_like(force),.005,eligible)
        tracker.observe(pos,quat,force,.005,eligible)
        self.assertEqual(tracker.qualified_events.tolist(),[1.,0.])

    def test_reset_latches_new_goal_without_changing_other_env_flight(self):
        _,_,env,_,_=fixture();s=state_class()(env,settings(),None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
        s.curriculum.index=8;s.curriculum.cap=3.;s.curriculum.level=1
        s.leap.active[1]=True;s.leap.current_distance[1]=.1
        s.reset(torch.tensor([0]))
        self.assertEqual(s.profile.index.tolist(),[8,0,0])
        torch.testing.assert_close(s.leap.distance_target,torch.tensor([.8,.18,.18]))
        self.assertTrue(s.leap.active[1]);self.assertAlmostEqual(s.leap.current_distance[1].item(),.1)
        self.assertAlmostEqual(s.flight.cfg.flight_max_landing_vz[0].item(),1.8,places=6)
        self.assertAlmostEqual(s.flight.cfg.flight_max_landing_vz[1].item(),1.2,places=6)

    def test_restore_stage_distinguishes_multiple_3mps_goals_and_freezes_evaluation(self):
        _,_,env,_,_=fixture();s=state_class()(env,settings(),None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
        s.restore_staged_progress(6,5)
        self.assertFalse(s.curriculum_enabled);self.assertEqual(s.curriculum.index,6)
        torch.testing.assert_close(s.leap.distance_target,torch.full((3,),.5))
        torch.testing.assert_close(s.startup.goal_speed,torch.full((3,),3.))
        with self.assertRaises(ValueError):s.set_evaluation_speed(3.)

    def test_failed_initial_episode_is_counted_not_silently_excluded(self):
        _,_,env,_,_=fixture();s=state_class()(env,settings(staged_min_dwell_seconds=0.),None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
        s.age[:]=1;env.reset_terminated[:]=True;s.complete_episodes([0,1,2])
        self.assertEqual(s.curriculum.count,3);self.assertEqual(s.curriculum.successes,0)


class StagedSuccessTests(unittest.TestCase):
    def ready(self,index=0,**overrides):
        _,_,env,_,_=fixture();s=state_class()(env,settings(staged_min_dwell_seconds=0.,**overrides),None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
        if index:s.restore_staged_progress(index,freeze=False)
        s.age[:]=1500;s.first[:]=False;s.startup.stage[:]=3;s.startup.times[:,3]=20
        env.reset_terminated.zero_();env.reset_time_outs[:]=True
        s.sums['points'][:]=1000;s.sums['match'][:]=1000;s.sums['command'][:]=1000*s.episode_speed
        s.sums['roll2'][:]=1;s.sums['pitch2'][:]=1
        s.cycles[:]=20;s.flight_cycles[:]=20;s.rom_sum[:]=20;s.fore_rom_sum[:]=20;s.fore_participation[:]=20
        s.total_impulse[:]=25;s.events[:]=20;s.good_events[:]=20;s.sequences[:]=20
        s.flight.totals['stable_landings'][:]=20;s.flight.totals['aborted_landings'][:]=2
        s.leap.events[:]=20;s.leap.qualified_events[:]=16;s.guard_sums['intervention'][:]=1000
        for name in ('front','hind'):
            s.stability_sums[name+'_support'][:]=100;s.stability_sums[name+'_width_good'][:]=100
        return s

    def test_short_measurement_heading_speed_front_width_landing_and_clipping_block_promotion(self):
        for gate in (None,'seconds','heading','speed','width','landing','forelimb','raw_clip','actual'):
            s=self.ready()
            if gate=='seconds':s.sums['points'][:]=450
            if gate=='heading':s.sums['heading2'][:]=10
            if gate=='speed':s.sums['vx2'][:]=90
            if gate=='width':s.stability_sums['front_width_good'][:]=70
            if gate=='landing':s.flight.totals['aborted_landings'][:]=20
            if gate=='forelimb':s.fore_participation[:]=0
            if gate=='raw_clip':s.sums['clip'][:]=200
            if gate=='actual':s.guard_sums['actual'][:]=1
            s.complete_episodes([0]);self.assertEqual(bool(s.last_episode['success']),gate is None,gate)

    def test_final_success_requires80cm_stage_and_keeps100percent_filter_allowance(self):
        for index in (0,8):
            s=self.ready(index);s.complete_episodes([0])
            self.assertTrue(s.last_episode['success'])
            self.assertEqual(bool(s.last_episode['staged_final_success']),index==8)
            self.assertEqual(s.last_episode['guarded_intervention'],1.)


class StagedCheckpointTests(unittest.TestCase):
    def test_production_save_callback_keeps_intermediate_progress_and_iteration_lineage(self):
        tree=ast.parse((ROOT/'scripts/training/gallop_baseline.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='runner_emit')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);plan=staged_gallop_plan()
            curriculum=StagedGallopCurriculum(settings());curriculum.restore(5,4)
            manifest=dict(staged_design=dict(plan=plan),fine_tuning=dict(parent_cumulative_iterations=30500))
            runner=NS(current_learning_iteration=249)
            ns=dict(Path=Path,json=json,staged=True,relearn=False,manifest=manifest,out=out,runner=runner,
                env=NS(gallop=NS(curriculum=curriculum)),log=NS(emit=lambda *a,**kw:None),
                sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
            exec(compile(ast.Module(body=[node],type_ignores=[]),'production_callback','exec'),ns)
            first=out/'model_249.pt';first.write_bytes(b'intermediate')
            ns['runner_emit']('CHECKPOINT_SAVED',path=str(first))
            curriculum.restore(8,7);runner.current_learning_iteration=499
            final=out/'model_final.pt';final.write_bytes(b'final')
            ns['runner_emit']('CHECKPOINT_SAVED',path=str(final))
            self.assertEqual(staged_checkpoint_progress(first,manifest)['index'],5)
            self.assertEqual(staged_checkpoint_progress(first,manifest)['cumulative_iterations'],30750)
            self.assertEqual(staged_checkpoint_progress(final,manifest)['index'],8)
            milestone=out/'stage_passed_05.pt';milestone.write_bytes(b'milestone')
            ns['runner_emit']('CHECKPOINT_SAVED',path=str(milestone))
            self.assertEqual(staged_checkpoint_progress(milestone,manifest)['index'],5)
            self.assertEqual(manifest['trained_staged_index'],8)

    def test_production_stage_emitter_saves_once_only_when_training_promotes(self):
        tree=ast.parse((ROOT/'scripts/training/gallop_baseline.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='stage_emit')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);saved=[]
            def save(path):Path(path).write_bytes(b'policy');saved.append(path)
            ns=dict(staged=True,a=NS(mode='train'),out=out,log=NS(emit=lambda *a,**kw:None),runner=NS(save=save))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'stage_emit','exec'),ns)
            for reason in ('hold','confirming_success','promote_stage','promote_stage'):
                ns['stage_emit']('GALLOP_CURRICULUM',reason=reason,staged_index_before=0)
            self.assertEqual(len(saved),1)
            ns['a'].mode='check';ns['stage_emit']('GALLOP_CURRICULUM',reason='promote_stage',staged_index_before=1)
            self.assertEqual(len(saved),1)

    def test_each_checkpoint_restores_own_goal_and_hash_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'model_500.pt';p.write_bytes(b'checkpoint')
            plan=staged_gallop_plan();record=dict(index=5,goal=plan[5],highest_confirmed_stage=4,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            m=dict(staged_design=dict(plan=plan),staged_checkpoints={p.name:record},trained_staged_index=8)
            self.assertEqual(staged_checkpoint_progress(p,m)['index'],5)
            with self.assertRaises(ValueError):staged_checkpoint_progress(Path(directory)/'model_final.pt',m)
            record['goal']=plan[6]
            with self.assertRaises(ValueError):staged_checkpoint_progress(p,m)
            record['goal']=plan[5];p.write_bytes(b'changed')
            with self.assertRaises(ValueError):staged_checkpoint_progress(p,m)


if __name__=='__main__':unittest.main()
