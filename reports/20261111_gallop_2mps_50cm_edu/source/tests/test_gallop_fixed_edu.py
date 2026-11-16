"""Fixed goal, actual-velocity monitoring, reset independence and flight-height credit."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,OfficialJointMonitor,GuardedJointController,phase_targets,stabilization_leap_target,SuccessSpeedCurriculum,FixedGallopGoal
from tests.test_gallop_guarded import fixture
from tests.test_gallop_pair_env import state_class


def settings():return GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_edu.json')


class FixedEduTests(unittest.TestCase):
    def monitor(self):return OfficialJointMonitor(torch.zeros(2,12),torch.arange(8,12),settings())

    def test_goal_stays2m_and50cm_after_repeated_success_windows(self):
        s=settings();c=FixedGallopGoal(s)
        for step in (1000,2000,3000,4000):
            for _ in range(s.min_episodes):c.record(True,c.level,{})
            self.assertIsNone(c.decide(step));self.assertEqual(c.cap,2.);self.assertEqual(c.events,[])
        torch.testing.assert_close(stabilization_leap_target(torch.tensor([1.,2.,3.]),s),torch.full((3,),.5))
        t=phase_targets(torch.tensor([.1]),torch.tensor([2.]),s)
        self.assertAlmostEqual(t['period'].item(),.60,places=6)
        self.assertAlmostEqual((1-max(a+b for a,b in zip(s.touchdown,s.duty)))*s.fast_period,.30)

    def test_actual_calf_speed_cost_starts_before_hard_boundary_and_never_clips_state(self):
        m=self.monitor();dq=torch.zeros(2,12);dq[0,8]=15.;before=dq.clone()
        m.observe(dq,torch.zeros_like(dq),torch.zeros_like(dq),.005)
        self.assertGreater(m.costs()[0,0],0.);self.assertEqual(m.costs()[1,0],0.)
        self.assertTrue(m.diagnostics()['edu_limits'].all());torch.testing.assert_close(dq,before)

    def test_one_bad_physics_sample_disqualifies_success_but_sustained_run_terminates(self):
        m=self.monitor();dq=torch.zeros(2,12);dq[0,8]=16.
        for _ in range(3):m.observe(dq,dq*0,dq*0,.005)
        self.assertFalse(m.failed[0]);self.assertFalse(m.diagnostics()['edu_limits'][0])
        m.observe(dq,dq*0,dq*0,.005);self.assertTrue(m.failed[0]);self.assertFalse(m.failed[1])
        m.reset(torch.tensor([0]));self.assertTrue((m.bad_samples==0).all())

    def test_urdf_efforts_are_joint_specific_and_requested_saturation_is_penalized(self):
        m=self.monitor();dq=torch.zeros(2,12);applied=dq.clone();computed=dq.clone()
        applied[0,0]=24.;applied[1,8]=24.;computed.copy_(applied);computed[1,8]=60.
        m.observe(dq,computed,applied,.005)
        self.assertEqual(m.bad_samples[0],1.);self.assertEqual(m.bad_samples[1],0.)
        self.assertGreater(m.costs()[1,1],0.);self.assertGreater(m.costs()[1,2],0.)

    def test_nonfinite_sample_fails_with_finite_reward_cost(self):
        m=self.monitor();dq=torch.zeros(2,12);dq[0,8]=float('nan')
        m.observe(dq,torch.zeros_like(dq),torch.zeros_like(dq),.005)
        self.assertTrue(m.failed[0]);self.assertTrue(torch.isfinite(m.costs()).all())

    def test_calf_target_rate_is12_without_changing_hip_or_thigh_rates(self):
        s=settings();limits=torch.tensor([-2.,2.]).expand(2,12,2).clone()
        g=GuardedJointController(torch.zeros(2,12),limits,torch.arange(4),torch.arange(8,12),s)
        torch.testing.assert_close(g.rate,torch.tensor([6.]*4+[16.]*4+[12.]*4))

    def test_production_state_samples_all_physics_and_logs_raw_and_compensated_height(self):
        _,_,env,data,sensor=fixture();n=len(data.joint_pos)
        data.joint_vel=torch.zeros_like(data.joint_pos);data.computed_torque=torch.zeros_like(data.joint_pos);data.applied_torque=torch.zeros_like(data.joint_pos)
        state=state_class()(env,settings(),None,None,False,False,walking_policy=lambda obs:torch.zeros(n,12));env.gallop=state
        state.startup.stage[:]=3;state.startup.blend_seconds[:]=settings().startup_blend_seconds
        data.root_pos_w[:,2]=.32;data.body_pos_w[:,:,2]=.222
        state.leap.takeoff[:,2]=.2;state.leap.active[:]=True
        state.begin_step()
        for _ in range(env.cfg.decimation):state.observe_leap_physics(.005)
        state.update_measurements()
        torch.testing.assert_close(state.official.step_seconds,torch.full((n,),.02))
        torch.testing.assert_close(state.guard_height,torch.full((n,4),.08),atol=1e-6,rtol=0)
        torch.testing.assert_close(torch.from_numpy(state.snapshot()['foot_height']),torch.full((n,4),.2),atol=1e-6,rtol=0)
        self.assertIn('edu_velocity',state.components);self.assertIn('joint_velocity',state.snapshot())

    def test_explicit_parent_stage_reset_and_legacy_stage_restore_use_production_cli_block(self):
        tree=ast.parse((ROOT/'scripts/training/gallop_baseline.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="parent.get('gallop_settings', {}).get('staged_enabled')")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'model_final.pt';checkpoint.write_bytes(b'fake immutable weights')
            parent={'gallop_settings':{'staged_enabled':True},'staged_design':{'plan':[{'speed':2.5},{'speed':2.75}]},
                'staged_checkpoints':{'model_final.pt':{'index':1,'goal':{'speed':2.75},'highest_confirmed_stage':0,
                    'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}}
            def reject(message):raise ValueError(message)
            def run(reset,staged):
                ns=dict(a=NS(finetune_from=checkpoint,reset_curriculum=reset),parent=parent,source_checkpoint=checkpoint,
                    settings=settings(),p=NS(error=reject),staged=staged,staged_parent_progress=None,reset_parent_progress=None)
                exec(compile(ast.Module(body=[node],type_ignores=[]),'production_parent_reset','exec'),ns);return ns
            with self.assertRaises(ValueError):run(False,False)
            ns=run(True,False);self.assertEqual(ns['reset_parent_progress']['index'],1);self.assertIsNone(ns['staged_parent_progress'])
            ns=run(False,True);self.assertEqual(ns['staged_parent_progress']['index'],1)


if __name__=='__main__':unittest.main()
