"""Measured-velocity braking, infeasible bounds, peak costs and fault latching."""
from dataclasses import replace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,GuardedJointController,OfficialJointMonitor
from tests.test_gallop_guarded import fixture


def settings():return GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_velocity_guard.json')


def guard(n=2):
    return GuardedJointController(torch.zeros(n,12),torch.tensor([-2.,2.]).expand(n,12,2),
        torch.arange(4),torch.arange(8,12),settings())


class VelocityGuardTests(unittest.TestCase):
    def test_feasible_brake_opposes_velocity_and_keeps_all_bounds(self):
        g=guard();q=torch.zeros(2,12);dq=q.clone();dq[0,8]=14.;dq[1,8]=-14.
        request=torch.ones(2,12);request[1]*=-1;before=request.clone()
        target=g.process(request,.02,q,dq)
        tau=25*(target-q)-.5*dq
        self.assertLess(tau[0,8],0.);self.assertGreater(tau[1,8],0.)
        self.assertTrue((g.brake_intervention>0).all());self.assertFalse(g.brake_infeasible.any())
        self.assertTrue(((target-g.default).abs()<=g.rate*.02+1e-6).all())
        self.assertTrue((target>=g.lower).all() and (target<=g.upper).all());torch.testing.assert_close(request,before)

    def test_infeasible_brake_is_logged_without_breaking_angle_or_rate_limits(self):
        g=guard();q=torch.zeros(2,12);dq=q.clone();q[:,8]=.8;dq[:,8]=-14.
        original_q=q.clone();original_dq=dq.clone();target=g.process(torch.zeros_like(q),.02,q,dq)
        self.assertTrue(g.brake_infeasible[:,8].all())
        torch.testing.assert_close(target[:,8],torch.full((2,),.2))
        torch.testing.assert_close(q,original_q);torch.testing.assert_close(dq,original_dq)

    def test_below_margin_matches_old_projection_and_limits_are_joint_specific(self):
        g=guard();q=torch.zeros(2,12);dq=torch.full_like(q,12.)
        result=g.process(torch.ones_like(q),.02,q,dq)
        torch.testing.assert_close(result,g.rate.expand(2,12)*.02)
        self.assertFalse(g.brake_active.any())
        g.process(torch.ones_like(q),.02,q,torch.full_like(q,14.))
        self.assertFalse(g.brake_active[:,:8].any());self.assertTrue(g.brake_active[:,8:].all())

    def test_missing_state_rejected_and_reset_clears_only_selected_environment(self):
        g=guard()
        with self.assertRaises(ValueError):g.process(torch.zeros(2,12),.02)
        g.process(torch.ones(2,12),.02,torch.zeros(2,12),torch.full((2,12),14.))
        g.reset(torch.tensor([0]));self.assertFalse(g.brake_active[0].any());self.assertTrue(g.brake_active[1,8:].all())
        self.assertEqual(g.brake_intervention[0],0.)

    def test_one_physics_sample_faults_at_both5ms_and1ms_and_remains_latched(self):
        for dt in (.005,.001):
            m=OfficialJointMonitor(torch.zeros(2,12),torch.arange(8,12),settings())
            dq=torch.zeros(2,12);dq[0,8]=15.8;m.observe(dq,torch.zeros_like(dq),torch.zeros_like(dq),dt)
            self.assertTrue(m.failed[0]);self.assertFalse(m.failed[1])
            m.observe(torch.zeros_like(dq),torch.zeros_like(dq),torch.zeros_like(dq),dt)
            self.assertTrue(m.failed[0]);self.assertEqual(m.run[0],0.)
            m.reset(torch.tensor([0]));self.assertFalse(m.failed[0]);self.assertEqual(m.step_velocity_peak[0],0.)

    def test_physics_peak_cost_is_not_diluted_and_control_window_resets(self):
        cfg=settings();a=OfficialJointMonitor(torch.zeros(1,12),torch.arange(8,12),cfg)
        b=OfficialJointMonitor(torch.zeros(1,12),torch.arange(8,12),replace(cfg,edu_velocity_peak_weight=0.))
        dq=torch.zeros(1,12);dq[:,8]=15.
        for i in range(4):
            for m in (a,b):m.observe(dq if i==0 else torch.zeros_like(dq),torch.zeros_like(dq),torch.zeros_like(dq),.005)
        torch.testing.assert_close(a.costs()[:,0],b.costs()[:,0]+a.step_velocity_peak)
        torch.testing.assert_close(a.step_velocity_peak,b.costs()[:,0]*4)
        a.begin_step();self.assertEqual(a.step_velocity_peak[0],0.);self.assertEqual(a.costs().sum(),0.)

    def test_production_action_reads_actual_state_and_preserves_latent_sample(self):
        term,state,env,data,_=fixture();cfg=settings();state.cfg=cfg
        state.guard=GuardedJointController(data.default_joint_pos,data.soft_joint_pos_limits,state.joint_ids[:,0],state.calf_ids,cfg)
        data.joint_vel=torch.zeros_like(data.joint_pos);data.joint_vel[:,state.calf_ids]=14.
        state.startup.stage.fill_(3);state.startup.blend_seconds.fill_(cfg.startup_blend_seconds)
        sample=torch.full((3,12),10.);before=sample.clone();term.process_actions(sample)
        self.assertTrue(state.guard.brake_active[:,state.calf_ids].all())
        self.assertTrue((state.guard.brake_intervention>0).all())
        torch.testing.assert_close(sample,before)
        torch.testing.assert_close(data.default_joint_pos+cfg.action_scale*env.action_manager.action,term._processed_actions)

    def test_invalid_brake_configuration_rejected(self):
        for changes in (dict(velocity_brake_gain=0.),dict(velocity_brake_margin=1.),dict(edu_velocity_peak_weight=-1.),
                        dict(velocity_brake_enabled=1),dict(edu_limits_enabled=False)):
            with self.assertRaises(ValueError):replace(settings(),**changes)


if __name__=='__main__':unittest.main()
