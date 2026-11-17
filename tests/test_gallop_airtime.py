"""True-flight objective, contact-foot cost and speed forensics; CPU only."""
import ast
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings, LeapDistanceTracker, FlightStabilityTracker, OfficialJointMonitor
from isaac_task.staged_adaptation import StagedSpeedAdaptation
from tests.test_gallop_flight import Trajectory, settings as old_flight_settings
from tests.test_gallop_leap_env import make_state
from tests.test_gallop_pair_env import state_class


def settings(**changes):
    return replace(GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_airtime.json'),
                   metric_warmup_seconds=0.,**changes)


def trajectory(control_dt=.02,n=1,**changes):
    t=Trajectory(n,control_dt);t.s=settings(**changes)
    t.leap=LeapDistanceTracker(t.position,t.s)
    t.flight=FlightStabilityTracker(t.position,t.s,torch.full((n,),150.))
    t.earned={key:torch.zeros(n) for key in t.flight.points}
    return t


def fly(t,steps=12,takeoff_vz=.4,air_vx=2.5,**landing):
    t.sample(phase=.7);t.sample(ground=False,phase=.9,vz=takeoff_vz)
    for _ in range(steps-1):t.sample(ground=False,phase=.9,vz=-.2,vx=air_vx)
    t.sample(phase=.9,vz=-.2,vx=air_vx,**landing)


def settle(t):
    for _ in range(24):t.sample(phase=.1)


class AirtimeTests(unittest.TestCase):
    def test_longer_actual_forward_flights_receive_more_airtime_credit(self):
        scores=[]
        for steps in (12,24,48):
            t=trajectory();fly(t,steps);settle(t)
            scores.append(t.earned['airtime'].item())
            torch.testing.assert_close(t.earned['airtime'],t.earned['airtime_touchdown']+t.earned['airtime_recovery'])
        self.assertLess(scores[0],scores[1]);self.assertLess(scores[1],scores[2])
        self.assertGreater(scores[2],.9);self.assertLessEqual(scores[2],1.)
        self.assertEqual(t.leap.qualified_events.item(),1.)

    def test_grounded_unloading_no_longer_gets_clearance_credit(self):
        new=trajectory();old=trajectory(flight_clearance_requires_airborne=False)
        new.sample(phase=.9);old.sample(phase=.9)
        self.assertEqual(new.flight.rates['clearance'].item(),0.)
        self.assertGreater(old.flight.rates['clearance'].item(),0.)
        new.sample(ground=False,phase=.9,vz=.4)
        self.assertGreater(new.flight.rates['clearance'].item(),0.)

    def test_initial_fall_downward_takeoff_and_slow_vertical_hops_get_no_airtime_credit(self):
        initial=trajectory()
        for _ in range(12):initial.sample(ground=False,phase=.9)
        initial.sample();settle(initial)
        self.assertEqual(initial.earned['airtime'].item(),0.)
        for change in (dict(takeoff_vz=-.1),dict(air_vx=.2)):
            t=trajectory();fly(t,**change);settle(t)
            self.assertEqual(t.earned['airtime'].item(),0.,change)

    def test_forelimb_failure_and_subminimum_air_are_not_rewarded(self):
        t=trajectory();t.sample(phase=.7,fore=False);t.sample(ground=False,phase=.9,vz=.4,fore=False)
        for _ in range(12):t.sample(ground=False,phase=.9,fore=False)
        t.sample(phase=.9,fore=False);settle(t)
        self.assertEqual(t.earned['airtime'].item(),0.)
        short=trajectory();fly(short,steps=1);settle(short)
        self.assertEqual(short.earned['airtime'].item(),0.)

    def test_unsafe_contact_and_recovery_failure_cannot_collect_full_credit(self):
        unsafe=trajectory();fly(unsafe,slide=.8);settle(unsafe)
        self.assertEqual(unsafe.earned['airtime'].item(),0.)
        t=trajectory();fly(t);immediate=t.earned['airtime'].clone()
        self.assertGreater(immediate.item(),0.)
        self.assertEqual(t.earned['airtime_recovery'].item(),0.)
        t.sample(slide=.8);settle(t)
        torch.testing.assert_close(t.earned['airtime'],immediate)
        self.assertEqual(t.flight.airtime_recovery_bank.item(),0.)

    def test_same_reference_cycle_has_one_best_score_budget(self):
        t=trajectory();fly(t,steps=12);settle(t);first=t.earned['airtime'].clone()
        fly(t,steps=12);settle(t)
        torch.testing.assert_close(t.earned['airtime'],first,rtol=0,atol=1e-6)
        fly(t,steps=24);settle(t)
        self.assertGreater(t.earned['airtime'].item(),first.item())
        self.assertLessEqual(t.earned['airtime'].item(),.5+1e-5)

    def test_phase_wrap_preserves_pending_recovery_credit(self):
        t=trajectory();fly(t);before=t.flight.airtime_recovery_bank.clone()
        t.flight.begin_step(torch.ones(1,dtype=torch.bool))
        torch.testing.assert_close(t.flight.airtime_recovery_bank,before)
        settle(t);self.assertGreater(t.earned['airtime_recovery'].item(),0.)

    def test_termination_and_environment_reset_clear_new_state(self):
        t=trajectory(n=2);fly(t)
        self.assertGreater(t.flight.points['airtime'][0].item(),0.)
        t.flight.reject_terminated(torch.tensor([True,False]))
        self.assertEqual(t.flight.points['airtime'][0].item(),0.)
        self.assertEqual(t.flight.airtime_recovery_bank[0].item(),0.)
        original=t.flight.airtime_recovery_bank[1].clone()
        t.flight.reset(torch.tensor([0]))
        self.assertFalse(t.flight.airtime_valid[0]);self.assertEqual(t.flight.airtime_best[0].item(),0.)
        torch.testing.assert_close(t.flight.airtime_recovery_bank[1],original)

    def test_slide_cost_starts_before_gate_and_ignores_airborne_feet(self):
        low=trajectory();low.sample(slide=.2)
        early=trajectory();early.sample(slide=.35)
        self.assertEqual(low.flight.rates['slide'].item(),0.)
        self.assertGreater(early.flight.rates['slide'].item(),0.)
        self.assertEqual(early.flight.totals['aborted_landings'].item(),0.)
        air=trajectory();air.sample(phase=.7);air.sample(ground=False,phase=.9,vz=.4,slide=2.)
        self.assertEqual(air.flight.rates['slide'].item(),0.)

    def test_airtime_credit_is_independent_of_control_decimation(self):
        scores=[]
        for dt in (.01,.02):
            t=trajectory(control_dt=dt);fly(t);settle(t);scores.append(t.earned['airtime'])
        torch.testing.assert_close(*scores)

    def test_config_preserves_goal_and_all_mechanical_and_landing_gates(self):
        old=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_landing_credit.json');new=settings()
        changes={key for key in old.to_dict() if getattr(old,key)!=getattr(new,key)}
        self.assertEqual(changes,{'metric_warmup_seconds','flight_airtime_weight','flight_clearance_requires_airborne',
                                  'flight_slide_weight','flight_drive_weight','flight_landing_weight','staged_noise_target_std'})
        self.assertEqual((new.start_speed,new.leap_min_distance,new.flight_min_seconds),(2.,.5,.24))

    def test_invalid_new_settings_are_rejected(self):
        for change in (dict(flight_airtime_weight=-1.),dict(flight_airtime_weight=float('nan')),
                       dict(flight_clearance_requires_airborne=1),dict(flight_slide_soft_speed=0.),
                       dict(flight_slide_soft_speed=.5),dict(flight_shaping_enabled=False)):
            with self.assertRaises(ValueError):settings(**change)


class SpeedDiagnosticsTests(unittest.TestCase):
    def test_peak_joint_actual_limit_time_and_reset_are_recorded_without_clipping(self):
        m=OfficialJointMonitor(torch.zeros(2,12),torch.arange(8,12),settings())
        dq=torch.zeros(2,12);dq[0,8]=-16.;original=dq.clone();m.observe(dq,dq*0,dq*0,.005)
        self.assertEqual(m.peak_velocity_joint[0].item(),8);self.assertEqual(m.peak_velocity_actual[0].item(),-16.)
        self.assertAlmostEqual(m.peak_velocity_limit[0].item(),15.7,places=5)
        self.assertAlmostEqual(m.peak_velocity_seconds[0].item(),.005,places=6)
        torch.testing.assert_close(dq,original)
        dq.zero_();dq[:,0]=34.;m.observe(dq,dq*0,dq*0,.005)
        self.assertEqual(m.peak_velocity_joint[0].item(),0)
        self.assertAlmostEqual(m.peak_velocity_seconds[0].item(),.01,places=6)
        m.reset(torch.tensor([0]));self.assertEqual(m.peak_velocity_joint[0].item(),-1)
        self.assertEqual(m.peak_velocity_joint[1].item(),0)

    def test_nonfinite_peak_has_explicit_finite_flag_and_still_faults(self):
        m=OfficialJointMonitor(torch.zeros(1,12),torch.arange(8,12),settings())
        dq=torch.zeros(1,12);dq[0,8]=float('nan');zero=torch.zeros_like(dq)
        m.observe(dq,zero,zero,.005)
        self.assertTrue(m.failed[0]);self.assertFalse(m.peak_velocity_sample_finite[0])
        self.assertEqual(m.peak_velocity_joint[0].item(),8)
        self.assertEqual(m.peak_velocity_actual[0].item(),0.)

    def test_smaller_exploration_retains_joint_ratios_and_full_rollout_boundaries(self):
        initial=torch.linspace(.12,.20,12)
        policy=NS(std=torch.nn.Parameter(initial.clone()),state_dependent_std=False,is_recurrent=False)
        alg=NS(policy=policy,entropy_coef=.01,is_multi_gpu=False,act=lambda obs:policy.std.clone(),update=lambda:None)
        adaptation=StagedSpeedAdaptation(NS(alg=alg,num_steps_per_env=24),settings())
        first=[alg.act(None) for _ in range(24)]
        for value in first:torch.testing.assert_close(value,initial)
        next_value=alg.act(None)
        self.assertLess(float(next_value.mean()),.16)
        adaptation.apply(500)
        self.assertAlmostEqual(float(policy.std.mean()),.10,places=6)
        torch.testing.assert_close(policy.std/policy.std.mean(),initial/initial.mean())
        self.assertFalse(policy.std.requires_grad);self.assertEqual(alg.entropy_coef,0.)


class AirtimeWiringTests(unittest.TestCase):
    def test_actual_environment_pays_airtime_masks_failed_envs_and_uses_negative_slide(self):
        results=[]
        for dt in (.01,.02):
            _,env,data,sensor=make_state(dt)
            data.root_lin_vel_b=torch.tensor([[2.5,0.,0.]]).repeat(3,1);data.root_lin_vel_w=data.root_lin_vel_b.clone()
            data.body_lin_vel_w=torch.zeros(3,4,3)
            data.body_lin_vel_w[:,:,0]=.35
            env.scene['robot'].root_physx_view=NS(get_masses=lambda:torch.full((3,1),15.))
            env.cfg.sim=NS(gravity=(0.,0.,-9.81))
            env.command_manager.get_command=lambda name:torch.tensor([[2.5,0.,0.]]).repeat(3,1)
            s=replace(old_flight_settings(),flight_airtime_weight=12.,flight_slide_weight=2.,flight_clearance_requires_airborne=True)
            state=state_class()(env,s,None,None,False,False);state.phase[:]=.75;earned=torch.zeros(3)
            for step in range(40//env.cfg.decimation):
                env.common_step_counter=step+1;state.begin_step();state.last_fore_ok[:]=torch.tensor([True,False,True])
                for substep in range(env.cfg.decimation):
                    index=step*env.cfg.decimation+substep+1;airborne=5<=index<19
                    data.root_pos_w[:,0]=.0125*index;data.root_lin_vel_w[:,2]=.4 if index<7 else -.3 if index<=19 else 0.
                    data.root_lin_vel_b[:,2]=data.root_lin_vel_w[:,2]
                    data.body_pos_w[:,:,2]=.062 if airborne else .022
                    sensor.data.net_forces_w[:,:,2]=0. if airborne else 25.
                    state.observe_leap_physics(.005)
                state.update_measurements();earned+=s.flight_airtime_weight*state.components['flight_airtime']*dt
                self.assertTrue((state.components['flight_slide']<=0).all())
            self.assertGreater(earned[0].item(),0.);self.assertEqual(earned[1:].tolist(),[0.,0.])
            self.assertLess(state.components['flight_slide'][0].item(),0.)
            self.assertEqual(state.snapshot()['flight_abort_sample_contact'].shape,(3,4))
            results.append(earned)
        torch.testing.assert_close(*results,atol=1e-5,rtol=0)

    def test_reward_config_registers_positive_airtime_and_signed_slide_terms(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_gallop')
        block=next(n for n in func.body if isinstance(n,ast.If) and ast.unparse(n.test)=='settings.flight_shaping_enabled')
        ns={'settings':settings(),'weights':{}}
        exec(compile(ast.Module(body=[block],type_ignores=[]),'reward_weights','exec'),ns)
        self.assertEqual(ns['weights']['flight_airtime'],12.)
        self.assertEqual(ns['weights']['flight_slide'],2.)


if __name__=='__main__':unittest.main()
