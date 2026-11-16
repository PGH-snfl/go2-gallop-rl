"""Flight reward ordering, recovery credit and reset isolation; no physics/PPO."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings, LeapDistanceTracker, FlightStabilityTracker,
    PairBalanceTracker, SuccessSpeedCurriculum, flight_phase_targets, phase_targets)
from tests.test_gallop_leap_env import make_state
from tests.test_gallop_pair_env import state_class


def settings(**changes):
    return replace(GallopSettings.load(ROOT/'configs/gallop_2p5mps_flight.json'),metric_warmup_seconds=0.,**changes)


class Trajectory:
    def __init__(self,n=1,control_dt=.02):
        self.s=settings();self.n=n;self.index=0;self.decimation=round(control_dt/.005)
        self.position=torch.tensor([[0.,0.,.35]]).repeat(n,1)
        self.leap=LeapDistanceTracker(self.position,self.s)
        self.flight=FlightStabilityTracker(self.position,self.s,torch.full((n,),150.))
        self.earned={key:torch.zeros(n) for key in self.flight.points}

    def sample(self,ground=True,phase=.9,vz=0.,roll=0.,vy=0.,force=150.,slide=0.,fore=True,vx=2.5):
        if self.index%self.decimation==0:
            self.leap.begin_step(torch.zeros(self.n,dtype=torch.bool))
            self.flight.begin_step(torch.zeros(self.n,dtype=torch.bool))
        self.index+=1;self.position[:,0]+=vx*.005
        forces=torch.zeros(self.n,4)
        if ground:forces[:,2]=force  # One foot is enough: no simultaneous landing requirement.
        eligible=torch.full((self.n,),fore,dtype=torch.bool)
        q=torch.tensor([[1.,0.,0.,0.]]).repeat(self.n,1)
        start,finish,gain=self.leap.observe(self.position,q,forces,.005,eligible)
        before={key:value.clone() for key,value in self.flight.points.items()}
        rpy=torch.zeros(self.n,3);rpy[:,0]=roll
        velocity=torch.tensor([[vx,vy,vz]]).repeat(self.n,1)
        foot_v=torch.zeros(self.n,4,3);foot_v[:,:,0]=slide
        self.flight.observe(self.leap,start,finish,gain,torch.full((self.n,),phase),torch.full((self.n,),.345),
            self.position,velocity[:,2],velocity,torch.zeros(self.n,3),rpy,
            torch.full((self.n,4),0. if ground else .04),foot_v,forces,
            torch.tensor([[2.5,0.,0.]]).repeat(self.n,1),eligible,.005)
        for key in before:self.earned[key]+=self.flight.points[key]-before[key]

    def fly(self,**landing):
        self.sample(ground=True,phase=.7)
        self.sample(ground=False,vz=.4)
        for _ in range(13):self.sample(ground=False,vz=-.3)
        self.sample(ground=True,vz=-.3,**landing)


class GallopFlightTests(unittest.TestCase):
    def test_fixed_2p5_command_keeps_reference_anchors_and_has_enough_air_time(self):
        s=settings();self.assertEqual(SuccessSpeedCurriculum(s).bounds(),(2.5,2.5))
        t=phase_targets(torch.zeros(1),torch.tensor([2.5]),s)
        self.assertAlmostEqual(t['period'].item(),.345,places=6)
        self.assertGreater((1-max(t+d for t,d in zip(s.touchdown,s.duty)))*t['period'].item(),.06)
        prepare,air,recover,stage=flight_phase_targets(torch.tensor([.4,.7,.9,.1]),t['period'],s)
        self.assertEqual(stage.tolist(),[0,1,2,3])
        self.assertEqual(prepare.tolist(),[False,True,False,False])
        self.assertEqual(air.tolist(),[False,False,True,False])
        self.assertEqual(recover.tolist(),[False,False,False,True])

    def test_upward_preparation_is_rewarded_before_any_takeoff_even_without_fore_credit(self):
        stationary=Trajectory();stationary.sample(phase=.7,fore=False)
        upward=Trajectory();upward.sample(phase=.7,vz=.4,fore=False)
        self.assertEqual(stationary.flight.rates['drive'].item(),0.)
        self.assertGreater(upward.flight.rates['drive'].item(),0.)
        self.assertEqual(upward.leap.events.item(),0.)
        self.assertEqual(upward.earned['distance'].item(),0.)

    def test_remaining_support_in_shared_flight_has_a_cost_and_no_landing_credit(self):
        grounded=Trajectory();grounded.sample(phase=.9)
        airborne=Trajectory();airborne.sample(phase=.9,ground=False)
        self.assertGreater(grounded.flight.rates['support'].item(),airborne.flight.rates['support'].item())
        self.assertGreater(airborne.flight.rates['clearance'].item(),grounded.flight.rates['clearance'].item())
        self.assertEqual(grounded.earned['landing'].item(),0.)

    def test_roll_and_lateral_drift_increase_balance_cost(self):
        level=Trajectory();level.sample()
        tilted=Trajectory();tilted.sample(roll=.2,vy=.25)
        self.assertGreater(tilted.flight.rates['balance'].item(),level.flight.rates['balance'].item())

    def test_distance_is_held_until_stable_supported_recovery_including_phase_wrap(self):
        t=Trajectory();t.fly()
        self.assertEqual(t.leap.qualified_events.item(),1.)
        self.assertEqual(t.earned['distance'].item(),0.)
        self.assertTrue(t.flight.recovering.item())
        t.flight.begin_step(torch.ones(1,dtype=torch.bool))
        self.assertGreater(t.flight.recovery_bank.item(),0.)
        for _ in range(15):t.sample(phase=.1)
        self.assertEqual(t.earned['distance'].item(),0.)
        t.sample(phase=.1)
        self.assertGreater(t.earned['distance'].item(),.9)
        self.assertGreater(t.earned['landing'].item(),.9)
        self.assertEqual(t.flight.totals['stable_landings'].item(),1.)

    def test_hard_impact_tilt_or_slip_blocks_distance_and_landing_bonus(self):
        for change in ({'force':500.},{'roll':.4},{'slide':.8}):
            t=Trajectory();t.fly(**change)
            for _ in range(20):t.sample(phase=.1)
            self.assertEqual(t.earned['distance'].item(),0.,change)
            self.assertEqual(t.earned['landing'].item(),0.,change)
            self.assertEqual(t.flight.totals['stable_landings'].item(),0.)
            self.assertEqual(t.flight.totals['aborted_landings'].item(),1.)

    def test_preimpact_downward_velocity_is_not_hidden_by_touchdown_deceleration(self):
        t=Trajectory();t.sample(phase=.7);t.sample(ground=False,vz=.4)
        for _ in range(13):t.sample(ground=False,vz=-2.)
        t.sample(vz=0.)
        for _ in range(20):t.sample(phase=.1)
        self.assertEqual(t.earned['distance'].item(),0.)
        self.assertEqual(t.flight.totals['landing_peak_down_vz'].item(),2.)

    def test_retakeoff_before_recovery_cancels_pending_distance(self):
        t=Trajectory();t.fly()
        for _ in range(5):t.sample(phase=.1)
        t.sample(ground=False,vz=.4)
        self.assertEqual(t.flight.recovery_bank.item(),0.)
        self.assertEqual(t.flight.totals['aborted_landings'].item(),1.)
        self.assertEqual(t.earned['distance'].item(),0.)

    def test_initial_fall_fore_failure_and_backward_motion_cannot_earn_landing_bonus(self):
        initial=Trajectory()
        for _ in range(14):initial.sample(ground=False)
        for _ in range(20):initial.sample(phase=.1)
        self.assertEqual(initial.earned['landing'].item(),0.)
        for fore,vx in [(False,2.5),(True,-2.5)]:
            t=Trajectory();t.sample(phase=.7,fore=fore,vx=vx)
            for _ in range(14):t.sample(ground=False,fore=fore,vx=vx,vz=.4)
            for _ in range(20):t.sample(phase=.1,fore=fore,vx=vx)
            self.assertEqual(t.earned['distance'].item(),0.)
            self.assertEqual(t.earned['landing'].item(),0.)

    def test_termination_on_confirmation_step_revokes_all_positive_event_points(self):
        t=Trajectory();t.fly()
        for _ in range(16):t.sample(phase=.1)
        self.assertGreater(t.flight.points['distance'].item(),0.)
        t.flight.reject_terminated(torch.ones(1,dtype=torch.bool))
        self.assertEqual(t.flight.points['distance'].item(),0.)
        self.assertEqual(t.flight.points['landing'].item(),0.)
        self.assertEqual(t.flight.totals['stable_landings'].item(),0.)

    def test_reset_drops_one_environment_bank_without_touching_the_other(self):
        t=Trajectory(2);t.fly();t.flight.reset(torch.tensor([0]));t.leap.reset(torch.tensor([0]))
        for _ in range(16):t.sample(phase=.1)
        self.assertEqual(t.earned['distance'][0].item(),0.)
        self.assertGreater(t.earned['distance'][1].item(),0.)

    def test_short_flights_have_continuous_partial_credit_and_cycle_budget(self):
        s=settings();t=LeapDistanceTracker(torch.zeros(1,3),s)
        q=torch.tensor([[1.,0.,0.,0.]]);force=torch.tensor([[100.,0.,0.,0.]])
        eligible=torch.ones(1,dtype=torch.bool);p=torch.tensor([[0.,0.,.35]])
        t.begin_step(torch.zeros(1,dtype=torch.bool));t.observe(p,q,force,.005,eligible)
        t.observe(p,q,force*0,.005,eligible)
        for i in range(1,7):
            p[:,0]=.08*i/6
            t.observe(p,q,force if i==6 else force*0,.005,eligible)
        self.assertAlmostEqual(t.pending_reward.item(),.25,places=5)
        self.assertEqual(t.qualified_events.item(),0.)

    def test_hip_rom_cost_detects_persistent_hind_asymmetry_but_allows_mirrored_motion(self):
        q=torch.zeros(2,4,3);t=PairBalanceTracker(q,settings())
        wrapped=torch.ones(2,dtype=torch.bool)
        t.finish_cycles(wrapped,q)
        for value in torch.linspace(0,1,20):
            q[:,:,0]=value*torch.tensor([[.2,-.2,.2,-.2],[.2,-.2,.1,-.5]])
            t.observe(q,torch.ones(2,4),torch.ones(2,4,dtype=torch.bool),.005,wrapped)
        t.finish_cycles(wrapped,q)
        self.assertEqual(t.hip_cost[0].item(),0.)
        self.assertGreater(t.hip_cost[1].item(),.1)
        t.reset(torch.tensor([1]),q);self.assertEqual(t.hip_cost[1].item(),0.)

    def test_invalid_flight_config_rejected(self):
        for change in ({'flight_support_weight':-1.},{'flight_max_force_bw':0.},
                       {'flight_recovery_seconds':float('nan')},{'flight_prepare_seconds':.3},
                       {'reference_max_speed':1.},{'leap_partial_credit':2.}):
            with self.assertRaises(ValueError):settings(**change)


class GallopFlightEnvironmentTests(unittest.TestCase):
    def test_production_wiring_has_same_event_points_at_10_and_20ms_and_masks_failed_envs(self):
        results=[]
        for control_dt in (.01,.02):
            _,env,data,sensor=make_state(control_dt)
            data.root_lin_vel_b=torch.tensor([[2.5,0.,0.]]).repeat(3,1)
            data.root_lin_vel_w=data.root_lin_vel_b.clone()
            data.body_lin_vel_w=torch.zeros(3,4,3)
            env.scene['robot'].root_physx_view=SimpleNamespace(get_masses=lambda:torch.full((3,1),15.))
            env.cfg.sim=SimpleNamespace(gravity=(0.,0.,-9.81))
            env.command_manager.get_command=lambda name:torch.tensor([[2.5,0.,0.]]).repeat(3,1)
            s=settings();state=state_class()(env,s,None,None,training_targets=False,curriculum_enabled=False)
            state.phase[:]=.75
            earned=torch.zeros(3);landing=torch.zeros(3);drive=torch.zeros(3)
            for step in range(40//env.cfg.decimation):
                env.common_step_counter=step+1;state.begin_step()
                state.last_fore_ok[:]=torch.tensor([True,False,True])
                for substep in range(env.cfg.decimation):
                    index=step*env.cfg.decimation+substep+1
                    airborne=5<=index<19
                    data.root_pos_w[:,0]=.0125*index
                    data.root_lin_vel_w[:,2]=.4 if index<7 else -.3 if index<=19 else 0.
                    data.root_lin_vel_b[:,2]=data.root_lin_vel_w[:,2]
                    data.body_pos_w[:,:,2]=.062 if airborne else .022
                    sensor.data.net_forces_w[:,:,2]=0. if airborne else 25.
                    state.observe_leap_physics(.005)
                state.update_measurements()
                earned+=s.leap_reward_weight*state.components['leap_distance']*control_dt
                landing+=state.components['flight_landing']*control_dt
                drive+=state.components['flight_drive']*control_dt
            results.append(earned)
            self.assertGreater(earned[0].item(),2.5)
            self.assertEqual(earned[1:].tolist(),[0.,0.])
            self.assertGreater(landing[0].item(),.9)
            self.assertGreater(drive[1].item(),0.)
            state.complete_episodes([0])
            self.assertEqual(state.last_episode['flight_stable_landings'],1.)
            self.assertEqual(state.last_episode['leap_physics_samples'],40.)
            self.assertIn('flight_planned_stage',state.snapshot())
            state.reset(torch.tensor([0]))
            self.assertEqual(state.flight.totals['stable_landings'][0].item(),0.)
        torch.testing.assert_close(results[0],results[1],rtol=0,atol=1e-5)


if __name__=='__main__':unittest.main()
