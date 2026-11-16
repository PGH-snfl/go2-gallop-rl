"""3 m/s and 80 cm goal consistency; synthetic events, never physics or PPO."""
from dataclasses import replace
import ast
from types import SimpleNamespace as NS
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import (GallopSettings,SuccessSpeedCurriculum,phase_targets,
    stabilization_leap_target,LeapDistanceTracker,FlightStabilityTracker)
from tests.test_gallop_filter_free import passing_episode


def settings():return GallopSettings.load(ROOT/'configs/gallop_3ms_80cm_stable.json')


class SyntheticFlight:
    def __init__(self):
        self.cfg=replace(settings(),metric_warmup_seconds=0.)
        self.position=torch.tensor([[0.,0.,.28]])
        self.leap=LeapDistanceTracker(self.position,self.cfg)
        self.flight=FlightStabilityTracker(self.position,self.cfg,torch.tensor([150.]))
        self.distance_points=0.

    def sample(self,ground,phase,vz=0.,height=.28,force=150.,roll=0.):
        self.leap.begin_step(torch.zeros(1,dtype=torch.bool));self.flight.begin_step(torch.zeros(1,dtype=torch.bool))
        self.position[:,0]+=3*.005;self.position[:,2]=height
        forces=torch.zeros(1,4)
        if ground:forces[:,2]=force
        eligible=torch.ones(1,dtype=torch.bool)
        start,finish,gain=self.leap.observe(self.position,torch.tensor([[1.,0.,0.,0.]]),forces,.005,eligible)
        rpy=torch.tensor([[roll,0.,0.]])
        self.flight.observe(self.leap,start,finish,gain,torch.tensor([phase]),torch.tensor([.6]),
            self.position,torch.tensor([vz]),torch.tensor([[3.,0.,vz]]),torch.zeros(1,3),rpy,
            torch.full((1,4),0. if ground else .04),torch.zeros(1,4,3),forces,
            torch.tensor([[3.,0.,0.]]),eligible,.005)
        self.distance_points+=float(self.flight.points['distance'][0])

    def fly(self,landing_force=150.):
        duration=.30;upward=9.81*duration/2
        self.sample(True,.49)
        for index in range(60):
            time=index*.005
            self.sample(False,(.501+time/.6)%1,upward-9.81*time,.28+upward*time-.5*9.81*time*time)
        self.sample(True,.001,vz=-upward,force=landing_force)


class LongLeapTests(unittest.TestCase):
    def test_actual_initial_goal_is_three_and_point_eight(self):
        cfg=settings();self.assertEqual(SuccessSpeedCurriculum(cfg).bounds(),(3.,3.))
        torch.testing.assert_close(stabilization_leap_target(torch.tensor([3.]),cfg),torch.tensor([.8]))
        state,_=passing_episode(cfg,1.)
        torch.testing.assert_close(state.episode_speed,torch.full((3,),3.))
        torch.testing.assert_close(state.leap.distance_target,torch.full((3,),.8))
        torch.testing.assert_close(state.startup.goal_speed,torch.full((3,),3.))

    def test_reference_air_window_and_ballistic_consistency(self):
        cfg=settings();period=phase_targets(torch.tensor([0.]),torch.tensor([3.]),cfg)['period'].item()
        air=period*(1-max(t+d for t,d in zip(cfg.touchdown,cfg.duty)))
        self.assertAlmostEqual(period,.6,places=6);self.assertAlmostEqual(air,.3,places=6)
        self.assertGreaterEqual(air,cfg.flight_min_seconds);self.assertGreaterEqual(air*3,cfg.leap_min_distance)
        required_air=cfg.leap_min_distance/3;required_vz=9.81*required_air/2
        self.assertGreater(required_vz,1.2)
        self.assertLess(9.81*air/2,cfg.flight_max_landing_vz)
        self.assertLess(.28+9.81*air*air/8,cfg.max_height)
        self.assertLess(1+(9.81*air/2)/(9.81*cfg.flight_recovery_seconds),cfg.flight_max_force_bw)

    def test_air_template_has_no_required_contacts_and_recovery_does_not_overlap_preparation(self):
        cfg=settings();phase=torch.tensor([.55,.70,.95]);t=phase_targets(phase,torch.full_like(phase,3.),cfg)
        self.assertFalse(t['stance'].any())
        self.assertLess(cfg.flight_prepare_seconds+cfg.flight_recovery_seconds,.3)

    def test_hardware_filters_and_filter_free_gate_are_preserved(self):
        old=GallopSettings.load(ROOT/'configs/gallop_guarded_filter_free.json');new=settings()
        keys=('action_scale','guarded_sim_limit_factor','guarded_target_margin','guarded_hip_rate','guarded_leg_rate',
            'guarded_actual_tolerance','guarded_violation_seconds','guarded_max_intervention','max_action_clipping',
            'guarded_front_height','guarded_hind_height','flight_max_force_bw','stabilization_width_fraction','stabilization_landing_fraction')
        for key in keys:self.assertEqual(getattr(old,key),getattr(new,key),key)
        self.assertFalse(new.relearn_enabled)

    def test_stability_rewards_increase_with_modest_width_and_height_changes(self):
        old=GallopSettings.load(ROOT/'configs/gallop_guarded_filter_free.json');new=settings()
        for key in ('yaw_penalty_weight','stabilization_yaw_rate_weight','roll_pitch_quiet_weight',
                    'flight_balance_weight','flight_landing_weight','flight_impact_weight'):
            self.assertGreater(getattr(new,key),getattr(old,key),key)
        for key in ('stance_width_weight','guarded_height_weight'):
            self.assertLessEqual(getattr(new,key),getattr(old,key)*1.2)

    def test_production_configuration_registers_fixed_speed_and_stability_weights(self):
        tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_gallop')
        cfg=NS(scene=NS(terrain=NS(),contact_forces=NS(),robot=NS()),curriculum=NS(),
            commands=NS(base_velocity=NS(ranges=NS(),limit_ranges=NS())),
            actions=NS(JointPositionAction=NS()),sim=NS(dt=.005),
            events=NS(reset_base=NS(params={'pose_range':{}}),reset_robot_joints=NS(params={})),
            observations=NS(policy=NS(),critic=NS()),terminations=NS(),
            rewards=NS(**{k:NS(params={}) for k in ('track_lin_vel_xy','base_linear_velocity','base_angular_velocity','action_rate')}))
        namespace=dict(StartupVelocityCommand=object,BoundedJointPositionAction=object,
            ObservationTermCfg=NS,RewardTermCfg=NS,TerminationTermCfg=NS)
        for name in ('gallop_reward','phase_observation','heading_observation','guarded_limit_termination','height_termination'):
            namespace[name]=lambda *args:None
        exec(compile(ast.Module(body=[function],type_ignores=[]),'configure_long_leap','exec'),namespace)
        namespace['configure_gallop'](cfg,settings())
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_x,(3.,3.))
        self.assertEqual(cfg.scene.robot.soft_joint_pos_limit_factor,1.)
        for key,weight in dict(leap_distance=8.,yaw_heading=4.,roll_pitch_quiet=5.,front_width=4.,hind_width=4.,
            flight_balance=5.,flight_landing=6.,flight_impact=4.,guarded_projection=8.,guarded_front_height=4.,guarded_hind_height=4.).items():
            term=getattr(cfg.rewards,'gallop_'+key)
            self.assertEqual(term.weight,weight,key);self.assertEqual(term.params,{'key':key})

    def test_long_flight_distance_waits_for_stable_120ms_recovery(self):
        probe=SyntheticFlight();probe.fly()
        self.assertEqual(probe.leap.qualified_events.item(),1.)
        self.assertGreater(probe.leap.last_distance.item(),.8);self.assertEqual(probe.distance_points,0.)
        for _ in range(23):probe.sample(True,.1)
        self.assertEqual(probe.distance_points,0.)
        probe.sample(True,.1)
        self.assertGreater(probe.distance_points,0.)
        self.assertEqual(probe.flight.totals['stable_landings'].item(),1.)

    def test_excessive_landing_force_still_blocks_distance_credit(self):
        probe=SyntheticFlight();probe.fly(landing_force=500.)
        for _ in range(30):probe.sample(True,.1)
        self.assertEqual(probe.distance_points,0.)
        self.assertEqual(probe.flight.totals['stable_landings'].item(),0.)
        self.assertEqual(probe.flight.totals['aborted_landings'].item(),1.)


if __name__=='__main__':unittest.main()
