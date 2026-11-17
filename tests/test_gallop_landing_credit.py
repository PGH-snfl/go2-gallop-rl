"""Reward accounting and physics-step diagnostics; CPU only, no simulator/PPO."""
from dataclasses import replace
import ast
from types import SimpleNamespace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings, LeapDistanceTracker, FlightStabilityTracker
from tests.test_gallop_distance_focus import leap_score
from tests.test_gallop_flight import Trajectory


def settings(**changes):
    return replace(GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_landing_credit.json'),
                   metric_warmup_seconds=0.,**changes)


def trajectory(control_dt=.02,n=1,**changes):
    t=Trajectory(n,control_dt)
    t.s=settings(**changes)
    t.leap=LeapDistanceTracker(t.position,t.s)
    t.flight=FlightStabilityTracker(t.position,t.s,torch.full((n,),150.))
    t.earned={key:torch.zeros(n) for key in t.flight.points}
    return t


class LandingCreditTests(unittest.TestCase):
    def test_safe_touchdown_pays_fraction_and_recovery_pays_remainder_without_double_credit(self):
        split=trajectory();legacy=trajectory(flight_touchdown_distance_fraction=0.)
        for t in (split,legacy):t.fly()
        immediate=split.earned['distance'].clone()
        self.assertGreater(immediate.item(),0.)
        self.assertEqual(split.earned['landing'].item(),0.)
        self.assertEqual(legacy.earned['distance'].item(),0.)
        self.assertAlmostEqual(split.flight.recovery_bank.item()/legacy.flight.recovery_bank.item(),.75,places=6)
        for t in (split,legacy):
            for _ in range(24):t.sample(phase=.1)
            torch.testing.assert_close(t.earned['distance'],t.earned['distance_touchdown']+t.earned['distance_recovery'])
            self.assertEqual(t.flight.totals['stable_landings'].item(),1.)
        torch.testing.assert_close(split.earned['distance'],legacy.earned['distance'])
        torch.testing.assert_close(split.earned['distance_recovery'],3*immediate)

    def test_retakeoff_retains_only_safe_touchdown_credit_and_cancels_remainder(self):
        t=trajectory();t.fly();immediate=t.earned['distance'].clone()
        for _ in range(5):t.sample(phase=.1)
        t.sample(ground=False,vz=.4)
        torch.testing.assert_close(t.earned['distance'],immediate)
        self.assertEqual(t.earned['distance_recovery'].item(),0.)
        self.assertEqual(t.earned['landing'].item(),0.)
        self.assertEqual(t.flight.recovery_bank.item(),0.)
        self.assertEqual(t.flight.totals['abort_retakeoff'].item(),1.)
        self.assertEqual(t.flight.totals['abort_contact'].item(),1.)

    def test_unsafe_touchdown_pays_nothing_and_retains_overlapping_physics_reasons(self):
        t=trajectory();t.fly(force=500.,slide=.8,roll=.4)
        self.assertEqual(t.earned['distance'].item(),0.)
        self.assertEqual(t.flight.totals['aborted_landings'].item(),1.)
        for reason in ('force','slide','tilt'):
            self.assertEqual(t.flight.totals['abort_'+reason].item(),1.)
            self.assertTrue(t.flight.abort_reasons[0,t.flight.ABORT_REASONS.index(reason)])
        self.assertAlmostEqual(t.flight.abort_sample['foot_slide'].item(),.8,places=6)
        for _ in range(3):t.sample()
        self.assertEqual(t.flight.sample['foot_slide'].item(),0.)
        self.assertAlmostEqual(t.flight.abort_sample['foot_slide'].item(),.8,places=6)
        self.assertAlmostEqual(t.flight.totals['landing_peak_foot_slide'].item(),.8,places=6)

    def test_slip_during_recovery_blocks_remaining_credit_but_not_prior_touchdown(self):
        t=trajectory();t.fly();immediate=t.earned['distance'].clone()
        t.sample(slide=.8)
        for _ in range(30):t.sample()
        torch.testing.assert_close(t.earned['distance'],immediate)
        self.assertEqual(t.flight.totals['abort_slide'].item(),1.)
        self.assertEqual(t.flight.totals['stable_landings'].item(),0.)

    def test_termination_on_touchdown_step_revokes_both_event_parts(self):
        t=trajectory();t.fly()
        self.assertGreater(t.flight.points['distance_touchdown'].item(),0.)
        t.flight.reject_terminated(torch.ones(1,dtype=torch.bool))
        for value in t.flight.points.values():self.assertEqual(value.item(),0.)
        self.assertEqual(t.flight.recovery_bank.item(),0.)
        self.assertEqual(t.flight.totals['abort_termination'].item(),1.)

    def test_initial_fall_backward_fore_failure_and_subminimum_flight_pay_nothing(self):
        initial=trajectory()
        for _ in range(14):initial.sample(ground=False)
        initial.sample()
        self.assertEqual(initial.earned['distance'].item(),0.)
        for fore,vx in [(False,2.5),(True,-2.5)]:
            t=trajectory();t.sample(phase=.7,fore=fore,vx=vx)
            for _ in range(14):t.sample(ground=False,fore=fore,vx=vx)
            for _ in range(25):t.sample(fore=fore,vx=vx)
            self.assertEqual(t.earned['distance'].item(),0.)
        short=trajectory();short.sample(phase=.7);short.sample(ground=False,vz=.4);short.sample()
        self.assertEqual(short.earned['distance'].item(),0.)

    def test_cycle_best_bounds_repeated_flights_and_reset_isolates_environments(self):
        t=trajectory(n=2);t.fly()
        for _ in range(24):t.sample(phase=.1)
        first=t.earned['distance'].clone()
        t.fly()
        for _ in range(24):t.sample(phase=.1)
        torch.testing.assert_close(t.earned['distance'],first,atol=1e-6,rtol=0)
        t.sample(slide=.8)  # No pending recovery: no extra abort.
        before=t.position.clone();t.flight.reset(torch.tensor([0]))
        torch.testing.assert_close(t.position,before)
        for value in (*t.flight.totals.values(),*t.flight.sample.values(),*t.flight.abort_sample.values()):
            self.assertEqual(torch.count_nonzero(value[0]).item(),0)
        self.assertEqual(t.flight.totals['stable_landings'][1].item(),2.)

    def test_event_payout_is_independent_of_control_step_duration(self):
        results=[]
        for dt in (.01,.02):
            t=trajectory(control_dt=dt);t.fly()
            for _ in range(24):t.sample(phase=.1)
            results.append(t.earned)
        for key in results[0]:torch.testing.assert_close(results[0][key],results[1][key])

    def test_new_config_only_changes_reward_credit_and_preserves_goals_and_safety(self):
        old=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_velocity_guard.json')
        new=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_landing_credit.json')
        changed={key for key in old.to_dict() if getattr(old,key)!=getattr(new,key)}
        self.assertEqual(changed,{'leap_progress_exponent','flight_touchdown_distance_fraction',
                                  'flight_landing_distance_scaled','flight_takeoff_weight'})
        for distance,seconds in ((.07,.04),(.17,.09),(.4,.20)):
            self.assertGreater(leap_score(new,distance,seconds).pending_reward.item(),
                               leap_score(old,distance,seconds).pending_reward.item())
            self.assertEqual(leap_score(new,distance,seconds).qualified_events.item(),0.)
        full=leap_score(new,.51,.255)
        self.assertEqual(full.qualified_events.item(),1.)
        self.assertEqual(full.pending_reward.item(),1.)

    def test_invalid_fraction_rejected_and_old_config_keeps_deferred_behavior(self):
        for value in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):settings(flight_touchdown_distance_fraction=value)
        with self.assertRaises(ValueError):settings(flight_shaping_enabled=False)
        old=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_velocity_guard.json')
        self.assertEqual(old.flight_touchdown_distance_fraction,0.)

    def test_manifest_remains_compatible_with_checkpoint_original_source(self):
        tree=ast.parse((ROOT/'scripts/training/gallop_baseline.py').read_text())
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        block=next(node for node in main.body if isinstance(node,ast.If)
                   and ast.unparse(node.test)=="getattr(settings, 'flight_shaping_enabled', False)")
        old=settings().to_dict()
        for key in ('flight_airtime_weight','flight_clearance_requires_airborne','flight_slide_weight','flight_slide_soft_speed'):
            del old[key]
        oldest={key:value for key,value in old.items() if key!='flight_touchdown_distance_fraction'}
        for core,s,version in ((SimpleNamespace(FlightStabilityTracker=object),SimpleNamespace(**oldest),1),
                               (SimpleNamespace(FlightStabilityTracker=FlightStabilityTracker),SimpleNamespace(**old),2),
                               (SimpleNamespace(FlightStabilityTracker=FlightStabilityTracker),settings(),3)):
            namespace={'manifest':{},'gallop_core':core,'settings':s}
            exec(compile(ast.Module(body=[block],type_ignores=[]),'flight_manifest','exec'),namespace)
            design=namespace['manifest']['flight_design']
            self.assertEqual(design['version'],version)
            if version==1:self.assertNotIn('touchdown_distance_fraction',design)
            else:self.assertEqual(design['touchdown_distance_fraction'],.25)


if __name__=='__main__':unittest.main()
