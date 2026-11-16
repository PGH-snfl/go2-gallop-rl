"""Distance preference, recovery credit and compatibility for the fixed50cm task."""
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,LeapDistanceTracker,HeadingMotionTracker,FixedGallopGoal
from tests.test_gallop_flight import Trajectory


def settings():return GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_distance_focus.json')


def leap_score(s,distance,seconds):
    s=replace(s,metric_warmup_seconds=0.)
    p=torch.tensor([[0.,0.,.35]]);t=LeapDistanceTracker(p,s)
    q=torch.tensor([[1.,0.,0.,0.]]);f=torch.tensor([[100.,0.,0.,0.]]);eligible=torch.ones(1,dtype=torch.bool)
    t.begin_step(torch.zeros(1,dtype=torch.bool));t.observe(p,q,f,.005,eligible)
    t.observe(p,q,f*0,.005,eligible)
    count=round(seconds/.005)
    for i in range(1,count+1):
        p[:,0]=distance*i/count
        t.observe(p,q,f if i==count else f*0,.005,eligible)
    return t


class DistanceFocusTests(unittest.TestCase):
    def test_short_jump_credit_falls_but_near_target_and_full_credit_rise(self):
        old=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_edu.json');new=settings()
        scores=[]
        for distance,seconds in [(.17,.09),(.4,.20),(.51,.255)]:
            a=leap_score(old,distance,seconds);b=leap_score(new,distance,seconds)
            scores.append((float(a.pending_reward)*old.leap_reward_weight,float(b.pending_reward)*new.leap_reward_weight))
        self.assertLess(scores[0][1],scores[0][0])
        self.assertGreater(scores[1][1],scores[1][0])
        self.assertEqual(scores[2],(8.,24.))
        self.assertLess(scores[0][1],scores[1][1]);self.assertLess(scores[1][1],scores[2][1])

    def test_distance_alone_or_time_alone_does_not_qualify(self):
        for distance,seconds in [(.51,.10),(.2,.25)]:
            self.assertEqual(leap_score(settings(),distance,seconds).qualified_events.item(),0.)

    def test_landing_scaling_keeps_recovery_and_distance_credit_separate(self):
        original=Trajectory();scaled=Trajectory()
        scaled.s=replace(scaled.s,flight_landing_distance_scaled=True)
        scaled.leap.cfg=scaled.s;scaled.flight.cfg=scaled.s
        for t in (original,scaled):
            t.sample(phase=.7);t.sample(ground=False,vz=.4)
            for _ in range(5):t.sample(ground=False,vz=-.2)
            t.sample(ground=True,vz=-.2)
            self.assertEqual(t.earned['landing'].item(),0.)
            for _ in range(20):t.sample(phase=.1)
        self.assertGreater(scaled.earned['landing'].item(),0.)
        self.assertLess(scaled.earned['landing'].item(),original.earned['landing'].item())
        torch.testing.assert_close(scaled.earned['distance'],original.earned['distance'])
        self.assertEqual(scaled.flight.totals['stable_landings'].item(),original.flight.totals['stable_landings'].item())
        scaled.flight.reset(torch.tensor([0]));self.assertEqual(scaled.flight.recovery_progress.item(),0.)

    def test_forward_progress_is_broader_without_stationary_or_backward_credit(self):
        def score(s,vx):
            root=torch.tensor([[0.,0.,.35]]);tracker=HeadingMotionTracker(root,s)
            feet=torch.tensor([[[.2,.135,.02],[.2,-.135,.02],[-.2,.135,.02],[-.2,-.135,.02]]])
            terms,_=tracker.observe(torch.tensor([[1.,0.,0.,0.]]),root,feet,torch.ones(1,4,dtype=torch.bool),
                torch.tensor([[vx,0.,0.]]),torch.tensor([2.]),.02,angular_velocity=torch.zeros(1,3))
            return float(terms['forward_progress'])
        new=settings();old=replace(new,progress_speed_scale_fraction=.25)
        self.assertGreater(score(new,1.4),score(old,1.4));self.assertGreater(score(new,2.6),score(old,2.6))
        self.assertEqual(score(new,0.),0.);self.assertEqual(score(new,-1.),0.)

    def test_goals_and_all_safety_limits_are_preserved(self):
        new=settings();old=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_edu.json')
        for key in old.to_dict():
            if key.startswith(('edu_','guarded_','startup_')):self.assertEqual(getattr(old,key),getattr(new,key),key)
        goal=FixedGallopGoal(new)
        for _ in range(1024):goal.record(True,0,{})
        self.assertIsNone(goal.decide(10000));self.assertEqual(goal.events,[]);self.assertEqual(goal.cap,2.)
        self.assertEqual(new.leap_min_distance,.5);self.assertEqual(new.flight_min_seconds,.24)

    def test_invalid_reward_scales_are_rejected(self):
        for change in [dict(velocity_tracking_std=0.),dict(velocity_tracking_weight=-1.),dict(progress_speed_scale_fraction=float('nan')),
                       dict(leap_progress_exponent=.5),dict(flight_landing_distance_scaled=1)]:
            with self.assertRaises(ValueError):replace(settings(),**change)


if __name__=='__main__':unittest.main()
