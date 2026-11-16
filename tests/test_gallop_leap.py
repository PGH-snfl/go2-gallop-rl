"""Physics-sampled takeoff/landing distances and bounded event rewards; no PPO."""
from dataclasses import replace
import math
from pathlib import Path
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,LeapDistanceTracker,SuccessSpeedCurriculum,phase_targets


def tracker(n=1,warmup=0.):
    settings=replace(GallopSettings(),metric_warmup_seconds=warmup,leap_reward_weight=1.)
    return LeapDistanceTracker(torch.zeros(n,3),settings)


def sample(t,position=(0.,0.,.35),ground=True,yaw=0.,eligible=True):
    n=len(t.elapsed)
    p=torch.tensor(position).expand(n,3).clone()
    q=torch.tensor([math.cos(yaw/2),0.,0.,math.sin(yaw/2)]).expand(n,4)
    force=torch.zeros(n,4)
    if ground:force[:,0]=100.
    t.observe(p,q,force,.005,torch.full((n,),eligible,dtype=torch.bool))


def flight(t,distance=.12,seconds=.04,start=0.,yaw=0.,eligible=True,sideways=False):
    t.begin_step(torch.zeros(len(t.elapsed),dtype=torch.bool))
    axis=(-math.sin(yaw),math.cos(yaw)) if sideways else (math.cos(yaw),math.sin(yaw))
    sample(t,(start,0.,.35),ground=False,yaw=yaw,eligible=eligible)
    steps=round(seconds/.005)
    for i in range(1,steps+1):
        sample(t,(start+axis[0]*distance*i/steps,axis[1]*distance*i/steps,.35),ground=i==steps,yaw=yaw,eligible=eligible)


class GallopLeapTests(unittest.TestCase):
    def test_three_mps_preset_is_fixed_and_keeps_fast_reference(self):
        s=GallopSettings.load(ROOT/'configs/gallop_3mps_leap.json')
        self.assertEqual(SuccessSpeedCurriculum(s).bounds(),(3.,3.))
        self.assertTrue((torch.empty(100).uniform_(*SuccessSpeedCurriculum(s).bounds())==3.).all())
        target=phase_targets(torch.zeros(1),torch.tensor([3.]),s)
        self.assertAlmostEqual(target['period'].item(),.32,places=6)
        self.assertAlmostEqual((target['rom_thigh']/1.8).item(),.75,places=6)

    def test_twelve_cm_and_forty_ms_earns_full_score_on_landing(self):
        t=tracker();sample(t);flight(t)
        self.assertEqual(t.events.item(),1);self.assertEqual(t.qualified_events.item(),1)
        self.assertAlmostEqual(t.last_distance.item(),.12,places=6)
        self.assertAlmostEqual(t.last_seconds.item(),.04,places=6)
        self.assertAlmostEqual(t.pending_reward.item(),1.,places=6)

    def test_below_distance_or_time_has_only_small_shaping_credit(self):
        for distance,seconds in [(.06,.04),(.12,.02),(.06,.02)]:
            t=tracker();sample(t);flight(t,distance,seconds)
            self.assertEqual(t.qualified_events.item(),0)
            self.assertGreater(t.pending_reward.item(),0)
            self.assertLessEqual(t.pending_reward.item(),.25)

    def test_grounded_forward_motion_cannot_be_rewarded_as_a_leap(self):
        t=tracker();sample(t)
        for i in range(1,20):sample(t,(float(i),0.,.35))
        self.assertEqual(t.events.item(),0);self.assertEqual(t.pending_reward.item(),0)

    def test_initial_free_fall_does_not_count(self):
        t=tracker()
        for i in range(20):sample(t,(i*.1,0.,.35),ground=False)
        sample(t,(3.,0.,.35),ground=True)
        self.assertEqual(t.events.item(),0);self.assertEqual(t.pending_reward.item(),0)

    def test_flight_started_before_warmup_is_ignored(self):
        t=tracker(warmup=.05);sample(t);flight(t,.3,.1)
        self.assertEqual(t.events.item(),0)
        flight(t,.12,.04,start=.3)
        self.assertEqual(t.qualified_events.item(),1)

    def test_backward_and_sideways_displacement_do_not_earn_reward(self):
        for distance,sideways in [(-.12,False),(.12,True)]:
            t=tracker();sample(t);flight(t,distance,sideways=sideways)
            self.assertEqual(t.qualified_events.item(),0)
            self.assertEqual(t.pending_reward.item(),0)

    def test_distance_uses_takeoff_heading_even_if_body_turns(self):
        t=tracker();sample(t,yaw=math.pi/2)
        sample(t,ground=False,yaw=math.pi/2)
        for i in range(1,9):sample(t,(0.,.12*i/8,.35),ground=i==8,yaw=0.)
        self.assertAlmostEqual(t.last_distance.item(),.12,places=6)
        self.assertEqual(t.qualified_events.item(),1)

    def test_phase_wrap_does_not_split_a_real_flight(self):
        t=tracker();sample(t);sample(t,ground=False)
        for i in range(1,4):sample(t,(.015*i,0.,.35),ground=False)
        t.begin_step(torch.ones(1,dtype=torch.bool))
        self.assertTrue(t.active.item())
        for i in range(4,9):sample(t,(.015*i,0.,.35),ground=i==8)
        self.assertEqual(t.qualified_events.item(),1)
        self.assertAlmostEqual(t.last_distance.item(),.12,places=6)

    def test_reward_is_capped_per_phase_cycle(self):
        t=tracker();sample(t);flight(t)
        first=t.pending_reward.item();flight(t,start=.12,distance=.3,seconds=.1)
        self.assertEqual(t.pending_reward.item(),0)
        self.assertEqual(t.qualified_events.item(),2)
        t.begin_step(torch.ones(1,dtype=torch.bool))
        # flight() preserves the reset budget while clearing step pulses.
        flight(t,start=.42)
        self.assertAlmostEqual(first+t.pending_reward.item(),2.,places=6)

    def test_forelimb_ineligible_flight_does_not_consume_reward_budget(self):
        t=tracker();sample(t);flight(t,eligible=False)
        self.assertEqual(t.pending_reward.item(),0);self.assertEqual(t.cycle_best.item(),0)
        flight(t,start=.12,eligible=True)
        self.assertAlmostEqual(t.pending_reward.item(),1.,places=6)

    def test_unfinished_flight_and_one_environment_reset_cannot_fake_distance(self):
        t=tracker(2);sample(t);sample(t,ground=False)
        for i in range(1,5):sample(t,(.015*i,0.,.35),ground=False)
        self.assertEqual(t.events.sum().item(),0);self.assertEqual(t.pending_reward.sum().item(),0)
        t.reset(torch.tensor([0]))
        for i in range(5,9):sample(t,(.015*i,0.,.35),ground=i==8)
        self.assertEqual(t.qualified_events.tolist(),[0.,1.])
        self.assertEqual(t.pending_reward[0].item(),0)

    def test_one_remaining_contact_blocks_takeoff(self):
        t=tracker();sample(t)
        q=torch.tensor([[1.,0.,0.,0.]])
        for i in range(20):t.observe(torch.tensor([[.1*i,0.,.35]]),q,torch.tensor([[4.,0.,0.,0.]]),.005,torch.ones(1,dtype=torch.bool))
        self.assertFalse(t.active.item());self.assertEqual(t.events.item(),0)

    def test_invalid_distance_and_weight_are_rejected(self):
        for values in ({'leap_min_distance':0.},{'leap_min_distance':float('nan')},{'leap_reward_weight':-1.}):
            with self.assertRaises(ValueError):replace(GallopSettings(),**values)


if __name__=='__main__':unittest.main()
