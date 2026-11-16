"""Opt-in startup recovery: speed mismatch, pose failure and reset behavior."""
from dataclasses import replace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,WalkingToGallop
from tests.test_gallop_startup import advance


class PilotTests(unittest.TestCase):
    def cfg(self):return GallopSettings.load(ROOT/'configs/gallop_pilot_18cm.json')

    def test_speed_mismatch_does_not_lock_handover(self):
        c=WalkingToGallop(torch.zeros(1),self.cfg())
        advance(c,4.)
        self.assertEqual(c.stage.item(),2)
        advance(c,3.2,velocity=1.)
        self.assertEqual(c.stage.item(),3)
        self.assertAlmostEqual(c.command.item(),2.5)
        self.assertFalse(c.failed.item())

    def test_bad_pose_pauses_mixture_and_expires_as_failure(self):
        c=WalkingToGallop(torch.zeros(1),self.cfg());advance(c,4.)
        mix=c.mix.clone();advance(c,3.,velocity=1.,tilt=.4)
        torch.testing.assert_close(c.mix,mix)
        self.assertEqual(c.stage.item(),2)
        advance(c,5.1,velocity=1.,tilt=.4)
        self.assertTrue(c.failed.item())
        c.reset(torch.tensor([0]));self.assertFalse(c.failed.item())
        self.assertEqual(c.command.item(),0.)

    def test_legacy_still_waits_for_speed_and_inactive_stays_newborn(self):
        legacy=replace(self.cfg(),startup_pose_handover=False)
        c=WalkingToGallop(torch.zeros(1),legacy);advance(c,4.)
        advance(c,15.,velocity=0.)
        self.assertEqual(c.stage.item(),2);self.assertFalse(c.failed.item())
        d=WalkingToGallop(torch.zeros(1),self.cfg())
        advance(d,20.,active=torch.tensor([False]))
        self.assertEqual(d.times.sum().item(),0.);self.assertFalse(d.failed.item())

    def test_fixed_pilot_preserves_every_motor_guard_and_reward_weight(self):
        old=GallopSettings.load(ROOT/'configs/gallop_guarded_filter_free.json').to_dict()
        new=self.cfg().to_dict()
        changed={k for k in old if old[k]!=new[k]}
        self.assertEqual(changed,{'max_speed','stabilization_min_leap_distance','leap_min_distance','startup_pose_handover'})
        self.assertEqual(new['guarded_max_intervention'],1.)


if __name__=='__main__':unittest.main()
