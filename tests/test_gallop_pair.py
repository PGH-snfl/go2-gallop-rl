"""Pair coordination costs preserve lead/lag and detect persistent one-leg use."""
from dataclasses import replace
import unittest

import torch

from isaac_task.gallop_core import GallopSettings,PairBalanceTracker,pair_cycle_penalties,pair_joint_penalty,phase_targets


class GallopPairTests(unittest.TestCase):
    def setUp(self):self.settings=GallopSettings()

    def test_reference_gallop_lead_lag_is_not_punished(self):
        phase=torch.linspace(0,1,1000)[:-1]
        t=phase_targets(phase,torch.full_like(phase,2.),self.settings)
        reference=torch.stack((torch.zeros_like(t['thigh']),t['thigh'],t['calf']),dim=2)
        self.assertGreater((reference[:,0,1]-reference[:,1,1]).abs().max().item(),.5)
        self.assertEqual(pair_joint_penalty(reference,reference,self.settings).max().item(),0.)
        excursion=(reference.max(dim=0).values-reference.min(dim=0).values)[None,:,1:]
        stance=t['stance'].float().mean(dim=0)[None]
        costs,_=pair_cycle_penalties(excursion,stance*100.,stance,torch.ones(1),self.settings)
        for cost in costs.values():self.assertEqual(cost.item(),0.)

    def test_frozen_right_foreleg_has_motion_cost(self):
        phase=torch.linspace(0,1,1000)[:-1]
        t=phase_targets(phase,torch.full_like(phase,2.),self.settings)
        reference=torch.stack((torch.zeros_like(t['thigh']),t['thigh'],t['calf']),dim=2)
        joints=reference.clone();joints[:,1]=reference[:,1].mean(dim=0)
        self.assertGreater(pair_joint_penalty(joints,reference,self.settings).mean().item(),.01)
        excursion=(joints.max(dim=0).values-joints.min(dim=0).values)[None,:,1:]
        costs,_=pair_cycle_penalties(excursion,torch.ones(1,4),torch.full((1,4),.4),torch.ones(1),self.settings)
        self.assertGreater(costs['rom'].item(),.4)

    def test_hip_mirror_and_common_offsets_are_allowed(self):
        reference=torch.zeros(1,4,3)
        q=reference+torch.tensor([[[.4,.3,.3],[-.4,.3,.3],[.4,.3,.3],[-.4,.3,.3]]])
        self.assertEqual(pair_joint_penalty(q,reference,self.settings).item(),0.)
        q[:,1,0]=.4
        self.assertGreater(pair_joint_penalty(q,reference,self.settings).item(),0.)

    def test_one_leg_load_dominance_is_detected_in_each_pair(self):
        impulse=torch.tensor([[1.,1.,1.,1.],[2.,0.,1.,1.],[1.,1.,0.,2.]])
        costs,metrics=pair_cycle_penalties(torch.ones(3,4,2),impulse,torch.full((3,4),.4),torch.ones(3),self.settings)
        self.assertEqual(costs['load'][0].item(),0.)
        torch.testing.assert_close(costs['load'][1:],torch.tensor([.5,.5]))
        torch.testing.assert_close(metrics['load'][1:],torch.tensor([[1.,0.],[0.,1.]]))

    def test_persistent_stance_difference_is_detected_without_force_difference(self):
        costs,_=pair_cycle_penalties(torch.ones(1,4,2),torch.ones(1,4),torch.tensor([[1.,0.,.35,.45]]),torch.ones(1),self.settings)
        self.assertEqual(costs['load'].item(),0.)
        self.assertGreater(costs['stance'].item(),.4)

    def test_empty_airborne_cycle_has_finite_zero_costs(self):
        costs,metrics=pair_cycle_penalties(torch.zeros(2,4,2),torch.zeros(2,4),torch.zeros(2,4),torch.zeros(2),self.settings)
        for value in (*costs.values(),*metrics.values()):
            self.assertTrue(torch.isfinite(value).all());self.assertEqual(value.count_nonzero().item(),0)

    def test_partial_warmup_and_reset_cycles_cannot_leak_between_environments(self):
        q=torch.zeros(2,4,3);tracker=PairBalanceTracker(q,self.settings)
        wrapped=torch.ones(2,dtype=torch.bool)
        for _ in range(3):tracker.observe(q,torch.ones(2,4),torch.ones(2,4,dtype=torch.bool),.02,wrapped)
        tracker.finish_cycles(wrapped,q)
        self.assertEqual(tracker.valid_cycles.count_nonzero().item(),0)
        force=torch.tensor([[100.,0.,50.,50.],[50.,50.,50.,50.]])
        for _ in range(20):tracker.observe(q,force,force>0,.02,wrapped)
        tracker.finish_cycles(wrapped,q)
        self.assertGreater(tracker.costs['load'][0].item(),0)
        self.assertEqual(tracker.costs['load'][1].item(),0)
        self.assertEqual(tracker.valid_cycles.tolist(),[1.,1.])
        tracker.reset(torch.tensor([0]),q)
        self.assertEqual(tracker.valid_cycles.tolist(),[0.,1.])
        for _ in range(5):tracker.observe(q,force,force>0,.02,torch.tensor([False,True]))
        tracker.finish_cycles(wrapped,q)
        self.assertEqual(tracker.valid_cycles.tolist(),[0.,2.])
        self.assertEqual(tracker.costs['load'][0].item(),0.)

    def test_invalid_penalty_configuration_is_rejected(self):
        for change in ({'pair_rom_tolerance':1.},{'pair_load_weight':-1.},{'pair_joint_scale':0.},{'pair_load_weight':float('nan')}):
            with self.assertRaises(ValueError):replace(self.settings,**change)


if __name__=='__main__':unittest.main()
