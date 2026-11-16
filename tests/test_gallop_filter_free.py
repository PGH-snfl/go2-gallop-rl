"""Free filter intervention from promotion while preserving physical gates."""
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings
from tests.test_gallop_guarded import fixture
from tests.test_gallop_pair_env import state_class


def passing_episode(config,intervention):
    _,_,env,_,_=fixture()
    state=state_class()(env,config,None,None,False,True,walking_policy=lambda obs:torch.zeros(3,12))
    env.gallop=state
    state.age[:]=1500;state.first[:]=False;state.startup.stage[:]=3;state.startup.times[:,3]=20
    env.reset_terminated.zero_();env.reset_time_outs[:]=True
    for key,value in dict(points=1000,match=1000,command=2500,clip=50,roll2=1,pitch2=1).items():state.sums[key][:]=value
    state.cycles[:]=20;state.flight_cycles[:]=20;state.rom_sum[:]=20;state.fore_rom_sum[:]=20;state.fore_participation[:]=20
    state.total_impulse[:]=25;state.events[:]=20;state.good_events[:]=20;state.sequences[:]=20
    state.flight.totals['stable_landings'][:]=20;state.flight.totals['aborted_landings'][:]=2
    state.leap.events[:]=20;state.leap.qualified_events[:]=16
    state.guard_sums['intervention'][:]=intervention*1000
    for name in ('front','hind'):
        state.stability_sums[name+'_support'][:]=100;state.stability_sums[name+'_width_good'][:]=100
    return state,env


class FilterFreePromotionTests(unittest.TestCase):
    def test_profile_changes_only_the_intervention_threshold(self):
        old=GallopSettings.load(ROOT/'configs/gallop_guarded_stability.json').to_dict()
        new=GallopSettings.load(ROOT/'configs/gallop_guarded_filter_free.json').to_dict()
        self.assertEqual([k for k in old if old[k]!=new[k]],['guarded_max_intervention'])
        self.assertEqual(new['guarded_max_intervention'],1.)
        self.assertFalse(new['relearn_enabled'])

    def test_real_episode_accepts_50_and_100_percent_intervention(self):
        for fraction in (0.,.5088516746,1.):
            for profile in ('stability','filter_free'):
                with self.subTest(fraction=fraction,profile=profile):
                    cfg=GallopSettings.load(ROOT/f'configs/gallop_guarded_{profile}.json')
                    state,_=passing_episode(cfg,fraction);state.complete_episodes([0])
                    expected=fraction<=cfg.guarded_max_intervention
                    self.assertEqual(state.last_episode['guarded_limits'],float(expected))
                    self.assertEqual(state.last_episode['success'],float(expected))
                    self.assertEqual(state.curriculum.successes,int(expected))

    def test_safety_landing_width_and_angle_request_checks_still_block_success(self):
        cfg=GallopSettings.load(ROOT/'configs/gallop_guarded_filter_free.json')
        for failure in ('actual','landing','width','raw_angle','termination'):
            with self.subTest(failure=failure):
                state,env=passing_episode(cfg,1.)
                if failure=='actual':state.guard_sums['actual'][:]=1
                elif failure=='landing':state.flight.totals['aborted_landings'][:]=10
                elif failure=='width':state.stability_sums['front_width_good'][:]=70
                elif failure=='raw_angle':state.sums['clip'][:]=110
                else:env.reset_terminated[:]=True
                state.complete_episodes([0]);self.assertEqual(state.last_episode['success'],0.)
                self.assertEqual(state.curriculum.successes,0)


if __name__=='__main__':unittest.main()
