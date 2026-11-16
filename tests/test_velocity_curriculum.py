"""Regression tests for asynchronous episode ends, partial episodes and stage credit."""
from types import SimpleNamespace as NS
import unittest

import torch

from isaac_task.velocity_curriculum import PeriodicVelocityCurriculum, VelocityCurriculumSettings


class VelocityCurriculumTests(unittest.TestCase):
    def make(self, min_episodes=2):
        command = NS(cfg=NS(ranges=NS(lin_vel_x=[-.1, .1], lin_vel_y=[-.1, .1]),
                            limit_ranges=NS(lin_vel_x=[-1., 1.], lin_vel_y=[-.4, .4])))
        env = NS(num_envs=2, device='cpu', max_episode_length=10, max_episode_length_s=1., common_step_counter=0,
                 command_manager=NS(get_term=lambda name: command),
                 reward_manager=NS(_episode_sums={'track_lin_vel_xy': torch.zeros(2)},
                                   get_term_cfg=lambda name: NS(weight=1.5)))
        controller = PeriodicVelocityCurriculum(env, settings=VelocityCurriculumSettings(interval_steps=10,min_episodes=min_episodes))
        return env, controller, command.cfg.ranges

    def test_async_resets_expand_without_a_reset_at_decision_and_reject_old_stage(self):
        env, c, ranges = self.make()
        resets = {3: [0], 7: [1], 13: [0], 17: [1], 23: [0], 27: [1], 33: [0], 37: [1]}
        for step in range(1, 41):
            env.common_step_counter = step
            c.begin_step()
            env.reward_manager._episode_sums['track_lin_vel_xy'] += .145
            if step in resets:
                ids = torch.tensor(resets[step])
                c.collect_before_reset(ids)
                env.reward_manager._episode_sums['track_lin_vel_xy'][ids] = 0
            c.after_step()
        self.assertEqual([e['step'] for e in c.events if e['expanded']], [20, 40])
        self.assertEqual(c.discarded_initial, 2)
        self.assertEqual(c.discarded_cross_stage, 2)
        self.assertAlmostEqual(ranges.lin_vel_x[1], .3)
        self.assertEqual(c.events[0]['reason'], 'waiting_for_episodes')
        self.assertEqual(c.events[2]['reason'], 'waiting_for_episodes')

    def test_short_failures_do_not_receive_full_episode_credit(self):
        env, c, ranges = self.make()
        c.first_episode[:] = False
        c.actual_steps[:] = 2
        env.reward_manager._episode_sums['track_lin_vel_xy'][:] = .29
        c.collect_before_reset(torch.arange(2))
        env.common_step_counter = 10
        c.after_step()
        self.assertAlmostEqual(c.last_score, .29, places=6)
        self.assertEqual(c.events[-1]['reason'], 'reward_below_threshold')
        self.assertEqual(ranges.lin_vel_x[1], .1)
        self.assertEqual(c.episodes, 0)

    def test_evidence_survives_a_window_with_too_few_episodes(self):
        env, c, ranges = self.make()
        c.first_episode[:] = False
        for i, step in enumerate([10, 20]):
            c.actual_steps[i] = 10
            env.reward_manager._episode_sums['track_lin_vel_xy'][i] = 1.45
            c.collect_before_reset(torch.tensor([i]))
            env.common_step_counter = step
            c.after_step()
        self.assertEqual([e['reason'] for e in c.events], ['waiting_for_episodes', 'expanded'])
        self.assertAlmostEqual(ranges.lin_vel_x[1], .2)

    def test_caps_and_repeated_calls_do_not_create_extra_stages(self):
        env, c, ranges = self.make()
        ranges.lin_vel_x = [-1., 1.]
        ranges.lin_vel_y = [-.4, .4]
        c.actual_steps[:] = 10
        env.reward_manager._episode_sums['track_lin_vel_xy'][:] = 1.45
        c.collect_before_reset(torch.arange(2))
        env.common_step_counter = 10
        c.after_step()
        c.after_step()
        self.assertEqual(len(c.events), 1)
        self.assertEqual(c.events[-1]['reason'], 'at_limit')
        self.assertEqual(c.stage, 0)
        self.assertEqual(c.expansions, 0)


if __name__ == '__main__':
    unittest.main()
