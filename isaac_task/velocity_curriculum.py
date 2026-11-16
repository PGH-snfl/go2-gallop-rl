"""Periodic command curriculum; independent of reset timing and the episode buffer phase."""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class VelocityCurriculumSettings:
    interval_steps: int = 1000
    min_episodes: int = 32
    reward_fraction: float = .8
    range_increment: float = .1


class PeriodicVelocityCurriculum:
    """Collect before reset, decide after every step, and keep stage evidence separate.

    Rewards retain upstream normalization by the maximum episode duration. Early
    failures therefore lower the score instead of receiving full-duration credit.
    The first shortened episode and episodes crossing a range change are excluded.
    """

    def __init__(self, env, emit=None, settings=None):
        self.env = env
        self.settings = settings or VelocityCurriculumSettings(min_episodes=min(32, max(1, env.num_envs // 4)))
        if self.settings.interval_steps < 1 or self.settings.min_episodes < 1:
            raise ValueError('positive curriculum interval and episode count required')
        self.emit = emit
        self.actual_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.first_episode = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.episode_stage = torch.zeros_like(self.actual_steps)
        self.stage = self.expansions = self.episodes = 0
        self.reward_sum = 0.
        self.discarded_initial = self.discarded_cross_stage = 0
        self.next_decision_step = self.settings.interval_steps
        self.last_score = None
        self.events = []

    def begin_step(self):
        self.actual_steps += 1

    def collect_before_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        ids = ids[self.actual_steps[ids] > 0]
        if not len(ids):
            return
        initial_partial = self.first_episode[ids] & (self.actual_steps[ids] < self.env.max_episode_length)
        cross_stage = self.episode_stage[ids] != self.stage
        eligible = ~initial_partial & ~cross_stage
        self.discarded_initial += int(initial_partial.sum())
        self.discarded_cross_stage += int((cross_stage & ~initial_partial).sum())
        eligible_ids = ids[eligible]
        if len(eligible_ids):
            sums = self.env.reward_manager._episode_sums['track_lin_vel_xy'][eligible_ids]
            if not torch.isfinite(sums).all():
                raise RuntimeError('nonfinite curriculum reward')
            self.reward_sum += float(sums.sum())
            self.episodes += len(eligible_ids)
        self.actual_steps[ids] = 0
        self.first_episode[ids] = False
        self.episode_stage[ids] = self.stage

    def after_step(self):
        step = self.env.common_step_counter
        if step < self.next_decision_step:
            return
        self.next_decision_step = (step // self.settings.interval_steps + 1) * self.settings.interval_steps
        term = self.env.command_manager.get_term('base_velocity')
        ranges, limits = term.cfg.ranges, term.cfg.limit_ranges
        before = dict(lin_vel_x=list(ranges.lin_vel_x), lin_vel_y=list(ranges.lin_vel_y))
        threshold = self.env.reward_manager.get_term_cfg('track_lin_vel_xy').weight * self.settings.reward_fraction
        self.last_score = self.reward_sum / (self.episodes * self.env.max_episode_length_s) if self.episodes else None
        expanded = False
        reason = 'waiting_for_episodes'
        if self.episodes >= self.settings.min_episodes:
            reason = 'reward_below_threshold'
            if self.last_score > threshold:
                for key in before:
                    lo, hi = getattr(ranges, key)
                    cap_lo, cap_hi = getattr(limits, key)
                    setattr(ranges, key, [max(cap_lo, lo - self.settings.range_increment),
                                          min(cap_hi, hi + self.settings.range_increment)])
                expanded = any(not math.isclose(getattr(ranges, key)[i], before[key][i], abs_tol=1e-7)
                               for key in before for i in [0, 1])
                reason = 'expanded' if expanded else 'at_limit'
                if expanded:
                    self.stage += 1
                    self.expansions += 1
        event = dict(step=int(step), stage=self.stage, eligible_episodes=self.episodes,
                     normalized_track_reward=self.last_score, threshold=threshold, expanded=expanded,
                     reason=reason, before=before,
                     after={key: list(getattr(ranges, key)) for key in before},
                     discarded_initial=self.discarded_initial, discarded_cross_stage=self.discarded_cross_stage,
                     expansions=self.expansions)
        self.events.append(event)
        if self.emit:
            self.emit('VELOCITY_CURRICULUM', **event)
        # Keep evidence while waiting; completed decisions use a fresh window.
        if self.episodes >= self.settings.min_episodes:
            self.reward_sum = 0.
            self.episodes = 0

    def metrics(self):
        ranges = self.env.command_manager.get_term('base_velocity').cfg.ranges
        values = {'Curriculum/lin_vel_cmd_levels': float(ranges.lin_vel_x[1]),
                  'Curriculum/eligible_episodes': self.episodes,
                  'Curriculum/expansions': self.expansions,
                  'Curriculum/discarded_initial': self.discarded_initial,
                  'Curriculum/discarded_cross_stage': self.discarded_cross_stage}
        if self.last_score is not None:
            values['Curriculum/window_track_reward'] = self.last_score
        return values

    def summary(self):
        return dict(settings=asdict(self.settings), **self.metrics(), events=self.events)


def curriculum_env_class(base_class):
    """Build only after AppLauncher starts; no Isaac Lab imports at module import."""
    class CurriculumEnv(base_class):
        def __init__(self, *args, curriculum_emit=None, curriculum_enabled=True, **kwargs):
            self.velocity_curriculum = None
            super().__init__(*args, **kwargs)
            if curriculum_enabled:
                self.velocity_curriculum = PeriodicVelocityCurriculum(self, curriculum_emit)

        def _reset_idx(self, env_ids):
            if self.velocity_curriculum is not None:
                self.velocity_curriculum.collect_before_reset(env_ids)
            return super()._reset_idx(env_ids)

        def step(self, action):
            if self.velocity_curriculum is not None:
                self.velocity_curriculum.begin_step()
            result = super().step(action)
            if self.velocity_curriculum is not None:
                self.velocity_curriculum.after_step()
            return result

    return CurriculumEnv
