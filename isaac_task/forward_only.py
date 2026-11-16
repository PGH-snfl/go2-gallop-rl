"""A separate command-distribution experiment for positive forward tracking."""
import math

FORWARD_SPEED_RANGE = (.2, .4)


def configure_forward_only(cfg, speed_range=FORWARD_SPEED_RANGE):
    speed_range=tuple(float(v) for v in speed_range)
    if len(speed_range)!=2 or not all(math.isfinite(v) for v in speed_range) or not (0 < speed_range[0] <= speed_range[1]):
        raise ValueError('forward speed range requires 0 < minimum <= maximum and finite values')
    command = cfg.commands.base_velocity
    command.rel_standing_envs = 0.
    command.heading_command = False
    for ranges in [command.ranges, command.limit_ranges]:
        ranges.lin_vel_x = speed_range
        ranges.lin_vel_y = (0., 0.)
        ranges.ang_vel_z = (0., 0.)
    # Upstream/fixed symmetric range expansion would add negative vx or lateral
    # commands. This experiment keeps one positive range until tracking works.
    cfg.curriculum.lin_vel_cmd_levels = None
    return dict(lin_vel_x=list(speed_range),lin_vel_y=[0.,0.],ang_vel_z=[0.,0.],
                rel_standing_envs=0.,command_curriculum=False)
