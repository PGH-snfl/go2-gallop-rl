"""Offline native-curriculum probes and aggregation of frozen-policy diagnostics."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from gpd.paths import ROOT
from gpd.portable import external_path, resolve_go2_asset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = external_path('unitree_rl_lab_official') / 'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py'
    spec = importlib.util.spec_from_file_location('native_curriculum_probe', source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    probes = []
    for step, reward in [(999, 1.45), (1000, 1.19), (1000, 1.45), (1001, 1.45)]:
        term = NS(cfg=NS(ranges=NS(lin_vel_x=(-.1, .1), lin_vel_y=(-.1, .1)),
                         limit_ranges=NS(lin_vel_x=(-1., 1.), lin_vel_y=(-.4, .4))))
        env = NS(command_manager=NS(get_term=lambda name: term),
                 reward_manager=NS(_episode_sums={'track_lin_vel_xy': torch.tensor([reward * 20])},
                                   get_term_cfg=lambda name: NS(weight=1.5)),
                 max_episode_length_s=20., max_episode_length=1000, common_step_counter=step, device='cpu')
        result = native.lin_vel_cmd_levels(env, torch.tensor([0]))
        probes.append(dict(step=step, normalized_track_reward=reward, threshold=1.2,
                           returned_vx_limit=float(result), expanded=float(result) > .10001))
    assert [p['expanded'] for p in probes] == [False, False, True, False]
    diagnostics = []
    for directory in a.diagnostics:
        status = json.loads((directory / 'status.json').read_text())
        if status['status'] != 'completed' or status['policy_updates'] != 0:
            raise RuntimeError(f'diagnostic did not complete without updates: {directory}')
        s = json.loads((directory / 'summary.json').read_text())
        calls = [c for c in json.loads((directory / 'curriculum_calls.json').read_text()) if c['step'] > 0]
        reward_pass = [c for c in calls if c['reward_gate']]
        s['reward_gate_pass_but_global_gate_fail'] = sum(not c['global_step_gate'] for c in reward_pass)
        s['reward_gate_pass_fraction'] = len(reward_pass) / len(calls) if calls else None
        s['mean_sampled_xy_speed'] = float(np.mean([x['mean_xy_speed'] for x in s['samples']]))
        s['mean_sampled_all_feet_contact_fraction'] = float(np.mean([x['all_feet_contact_fraction'] for x in s['samples']]))
        s['mean_sampled_air_time_gate_fraction'] = float(np.mean([x['air_time_command_gate_fraction'] for x in s['samples']]))
        s['directory'] = str(directory.resolve())
        s.pop('samples')
        diagnostics.append(s)
    report = dict(native_curriculum_source=str(source),
        native_curriculum_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_function_probes=probes,
        stationary_linear_reward=[dict(vx_command=v, actual_vx=0., weighted_reward=1.5*math.exp(-v*v/.25),
                                     fraction_of_maximum=math.exp(-v*v/.25)) for v in [.05, .1, .2, .4, .8]],
        expected_initial_air_time_gate_fraction=.9 * (1 - math.pi/4),
        air_time_gate_assumption='independent uniform vx,vy in [-.1,.1], 10% standing, norm(vxy)>0.1',
        zero_offset_absence_probability_under_iid_uniform={str(n): (999/1000)**n for n in [256,4096]},
        phase_probability_caveat='Illustrative initial iid phase probabilities, not a reconstruction of the past run.',
        diagnostics=diagnostics,
        limits='Frozen final policy and unchanged native callbacks; historical PPO reset/curriculum callbacks were not saved.')
    (a.output / 'diagnosis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
