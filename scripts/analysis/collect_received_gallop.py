"""Analyze received run artifacts without Torch, Isaac or learning.

Requires NumPy, PyYAML, matplotlib, tensorboard and absl-py. Raw artifacts
remain under the selected output directory's inputs/server folder.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import pickletools
import subprocess
import zipfile

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gpd.paths import ROOT


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checkpoint_iteration(path):
    # Read the plain integer opcode; never execute the checkpoint pickle.
    with zipfile.ZipFile(path) as archive:
        key = next(k for k in archive.namelist() if k.endswith('/data.pkl'))
        ops = list(pickletools.genops(archive.read(key)))
    values = []
    for i, (op, arg, _) in enumerate(ops):
        if op.name in ('BINUNICODE', 'SHORT_BINUNICODE') and arg == 'iter':
            j = i + 1
            while ops[j][0].name in ('BINPUT', 'LONG_BINPUT', 'MEMOIZE'):
                j += 1
            if ops[j][0].name not in ('BININT', 'BININT1', 'BININT2', 'INT'):
                raise ValueError('Unexpected iteration representation')
            values.append(int(ops[j][1]))
    if len(values) != 1:
        raise ValueError('Ambiguous checkpoint iteration')
    return values[0]


def group_stats(rows):
    measured = [r for r in rows if r['gallop_metric_seconds'] > 0]
    events = sum(r['leap_events'] for r in rows)
    takeoffs = sum(r['flight_takeoffs'] for r in rows)
    criteria = ('tracking', 'pattern', 'flight', 'motion', 'front_load',
                'front_motion', 'front_cycle', 'stable', 'mature',
                'stabilization_pose', 'stabilization_width',
                'stabilization_landing', 'stabilization_distance', 'edu_limits')
    return {
        'episodes': len(rows),
        'eligible': sum(bool(r['eligible']) for r in rows),
        'successes': sum(bool(r['success']) for r in rows),
        'terminated': sum(bool(r['terminated']) for r in rows),
        'timed_out': sum(bool(r['timed_out']) for r in rows),
        'termination_timeout_overlap': sum(bool(r['terminated'] and r['timed_out']) for r in rows),
        'edu_terminated': sum(bool(r['edu_terminated']) for r in rows),
        'edu_termination_fraction': float(np.mean([r['edu_terminated'] for r in rows])),
        'startup_completed': sum(bool(r['startup_completed']) for r in rows),
        'startup_failed': sum(bool(r['startup_failed']) for r in rows),
        'startup_stage_counts': dict(Counter(int(r['startup_stage']) for r in rows)),
        'measured_episodes': len(measured),
        'measured_episode_mean_vx': float(np.mean([r['gallop_mean_vx'] for r in measured])),
        'maximum_leap_m': max(r['leap_distance_max'] for r in rows),
        'leap_events': events,
        'event_weighted_mean_leap_m': sum(r['leap_distance_mean'] * r['leap_events'] for r in rows) / events if events else None,
        'event_weighted_mean_air_seconds': sum(r['leap_air_seconds_mean'] * r['leap_events'] for r in rows) / events if events else None,
        'takeoff_weighted_mean_vz': sum(r['flight_takeoff_vz_mean'] * r['flight_takeoffs'] for r in rows) / takeoffs if takeoffs else None,
        'max_runtime_velocity_ratio': max(r['edu_peak_velocity_ratio'] for r in rows),
        'max_runtime_effort_ratio': max(r['edu_peak_effort_ratio'] for r in rows),
        'official_termination_by_startup_stage': {str(stage): sum(bool(r['edu_terminated']) for r in rows if int(r['startup_stage']) == stage)
                                                  for stage in sorted({int(r['startup_stage']) for r in rows})},
        'qualified_leap_events': sum(r['leap_qualified_events'] for r in rows),
        'measured_criterion_pass_fraction': {
            k: float(np.mean([r[k] for r in measured])) for k in criteria
        },
        'note': 'Episode-weighted fractions; terminated and timed_out can overlap. Not an independent evaluation success rate.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/20261117_server_results')
    parser.add_argument('--train', default='runs/isaac/train/unitree_go2_gallop/20261117_160234_3e2363')
    parser.add_argument('--play', default='runs/isaac/play/unitree_go2_gallop/20261117_190513_06271d')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    raw = out / 'inputs/server'
    train_rel, play_rel = args.train, args.play
    train, play = raw / train_rel, raw / play_rel
    inventory = read(out / 'server_archive_manifest.json')
    assert all(sha(raw / r['path']) == r['sha256'] for r in inventory['files'])
    ts, ps = read(train / 'status.json'), read(play / 'status.json')
    tm, pm = read(train / 'manifest.json'), read(play / 'manifest.json')
    settings = tm['gallop_settings']
    agent = yaml.safe_load((train / 'params/agent.yaml').read_text())
    # env.yaml contains Python tuple/slice tags: use a data-only loader.
    class DataLoader(yaml.SafeLoader):
        pass
    def tagged(loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)
    DataLoader.add_multi_constructor('tag:yaml.org,2002:python/', tagged)
    env_cfg = yaml.load((train / 'params/env.yaml').read_text(), Loader=DataLoader)
    dt = float(env_cfg['sim']['dt'] * env_cfg['decimation'])
    steps = agent['num_steps_per_env']
    model_hash = sha(train / 'model_final.pt')
    iterations = checkpoint_iteration(train / 'model_final.pt') + 1
    reader = EventAccumulator(str(train), size_guidance={'scalars': 0})
    reader.Reload()
    scalars = {tag: reader.Scalars(tag) for tag in reader.Tags()['scalars']}
    completion = {
        'status_completed': ts['status'] == ps['status'] == 'completed',
        'checkpoint_iteration_counter': iterations == tm['stage_iterations'],
        'status_policy_updates': ts['policy_updates'] == iterations,
        'all_loss_steps_present': [v.step for v in scalars['Loss/value_function']] == list(range(iterations)),
        'play_uses_received_model': pm['source_checkpoint']['sha256'] == model_hash,
        'settings_unchanged_between_train_play': settings == pm['gallop_settings'],
        'play_has_no_policy_updates': ps['policy_updates'] == 0 and ps['policy_state_unchanged'],
    }
    for key in ('gallop_core', 'gallop_env'):
        completion[key + '_source_hash_matches'] = sha(train / 'source' / (key + '.py')) == tm[key + '_sha256']
    assert all(completion.values()), completion
    episodes = [json.loads(line) for line in (train / 'episodes.jsonl').read_text().splitlines() if line]
    pe = [json.loads(line) for line in (play / 'episodes.jsonl').read_text().splitlines() if line]
    assert len(pe) == 1
    assert all(r['stabilization_target_speed'] == 2 and r['stabilization_leap_target'] == .5 for r in episodes)
    with np.load(play / 'gallop_trace.npz', allow_pickle=False) as saved:
        d = {key: saved[key] for key in saved.files}
    valid = d['valid'][:, 0].astype(bool)
    measured = valid & d['gallop_metric_active'][:, 0].astype(bool)
    assert len(d['time']) == 1500 and measured.any()
    # Reuse only the existing pure NumPy rollout functions; no import of Torch.
    collector = ROOT / 'scripts/analysis/collect_gallop.py'
    tree = ast.parse(collector.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ('longest_run', 'rollout_metric_mask', 'analyze_rollout')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(collector), 'exec'), namespace)
    rollout = namespace['analyze_rollout'](d, settings, pe)
    names = tm['official_joint_limits']['joint_order']
    canonical = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')]
    q = d['joint_pos'][valid, 0].reshape(-1, 12)[:, [canonical.index(n) for n in names]]
    dq, tau = d['joint_velocity'][valid, 0], d['applied_torque'][valid, 0]
    official = tm['official_joint_limits']
    bounds = np.array(tm['guarded_runtime_joint_contract']['original_limits'])
    q_excess = np.maximum(bounds[:, 0] - q, 0) + np.maximum(q - bounds[:, 1], 0)
    limits = np.array(official['velocity_rad_s'])
    efforts = np.array(official['effort_nm'])
    joints = []
    for i, name in enumerate(names):
        joints.append({'joint': name, 'max_abs_velocity_rad_s': float(np.abs(dq[:, i]).max()),
                       'velocity_limit_rad_s': float(limits[i]),
                       'max_abs_applied_torque_nm': float(np.abs(tau[:, i]).max()),
                       'effort_limit_nm': float(efforts[i]),
                       'max_position_excess_rad': float(q_excess[:, i].max())})
    joint_check = {
        'control_samples': int(valid.sum()), 'control_dt_s': dt,
        'control_sample_position_pass': bool((q_excess <= 1e-4).all()),
        'control_sample_velocity_pass': bool((np.abs(dq) <= limits + 1e-3).all()),
        'control_sample_effort_pass': bool((np.abs(tau) <= efforts + 1e-3).all()),
        'runtime_bad_physics_samples': int(pe[0]['edu_bad_physics_samples']),
        'runtime_peak_velocity_ratio': pe[0]['edu_peak_velocity_ratio'],
        'runtime_peak_effort_ratio': pe[0]['edu_peak_effort_ratio'],
        'per_joint': joints,
        'scope': 'Independent audit at 20ms control samples; 5ms physics peaks rely on saved runtime counters, not a separately received physics trace.',
    }
    groups = {'all': group_stats(episodes),
              'first_1000_updates': group_stats([r for r in episodes if r['step'] <= 1000 * steps]),
              'last_1000_updates': group_stats([r for r in episodes if r['step'] > (iterations - 1000) * steps])}
    scalar_summary = {}
    for tag, records in scalars.items():
        values = np.array([r.value for r in records], dtype=float)
        scalar_summary[tag] = {'count': len(records), 'first': float(values[0]), 'last': float(values[-1]),
                               'first_1000_mean': float(values[:1000].mean()),
                               'last_1000_mean': float(values[-1000:].mean()),
                               'minimum': float(values.min()), 'maximum': float(values.max())}
    video = next((play / 'videos').glob('*.mp4'))
    video_probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_name,width,height,avg_frame_rate', '-of', 'json', str(video)], text=True))
    downloaded = out / 'inputs/gallop-play-step-0.mp4'
    provenance = {'checkpoint_sha256': model_hash, 'checkpoint_pickle_executed': False,
                  'downloaded_video_matches_server_archive': sha(video) == sha(downloaded) if downloaded.exists() else None,
                  'archive_files': len(inventory['files']), 'archive_sha256': inventory['archive_sha256'],
                  'validation': completion}
    if downloaded.exists():
        assert provenance['downloaded_video_matches_server_archive']
    summary = {'train': train_rel, 'play': play_rel, 'provenance': provenance,
               'training': {'additional_updates': iterations, 'inherited_updates': tm['fine_tuning']['parent_cumulative_iterations'],
                            'cumulative_updates': tm['cumulative_iterations'], 'num_envs': ts['num_envs'],
                            'rollout_steps_per_update': steps, 'new_transitions': iterations * steps * ts['num_envs'],
                            'elapsed_seconds': ts['last_event']['elapsed_s'], 'control_dt_s': dt,
                            'physics_dt_s': env_cfg['sim']['dt'], 'episodes': groups,
                            'scalar_summary': scalar_summary},
               'playback': {'status': ps, 'reported_episode': pe[0], 'recomputed_rollout': rollout,
                            'joint_audit': joint_check, 'video': video_probe},
               'verdict': {'training_completed': True, 'fixed_goal_achieved': bool(pe[0]['success']),
                           'sim2sim_evaluated_for_this_model': False, 'hardware_deployment_accepted': False},
               'limitations': ['Single saved play per policy; no independent multi-seed evaluation or causal reward ablation.',
                               'Bad orientation/height/contact exact per-episode causes are not separately recorded; TensorBoard cause rates are rolling statistics.',
                               'Training environment count remained 256; no 1024-environment run was received.',
                               'No new simulation, policy update or physical robot command was executed during collection.']}
    from scripts.analysis.compare_gallop_results import compare, landing_diagnostics
    diagnostics = landing_diagnostics(play)
    for name, rows in [('all', episodes),
                       ('first_1000_updates', [r for r in episodes if r['step'] <= 1000 * steps]),
                       ('last_1000_updates', [r for r in episodes if r['step'] > (iterations - 1000) * steps])]:
        keys = ('flight_landings', 'flight_stable_landings', 'flight_aborted_landings',
                'leap_reward_points', 'leap_reward_steps')
        totals = {key: float(sum(r.get(key, 0) for r in rows)) for key in keys}
        for key in sorted({key for row in rows for key in row if key.startswith('flight_abort_')}):
            totals[key] = float(sum(row.get(key, 0) for row in rows))
        settled = totals['flight_stable_landings'] + totals['flight_aborted_landings']
        totals['stable_fraction_finished'] = totals['flight_stable_landings'] / settled if settled else None
        groups[name]['landing_totals'] = totals
    summary['playback']['landing_diagnostics'] = diagnostics
    write_json(out / 'landing_diagnostics.json', diagnostics)
    helper_tree = ast.parse((ROOT / 'scripts/analysis/collect_flight_diagnostics.py').read_text())
    event_helpers = {'np': np, 'json': json}
    selected = [n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('flight_event_rows', 'flight_trace_rows')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'offline_flight_helpers', 'exec'), event_helpers)
    event_rows = event_helpers['flight_event_rows'](d, settings)
    write_json(out / 'flight_events.json', event_rows)
    with (out / 'flight_events.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0]))
        writer.writeheader(); writer.writerows(event_rows)
    summary['playback']['event_audit'] = {
        'events': len(event_rows), 'maximum_air_seconds': max(r['air_seconds'] for r in event_rows),
        'distance_reward_sum': sum(r['distance_reward_points'] for r in event_rows),
        'touchdown_reward_sum': sum(r['touchdown_distance_reward_points'] for r in event_rows),
        'recovery_reward_sum': sum(r['recovery_distance_reward_points'] for r in event_rows),
        'status_counts': dict(Counter(r['status'] for r in event_rows)),
        'aborted_with_positive_touchdown_credit': sum(r['status'] == 'aborted' and r['touchdown_distance_reward_points'] > 0 for r in event_rows),
    }
    assert len(event_rows) == pe[0]['leap_events']
    assert np.isclose(summary['playback']['event_audit']['distance_reward_sum'], pe[0]['leap_reward_points'], rtol=1e-5)
    if args.baseline:
        baseline = read(args.baseline)
        summary['comparison'] = compare(baseline, summary)
        summary['comparison']['parent_checkpoint_matches'] = tm['source_checkpoint']['sha256'] == baseline['provenance']['checkpoint_sha256']
        summary['comparison']['baseline_summary'] = str(args.baseline.resolve())
        summary['comparison']['settings_changes'] = {key: {'before': value, 'after': settings.get(key)} for key, value in
            read(args.baseline.parent / 'inputs/server' / baseline['train'] / 'manifest.json')['gallop_settings'].items()
            if value != settings.get(key)}
        summary['comparison']['new_settings'] = {key: value for key, value in settings.items() if key not in
            read(args.baseline.parent / 'inputs/server' / baseline['train'] / 'manifest.json')['gallop_settings']}
        assert summary['comparison']['parent_checkpoint_matches']
        write_json(out / 'comparison.json', summary['comparison'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'play_episode.json', pe[0])
    write_json(out / 'joint_audit.json', joint_check)
    extreme_keys = ('step', 'env', 'age', 'startup_stage', 'gallop_metric_seconds',
                    'edu_peak_velocity_ratio', 'edu_peak_effort_ratio', 'edu_bad_physics_samples',
                    'edu_terminated', 'terminated', 'timed_out', 'leap_distance_max')
    extra_keys = ('edu_peak_velocity_joint', 'edu_peak_velocity_actual', 'edu_peak_velocity_limit',
                  'edu_peak_velocity_seconds', 'edu_peak_velocity_sample_finite')
    extreme_keys += tuple(key for key in extra_keys if key in episodes[0])
    extremes = sorted(episodes, key=lambda row: row['edu_peak_velocity_ratio'], reverse=True)[:5]
    extreme_records = [{key: row[key] for key in extreme_keys} for row in extremes]
    for row in extreme_records:
        if 'edu_peak_velocity_joint' in row:
            index = int(row['edu_peak_velocity_joint'])
            row['edu_peak_velocity_joint_name'] = names[index] if 0 <= index < len(names) else None
    write_json(out / 'training_velocity_extremes.json', extreme_records)
    phases = {}
    for stage in range(4):
        mask = measured & (d['flight_planned_stage'][:, 0] == stage)
        if mask.any():
            phases[str(stage)] = {'samples': int(mask.sum()),
                'grounded_fraction': float(d['contact'][mask, 0].any(axis=1).mean()),
                'world_vz_mean': float(d['world_vertical_velocity'][mask, 0].mean()),
                'world_vz_max': float(d['world_vertical_velocity'][mask, 0].max())}
    write_json(out / 'phase_statistics.json', phases)
    with (out / 'training_scalars.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['tag', 'iteration_zero_based', 'wall_time', 'value'])
        for tag, records in scalars.items():
            writer.writerows((tag, r.step, r.wall_time, r.value) for r in records)
    with (out / 'episode_windows.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['window', 'episodes', 'edu_termination_fraction',
            'startup_completed', 'measured_episodes', 'maximum_leap_m', 'qualified_leap_events', 'successes'])
        writer.writeheader()
        for name, row in groups.items():
            writer.writerow({'window': name, **{k: row[k] for k in writer.fieldnames if k != 'window'}})
    with (out / 'play_trace.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sim_time_s', 'formal_student_segment', 'vx_command', 'vx', 'height',
                         'roll_deg', 'pitch_deg', 'max_joint_speed_ratio', 'leap_events', 'max_leap_m'])
        for i, time in enumerate(d['time']):
            writer.writerow([time, int(measured[i]), d['command'][i, 0, 0], d['body_velocity'][i, 0, 0],
                             d['position'][i, 0, 2], *np.rad2deg(d['rpy'][i, 0, :2]),
                             np.max(np.abs(d['joint_velocity'][i, 0]) / limits),
                             d['leap_events'][i, 0], d['leap_distance_max'][i, 0]])
    make_plots(out, scalars, d, measured, summary)
    make_diagnostic_plots(out, d, summary)
    make_video_sheet(out, video)
    write_report(out, summary)
    versions = ['numpy', 'PyYAML', 'matplotlib', 'tensorboard', 'absl-py']
    (out / 'analysis_requirements.txt').write_text('\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in versions) + '\n')
    status = read(out / 'collection_status.json')
    status.update(server_access='SSH denied; user transferred archive via SCP',
                  training_performance_assessed=True, server_archive_sha256=inventory['archive_sha256'],
                  verified_server_files=len(inventory['files']), report='report.html',
                  video_server_association_verified=True, status='analyzed', analysis_completed=True)
    write_json(out / 'collection_status.json', status)
    assert all(sha(raw / r['path']) == r['sha256'] for r in inventory['files'])
    print(json.dumps({'report': str(out / 'report.html'), 'checks': completion,
                      'episodes': len(episodes), 'play_mean_vx': pe[0]['gallop_mean_vx'],
                      'play_max_leap_m': pe[0]['leap_distance_max'], 'raw_files_unchanged': True}, ensure_ascii=False))


def make_plots(out, scalars, d, mask, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    tags = [('Train/mean_reward', 1, 'Mean episode reward'),
            ('Train/mean_episode_length', .02, 'Mean episode duration (s)'),
            ('Episode_Termination/official_joint_limit', 100, 'Rolling official-limit termination (%)'),
            ('Episode_Termination/bad_orientation', 100, 'Rolling bad-orientation termination (%)'),
            ('Perf/total_fps', 1, 'Transitions per wall-clock second'),
            ('Loss/learning_rate', 1, 'Adaptive learning rate')]
    for ax, (tag, scale, title) in zip(axes.flat, tags):
        rows = scalars[tag]
        x = np.array([r.step + 1 for r in rows])
        y = np.array([r.value * scale for r in rows])
        ax.plot(x, y, alpha=.22, lw=.6)
        if len(y) >= 100:
            ax.plot(x[99:], np.convolve(y, np.ones(100) / 100, 'valid'), lw=1.4)
        ax.set_title(title); ax.set_xlabel('Additional PPO update'); ax.grid(alpha=.2)
    fig.savefig(out / 'training_curves.png', dpi=140); plt.close(fig)
    t = d['time']; lim = np.array([r['velocity_limit_rad_s'] for r in summary['playback']['joint_audit']['per_joint']])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout='constrained')
    axes[0].plot(t, d['command'][:, 0, 0], '--', label='command')
    axes[0].plot(t, d['body_velocity'][:, 0, 0], label='measured vx'); axes[0].set_ylabel('m/s')
    axes[1].plot(t, d['leap_last_distance'][:, 0] * 100, label='last flight distance')
    axes[1].axhline(50, ls='--', color='red', label='50 cm goal'); axes[1].set_ylabel('cm')
    axes[2].plot(t, np.max(np.abs(d['joint_velocity'][:, 0]) / lim, axis=1), label='20ms sample speed ratio')
    axes[2].axhline(1, ls='--', color='red', label='official speed limit'); axes[2].set_ylabel('ratio')
    axes[3].imshow(d['contact'][:, 0].T, origin='lower', aspect='auto', interpolation='nearest',
                   extent=[t[0], t[-1], -.5, 3.5], cmap='Blues')
    axes[3].set_yticks(range(4), ['FL', 'FR', 'RL', 'RR']); axes[3].set_ylabel('foot contact'); axes[3].set_xlabel('Simulation time (s)')
    for ax in axes:
        if mask.any(): ax.axvspan(t[0], t[mask][0], color='grey', alpha=.12)
        ax.grid(alpha=.2)
    for ax in axes[:3]: ax.legend(loc='upper right')
    fig.savefig(out / 'playback_metrics.png', dpi=140); plt.close(fig)


def make_video_sheet(out, video):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    frames = out / 'frames'; frames.mkdir(exist_ok=True)
    times = [2, 8, 10, 15, 20, 25]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for ax, sec in zip(axes.flat, times):
        path = frames / f'frame_{sec:02d}s.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(sec), '-i', str(video),
                        '-frames:v', '1', str(path)], check=True)
        with Image.open(path) as frame: ax.imshow(frame)
        ax.set_title(f'Video {sec:.0f} s'); ax.axis('off')
    fig.savefig(out / 'video_preview.png', dpi=130); plt.close(fig)


def make_diagnostic_plots(out, d, s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'comparison' in s:
        comparison = s['comparison']['playback']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
        for ax, (key, scale, title, goal) in zip(axes, [
            ('leap_distance_mean', 100, 'Mean flight distance (cm)', 50),
            ('leap_air_seconds_mean', 1000, 'Mean flight duration (ms)', 240),
            ('flight_stable_landings', 1, 'Stable landings (count)', None)]):
            values = [comparison[key][name] * scale for name in ('previous', 'current')]
            bars = ax.bar(['62k', str(s['training']['cumulative_updates'] // 1000) + 'k'], values,
                          color=['#92a7be', '#148673'])
            ax.bar_label(bars, fmt='%.2f'); ax.set_title(title); ax.grid(axis='y', alpha=.2)
            if goal:
                ax.axhline(goal, color='#bd542b', ls='--', label='goal'); ax.legend()
        fig.savefig(out / 'comparison.png', dpi=150); plt.close(fig)
    diag = s['playback']['landing_diagnostics']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout='constrained')
    time = d['time']
    for key, label in [('flight_landings', 'candidates'), ('flight_stable_landings', 'stable'),
                        ('flight_aborted_landings', 'aborted')]:
        axes[0].step(time, d[key][:, 0], label=label)
    axes[0].set_ylabel('Cumulative events'); axes[0].legend()
    events = diag['abort_events']
    if diag['available'] and events:
        axes[1].scatter([e['control_snapshot_time_s'] for e in events],
                        [e['flight_abort_sample_foot_slide'] for e in events], label='5ms abort sample')
        axes[1].axhline(.5, color='#bd542b', ls='--', label='0.5 m/s gate')
        axes[1].legend()
    axes[1].set_ylabel('Contact foot speed (m/s)')
    for key, label in [('flight_step_touchdown_distance_score', 'safe-touchdown distance'),
                        ('flight_step_recovery_distance_score', 'recovered distance')]:
        if key in d:
            axes[2].plot(time, np.cumsum(d[key][:, 0]) * 24, label=label)
    axes[2].set_ylabel('Weighted distance points'); axes[2].set_xlabel('Simulation time (s)')
    if 'flight_step_touchdown_distance_score' in d: axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out / 'landing_events.png', dpi=140); plt.close(fig)


def write_report(out, s):
    tr, ep = s['training'], s['playback']['reported_episode']
    first, last = tr['episodes']['first_1000_updates'], tr['episodes']['last_1000_updates']
    rec = s['playback']['recomputed_rollout']['environments'][0]
    audit = s['playback']['joint_audit']
    diag = s['playback']['landing_diagnostics']
    candidate = int(ep['flight_landings'])
    stable, aborted = int(ep['flight_stable_landings']), int(ep['flight_aborted_landings'])
    pending = candidate - stable - aborted
    label = f"{tr['cumulative_updates']:,}회"

    def table(rows, headings=('항목', '결과')):
        header = '<tr>' + ''.join('<th>' + html.escape(x) + '</th>' for x in headings) + '</tr>'
        return '<table>' + header + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>'
            for x in row) + '</tr>' for row in rows) + '</table>'

    rows = [
        ('추가 / 누적 학습', f"{tr['additional_updates']:,}회 / {label}"),
        ('환경 / 새 transition', f"{tr['num_envs']}개 / {tr['new_transitions']:,}"),
        ('학습 실행 시간', f"{tr['elapsed_seconds']/60:.2f}분"),
        ('플레이 / 정식 학생 계측', f"{s['playback']['recomputed_rollout']['duration_seconds']:.2f}초 / {ep['gallop_metric_seconds']:.2f}초"),
        ('평균 전진속도 / 목표', f"{rec['mean_body_velocity'][0]:.4f} / 2.0m/s"),
        ('전진속도 RMSE', f"{rec['planar_velocity_rmse'][0]:.4f}m/s (저장 trace 재계산)"),
        ('평균 / 최대 실제 도약 거리', f"{ep['leap_distance_mean']*100:.2f} / {ep['leap_distance_max']*100:.2f}cm (목표 50cm)"),
        ('평균 체공 시간', f"{ep['leap_air_seconds_mean']*1000:.2f}ms (목표 240ms)"),
        ('실제 비행 이벤트 최대 체공', f"{s['playback']['event_audit']['maximum_air_seconds']*1000:.2f}ms (5ms 이벤트 계측)"),
        ('목표 도약 / 전체 비행 이벤트', f"{int(ep['leap_qualified_events'])} / {int(ep['leap_events'])}"),
        ('안정 착지 / 착지 후보', f"{stable} / {candidate}; 취소 {aborted}, 기록 종료 시 진행 중 {pending}"),
        ('판정 완료된 착지 중 안정 비율', f"{stable/max(stable+aborted,1):.2%} (성공 기준 80%)"),
        ('거리 보상 / 보상 지급 스텝', f"{ep['leap_reward_points']:.6f} / {int(ep['leap_reward_steps'])}"),
        ('앞발 / 뒷발 폭 충족률', f"{ep['stabilization_front_width_fraction']:.2%} / {ep['stabilization_hind_width_fraction']:.2%} (각 80% 필요)"),
        ('접지 패턴 일치율', f"{ep['contact_accuracy']:.2%} (기준 60%)"),
        ('종료 원인', f"terminated={int(ep['terminated'])}, timed_out={int(ep['timed_out'])}"),
        ('5ms 속도·토크 한도 위반 기록', f"{int(ep['edu_bad_physics_samples'])}개"),
        ('속도 / 토크 한도 대비 최대 비율', f"{ep['edu_peak_velocity_ratio']:.2%} / {ep['edu_peak_effort_ratio']:.2%}"),
        ('20ms 저장 각도·속도·적용토크 독립 검사', f"{audit['control_sample_position_pass']} / {audit['control_sample_velocity_pass']} / {audit['control_sample_effort_pass']}"),
    ]
    comparison = ''
    if 'comparison' in s:
        cmp = s['comparison']['playback']
        specs = [('평균 전진속도 (m/s)', 'gallop_mean_vx', 1),
                 ('평균 도약 거리 (cm)', 'leap_distance_mean', 100),
                 ('최대 도약 거리 (cm)', 'leap_distance_max', 100),
                 ('평균 체공 (ms)', 'leap_air_seconds_mean', 1000),
                 ('안정 착지 (회)', 'flight_stable_landings', 1),
                 ('착지 후보 (회)', 'flight_landings', 1),
                 ('자격 도약 (회)', 'leap_qualified_events', 1),
                 ('앞발 폭 충족률 (%)', 'stabilization_front_width_fraction', 100)]
        comparison = '<h2>62k와 단일 플레이 비교</h2>' + table([
            (name, f"{cmp[key]['previous']*scale:.3f}", f"{cmp[key]['current']*scale:.3f}")
            for name, key, scale in specs], ('항목', '62k', '이번 모델'))
        comparison += '<p>부모 체크포인트 해시가 이전 62k 최종 모델과 일치합니다. 두 모델의 목표·관절 한도와 기록 길이는 같지만 각 모델의 seed 42 플레이 한 번씩의 관찰입니다. 다중 seed 성공률이나 보상 수정만의 인과 효과를 입증하지 않습니다. 보상식이 바뀌었으므로 모델 간 보상 크기는 직접 성능 비교에 쓰지 않습니다.</p><img src="comparison.png" alt="62k와 새 모델의 거리 체공 착지 비교">'
    reasons = sorted(diag['abort_counts'].items(), key=lambda x: -x[1])
    reason_table = table([(key, count) for key, count in reasons]) if reasons else '<p>이전 소스에는 사유별 기록이 없습니다.</p>'
    scores = diag['reward_score_sums']
    score_rows = [(key, f"{value:.6f}") for key, value in scores.items()]
    video = next((out / 'inputs/server' / s['play'] / 'videos').glob('*.mp4'))
    video_relative = video.relative_to(out).as_posix()
    probe = s['playback']['video']
    stream = probe['streams'][0]
    verdict = '고정 목표 과제 성공 기록이 있습니다.' if s['verdict']['fixed_goal_achieved'] else '학습은 완료됐으나 2m/s·50cm·체공 0.24초의 전체 과제는 미달입니다.'
    note = '이번 단일 플레이에서 거리는 일부 늘고 안정 착지·속도 추종은 개선됐지만 50cm와 240ms에 도달하지 못했습니다.' if 'comparison' in s else '단일 플레이의 거리·체공·착지·관절 조건을 아래에서 구분해 확인합니다.'
    history = tr['episodes']['all']
    training_rows = []
    for name, key, scale in [('학습 episode', 'episodes', 1), ('관절 한도 종료율 (%)', 'edu_termination_fraction', 100),
                              ('이벤트 가중 평균 거리 (cm)', 'event_weighted_mean_leap_m', 100),
                              ('이벤트 가중 평균 체공 (ms)', 'event_weighted_mean_air_seconds', 1000),
                              ('전체 최대 도약 거리 (cm)', 'maximum_leap_m', 100), ('자격 도약', 'qualified_leap_events', 1)]:
        training_rows.append((name, f"{first[key]*scale:.3f}", f"{last[key]*scale:.3f}"))
    f = first['landing_totals']; l = last['landing_totals']
    training_rows.append(('판정 완료 착지 중 안정 비율 (%)', f"{(f['stable_fraction_finished'] or 0)*100:.2f}", f"{(l['stable_fraction_finished'] or 0)*100:.2f}"))
    train_table = table(training_rows, ('항목', '초반 1,000회', '후반 1,000회'))
    summary_checks = html.escape(json.dumps(s['provenance']['validation'], ensure_ascii=False, indent=2))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 갤롭 {label} 결과</title><style>
body{{font:16px/1.75 sans-serif;max-width:1060px;margin:36px auto;padding:0 22px;color:#182334;background:#fafbfd}}
h1,h2{{line-height:1.35}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #dce2e9;padding:9px 12px;text-align:left}}img{{width:100%;height:auto;background:white}}video{{width:100%;max-width:800px}}.result{{border-left:5px solid #be5d21;padding:14px;background:#fff1e6}}code{{overflow-wrap:anywhere}}pre{{overflow:auto}}a{{color:#075aaf}}
</style><h1>Go2 갤롭 {label} 학습 결과</h1>
<p class="result"><strong>{verdict}</strong> {note}</p>
<h2>완료·출처 검증</h2><p>수집일 2026-11-17. 부모 {tr['inherited_updates']:,}회에서 {tr['additional_updates']:,}회 추가 학습해 누적 {tr['cumulative_updates']:,}회입니다. 최종 체크포인트의 iter={tr['additional_updates']-1}, status 업데이트 수, 전체 손실 로그 스텝을 대조했습니다. pickle을 실행하거나 모델을 불러오지 않고 정수 카운터만 읽었습니다.</p>
<p>받은 원본 {s['provenance']['archive_files']}개 파일의 SHA256을 확인했습니다. 플레이가 참조한 모델 해시와 최종 모델이 일치하고 학습·플레이 설정 및 저장된 core/env 소스 해시도 일치합니다. 이 수집에서 학습·시뮬레이션·실기 명령과 보상·훈련 설정 수정은 수행하지 않았습니다.</p>
<p>학습 <code>{html.escape(s['train'])}</code><br>플레이 <code>{html.escape(s['play'])}</code><br>모델 SHA256 <code>{s['provenance']['checkpoint_sha256']}</code></p>
<details><summary>검증 항목</summary><pre>{summary_checks}</pre></details>
<h2>현재 플레이 결과</h2>{table(rows)}
<p>종료된 episode는 1개이며 전체 기록을 확인했습니다. 정식 학생 구간은 출발 보행·전환·워밍업을 제외합니다. 위 평균 속도·RMSE는 저장 trace 재계산값이고 실행기 값과 기록 시점 차이가 있습니다. 20ms 독립 관절 검사와 5ms 실행기 누적 최고값은 구분합니다.</p>
{comparison}
<h2>착지 실패 원인과 새 보상</h2>{reason_table}
<p>사유는 물리 5ms마다 집계하며 여러 조건이 동시에 실패할 수 있습니다. 종료 취소는 20ms 제어 경계에서 집계합니다. <code>slide</code>는 접촉한 발 중심의 월드 좌표계 수평 속도가 0.5m/s를 넘었다는 뜻입니다. 실제 접촉점의 미끄럼이나 마찰 부족을 직접 측정한 값은 아닙니다.</p>
<p>현재 기록의 안정 착지 {stable}회·취소 {aborted}회·진행 중 {pending}회를 구분합니다. 착지시 최대 접촉 발 속도는 {ep.get('flight_landing_peak_foot_slide',0):.3f}m/s, 최대 힘은 체중의 {ep['flight_landing_peak_force_bw']:.3f}배(한도 3배), 최대 하강 속도는 {ep['flight_landing_peak_down_vz']:.3f}m/s(한도 1.65m/s)입니다.</p>
{table(score_rows)}<p>점수 표는 제어 스텝 점수 합계이며 가중치 적용 전입니다. 총 거리 보상은 <code>leap_reward_score × 24</code>를 합한 {ep['leap_reward_points']:.6f}입니다. 안전 접촉분과 안정 회복분을 나눠 검증했습니다. 접촉 보상을 받았더라도 이후 회복 실패로 착지 보너스와 남은 75%는 받지 못할 수 있습니다.</p>
<img src="landing_events.png" alt="착지 기록과 접촉 발 속도 및 보상 분리">
<h2>학습 중 변화</h2><p>전체 {history['episodes']:,}개 episode, 목표 성공 {history['successes']:,}개, 자격 도약 {int(history['qualified_leap_events']):,}회, 학습 중 최대 실제 도약 거리 {history['maximum_leap_m']*100:.2f}cm입니다.</p>
{train_table}<p>학습 중 서로 다른 길이의 episode 집계입니다. 동일 조건의 독립 평가가 아니며 학습 보상 상승만으로 도약 목표 달성을 판정하지 않습니다.</p>
<p><strong>플레이의 한도 준수와 학습 전체의 한도 준수는 다릅니다.</strong> 학습 중 관절 한도 종료는 {history['edu_terminated']:,}/{history['episodes']:,} episode ({history['edu_termination_fraction']:.2%})였으며 기록된 순간 관절속도 최고 비율은 공식 한도의 {history['max_runtime_velocity_ratio']:.3f}배입니다. 학습에는 탐색과 중간 정책이 포함되며 이 수치를 최종 결정론적 플레이와 혼동하지 않습니다. 초과한 물리 표본의 원시 관절 trace는 수신하지 않아 개별 관절과 원인은 이 집계만으로 확정할 수 없습니다.</p>
<p>종료 장치는 한도 초과를 감지해 episode를 끝냈지만 순간 초과 자체를 모두 예방한 것은 아닙니다. 학습 원본은 그대로 보존했습니다. 따라서 이번 30초 플레이의 초과 0회만으로 실기 안전을 입증하지 않습니다.</p>
<img src="training_curves.png" alt="학습 보상 종료율 처리량 학습률">
<h2>저장 trace·영상</h2><img src="playback_metrics.png" alt="속도 도약 거리 관절 한도 및 접지">
<p>MP4 {float(probe['format']['duration']):.2f}초 · {stream['width']}×{stream['height']} · 저장 프레임률 {stream['avg_frame_rate']}. 저장 FPS는 실제 시뮬레이터 처리 속도가 아닙니다.</p>
<video controls preload="metadata" src="{html.escape(video_relative)}"></video><img src="video_preview.png" alt="영상 2 8 10 15 20 25초 프레임">
<h2>판단과 다음 확인</h2><p>목표 거리·체공 시간과 관절·착지 한도는 그대로입니다. 이번 관찰에서 안정 착지와 평균 도약은 개선됐지만 짧은 비행에 머물렀습니다. 이륙 수직속도 평균 {ep['flight_takeoff_vz_mean']:.3f}m/s는 shaping 기준 1.4m/s보다 낮아 상승 추진과 발 회수 동작을 검토할 근거입니다. 단일 관찰만으로 원인을 확정하거나 안전 한도를 완화하지 않습니다.</p>
<p>다음에는 동일 초기 조건의 여러 seed에서 거리·체공·착지·속도 한도를 확인하고, 접촉 발 속도가 초과되는 발과 시점을 더 구체적으로 구분하는 것이 필요합니다. 이번 모델의 MuJoCo 전이와 실기 배포는 검증하지 않았습니다. 이번 요청에서는 코드 보상을 다시 수정하거나 추가 학습하지 않았습니다.</p>
<p><a href="summary.json">전체 수치</a> · <a href="landing_diagnostics.json">취소 시점의 물리 표본</a> · <a href="flight_events.csv">비행 이벤트별 거리·보상</a> · <a href="joint_audit.json">관절 한도 검사</a> · <a href="episode_windows.csv">학습 구간별 집계</a> · <a href="server_archive_manifest.json">원본 해시</a></p></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
