"""Offline flight-event joins, training cohorts and plots for a collected run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.collect_gallop import episode_stats, read_rows, rollout_metric_mask
from scripts.analysis.collect_unitree_baseline import load_yaml


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohort(rows, settings, max_episode_steps):
    result = episode_stats(rows)
    events = sum(r['leap_events'] for r in rows)
    result['event_totals'] = {key: sum(r[key] for r in rows) for key in (
        'leap_events', 'leap_qualified_events', 'leap_reward_points', 'flight_takeoffs',
        'flight_landings', 'flight_stable_landings', 'flight_aborted_landings')}
    reason_keys=sorted({key for row in rows for key in row if key.startswith('flight_abort_')})
    result['landing_abort_reasons']={key:sum(row.get(key,0) for row in rows) for key in reason_keys}
    result['landing_abort_reason_note']='5ms landing decisions; termination at control boundary; reasons can overlap.'
    result['event_weighted_distance_m'] = sum(r['leap_distance_mean'] * r['leap_events'] for r in rows) / events if events else None
    result['event_weighted_air_seconds'] = sum(r['leap_air_seconds_mean'] * r['leap_events'] for r in rows) / events if events else None
    failures = Counter()
    for r in rows:
        reasons = []
        if r['action_clip_fraction'] > settings['max_action_clipping']:
            reasons.append('target_clipping')
        if r['attitude_rms'] > settings['max_attitude_rms']:
            reasons.append('attitude')
        if r['terminated']:
            reasons.append('terminated')
        if r['age'] < max_episode_steps - 1:
            reasons.append('short_episode')
        if reasons:
            failures[','.join(reasons)] += 1
    result['stable_failure_combinations'] = dict(failures)
    return result


def flight_event_rows(z,settings):
    if z['valid'].shape[1] != 1:
        raise ValueError('This event join currently requires one playback environment')
    valid = z['valid'][:, 0].astype(bool)
    counters = ('leap_events', 'leap_qualified_events', 'flight_landings', 'flight_stable_landings', 'flight_aborted_landings')
    increments = {key: np.diff(np.r_[0, z[key][valid, 0]]) for key in counters}
    if any((delta < 0).any() or (delta > 1).any() for delta in increments.values()):
        raise ValueError('Cannot attribute multiple events or counter resets within one saved policy frame')
    indices = np.flatnonzero(valid)
    events = {}
    for local, i in enumerate(indices):
        event_id = int(z['leap_events'][i, 0])
        if increments['leap_events'][local]:
            events[event_id] = {
                'event_id': event_id, 'landing_snapshot_seconds': float(z['time'][i]),
                'distance_m': float(z['leap_last_distance'][i, 0]),
                'air_seconds': float(z['leap_last_seconds'][i, 0]),
                'target_distance_m': float(z['stabilization_leap_target'][i,0]) if 'stabilization_leap_target' in z else settings['leap_min_distance'],
                'distance_time_qualified': int(increments['leap_qualified_events'][local]),
                'recovery_monitored': int(increments['flight_landings'][local]),
                'stable_landing': 0, 'recovery_aborted': 0,
                'recovery_end_snapshot_seconds': None, 'distance_reward_points': 0.,
                'touchdown_distance_reward_points': 0., 'recovery_distance_reward_points': 0.,
                'abort_reason_counts': '{}',
            }
        if increments['flight_stable_landings'][local]:
            row = events[event_id]
            row['stable_landing'] = 1
            row['recovery_end_snapshot_seconds'] = float(z['time'][i])
        if increments['flight_aborted_landings'][local]:
            row = events[event_id]
            row['recovery_aborted'] = 1
            row['recovery_end_snapshot_seconds'] = float(z['time'][i])
            reasons={key:float(z[key][i,0]-(z[key][indices[local-1],0] if local else 0))
                     for key in z if key.startswith('flight_abort_') and not key.startswith('flight_abort_sample_') and z[key].ndim==2}
            row['abort_reason_counts']=json.dumps({key:value for key,value in reasons.items() if value},sort_keys=True)
        if event_id in events:
            row=events[event_id]
            row['distance_reward_points']+=float(z['leap_reward_score'][i,0])*settings['leap_reward_weight']
            for field,key in (('touchdown_distance_reward_points','flight_step_touchdown_distance_score'),
                              ('recovery_distance_reward_points','flight_step_recovery_distance_score')):
                if key in z:row[field]+=float(z[key][i,0])*settings['leap_reward_weight']
    for row in events.values():
        row['status'] = 'stable' if row['stable_landing'] else 'aborted' if row['recovery_aborted'] else 'pending_at_episode_end' if row['recovery_monitored'] else 'not_selected_for_recovery'
    event_rows = list(events.values())
    return event_rows


def flight_trace_rows(z):
    valid=z['valid'][:,0].astype(bool)
    extra_keys = [key for key in z if key.startswith(('leap_', 'flight_', 'pair_')) or key == 'world_vertical_velocity']
    traces = []
    for i, time in enumerate(z['time']):
        row = dict(time=float(time), env=0, first_segment_valid=int(valid[i]), episode_reset=int(z['episode_reset'][i, 0]))
        for key in extra_keys:
            values = np.asarray(z[key][i, 0])
            if values.ndim == 0:
                row[key] = values.item()
            else:
                row.update({key+'_'+'_'.join(map(str,index)):value.item() for index,value in np.ndenumerate(values)})
        traces.append(row)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--parent-report', type=Path)
    args = parser.parse_args()
    out = args.report.resolve()
    summary = json.loads((out / 'summary.json').read_text())
    settings = summary['provenance']['gallop_settings']
    train = Path(summary['train'])
    rows = read_rows(train / 'episodes.jsonl')
    eligible = [r for r in rows if r['eligible']]
    rollout = load_yaml(train / 'params/agent.yaml')['num_steps_per_env']
    env = load_yaml(train / 'params/env.yaml')
    max_episode_steps = round(env['episode_length_s'] / summary['policy_dt'])
    steps = summary['iterations'] * rollout
    groups = {
        'all_eligible': eligible,
        'first_200_iterations': [r for r in eligible if r['step'] <= 200 * rollout],
        'last_200_iterations': [r for r in eligible if r['step'] >= steps - 200 * rollout],
        'last_1000_iterations': [r for r in eligible if r['step'] >= steps - 1000 * rollout],
    }
    diagnostics = {'episode_rows': len(rows), 'groups': {name: cohort(group, settings, max_episode_steps) for name, group in groups.items()}}
    if settings.get('stabilization_enabled'):
        speeds=sorted({round(r['stabilization_target_speed'],5) for r in eligible})
        diagnostics['per_speed_stage']={str(speed):cohort([r for r in eligible if abs(r['stabilization_target_speed']-speed)<1e-4],settings,max_episode_steps) for speed in speeds}
        diagnostics['note']='Distance qualifications use each episode target; do not treat every stage as the final leap_min_distance.'
    write_json(out / 'training_diagnostics.json', diagnostics)
    terms = {tag: item['last_100_mean'] for tag, item in summary['tensorboard'].items() if tag.startswith('Episode_Reward/')}
    write_json(out / 'reward_contributions_last100.json', {
        'terms': terms, 'net_sum': sum(terms.values()),
        'leap_fraction_of_net_contribution': terms['Episode_Reward/gallop_leap_distance'] / sum(terms.values()),
        'note': 'Same TensorBoard normalization/window. This observed fraction does not isolate a causal reward-weight effect.'})

    with np.load(Path(summary['play']) / 'gallop_trace.npz') as archive:
        z = {key: archive[key] for key in archive.files}
    event_rows=flight_event_rows(z,settings)
    valid=z['valid'][:,0].astype(bool)
    indices=np.flatnonzero(valid)
    mask=rollout_metric_mask(z,settings,0)
    write_csv(out / 'play_flight_events.csv', event_rows)
    traces=flight_trace_rows(z)
    write_csv(out / 'play_flight_trace.csv', traces)

    phase_statistics = []
    for stage, name in enumerate(('support', 'prepare', 'shared_air', 'planned_recovery')):
        m = mask & (z['flight_planned_stage'][:, 0] == stage)
        if not m.any():
            continue
        contact = z['contact'][m, 0].astype(bool)
        phase_statistics.append({
            'stage': name, 'samples': int(m.sum()), 'airborne_sample_fraction': float((~contact).all(1).mean()),
            'roll_pitch_rms': np.sqrt(np.mean(z['rpy'][m, 0, :2] ** 2, axis=0)).tolist(),
            'angular_speed_max': float(np.linalg.norm(z['angular_velocity'][m, 0], axis=1).max()),
            'lateral_speed_abs_max': float(np.abs(z['body_velocity'][m, 0, 1]).max()),
            'supporting_feet_counts': dict(Counter(map(int, contact.sum(1)))),
            'contact_counts_FL_FR_RL_RR': contact.sum(axis=0).tolist(),
        })
    final = indices[-1]
    qualified_stable = [row for row in event_rows if row['distance_time_qualified'] and row['stable_landing']]
    result = dict(
        event_count=len(event_rows), status_counts=dict(Counter(row['status'] for row in event_rows)),
        qualified_events=sum(row['distance_time_qualified'] for row in event_rows),
        qualified_and_stable_events=len(qualified_stable), qualified_and_stable_event_ids=[row['event_id'] for row in qualified_stable],
        event_distance_quantiles_m=np.quantile([row['distance_m'] for row in event_rows], [0, .1, .5, .9, 1]).tolist() if event_rows else None,
        event_air_seconds_distribution={str(round(seconds, 3)): count for seconds, count in Counter(round(row['air_seconds'], 3) for row in event_rows).items()},
        terminal_recovering=bool(z['flight_recovering'][final, 0]), terminal_recovery_seconds=float(z['flight_recovery_elapsed'][final, 0]),
        terminal_pending_distance_score=float(z['flight_pending_distance_score'][final, 0]),
        phase_statistics=phase_statistics,
        physics_samples_per_policy_step=np.unique(np.diff(z['leap_physics_samples'][valid, 0])).tolist(),
        caveats=['Event geometry and counters come from the 5ms tracker; CSV timestamps are the following 20ms snapshots.',
                 'A pending recovery at natural timeout is incomplete, not an observed unstable landing.',
                 'No failure reason codes or per-foot velocity traces were saved; aborted recovery causes cannot be fully reconstructed.',
                 'Reference phase and actual recovery flags are different; phase statistics use reference phases.'])
    write_json(out / 'play_diagnostics.json', result)
    if args.parent_report:
        old = json.loads((args.parent_report / 'play_analysis.json').read_text())['environments'][0]
        new = summary['rollout']['environments'][0]
        write_json(out / 'parent_play_comparison.json', {
            'parent': old, 'current': new,
            'note': 'Check saved command profiles, flight thresholds, reference duty and reward tolerances for actual differences. Pair costs alone do not establish locomotion performance.'})
    render(out, z, mask, event_rows, summary)
    print(json.dumps({'report': str(out), 'training_eligible_episodes': len(eligible), 'play': result}, ensure_ascii=False))


def render(out, z, mask, events, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = summary['provenance']['gallop_settings']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout='constrained')
    colors = {'stable': 'tab:green', 'aborted': 'tab:orange', 'pending_at_episode_end': 'tab:blue', 'not_selected_for_recovery': 'grey'}
    for status in colors:
        selected = [row for row in events if row['status'] == status]
        if not selected:
            continue
        t = [row['landing_snapshot_seconds'] for row in selected]
        axes[0].scatter(t, [row['distance_m'] for row in selected], color=colors[status], label=status, s=25)
        axes[1].scatter(t, [row['air_seconds'] * 1000 for row in selected], color=colors[status], s=25)
    if 'stabilization_leap_target' in z:axes[0].plot(z['time'],z['stabilization_leap_target'][:,0],color='k',ls='--',label='episode distance target')
    else:axes[0].axhline(s['leap_min_distance'], color='k', ls='--')
    axes[0].set_ylabel('Actual flight distance (m)'); axes[0].legend(fontsize=8)
    axes[1].axhline(s['flight_min_seconds'] * 1000, color='k', ls='--'); axes[1].set_ylabel('Actual flight duration (ms)')
    t = z['time'][mask]
    axes[2].plot(t, z['leap_events'][mask, 0], label='all measured flights')
    axes[2].plot(t, z['flight_stable_landings'][mask, 0], label='stable landings')
    axes[2].plot(t, z['leap_qualified_events'][mask, 0], label='distance + duration qualified')
    axes[2].set_xlabel('Time (s), first episode'); axes[2].set_ylabel('Cumulative events'); axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out / 'play_flight_events.png', dpi=150); plt.close(fig)
    keys = ['drive', 'clearance', 'support', 'balance', 'takeoff', 'landing', 'impact', 'clip']
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), layout='constrained')
    with (out / 'tensorboard_scalars.csv').open() as stream:
        records = list(csv.DictReader(stream))
    for ax, key in zip(axes.flat, keys):
        rows = [row for row in records if row['tag'] == 'Episode_Reward/gallop_flight_' + key]
        ax.plot([int(row['iteration_zero_based']) + 1 for row in rows], [float(row['value']) for row in rows], lw=.8)
        ax.set_title('gallop_flight_' + key); ax.set_xlabel('Stage PPO iteration'); ax.grid(alpha=.2)
    fig.savefig(out / 'flight_reward_curves.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
