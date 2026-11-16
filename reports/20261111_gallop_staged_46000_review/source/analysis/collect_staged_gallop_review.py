"""Offline audit of saved staged-Gallop curriculum and a matching play run."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from scripts.analysis.collect_heading_diagnostics import archive, segment
from scripts.analysis.collect_gallop import rollout_metric_mask

CRITERIA = (
    'tracking', 'pattern', 'flight', 'motion', 'front_load', 'front_motion',
    'front_cycle', 'stable', 'mature', 'stabilization_pose',
    'stabilization_width', 'stabilization_landing', 'stabilization_distance',
    'stabilization_measured', 'guarded_limits', 'guarded_feet',
    'staged_measured', 'staged_tracking',
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(rows):
    measured = [r for r in rows if r['gallop_metric_seconds'] >= 10]
    def stats(batch):
        if not batch:
            return {'episodes': 0}
        n = len(batch)
        events = sum(r['leap_events'] for r in batch)
        landings = sum(r['flight_landings'] for r in batch)
        return dict(
            episodes=n, successes=sum(bool(r['success']) for r in batch),
            success_rate=sum(bool(r['success']) for r in batch)/n,
            startup_completed_rate=sum(bool(r['startup_completed']) for r in batch)/n,
            startup_failed=sum(bool(r['startup_failed']) for r in batch),
            terminated=sum(bool(r['terminated']) for r in batch),
            criteria={k:sum(bool(r[k]) for r in batch)/n for k in CRITERIA},
            leap_events=int(events),
            leap_mean_m=sum(r['leap_events']*r['leap_distance_mean'] for r in batch)/events if events else None,
            leap_max_m=max(r['leap_distance_max'] for r in batch),
            landing_events=int(landings),
            stable_landings=int(sum(r['flight_stable_landings'] for r in batch)),
            stable_landing_event_rate=sum(r['flight_stable_landings'] for r in batch)/landings if landings else None,
            measured_means={k:float(np.mean([r[k] for r in batch])) for k in (
                'stabilization_front_width_fraction', 'stabilization_hind_width_fraction',
                'stabilization_landing_fraction', 'guarded_front_high',
                'guarded_hind_high', 'guarded_intervention', 'action_clip_fraction',
                'vx_rmse', 'staged_yaw_rms')},
            failure_combinations=dict(Counter(
                ','.join(k for k in CRITERIA if not r[k])
                for r in batch if not r['success']
            )),
        )
    return {'all': stats(rows), 'measured_10s': stats(measured)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--parent-play', type=Path, required=True)
    args = parser.parse_args()
    out = args.report.resolve()
    summary = read(out/'summary.json')
    train, play = Path(summary['train']), Path(summary['play'])
    manifest = read(train/'manifest.json')
    cfg = manifest['gallop_settings']
    assert cfg['staged_enabled']
    source = train/'source/gallop_core.py'
    assert sha(source) == manifest['gallop_core_sha256']
    spec = importlib.util.spec_from_file_location('staged_audit_saved_core', source)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    controller = core.StagedGallopCurriculum(core.GallopSettings(**cfg))
    controller.configure_envs(summary['training_status']['num_envs'], summary['policy_dt'])
    events = read(train/'curriculum_summary.json')['events']
    controller.restore(events[0]['staged_index_before'])
    rows = [json.loads(line) for line in (train/'episodes.jsonl').read_text().splitlines() if line]
    accepted = []
    cursor = 0
    def record(row):
        if not row['eligible']:
            return
        chosen = controller.record(bool(row['success']), int(row['level']),
            {k:bool(row[k]) for k in CRITERIA}, env_id=int(row['env']), step=int(row['step']))
        assert chosen == row['staged_cohort_accepted'], ('cohort acceptance mismatch', row['step'], row['env'])
        if chosen:
            accepted.append(row)
    for saved in events:
        while cursor < len(rows) and rows[cursor]['step'] <= saved['step']:
            record(rows[cursor])
            cursor += 1
        actual = controller.decide(saved['step'])
        assert actual == saved, ('curriculum replay mismatch', saved['step'])
    while cursor < len(rows):
        record(rows[cursor])
        cursor += 1
    rollout = round(summary['new_transitions']/summary['iterations']/summary['training_status']['num_envs'])
    eligible = [r for r in rows if r['eligible']]
    windows = []
    for start in range(0, summary['iterations'], 1000):
        end = min(start+1000, summary['iterations'])
        window = [r for r in eligible if start*rollout < r['step'] <= end*rollout]
        windows.append(dict(start_iteration=start, end_iteration=end, **cohort(window)))
    closed = [e for e in events if e['cohort_completed'] == e['cohort_size'] and e['episodes'] >= cfg['min_episodes']]
    by_stage = {str(i):cohort([r for r in eligible if int(r['staged_index']) == i])
        for i in sorted({int(r['staged_index']) for r in eligible})}
    training = dict(
        all=cohort(eligible), accepted=cohort(accepted), windows=windows, by_stage=by_stage,
        recorded_episodes=len(rows), accepted_episodes=len(accepted),
        curriculum_decisions=len(events), complete_cohorts=len(closed),
        promotions=sum(e['reason']=='promote_stage' for e in events),
        reason_counts=dict(Counter(e['reason'] for e in events)),
        cohort_success_rate_max=max((e['success_rate'] for e in closed), default=None),
        cohort_success_rate_last=closed[-1]['success_rate'] if closed else None,
        cohort_success_rate_last10_mean=float(np.mean([e['success_rate'] for e in closed[-10:]])) if closed else None,
        last_complete_cohort=closed[-1] if closed else None,
        terminal_partial_cohort=events[-1],
        reproduced_all_decisions=True, reproduced_each_acceptance=True,
    )
    current = json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
    parent = json.loads((args.parent_play/'episodes.jsonl').read_text().splitlines()[0])
    parent_manifest = read(args.parent_play/'manifest.json')
    parent_data = archive(args.parent_play/'gallop_trace.npz')
    parent_mask = rollout_metric_mask(parent_data, parent_manifest['gallop_settings'], 0)
    current_data = archive(play/'gallop_trace.npz')
    current_mask = rollout_metric_mask(current_data, cfg, 0)
    metric_keys = (
        'leap_events', 'leap_distance_mean', 'leap_distance_max', 'leap_air_seconds_mean',
        'flight_landings', 'flight_stable_landings', 'stabilization_landing_fraction',
        'stabilization_front_width_fraction', 'stabilization_hind_width_fraction',
        'guarded_front_high', 'guarded_hind_high', 'guarded_intervention',
        'action_clip_fraction', 'guarded_actual', 'staged_yaw_rms',
        'vx_rmse', 'gallop_metric_seconds',
    )
    comparison = dict(
        current_play=str(play), parent_play=str(args.parent_play.resolve()),
        parent_checkpoint=parent_manifest['source_checkpoint'],
        metrics={k:{'parent':parent.get(k), 'current':current.get(k)} for k in metric_keys},
        current_formal=segment(current_data, current_mask, cfg),
        parent_formal=segment(parent_data, parent_mask, parent_manifest['gallop_settings']),
        current_failed_gates=[k for k in CRITERIA if not current[k]],
        note='Two saved stochastic play episodes, not a matched multi-seed evaluation. No causal or statistical significance claim.',
    )
    final_record = manifest['staged_checkpoints']['model_final.pt']
    result = dict(
        training=training, comparison=comparison,
        final_checkpoint=final_record,
        inactive_later_stage_reference_bug=dict(
            all_observed_stage_indices=sorted({int(r['staged_index']) for r in rows}),
            alpha_zero_all_observed=True if all(int(r['staged_index'])<4 for r in rows) else False,
            note='The previously documented missing-profile reference mismatch begins at stage4. All recorded episodes here stayed at stage0; it does not explain this run failing stage0.',
        ),
        limitations=[
            'Training success rates are on-policy randomized episode statistics, not independent frozen-policy evaluation.',
            'Stage0 target18cm is distinct from the configured final80cm.',
            'Timeout reset at30s is not a fall.',
            'Physical joint observations and filtered targets are different from raw requested angles.',
            'No PPO updates or physics rollouts were performed by this collector.',
        ],
    )
    write(out/'staged_review.json', result)
    summary['staged_review']={k:training[k] for k in (
        'recorded_episodes', 'accepted_episodes', 'complete_cohorts', 'promotions',
        'cohort_success_rate_max', 'cohort_success_rate_last', 'reproduced_all_decisions')}
    summary['staged_review']['final_checkpoint']=final_record
    write(out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout='constrained')
    x=[e['step']/rollout for e in closed]
    axes[0].plot(x,[100*e['success_rate'] for e in closed],label='Complete cohort success')
    axes[0].axhline(80,color='red',ls='--',label='Promotion80%')
    axes[0].set_ylabel('Success (%)');axes[0].legend()
    for key in ('stabilization_width','stabilization_landing','guarded_feet'):
        axes[1].plot(x,[100*e['criterion_rates'].get(key,np.nan) for e in closed],label=key)
    axes[1].set_ylabel('Gate pass (%)');axes[1].legend()
    wx=[w['end_iteration'] for w in windows]
    axes[2].plot(wx,[100*w['all']['leap_mean_m'] if w['all']['leap_mean_m'] else 0 for w in windows],label='Event-weighted mean air distance')
    axes[2].axhline(18,color='red',ls='--',label='Active stage0 goal18cm')
    axes[2].set_ylabel('Distance (cm)');axes[2].set_xlabel('Additional PPO iteration');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'staged_training_progress.png',dpi=140);plt.close(fig)
    print(json.dumps({k:summary['staged_review'][k] for k in ('recorded_episodes','complete_cohorts','promotions','cohort_success_rate_max')},indent=2))


if __name__ == '__main__':
    main()

