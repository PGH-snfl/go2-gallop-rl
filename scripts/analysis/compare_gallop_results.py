"""Compare two collected summaries and audit new landing diagnostics offline."""
import argparse
import json
from pathlib import Path

import numpy as np

from gpd.paths import ROOT


def compare(previous, current):
    keys = ('gallop_mean_vx', 'gallop_cycle_mean_vx_rmse', 'leap_distance_mean',
            'leap_distance_max', 'leap_air_seconds_mean', 'leap_events',
            'leap_qualified_events', 'flight_landings', 'flight_stable_landings',
            'flight_aborted_landings', 'leap_reward_points', 'edu_bad_physics_samples',
            'edu_peak_velocity_ratio', 'edu_peak_effort_ratio',
            'stabilization_front_width_fraction')
    old, new = (s['playback']['reported_episode'] for s in (previous, current))
    result = {key: {'previous': old[key], 'current': new[key], 'difference': new[key] - old[key]}
              for key in keys}
    return {
        'previous_checkpoint_sha256': previous['provenance']['checkpoint_sha256'],
        'current_checkpoint_sha256': current['provenance']['checkpoint_sha256'],
        'playback': result,
        'note': 'Single recorded play per policy. Observed differences do not establish a causal improvement or multi-seed success rate.',
        'reward_note': 'Reward functions changed between62k and66k; raw reward magnitudes are not directly comparable performance scores.',
    }


def landing_diagnostics(play):
    manifest = json.loads((play / 'manifest.json').read_text())
    with np.load(play / 'gallop_trace.npz', allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    valid = data['valid'][:, 0].astype(bool)
    indices = np.flatnonzero(valid)
    if not indices.size:
        return {'available': False, 'reason': 'No valid first-segment samples.'}
    reason_keys = sorted(key for key in data if key.startswith('flight_abort_')
                         and not key.startswith('flight_abort_sample_') and data[key].ndim == 2)
    counters = {}
    for key in reason_keys:
        values = data[key][valid, 0]
        if (np.diff(values) < 0).any():
            raise ValueError('Counter reset within valid segment: ' + key)
        counters[key] = int(values[-1])
    score_keys = ('flight_step_touchdown_distance_score', 'flight_step_recovery_distance_score',
                  'flight_step_confirmed_distance_score', 'flight_step_landing_score', 'leap_reward_score',
                  'flight_step_airtime_score', 'flight_step_touchdown_airtime_score', 'flight_step_recovery_airtime_score')
    scores = {key: float(data[key][valid, 0].sum()) for key in score_keys if key in data}
    records = []
    aborts = data['flight_aborted_landings'][valid, 0]
    increments = np.diff(np.r_[0, aborts])
    reason_order = manifest.get('flight_design', {}).get('abort_reason_order', [])
    for local in np.flatnonzero(increments > 0):
        i = indices[local]
        record = {'control_snapshot_time_s': float(data['time'][i]),
                  'aborts_in_control_interval': int(increments[local])}
        for key in data:
            if key.startswith('flight_abort_sample_'):
                value = data[key][i, 0]
                record[key] = value.item() if value.ndim == 0 else value.tolist()
        if 'flight_last_abort_reasons' in data:
            record['last_abort_reasons'] = [reason for reason, flag in
                zip(reason_order, data['flight_last_abort_reasons'][i, 0]) if flag]
        records.append(record)
    return {'available': bool(reason_keys), 'scope': 'First contiguous playback segment only.',
            'abort_counts': counters, 'reward_score_sums': scores, 'abort_events': records,
            'note': 'Reasons overlap. Counters retain all events; detailed samples show only the last abort in each control interval. Split scores are before the environment strength multiplier and reward weight.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT / 'reports/20261117_server_results/summary.json')
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--play', type=Path, help='Extracted current playback run directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.previous.read_text()), json.loads(args.current.read_text()))
    if args.play:
        result['landing_diagnostics'] = landing_diagnostics(args.play)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
