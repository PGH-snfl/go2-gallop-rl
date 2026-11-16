"""Summarize frozen Isaac candidate runs without updating any policy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpd.paths import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run, output):
    manifest = json.loads((run / 'manifest.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['status'] == 'completed' and status['checkpoints_unchanged']
    cases = json.loads((run / 'cases.json').read_text())
    assert len(cases) == 4 * len(manifest['labels']) * manifest['pairs']
    checks = {'run_completed': True, 'all_cases_present': True}
    summaries = []
    warmup_differences = {}
    for scenario in dict.fromkeys(c['scenario'] for c in cases):
        z = np.load(run / scenario / 'trace.npz')
        for label in manifest['labels']:
            rows = [r for r in cases if r['scenario'] == scenario and r['label'] == label]
            episodes = [r['episode'] for r in rows]
            assert all(e for e in episodes), 'Missing first-episode terminal record'
            n = len(rows)
            events = sum(e['leap_events'] for e in episodes)
            stable = sum(e['flight_stable_landings'] for e in episodes)
            aborted = sum(e['flight_aborted_landings'] for e in episodes)
            formal = []
            all_valid = []
            last_two = []
            for r in rows:
                i = r['env']
                valid = z['valid'][:, i].astype(bool)
                mask = valid & z['gallop_metric_active'][:, i].astype(bool)
                if scenario.startswith('warm'):
                    mask &= z['time'] > manifest['warmup_steps'] * .02 + 1.
                formal.append((i, mask))
                all_valid.append((i, valid))
                last = z['time'][valid][-1]
                last_two.append(z['body_velocity'][valid & (z['time'] > last - 2), i, 0])

            def values(key, axis=None, formal_only=True):
                arrays = []
                for i, mask in formal if formal_only else all_valid:
                    arr = z[key][mask, i]
                    arrays.append(arr if axis is None else arr[:, axis])
                return np.concatenate(arrays)

            def mean(key, axis=None):
                v = values(key, axis)
                return float(v.mean()) if v.size else None

            def width(pair):
                total = sum(e[f'stabilization_{pair}_support_samples'] for e in episodes)
                return (sum(e[f'stabilization_{pair}_width_fraction'] *
                            e[f'stabilization_{pair}_support_samples'] for e in episodes) / total
                        if total else None)

            angles = values('rpy')
            item = dict(scenario=scenario, label=label, starts=n,
                full_student_measured_count=sum(r['formal_seconds'] >= 5 for r in rows),
                full_student_seconds_mean=float(np.mean([r['formal_seconds'] for r in rows])),
                terminations=sum(bool(e['terminated']) for e in episodes),
                full_success_count=sum(bool(e['success']) for e in episodes),
                final_mix_mean=float(np.mean([r['final_mix'] for r in rows])),
                mean_vx=mean('body_velocity', 0),
                last_two_seconds_vx=float(np.concatenate(last_two).mean()),
                vx_rmse=(float(np.sqrt(np.mean((values('body_velocity', 0) - values('command', 0))**2)))
                         if angles.size else None),
                all_off_fraction=(float((~values('contact').astype(bool)).all(1).mean()) if angles.size else None),
                leap_events=int(events),
                leap_distance_mean=(sum(e['leap_distance_mean'] * e['leap_events'] for e in episodes) / events if events else None),
                leap_distance_max=max(e['leap_distance_max'] for e in episodes),
                leap_air_seconds_mean=(sum(e['leap_air_seconds_mean'] * e['leap_events'] for e in episodes) / events if events else None),
                stable_landings=int(stable), aborted_landings=int(aborted),
                landing_fraction=(stable / (stable + aborted) if stable + aborted else None),
                front_width_fraction=width('front'), hind_width_fraction=width('hind'),
                raw_clip_fraction=mean('action_clip'), filter_fraction=mean('guarded_intervention'),
                actual_guard_excess_max=float(values('guarded_actual_excess', formal_only=False).max()),
                attitude_rms_deg=(np.rad2deg(np.sqrt((angles**2).mean(0))).tolist() if angles.size else None))
            summaries.append(item)
        checks[scenario + '_trace_finite'] = all(np.isfinite(z[k]).all() for k in ('position', 'joint_pos', 'body_velocity', 'guarded_target'))
        if scenario.startswith('warm'):
            t = manifest['warmup_steps'] - 1
            for key in ('position', 'joint_pos', 'body_velocity', 'phase', 'guarded_target'):
                x = z[key][t].reshape(manifest['pairs'], len(manifest['labels']), -1)
                maximum = float(np.abs(x - x[:, :1]).max())
                # Matching initial requests and the same warmup actor do not
                # imply bit-identical trajectories in distinct PhysX instances.
                # Preserve this audit; warm tests are supporting evidence, not
                # strict common-state counterfactual experiments.
                warmup_differences[scenario + '/' + key] = dict(max_difference=maximum, strict_match_within_002=maximum < .02)
    for path, expected in zip(manifest['checkpoints'], manifest['checkpoint_sha256']):
        checks[Path(path).parent.name + '/' + Path(path).name + '_unchanged'] = sha(Path(path)) == expected
    assert all(v if isinstance(v, bool) else v['passes'] for v in checks.values()), checks
    output.mkdir(parents=True, exist_ok=True)
    contrasts = []
    for scenario in dict.fromkeys(c['scenario'] for c in cases):
        parent = sorted((r for r in cases if r['scenario'] == scenario and r['label'] == 'parent30k'), key=lambda r:r['pair'])
        final = sorted((r for r in cases if r['scenario'] == scenario and r['label'] == 'final34k'), key=lambda r:r['pair'])
        assert [r['pair'] for r in parent] == [r['pair'] for r in final]
        contrasts.append(dict(scenario=scenario, paired_starts=len(parent),
            parent_has_more_measured_flights=sum(p['episode']['leap_events'] > f['episode']['leap_events'] for p,f in zip(parent,final)),
            final_has_more_measured_flights=sum(p['episode']['leap_events'] < f['episode']['leap_events'] for p,f in zip(parent,final)),
            parent_measured_final_unmeasured=sum(p['formal_seconds']>=5 and f['formal_seconds']<5 for p,f in zip(parent,final))))
    result = dict(run=str(run.relative_to(ROOT)), policy_updates=0, summaries=summaries, paired_contrasts=contrasts,
        warmup_state_audit=warmup_differences, strict_common_state_warmup=False,
        limitations=['Single nominal simulator scene, eight small paired initial perturbations; not broad randomized validation.',
            'Warm tests bypass startup at 12s and diagnose retained ability only, not a deployable startup controller.',
            'Warm tests share the parent actor and requested initial states but do not clone a common physical state. Trajectories diverged before switching; inspect warmup_state_audit. They are supporting evidence, not strict common-state counterfactuals.',
            'New-task runtime commands/references/filters/measurements apply; fixed environment reward-manager weights remain old and are ignored.',
            'No new MuJoCo or hardware validation. Unmeasured metrics are null, never interpreted as perfect scores.'])
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    sources = [run / 'manifest.json', run / 'status.json', run / 'cases.json',
               ROOT / 'scripts/validation/compare_gallop_candidates.py', Path(__file__).resolve()]
    for scenario in dict.fromkeys(c['scenario'] for c in cases):
        sources += [run / scenario / 'trace.npz', run / scenario / 'episodes.jsonl']
    (output / 'evidence_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in sources}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/20261110_gallop_candidate_selection')
    a = p.parse_args()
    collect(a.run.resolve(), a.output.resolve())
