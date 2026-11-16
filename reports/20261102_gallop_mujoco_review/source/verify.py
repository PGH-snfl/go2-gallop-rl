"""Recheck saved evidence only: no simulation, training or policy update."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/mujoco/evaluate/unitree_go2_gallop/20261102_141502_01c472'
TRAIN = ROOT / 'runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54'
ISAAC = ROOT / 'runs/isaac/play/unitree_go2_gallop/20261102_140231_b1e215'
sys.path.insert(0, str(ROOT))
from isaac_task.gallop_runtime import checkpoint_source_directory


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = {}


def check(name, result):
    checks[name] = bool(result)


status, result = read(RUN / 'status.json'), read(RUN / 'report.json')
manifest, contract = read(TRAIN / 'manifest.json'), read(RUN / 'model_contract.json')
diag = read(REPORT / 'diagnostics.json')
asset = read(ROOT / 'assets/go2_isaac_contract.json')
checkpoint = TRAIN / 'model_final.pt'
check('checkpoint_hash_matches_run_and_isaac',
      sha(checkpoint) == result['checkpoint_sha256'] == read(ISAAC / 'manifest.json')['source_checkpoint']['sha256'])
source = checkpoint_source_directory(checkpoint, manifest)
check('frozen_sources_and_teacher_hash_verified', source.is_dir())
check('teacher_matches_original', sha(source / 'walking_teacher.pt') == sha(Path(manifest['walking_teacher']['original_checkpoint'])))
check('training_usd_unchanged', sha(Path(manifest['asset_path'])) == manifest['asset_sha256'] == asset['isaac_asset_sha256'])
original_xml = ROOT.parents[1] / 'unitree/unitree_mujoco/unitree_robots/go2/go2.xml'
check('original_xml_unchanged', sha(original_xml) == contract['source_model_sha256'])
check('physical_joint_order_evidence_unchanged', sha(Path(asset['order_evidence_path'])) == asset['order_evidence_sha256'])
check('completed_single_30_second_episode', status['status'] == 'completed' and status['simulated_seconds'] == 30 and status['steps'] == 1500)
check('no_training_or_weight_mutation', status['policy_updates'] == 0 and status['policy_state_unchanged'] and result['policy_state_unchanged'])
check('no_hardware_or_early_termination', not status['physical_hardware'] and not status['terminated'])

with np.load(RUN / 'gallop_trace.npz') as archive:
    trace = {key: archive[key] for key in archive.files}
with np.load(RUN / 'physics_trace.npz') as archive:
    physics = {key: archive[key] for key in archive.files}
check('finite_policy_and_physics_arrays', all(np.isfinite(v).all() for v in [*trace.values(), *physics.values()]))
check('policy_trace_1500_at_20ms', len(trace['time']) == 1500 and np.allclose(np.diff(trace['time']), .02))
check('physics_trace_6000_at_5ms', len(physics['time']) == 6000 and np.allclose(np.diff(physics['time']), .005))
check('no_formal_gallop_metric_segment', not trace['gallop_metric_active'].any() and result['episode']['gallop_metric_seconds'] == 0)
check('transition_incomplete', set(trace['startup_stage'].ravel()) == {0, 1, 2} and not result['episode']['startup_completed'])
check('mix_peak_matches_diagnostics', np.isclose(trace['startup_gallop_mix'].max(), diag['max_mix']))
tail = trace['time'] >= 20
velocity_error = np.abs(trace['body_velocity'][tail, 0, 0] - trace['command'][tail, 0, 0])
check('all_501_last10s_speed_errors_exceed_limit', tail.sum() == 501 and (velocity_error > .25).all())
check('tail_mean_and_command_match', np.isclose(trace['body_velocity'][tail, 0, 0].mean(), diag['last10_mean_vx']) and np.isclose(trace['command'][-1, 0, 0], diag['last_command']))

with (REPORT / 'physical_flight_events.csv').open() as stream:
    events = list(csv.DictReader(stream))
long_events = [e for e in events if float(e['air_seconds']) >= .02 - 1e-9 and float(e['distance_m']) > 0]
check('independent_flight_counts_123_52_8', len(events) == 123 and len(long_events) == 52 and sum(int(e['qualified']) for e in events) == 8)
check('independent_long_flight_averages_match',
      np.isclose(np.mean([float(e['distance_m']) for e in long_events]), diag['physical_long_distance_mean']) and
      np.isclose(np.mean([float(e['air_seconds']) for e in long_events]), diag['physical_long_air_seconds_mean']))
check('independent_recovery_51_fail_1_pending', sum(e['recovery'] == 'failed_proxy' for e in long_events) == 51 and sum(e['recovery'] == 'pending' for e in long_events) == 1)
video = read(REPORT / 'video_metadata.json')
check('video_has_1500_frames_30s_50fps', video['frames'] == 1500 and video['fps'] == 50 and video['duration'] == 30 and (RUN / 'mujoco_gallop.mp4').stat().st_size > 0)
check('nominal_mass_difference_disclosed', np.isclose(contract['nominal_mujoco_mass_kg'], 15.206408) and np.isclose(contract['nominal_usd_mass_kg'], 16.087000485276803))
official = subprocess.run(['git', '-C', manifest['official_repo'], 'status', '--porcelain'], capture_output=True, text=True, check=True)
check('official_checkout_clean', not official.stdout.strip())

missing = []
for target in re.findall(r'\]\(([^)]+)\)', (REPORT / 'README.md').read_text()):
    if not target.startswith('https://') and target != 'verification.json' and not (REPORT / target).exists():
        missing.append(target)
check('report_links_exist', not missing)
verification = dict(
    verified_at=datetime.now(timezone.utc).isoformat(), all_checks_passed=all(checks.values()),
    checks=checks, missing_report_links=missing, policy_updates=0, new_simulation_in_this_recheck=False,
    software_tests=dict(tests=5, passed=5,
        command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_rl_mujoco -v',
        note='Five adapter tests passed before the nominal run; not rerun by this evidence-only script.'),
    decision=dict(nominal_episode_completed=True, transition_passed=False, formal_gallop_metrics_available=False,
        physical_flight_proxy_is_not_a_formal_gallop_pass=True))
(REPORT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')

inputs = [checkpoint, TRAIN / 'manifest.json', TRAIN / 'params/env.yaml', TRAIN / 'params/agent.yaml',
          source / 'gallop_core.py', source / 'gallop_env.py', source / 'walking_teacher.pt',
          Path(manifest['asset_path']), Path(asset['order_evidence_path']), original_xml,
          ISAAC / 'manifest.json', ISAAC / 'episodes.jsonl', ISAAC / 'gallop_trace.npz']
files = [*inputs, *(p for p in RUN.rglob('*') if p.is_file()),
         *(p for p in REPORT.rglob('*') if p.is_file() and p.name != 'evidence_manifest.json' and '__pycache__' not in p.parts)]
evidence = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(files))]
(REPORT / 'evidence_manifest.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(dict(all_checks_passed=all(checks.values()), checks=len(checks),
                     failed=[k for k, v in checks.items() if not v], evidence_files=len(evidence))))
if not all(checks.values()):
    raise SystemExit(1)
