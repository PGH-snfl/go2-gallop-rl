"""Analyze received 2026-11-17 run artifacts without Torch, Isaac or learning.

Requires NumPy, PyYAML, matplotlib, tensorboard and absl-py. Raw artifacts
remain under reports/20261117_server_results/inputs/server.
"""
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
        'leap_events': sum(r['leap_events'] for r in rows),
        'qualified_leap_events': sum(r['leap_qualified_events'] for r in rows),
        'measured_criterion_pass_fraction': {
            k: float(np.mean([r[k] for r in measured])) for k in criteria
        },
        'note': 'Episode-weighted fractions; terminated and timed_out can overlap. Not an independent evaluation success rate.',
    }


def main():
    out = ROOT / 'reports/20261117_server_results'
    raw = out / 'inputs/server'
    train_rel = 'runs/isaac/train/unitree_go2_gallop/20261117_160234_3e2363'
    play_rel = 'runs/isaac/play/unitree_go2_gallop/20261117_190513_06271d'
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
        'checkpoint_iteration_counter': iterations == 4000,
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
                  'downloaded_video_matches_server_archive': sha(video) == sha(downloaded),
                  'archive_files': len(inventory['files']), 'archive_sha256': inventory['archive_sha256'],
                  'validation': completion}
    assert provenance['downloaded_video_matches_server_archive']
    summary = {'train': train_rel, 'play': play_rel, 'provenance': provenance,
               'training': {'additional_updates': iterations, 'inherited_updates': 58000,
                            'cumulative_updates': tm['cumulative_iterations'], 'num_envs': ts['num_envs'],
                            'rollout_steps_per_update': steps, 'new_transitions': iterations * steps * ts['num_envs'],
                            'elapsed_seconds': ts['last_event']['elapsed_s'], 'control_dt_s': dt,
                            'physics_dt_s': env_cfg['sim']['dt'], 'episodes': groups,
                            'scalar_summary': scalar_summary},
               'playback': {'status': ps, 'reported_episode': pe[0], 'recomputed_rollout': rollout,
                            'joint_audit': joint_check, 'video': video_probe},
               'verdict': {'training_completed': True, 'fixed_goal_achieved': False,
                           'sim2sim_evaluated_for_this_model': False, 'hardware_deployment_accepted': False},
               'limitations': ['Single saved play, seed 42; no independent multi-seed evaluation.',
                               '58k parent performance traces were not included; no causal before/after performance claim.',
                               'Bad orientation/height/contact exact per-episode causes are not separately recorded; TensorBoard cause rates are rolling statistics.',
                               'Training environment count remained 256; no 1024-environment run was received.',
                               'No new simulation, policy update or physical robot command was executed during collection.']}
    write_json(out / 'summary.json', summary)
    write_json(out / 'play_episode.json', pe[0])
    write_json(out / 'joint_audit.json', joint_check)
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
    make_video_sheet(out, video)
    write_report(out, summary)
    versions = ['numpy', 'PyYAML', 'matplotlib', 'tensorboard', 'absl-py']
    (out / 'analysis_requirements.txt').write_text('\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in versions) + '\n')
    status = read(out / 'collection_status.json')
    status.update(server_access='SSH denied; user transferred archive via SCP',
                  training_performance_assessed=True, server_archive_sha256=inventory['archive_sha256'],
                  verified_server_files=len(inventory['files']), report='report.html',
                  video_server_association_verified=True)
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
        ax.axvspan(t[0], t[mask][0], color='grey', alpha=.12)
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


def write_report(out, s):
    tr, ep = s['training'], s['playback']['reported_episode']
    recomputed = s['playback']['recomputed_rollout']['environments'][0]
    first, last = tr['episodes']['first_1000_updates'], tr['episodes']['last_1000_updates']
    target = .5
    rows = [
        ('추가 학습 / 누적', '4,000회 / 62,000회'),
        ('실제 환경 수 / 새 데이터', f"{tr['num_envs']}개 / {tr['new_transitions']:,} transition"),
        ('학습 실행 경과', f"{tr['elapsed_seconds']/60:.2f}분"),
        ('플레이 / 정식 학생 계측', f"30.00초 / {ep['gallop_metric_seconds']:.2f}초"),
        ('정식 구간 평균 전진속도', f"{recomputed['mean_body_velocity'][0]:.4f}m/s / 목표 2.0m/s"),
        ('전진속도 오차 RMSE', f"{recomputed['planar_velocity_rmse'][0]:.4f}m/s"),
        ('실제 비행 전진거리 평균 / 최대', f"{ep['leap_distance_mean']*100:.2f}cm / {ep['leap_distance_max']*100:.2f}cm / 목표 50cm"),
        ('평균 공중시간', f"{ep['leap_air_seconds_mean']*1000:.2f}ms / 목표 240ms"),
        ('조건을 충족한 도약 / 전체 비행 이벤트', f"{int(ep['leap_qualified_events'])} / {int(ep['leap_events'])}"),
        ('안정 착지 / 판정된 착지', f"{int(ep['flight_stable_landings'])} / {int(ep['flight_landings'])}"),
        ('플레이 종료', '30초 시간 제한 도달, terminated=0, timed_out=1'),
        ('관절속도 / 적용토크 한도 대비 최고 비율', f"{ep['edu_peak_velocity_ratio']*100:.2f}% / {ep['edu_peak_effort_ratio']*100:.2f}%"),
        ('5ms 한도 초과 기록', f"{int(ep['edu_bad_physics_samples'])}개 (실행기 누적 계측)"),
        ('앞발 / 뒷발 폭 조건 충족 비율', f"{ep['stabilization_front_width_fraction']*100:.2f}% / {ep['stabilization_hind_width_fraction']*100:.2f}%"),
    ]
    table = ''.join('<tr><th>'+html.escape(k)+'</th><td>'+html.escape(v)+'</td></tr>' for k, v in rows)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 갤롭 62000회 학습 결과</title><style>
body{{font:16px/1.7 sans-serif;max-width:1060px;margin:36px auto;padding:0 22px;color:#182334;background:#fafbfd}}
h1,h2{{line-height:1.35}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #dce2e9;padding:9px 12px;text-align:left}}th{{width:43%}}img{{width:100%;height:auto;background:white}}video{{width:100%;max-width:800px}}.result{{border-left:5px solid #be5d21;padding:14px;background:#fff1e6}}code{{overflow-wrap:anywhere}}a{{color:#075aaf}}
</style><h1>Go2 갤롭 62000회 학습 결과</h1>
<p class="result"><strong>학습은 완료됐지만 2m/s·50cm·공중 0.24초의 갤롭 과제는 미달입니다.</strong> 이번 단일 플레이는 제한 시간을 채웠고 관절 속도·토크 초과 기록은 0이었습니다. 짧은 비행과 안정 착지 미달이 남았습니다.</p>
<h2>완료와 출처</h2><p>수집일 2026년 11월 17일. 학습 4,000회의 손실 로그(0–3999), 최종 모델의 iter=3999, status의 4,000 업데이트를 대조했습니다. 부모 58,000회에서 누적 62,000회입니다. 모델 pickle을 실행하거나 Torch로 가중치를 불러오지 않고 카운터만 확인했습니다.</p>
<p>받은 43개 원본 파일의 SHA256을 수집 전후에 확인했습니다. 플레이가 참조하는 모델 해시와 받은 최종 모델 해시가 일치하며, 기존에 다운로드한 영상도 압축 안 영상과 동일합니다. 학습·플레이 설정과 저장된 core/env 코드 해시도 대조했습니다.</p>
<p>학습 실행: <code>{html.escape(s['train'])}</code><br>플레이 실행: <code>{html.escape(s['play'])}</code><br>모델 SHA256: <code>{s['provenance']['checkpoint_sha256']}</code></p>
<h2>핵심 결과</h2><table>{table}</table>
<p>속도는 정식 학생 구간의 저장 trace에서 독립 재계산한 값입니다. 실행기 집계 평균 {ep['gallop_mean_vx']:.4f}m/s 및 RMSE {ep['vx_rmse']:.4f}m/s와의 작은 차이는 상태 기록 시점 차이에 따른 것입니다. reset 1회는 마지막 30초 시간 제한이며 낙상 1회를 뜻하지 않습니다.</p>
<h2>학습 중 변화</h2><p>총 {tr['episodes']['all']['episodes']:,}개 episode가 기록됐고 목표 성공 및 자격 도약은 모두 0입니다. 전체 학습에서 관찰된 최대 비행 거리는 {tr['episodes']['all']['maximum_leap_m']*100:.2f}cm이며 50cm에 못 미쳤습니다.</p>
<p>초반 1,000회와 후반 1,000회의 실제 episode 집계를 비교하면 관절 한도 종료 비율이 {first['edu_termination_fraction']*100:.2f}% → {last['edu_termination_fraction']*100:.2f}%로 감소했습니다. 두 구간은 각 {first['episodes']:,}/{last['episodes']:,}개의 서로 다른 episode로, 고정 조건 독립 평가가 아닙니다. 방향은 개선됐지만 새 제약의 효과와 추가 학습 효과를 분리한 인과 실험은 아닙니다.</p>
<p>TensorBoard의 평균 보상은 초반/후반 1,000개 기록 평균 {tr['scalar_summary']['Train/mean_reward']['first_1000_mean']:.2f} → {tr['scalar_summary']['Train/mean_reward']['last_1000_mean']:.2f}, 평균 episode 길이는 {tr['scalar_summary']['Train/mean_episode_length']['first_1000_mean']*.02:.2f}초 → {tr['scalar_summary']['Train/mean_episode_length']['last_1000_mean']*.02:.2f}초입니다. 보상 상승은 갤롭 목표 달성으로 해석할 수 없습니다.</p>
<img src="training_curves.png" alt="학습 보상 종료 비율 처리량과 학습률">
<h2>플레이 계측</h2><p>회색 구간은 정식 학생 계측 시작 전입니다. 도약 거리는 물리 5ms 간격의 실행기 누적값을 읽었고, 속도·토크·각도는 저장된 20ms trace도 별도로 검사했습니다. 5ms 전체 원시 관절 trace를 받은 것은 아닙니다.</p>
<img src="playback_metrics.png" alt="플레이 속도 도약 거리 관절속도 비율 접지 상태">
<h2>영상</h2><p>서버에서 저장한 MP4는 29.98초·640×480·50fps입니다. 저장 영상의 프레임률은 실제 시뮬레이터 처리 성능과 다릅니다.</p>
<video controls preload="metadata" src="inputs/server/{s['play']}/videos/gallop-play-step-0.mp4"></video>
<img src="video_preview.png" alt="플레이 영상 2 8 10 15 20 25초 프레임">
<h2>남은 확인과 다음 검토</h2><ul><li>실제 학습은 256환경입니다. 1,024환경 확대나 새 GPU의 성능 우위는 이번 결과로 평가할 수 없습니다.</li><li>같은 62k 모델을 여러 seed에서 평가해 시간 제한 도달과 한도 준수를 재확인해야 합니다. 이 결과는 단일 seed 42의 한 플레이입니다.</li><li>50cm·240ms를 충족하는 비행과 착지 후 안정 회복을 함께 검토해야 합니다. 현재는 작은 비행 이벤트가 많고 안정 착지 0/56입니다.</li><li>받은 자료에 58k 부모의 동일 조건 평가가 없어 모델 간 성능 개선은 단정하지 않습니다. 이번 모델의 MuJoCo 검증과 실기 배포 적합성도 확인되지 않았습니다.</li></ul>
<p>이번 수집에서는 학습·시뮬레이션·실기 명령을 실행하지 않았고 보상·정책·훈련 설정을 수정하지 않았습니다.</p>
<h2>근거 파일</h2><p><a href="summary.json">전체 수치 JSON</a> · <a href="episode_windows.csv">학습 구간별 집계</a> · <a href="play_trace.csv">플레이 trace CSV</a> · <a href="joint_audit.json">관절별 검증</a> · <a href="server_archive_manifest.json">원본 파일 해시</a></p></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
