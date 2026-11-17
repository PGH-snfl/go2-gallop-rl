"""Audit received reward/landing logs offline; never load a policy or simulator."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/20261117_server_results'


def read(path):
    return json.loads(path.read_text())


def group(rows):
    keys = ('flight_stable_landings', 'flight_landings', 'flight_aborted_landings',
            'leap_reward_points', 'leap_reward_steps', 'leap_events')
    result = {key: float(sum(row[key] for row in rows)) for key in keys}
    result['episodes'] = len(rows)
    result['stable_fraction_of_candidates'] = result['flight_stable_landings'] / max(result['flight_landings'], 1)
    result['distance_reward_per_episode'] = result['leap_reward_points'] / len(rows)
    return result


def main():
    summary = read(OUT / 'summary.json')
    train = OUT / 'inputs/server' / summary['train']
    play = OUT / 'inputs/server' / summary['play']
    tm, pm = read(train / 'manifest.json'), read(play / 'manifest.json')
    settings = pm['gallop_settings']
    assert settings == tm['gallop_settings']
    source_matches = {}
    working_tree_matches = {}
    for key in ('gallop_core', 'gallop_env'):
        saved = (train / 'source' / (key + '.py')).read_bytes()
        local = (ROOT / 'isaac_task' / (key + '.py')).read_bytes()
        source_matches[key] = hashlib.sha256(saved).hexdigest() == tm[key + '_sha256']
        working_tree_matches[key] = saved == local
    assert all(source_matches.values()), source_matches
    rows = [json.loads(line) for line in (train / 'episodes.jsonl').read_text().splitlines() if line]
    episode = read(OUT / 'play_episode.json')
    with np.load(play / 'gallop_trace.npz', allow_pickle=False) as trace:
        data = {key: trace[key] for key in trace.files}

    def v(key):
        return data[key][:, 0]

    measured = v('valid').astype(bool) & v('gallop_metric_active').astype(bool)
    aborts = np.flatnonzero(np.diff(v('flight_aborted_landings'), prepend=0) > 0)
    evidence = []
    for i in aborts:
        evidence.append({
            'time_s': float(data['time'][i]),
            'previous_sample_recovering': bool(v('flight_recovering')[i - 1]) if i else False,
            'previous_sample_recovery_elapsed_s': float(v('flight_recovery_elapsed')[i - 1]) if i else 0,
            'sample_has_contact': bool(v('contact')[i].any()),
            'sample_tilt_max_rad': float(np.abs(v('rpy')[i, :2]).max()),
            'sample_angular_speed_rad_s': float(np.linalg.norm(v('angular_velocity')[i])),
            'sample_lateral_speed_m_s': float(abs(v('body_velocity')[i, 1])),
            'sample_height_m': float(v('position')[i, 2]),
        })
    updates = summary['training']['additional_updates']
    per_update = summary['training']['rollout_steps_per_update']
    training = {
        'all': group(rows),
        'first_1000_updates': group([row for row in rows if row['step'] <= 1000 * per_update]),
        'last_1000_updates': group([row for row in rows if row['step'] > (updates - 1000) * per_update]),
        'note': 'Episode cohorts are different lengths and not independent policy evaluations.',
    }
    rewards = summary['training']['scalar_summary']
    training['reward_logging'] = {key: rewards['Episode_Reward/' + key] for key in
                                ('gallop_leap_distance', 'gallop_flight_takeoff', 'gallop_flight_landing')}
    playback = {key: episode[key] for key in (
        'leap_reward_points', 'leap_reward_steps', 'flight_landings', 'flight_stable_landings',
        'flight_aborted_landings', 'leap_events', 'leap_qualified_events', 'leap_distance_mean',
        'leap_distance_max', 'leap_air_seconds_mean', 'flight_takeoff_vz_mean',
        'flight_landing_peak_force_bw', 'flight_landing_peak_down_vz')}
    playback['trace_reward_sums'] = {key: float(v(key).sum()) for key in (
        'flight_step_takeoff_score', 'flight_step_landing_score',
        'flight_step_confirmed_distance_score', 'leap_reward_score')}
    playback['max_sampled_recovery_s'] = float(v('flight_recovery_elapsed').max())
    playback['measured_snapshot_limits'] = {
        'max_tilt_rad': float(np.abs(v('rpy')[measured, :2]).max()),
        'max_angular_speed_rad_s': float(np.linalg.norm(v('angular_velocity')[measured], axis=1).max()),
        'max_lateral_speed_m_s': float(np.abs(v('body_velocity')[measured, 1]).max()),
        'height_min_m': float(v('position')[measured, 2].min()),
        'height_max_m': float(v('position')[measured, 2].max()),
    }
    result = {
        'source_matches_received_run': source_matches,
        'working_tree_matches_received_source': working_tree_matches,
        'training': training, 'playback': playback,
        'gates': {key: settings[key] for key in (
            'flight_recovery_seconds', 'flight_max_foot_slide', 'flight_max_tilt',
            'flight_max_angular_speed', 'flight_max_lateral_speed', 'flight_max_force_bw',
            'flight_max_landing_vz', 'flight_min_seconds', 'leap_partial_credit', 'leap_progress_exponent')},
        'abort_snapshot_evidence': evidence,
        'limitations': [
            'Landing checks run every 5 ms; saved pose/contact samples are only every 20 ms.',
            'Raw world foot velocities and per-condition abort reasons are not saved.',
            'Force/downward-speed runtime peaks are below their gates; exact remaining abort causes are unproven.',
            'Only one final-policy playback was received; there is no matched parent-policy evaluation.',
        ],
    }
    (OUT / 'reward_landing_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    first, last = training['first_1000_updates'], training['last_1000_updates']
    distance = training['reward_logging']['gallop_leap_distance']
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Gallop 거리 보상·착지 점검</title>
<style>body{{font:17px/1.75 sans-serif;max-width:940px;margin:40px auto;padding:0 20px;color:#182536}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4dd;padding:10px;text-align:left}}code{{background:#edf1f5;padding:2px 5px}}a{{color:#1261ad}}</style>
<h1>거리 보상·착지 점검</h1><p>추가 4,000회 학습과 최종 모델 플레이의 수신 로그·당시 보존 소스를 분석했습니다. 현재 작업 트리의 수정본을 평가한 결과는 아닙니다. 이 수집기는 학습 코드·설정·원본 결과를 수정하지 않습니다.</p>
<h2>확인된 결과</h2><table><tr><th>지표</th><th>초반 1,000회</th><th>후반 1,000회</th></tr>
<tr><td>안정 착지 / 착지 후보</td><td>{int(first['flight_stable_landings']):,} / {int(first['flight_landings']):,} ({first['stable_fraction_of_candidates']:.2%})</td><td>{int(last['flight_stable_landings']):,} / {int(last['flight_landings']):,} ({last['stable_fraction_of_candidates']:.2%})</td></tr>
<tr><td>TensorBoard 거리 보상 기록 평균</td><td>{distance['first_1000_mean']:.6f}</td><td>{distance['last_1000_mean']:.6f}</td></tr></table>
<p>위 비교는 길이와 표본 수가 다른 학습 중 episode 집계입니다. 동일 조건 독립 평가에서 정책이 퇴보했다고 단정할 수는 없습니다. 전체 학습에서는 안정 착지 {int(training['all']['flight_stable_landings']):,}회, 거리 보상 지급 {int(training['all']['leap_reward_steps']):,}개 제어 스텝이 확인되어 보상 경로가 항상 0인 것은 아닙니다.</p>
<p>최종 플레이: 착지 후보 56회, 안정 착지 0회, 중단 55회, 나머지 1회는 기록 종료 시 회복 진행 중입니다. 거리·착지 보상은 모두 0이며 이륙 보상은 지급됐습니다. 최대 도약 {episode['leap_distance_max']*100:.2f}cm, 평균 체공 {episode['leap_air_seconds_mean']*1000:.2f}ms입니다.</p>
<h2>코드에서 확인한 구조</h2><p><code>gallop_core.py</code>는 목표 미달 도약의 부분 점수를 <code>0.5 × ((거리/0.5m) × (체공/0.24초))²</code>로 계산합니다. 각 비율은 0~1로 제한되며 주기별 최고점 증가분만 적립합니다. 0.5m·0.24초를 모두 만족하면 원점수 1입니다.</p>
<p>거리 점수는 착지 뒤 회복 상태를 0.12초 유지한 경우에만 자세 품질을 곱해 확정됩니다. 회복 중 재이륙, 접촉 상실, 과도한 기울기·회전·옆속도·충격·접촉한 발의 수평 속도 초과 중 하나라도 있으면 적립분을 취소합니다. <code>gallop_env.py</code>는 이 확정 점수에 가중치 24를 적용합니다. 이륙 보상은 이보다 먼저 별도로 지급됩니다.</p>
<h2>원인을 확정할 수 있는 범위</h2><p>플레이 회복 시간의 저장 표본 최대는 {playback['max_sampled_recovery_s']:.3f}초로 요구 0.12초에 못 미쳤습니다. 물리 스텝에서 누적한 착지 최대 힘은 체중의 {episode['flight_landing_peak_force_bw']:.2f}배(한도 3배), 최대 하강 속도는 {episode['flight_landing_peak_down_vz']:.2f}m/s(한도 1.65m/s)여서 이 두 조건은 초과하지 않았습니다.</p>
<p>저장된 자세·회전·옆속도·높이 표본도 한도 안입니다. 다만 판정은 5ms마다, 저장은 20ms마다이며 접촉한 발의 수평 속도와 중단 사유별 기록이 없습니다. 따라서 발 미끄러짐이나 짧은 접촉 상실을 확정 원인으로 지목할 수 없습니다.</p>
<h2>권장 순서</h2><p>동일 설정으로 장시간 이어 학습하기 전에 5ms 착지 중단 사유와 발 속도를 기록해 가장 자주 막히는 조건을 확인해야 합니다. 이후 관절 안전 한도를 유지하면서 목표 미달 도약의 부분 보상과 착지 보상 배분을 검토하고, 같은 초기 조건의 평가로 거리·체공·착지를 함께 비교하는 편이 타당합니다.</p>
<p><a href="reward_landing_audit.json">계산값·중단 시점 JSON</a> · <a href="report.html">전체 결과 보고서</a></p>
<details><summary>상세 수치</summary><pre>{html.escape(json.dumps(result, ensure_ascii=False, indent=2))}</pre></details></html>'''
    (OUT / 'reward_landing_audit.html').write_text(body)
    print(json.dumps({'source_matches': source_matches, 'play_distance_reward': episode['leap_reward_points'],
                      'stable_fraction_first': first['stable_fraction_of_candidates'],
                      'stable_fraction_last': last['stable_fraction_of_candidates']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
