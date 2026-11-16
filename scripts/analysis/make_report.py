"""Render the full 40-clip comparison as a standalone research report."""
from pathlib import Path
import html
import json
from collections import defaultdict

from gpd.paths import ROOT

def category(name):
    if name in ['ai4_dog_canter','ai4_dog_run_00','ai4_dog_run_02']:
        return '높음 — 비대칭 달리기 reference', '앞·뒤 좌우 위상 모두 비대칭. 5개 높이 임계값에서 회전형 갤럽 유사 패턴 유지.'
    if name=='ai4_dog_run_04':
        return '대조군 — trot 중심', 'trot_01과 원본 배열이 같고 해당 trot 구간을 포함. run이라는 이름만으로 gallop 판정 불가.'
    if name in ['ai4_dog_trot_00','ai4_dog_trot_01']:
        return '대조군 — trot', '대각 다리 교대 달리기 비교용. 특히 trot_00은 높이 임계값만으로 라벨이 불안정해 원 라벨과 함께 해석.'
    if name in ['solo8_jump_forward_a','solo8_jump_forward_b']:
        return '부분 유사 — 도약', '몸통 수직 이동·공중 구간은 유사하나 안정된 비대칭 네 발 순서를 뜻하지 않음. 도약/착지 대조군.'
    if name=='solo8_scoot_forward':
        return '부분 유사 — scoot', '큰 몸통 pitch와 높이 변화가 있으나 대표 rollout의 발 위상은 trot 유사가 우세. gallop 직접 reference로 비추천.'
    if name=='solo8_crawl_fast':
        return '보조 관찰 — 빠른 crawl', '대표 rollout에 비대칭 추정 구간 일부가 있으나 여러 패턴 혼재·trot 유사가 우세. 숨은 gallop으로 확정 불가.'
    if name=='ai4_dog_synth_half_flip_jump':
        return '부분 유사 — 정책에서 생긴 점프', 'reference가 ellipse_walk와 바이트 단위 동일. 점프는 policy 추종 실패에서 생긴 180° 회전 동작(논문 Appendix A).'
    if name in ['ai4_dog_walk_03','ai4_dog_walk_04','ai4_dog_walk_06']:
        return '보조/대조 — 빠른 walk·trot 구간', 'walk_03은 trot_00과 원본 공유; walk_04·06은 빠른 trot 유사 구간. 일부 비대칭 판정은 임계값/전환에 민감.'
    if 'trot' in name:
        return '대조군 — horse trot', '말 trot만 제공되며 말 gallop은 목록에 없음. reference의 발 높이를 실제 공중기로 오인하면 안 됨.'
    if name=='ai4_dog_pace':
        return '낮음 — pace', '동측 다리 협응 비교용. 갤럽형 앞·뒤 leading/trailing 관계와 구별.'
    return '낮음 — 걷기·회전·횡이동', '현재 목록과 선택 구간에서 반복적 gallop reference를 뒷받침하는 근거 부족. 일부 단발 proxy 라벨로 재분류하지 않음.'

def main():
    (ROOT/"reports/kine2go").mkdir(parents=True,exist_ok=True)
    ref=json.loads((ROOT/'reports/kine2go/comparison.json').read_text())
    rollout=json.loads((ROOT/'reports/kine2go/rollout_comparison.json').read_text())
    byroll=defaultdict(list)
    for r in rollout['rollouts']:byroll[r['name']].append(r)
    rev=ref['dataset_revision']
    rows=[];details=[]
    for r in ref['records']:
        name=r['name'];m=r['selected'];b=byroll[name][0]['metrics'];cat,reason=category(name)
        url=f'https://huggingface.co/datasets/MIMUW-Robotics/kine2go/tree/{rev}/data/{name}'
        p=m['proxy'][2];patterns=html.escape(json.dumps(p['pattern_counts']))
        rows.append(f'<tr data-name="{name}" data-category="{html.escape(cat)}"><td><a href="{url}">{name}</a></td><td>{html.escape(cat)}</td><td>{r["config_motion_range"]}</td><td>{m["frames"]}</td><td>{m["mean_forward_speed_m_s"]:.2f}</td><td>{m["pitch_excursion_deg"]:.1f}</td><td>{m["base_height_excursion_m"]*100:.1f}</td><td>{b["mean_forward_speed_m_s"]:.2f}</td><td>{html.escape(reason)}</td></tr>')
        threshold_rows=''.join(f'<tr><td>{p["threshold_above_each_foot_baseline_m"]*100:.1f} cm</td><td>{html.escape(json.dumps(p["pattern_counts"]))}</td><td>{p["all_feet_above_threshold_fraction"]*100:.1f}%</td></tr>' for p in m['proxy'])
        rollrows=''.join(f'<tr><td>{x["trajectory"]}</td><td>{x["metrics"]["frames"]}</td><td>{x["metrics"]["mean_forward_speed_m_s"]:.2f}</td><td>{x["metrics"]["pitch_excursion_deg"]:.1f}</td><td>{html.escape(json.dumps(x["metrics"]["proxy"][1]["pattern_counts"]))}</td></tr>' for x in byroll[name])
        details.append(f'<details><summary>{name} — {html.escape(cat)}</summary><p>{html.escape(reason)}</p><p>원본 파일: {html.escape(r["catalogue"]["source"])} / catalogue: [{r["catalogue"]["start"]}, {r["catalogue"]["end"]}) / 실제 config: {r["config_motion_range"]}. 배열 {r["array_frames"]}프레임, 선택 구간 {m["frames"]}프레임, 메타데이터 n_frames {r["metadata"]["n_frames"]}는 확인한 rollout 길이와 일치.</p><p>속도는 저장된 Go2 reference의 root 궤적을 60 Hz로 재생한 값이며 원래 동물 실측 속도가 아니다. 선택 구간의 hip/thigh/calf 최대 수치 미분 속도: {m["max_joint_velocity_by_group_hip_thigh_calf_rad_s"]}. 위치 한계 초과: {m["joint_position_max_excess_rad"]:.6f} rad. FK 최대 오차 {r["fk_max_error_m"]:.8f} m.</p><table><tr><th>높이 임계값</th><th>추정 주기 패턴 수</th><th>네 발이 모두 높은 프레임 비율</th></tr>{threshold_rows}</table><p>마지막 열은 실제 공중기 비율이 아닌 운동학적 proxy다. 50프레임 이하 reference의 위상 추정은 가상 반복을 사용했으며 반복 주행 성공을 증명하지 않는다.</p><table><tr><th>공개 rollout</th><th>프레임</th><th>전진 속도</th><th>pitch 범위 폭(°)</th><th>발 높이 기반 패턴 수</th></tr>{rollrows}</table></details>')
    duplicates=''.join(f'<li>{" ↔ ".join(x)}</li>' for x in ref['identical_reference_groups'])
    narrative='''
<h1>Kine2Go 전체 40개 동작과 Gallop 유사성 비교</h1>
<p>분석일 2026-10-06 · 목표 기체 Go2 EDU · 입력은 공개 Kine2Go Go2 reference와 Genesis rollout. EDU 실물 검증 결과와 구별한다.</p>
<div class="conclusion"><strong>가장 가까운 reference는 ai4_dog_canter, ai4_dog_run_00, ai4_dog_run_02다.</strong> 원본의 발 높이로 추정한 앞·뒤 좌우 위상은 반복적 비대칭이고, 1–3 cm 임계값 변화에서도 rotary gallop 유사 분류가 유지됐다. 초기 주기 소재로는 <strong>run_02</strong>를 먼저 검토하고, run_00의 다른 주기와 canter를 비교하는 것이 타당하다. 실제 접촉력은 공개 파일에 없어 엄밀한 gallop 분류·실물 실행 가능성까지 확정하지 않는다.</div>
<h2>범위와 방법</h2>
<ul><li>공식 40개 폴더의 motion.npy·clip.json·config.json을 모두 확인했다. 40개는 policy/clip 이름 수이며 전체 reference 배열은 34종이다.</li><li>reference 전체와 policy_cfg.motion_range로 잘라 실제 loader가 사용할 구간을 각각 분석했다. Python의 끝점 제외 슬라이스를 사용했으며 -1은 마지막 프레임을 제외한다.</li><li>모든 clip의 대표 공개 rollout traj_0000을 확인하고 canter/run_00/run_02/run_04는 0001·0002까지 확인했다. 총 48개이며 공개 800개 전체의 성공률 분석은 아니다.</li><li>발 위치·발 순서·위상, root 이동, pitch, 몸통 높이 변화, 관절 범위·속도와 반복 경계를 비교했다. 속도는 reference/rollout root 궤적의 수치 미분을 몸통 축으로 회전해 평균한 값이다.</li><li>접촉력 필드는 export에 없다. reference에서는 각 발 높이의 하위 2%를 기준으로 +1/1.5/2/2.5/3 cm의 낮은 발 mask를 만들었다. rollout에서는 z=0 평지와 반지름 0.022 m에 +0/5/10/15 mm 여유를 적용했다.</li><li>위상은 RL의 연속 추정 touchdown 사이에서 각 발 이벤트가 하나인 주기에만 계산했다. 단일 프레임 토글을 정리했으며 동시에 가까운 위상의 허용오차는 주기의 0.08이다. 이것은 비교용 휴리스틱이며 생물학적 보편 임계값이 아니다.</li><li>GRF가 없어 발이 낮은 것과 실제 지지하는 것은 같지 않다. 말 reference에서 큰 가짜 공중 비율이 나온 사례, trot_00의 임계값 민감성, rollout 앞발의 추가 낮은 구간은 이 방법의 한계를 보여준다. 단발 분류로 clip 전체를 재라벨하지 않았다.</li></ul>
<h2>갤럽과 가까운 후보별 판단</h2>
<table><tr><th>후보</th><th>선택 범위 / 길이</th><th>reference 전진 속도</th><th>몸통 pitch 변화 폭</th><th>판단</th></tr>
<tr><td>ai4_dog_run_02</td><td>[35,196) / 161프레임</td><td>2.60 m/s</td><td>22.8°</td><td>가장 먼저 검토할 주기 소재. 2 cm 기준 5개의 완전한 비대칭 추정 주기, 1–3 cm 모두 rotary 유사. 독립된 run 파일이며 반복 경계가 좋은 27프레임 후보 존재.</td></tr>
<tr><td>ai4_dog_run_00</td><td>[399,535) / 136프레임</td><td>2.64 m/s</td><td>25.7°</td><td>반복 비대칭 달리기 소재. 2 cm 기준 4주기, 1–3 cm 모두 rotary 유사. canter 원본과 동일하며 canter 구간을 포함.</td></tr>
<tr><td>ai4_dog_canter</td><td>[430,459) / 29프레임</td><td>2.59 m/s</td><td>24.7°</td><td>공식 단일 짧은 달리기 reference. 높이 기반 위상에서는 네 발이 따로 나타나므로 이름만으로 엄격한 3박자 canter라고 가정하지 않는다. 가상 반복 결과를 자연 연속 주기 수로 세지 않았다.</td></tr>
<tr><td>ai4_dog_run_04</td><td>[493,716) / 223프레임</td><td>1.52 m/s</td><td>9.9°</td><td>trot 중심 대조군. trot_01의 원본과 동일하고 [630,663) 구간을 포함한다. 2 cm 기준 trot 유사 3주기·모호한 2주기.</td></tr>
<tr><td>ai4_dog_trot_00 / trot_01</td><td>[448,481) / [630,663)</td><td>1.46 / 1.44 m/s</td><td>7.4° / 3.4°</td><td>빠른 trot와 gallop의 구분용. trot_00은 높이 proxy만으로 분류하기 어려우므로 공식 라벨을 덮어쓰지 않는다.</td></tr>
<tr><td>solo8_jump_forward_a / b</td><td>[52,114) / [37,112)</td><td>1.50 / 1.17 m/s</td><td>13.8° / 5.2°</td><td>몸통 높이 변화가 22.2 / 23.0 cm로 크다. 도약·착지 및 공중 구간 비교에 유용하나 비대칭 gallop 주기의 직접 reference는 아니다.</td></tr>
<tr><td>solo8_scoot_forward</td><td>[69,119) / 50프레임</td><td>1.22 m/s</td><td>20.6°</td><td>큰 몸통 운동은 유사하지만 공개 대표 rollout에서는 trot 유사 패턴이 우세. 모양의 유사성과 발 순서를 구분해야 한다.</td></tr>
<tr><td>solo8_crawl_fast</td><td>[77,119) / 42프레임</td><td>1.44 m/s</td><td>2.9°</td><td>대표 rollout에 비대칭 proxy 주기가 일부 있지만 trot 등과 혼재. 이름 밖의 동작 변화 관찰용 보조 후보이며 깨끗한 gallop 소재로는 부족.</td></tr>
<tr><td>ai4_dog_synth_half_flip_jump</td><td>[116,1165) / 1049프레임 reference</td><td>0.87 m/s</td><td>5.4°</td><td>reference는 ellipse_walk와 동일. 점프는 별도 동물 갤럽 자료가 아니라 추종 실패에서 생긴 정책 동작. 공개 rollout 300프레임에서 pitch 변화 폭 37.2°·yaw 총 변화 약 180°.</td></tr>
<tr><td>ai4_dog_walk_03 / walk_04 / walk_06</td><td>각 config 구간</td><td>1.22 / 1.54 / 1.42 m/s</td><td>10.3° / 6.4° / 6.6°</td><td>이름이 walk여도 빠른 trot 구간이 있어 추가 비교했다. walk_03의 일부 비대칭 추정은 임계값에 민감하고 안정된 gallop 전체로 볼 근거는 부족하다.</td></tr></table>
<h2>발 순서와 공개 rollout의 차이</h2>
<p>주요 run reference의 반복 순서는 순환 시작점을 RR에 두면 대략 <strong>RR → RL → FL → FR</strong>이다. 뒷다리 lead는 RL, 앞다리 lead는 FR인 회전형 관계로 추정된다. canter는 같은 run_00 파일의 짧은 구간이다. 별도의 말 gallop이나 gallop이라는 이름의 clip은 공식 40개 목록에 없다.</p>
<p>세 후보의 대표 rollout은 전진 속도 약 2.5–2.7 m/s로 실행되지만, FL이 한 주기에 두 번 낮아지는 구간이 반복적으로 나타난다. canter/run_00/run_02의 세 표본에서 단순 저발높이 이벤트 수가 각각 다른 다리의 약 두 배인 경우가 있다. 실제 두 번의 지지 접촉인지, 순간적인 낮은 위치·접촉 변화인지는 저장된 힘이 없어 확정할 수 없다. 그래서 source의 4박자를 공개 policy가 완벽히 유지했다고 결론내리지 않는다.</p>
<p><img src="foot_patterns.png" alt="Selected references and Genesis rollout foot-height proxies"></p>
<h2>Go2 EDU용 주기 소재 선택</h2>
<p><strong>추천 순서: run_02의 좋은 주기 → run_00의 좋은 주기 → 공식 canter 구간을 비교군으로 유지.</strong> run_02에서 117→144 경계의 27프레임(0.45초) 후보는 관절 자세 최대 차이 약 1.82°, 높이 차이 약 1.81 mm다. run_00의 457→484 후보는 약 3.15°, 높이 차이 약 1.42 mm다. 이것은 경계 자세 비교이며 속도·접촉·힘까지 연속인 주기를 보증하지 않는다. 아직 NPZ 변환이나 학습 완료 결과가 아니다.</p>
<p>공식 canter의 430→459 경계는 관절 자세 차이 최대 약 18.38°다. 짧다는 이유로 그대로 반복하기 가장 좋은 clip이라고 단정하면 안 된다. 배열의 마지막 저장 표본(458)과 첫 표본(430) 차이 7.76°와, 한 주기 뒤 경계 표본(459)과 첫 표본 차이는 서로 다른 검사다.</p>
<p>세 후보의 무릎 최대 reference 속도는 수치 미분 기준 12.66 / 12.66 / 14.48 rad/s(canter/run_00/run_02)다. Unitree 공개 URDF 무릎 15.70 rad/s 아래이지만, 이 사실만으로 토크–속도·착지·전력 한계를 충족한다고 볼 수 없다. 몸통 pitch 변화가 큰데 실제 Go2 EDU의 안정성을 위해 얼마나 조정해야 하는지도 따로 평가해야 한다. EDU에 없는 척추 운동은 reference의 몸통 전체 pitch와 구별한다.</p>
<h2>이번 비교에서 확인한 데이터 해석 수정</h2>
<ol><li><strong>clip.json의 n_frames와 reference 길이가 다르다는 이유만으로 메타데이터 오류라고 단정하면 안 된다.</strong> 확인한 48개 rollout 길이는 각 clip의 n_frames와 모두 일치한다. 논문 Appendix B의 length도 rollout 프레임 수다. canter reference 567 vs n_frames 571, trot_01 reference 1263 vs 571 등은 서로 다른 데이터 단계의 길이로 해석해야 한다. 과거 로컬 audit의 “571과 다른 reference 길이이므로 metadata inconsistency” 판단은 이 비교로 수정한다.</li><li><strong>reference와 rollout의 관절 순서가 다르다.</strong> reference 6:18은 FL,FR,RL,RR 순으로 hip→thigh→calf의 group-major다. rollout dof_pos는 FR→FL→RR→RL의 leg-major다. 공개 metadata가 reference에도 leg-major라고 적은 부분은 실제 FK와 맞지 않는다. 40개 reference에서 올바른 순서의 FK 최대 오차는 약 3.3 μm, 모든 48개 rollout은 별도 순서와 발 link 인덱스를 검증했다.</li><li><strong>half_flip_jump의 출처 표가 논문과 로컬 catalogue에서 다르다.</strong> 로컬 motion_ranges.txt에는 half_flip_jump.txt가 쓰여 있지만 그 raw 파일이 없다. 논문은 ellipse_walk.txt를 사용했다고 밝히고, HF의 두 reference도 SHA-256이 동일하다. 따라서 점프 reference가 따로 존재한다고 가정하지 않는다.</li><li><strong>wide_strafe의 실제 config는 시작 12, 로컬 catalogue·논문은 121이다.</strong> 수치 비교는 실제 공개 config [12,1080)를 따랐으며 차이를 숨기지 않았다.</li><li>원본 저장 배열끼리 같은 경우가 6쌍 있다. clip 이름의 수, 원본 파일의 수, 학습에 선택한 구간 수, 공개 rollout 수를 구분해야 한다.</li></ol>
<h2>같은 원본 배열을 공유하는 쌍</h2><ul>__DUPLICATES__</ul>
<h2>전체 40개 비교표</h2><p>v_ref/v_roll은 전진 속도(m/s), pitch는 reference의 변화 폭(°), Δz는 reference 몸통 높이 변화(cm). rollout은 traj_0000 한 표본의 평균이다. 순위 점수나 성공률로 해석하지 않는다.</p>
<input id="filter" placeholder="동작 이름·판정 검색" aria-label="모션 검색"><p id="count">40개 / 40개</p>
<div class="scroll"><table id="all"><thead><tr><th>동작</th><th>갤럽 관련성</th><th>config 범위</th><th>선택 프레임</th><th>v_ref</th><th>pitch</th><th>Δz</th><th>v_roll</th><th>비교 판단</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<h2>개별 동작 상세 수치</h2>__DETAILS__
<h2>출처와 재현</h2><ul>
<li><a href="https://arxiv.org/html/2606.14433">Kine2Go 논문: Appendix A·B, 제외 동작과 future work</a></li>
<li><a href="https://huggingface.co/datasets/MIMUW-Robotics/kine2go">공식 공개 데이터셋</a> — 확인한 revision: <code>__REV__</code></li>
<li><a href="https://github.com/nomagiclab/kine2go-pipeline">공식 코드</a> — motion_ranges.txt, imitation_wrapper의 실제 슬라이스, trajectory export writer</li>
<li>다운로드 manifest: ../data/download_manifest.json, ../data/rollout_manifest.json. 전체 수치: comparison.json·csv, rollout_comparison.json·csv. 발 패턴 그림: foot_patterns.png·pdf.</li>
<li>분석 환경: /home/cvr/miniconda3/envs/kine2go/bin/python. 다시 계산: compare_kine2go.py → compare_rollouts.py → make_report.py.</li>
</ul><p>이 비교는 자료의 운동학·공개 시뮬레이션 결과를 분석한 것이다. 실제 Go2 EDU의 갤럽 성공, 힘 센서 기반 접촉 분류, 800개 전체의 성공률은 검증하지 않았다.</p>
'''
    narrative=narrative.replace('__DUPLICATES__',duplicates).replace('__ROWS__',''.join(rows)).replace('__DETAILS__',''.join(details)).replace('__REV__',rev)
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kine2Go 40개 Gallop 비교</title><style>body{font-family:system-ui,sans-serif;line-height:1.7;color:#162234;background:#f6f8fb;max-width:1320px;margin:auto;padding:28px}h1,h2{line-height:1.3}h2{margin-top:40px}.conclusion{background:#e2f3ed;border-left:5px solid #218464;padding:18px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #dce2e8;padding:10px;text-align:left;vertical-align:top}th{background:#e8eef5;position:sticky;top:0}a{color:#1c61ad}img{max-width:100%;background:white}details{background:white;padding:15px;margin:12px 0;border:1px solid #dce2e8}summary{cursor:pointer;font-weight:bold}.scroll{overflow:auto;max-height:950px}input{padding:12px;width:70%;font:inherit}code{overflow-wrap:anywhere}li{margin:8px 0}</style><body>'''+narrative+'''<script>const box=document.getElementById('filter');box.addEventListener('input',()=>{let n=0;document.querySelectorAll('#all tbody tr').forEach(row=>{const show=row.textContent.toLowerCase().includes(box.value.toLowerCase());row.hidden=!show;if(show)n++;});document.getElementById('count').textContent=n+'개 / 40개';});</script></body></html>'''
    (ROOT/'reports/kine2go/report.html').write_text(document)
    print('Wrote reports/kine2go/report.html; 40 rows and 40 detailed records.')

if __name__=='__main__':main()
