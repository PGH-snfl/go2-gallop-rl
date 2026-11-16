# 자동승급 갤롭 추가15,500회 결과 수집 — 2026-11-11

**판정: 학습은 정상 완료했지만 자동승급은0회다. 누적46,000회 최종 정책은 여전히0단계(2.5m/s·18cm)에 있다.** 저장된30초 플레이에서는 전진·공중기·도약을 유지했고 착지와 앞발 폭이 이전 저장 플레이보다 개선됐다. 다만 앞발 폭·발 높이 조건에 미달하여 전체 성공은 아니다.80cm 단계에는 진입하지 않았다.

이번 작업은 기존 자료의 오프라인 수집/계산이다. PPO0·새 물리 rollout0·모방0. 원본 모델·로그·영상·학습 소스는 보존했다. 선택 기록은 완료 사실과 보고서 경로만 갱신하고 보호 기준30k나 다음 학습 부모를 자동으로 교체하지 않았다.

## 실행과 완료 근거

|항목|결과|
|---|---|
|학습 실행| `runs/isaac/train/unitree_go2_gallop/20261110_223314_cef16b/` |
|모델| `model_final.pt` |
|부모|파일럿500회, 누적30,500회 `20261110_214726_dc4067/model_final.pt`|
|이번 PPO|15,500회 / 새95,232,000transition|
|누적|46,000회|
|상태|completed, 마지막COMPLETED2026-11-11 02:26:32 KST|
|소요|13,991.83초 ≈3시간53분12초|
|완료 검증|status15500·checkpoint iter15500·PPO loss15500step 모두 일치|
|저장 목표|0단계2.5m/s·18cm; 최고 확인 단계 없음|
|최종 SHA256|`32ac3e282cf014ec8ac342ca832a49ea809b1e9a158beb17b490c9fe6b22ecdb`|
|플레이| `runs/isaac/play/unitree_go2_gallop/20261111_134458_3dcc3c/` |

원본 정보: [summary.json](summary.json), [staged_review.json](staged_review.json), [원본 파일 해시](evidence_manifest.json), [최종 검증](verification.json).

## 이번 플레이

[플레이 영상](../../runs/isaac/play/unitree_go2_gallop/20261111_134458_3dcc3c/videos/gallop-play-step-0.mp4) · [동작 프레임](gallop_keyframes.png) · [접촉 패턴](play_contacts.png) · [발 높이·필터 그래프](guarded_height_and_intervention.png).

- 1500policy step=30.00초, 실패 종료0/자연 timeout1. 마지막 reset_count1은30초 시간 종료이며 낙상이 아니다.
- 학생 단독 제어는 약7.76초 후 진입했고 warmup 이후 **8.78~30.00초,1062표본=21.24초**를 정식 계측했다.
- 정식 구간 평균 몸체 전진속도 **2.529m/s**, 명령2.5m/s. 저장 상태 기준 RMSE0.159m/s. episode 누적 계산은0.1587m/s로 표본 시점 차이가 있다.
- 계측 구간 이동량 약53.82m. 위상당 이동량이나 공중 이동거리와는 다른 지표다.
- 도약55회, 평균20.24cm·최대23.80cm. 평균 공중시간85.18ms.
- 거리/시간 자격을 충족한 도약53회. 착지 후 안정 확인44회/착지 평가54회=**81.48%**. 거리/시간 자격과 안정 착지를 모두 만족한 사건은43회다.
- 20ms 스냅샷의 완전 공중 시간 비율20.72%; runtime의 공중기 포함 주기 비율96.23%(51/53주기). 두 비율의 분모가 다르다.
- 도약 이벤트 기하/카운터는5ms tracker에서 수집된6000physics 표본에 근거한다. 신규 simulation을 실행한 것이 아니다.
- 앞다리 impulse share 약35.74%(FL14.97%,FR20.77%); 앞다리 하중·운동·반복 참여 gate 통과.
- 접촉 순서 정확도4.85%는 현재 엄격 순서가 필수 조건이 아니므로 이번 실패 원인으로 삼지 않는다. strict gallop 분류를 통과했다고도 주장하지 않는다.

영상은1499frames/50fps/29.98초/640×480로 읽기 검증했다. 요청된1500step과 영상1499frame의 차이를 숨기지 않았다. 학습 자체에는 영상이 없고 이번 플레이 영상1개가 있다.

## 부모 파일럿의 저장 플레이와 비교

**각1회 저장된 stochastic play끼리의 비교**다. 과거 파일럿의 명목8시작 평균20.38cm·앞폭61.38%와 직접 섞지 않는다. 시작상태/무작위화·전환시간이 완전히 통제된 새 다중seed 평가가 아니므로 추가학습의 인과 효과나 통계적 유의성을 주장하지 않는다.

|물리 지표|부모 누적30,500회 저장 play|이번 누적46,000회 저장 play|
|---|---:|---:|
|정식 계측 구간|21.38초|21.24초|
|실제 전진속도|2.470m/s|2.529m/s|
|평균 도약거리|18.24cm|20.24cm|
|최대 도약거리|22.97cm|23.80cm|
|도약 사건|59|55|
|안정 착지율|68.52%|81.48%|
|앞발 폭24~30cm 통과율|52.56%|73.63%|
|뒷발 폭24~30cm 통과율|96.00%|99.83%|
|yaw RMS, 정식 구간|1.11°|1.37°|
|roll RMS|1.93°|1.82°|
|pitch RMS|1.55°|4.02°|
|앞발 높이 초과율, pair평균·기준+1mm|7.16%|10.40%|
|뒷발 높이 초과율, pair평균·기준+1mm|4.96%|5.37%|
|raw 요청각 초과율|5.18%|2.87%|
|전체 필터 개입률|50.55%|53.77%|
|전체 성공|실패|실패|

도약·착지·앞폭은 이 두 저장 플레이에서 개선됐지만 pitch 진동/편향과 앞발 높이는 악화됐다. **안정성 전체가 개선됐다거나 최신 모델이 최적이라고 단정하지 않는다.**

현재 formal 평균발폭 자체는 앞/뒤 기준 근처여도 통과율이 낮을 수 있다. 폭 gate는 각 쌍의 **두 발이 모두 지지하는 순간**에서24–30cm를 만족하는 비율을 보며, 전체 시간 평균폭과 다르다.

원본 비교: [heading_parent_comparison.json](heading_parent_comparison.json), [staged_review.json](staged_review.json).

## 현재 플레이의 실패 조건

|조건|현재|기준|판정|
|---|---:|---:|---|
|앞발 지지폭 통과율|73.63%|80% 이상|실패|
|뒷발 지지폭 통과율|99.83%|80% 이상|통과|
|안정 착지율|81.48%|80% 이상|통과|
|앞발 높이 초과율|10.40%|10% 이하|실패|
|뒷발 높이 초과율|5.37%|10% 이하|통과|
|요청각 초과율|2.87%|10% 이하|통과|
|실제 관절 지속 초과|0|없어야 함|통과|
|정식 계측 시간|21.24초|10초 이상|통과|
|전진속도 RMSE|0.159m/s|0.25m/s 이하|통과|
|yaw RMS|1.37°|3° 이하|통과|

기록된 실패 gate는 `stable`, `stabilization_width`, `guarded_feet`다. 여기서 `stable=false`는 낙상을 뜻하지 않는다. 코드에서 stable에 관절/발 높이 보호 gate도 연결되므로 이번에는 앞발 높이 초과가 영향을 준다.

앞발 높이별:

- FL cap16cm, 최대23.88cm, cap+1mm초과15.25%.
- FR cap16cm, 최대20.86cm, cap+1mm초과5.56%.
- RL/RR cap10cm, 최대12.24/13.30cm, cap+1mm초과3.48/7.25%.
- 앞쪽 두 발의 초과율 평균이10.40%로10% gate를 넘었다.16/10cm는 이 보상의 soft height 기준으로 관절의 물리 한계와 구분한다.

## 관절·필터 검증

[guarded_play_diagnostics.json](guarded_play_diagnostics.json)에서 저장된 요청→명령 필터, 변화율, 보정 비용, 실제 관절 초과와 발 높이 비용을 재계산해 원래 기록과 대조했다.

- 실행 target의 작업/물리 관절 범위 초과0.
- target 변화율 초과0; hip6, thigh/calf16rad/s 제한 유지.
- 원래 기계적 범위 초과0; 축소된 물리 범위에서도1mrad/허용.01rad 이상 초과0.
- 수치 정밀도 수준의 최대 초과2.98e-8rad; 지속 위반 시간0.
- raw 요청각 초과2.87%; 전체 필터 개입53.77%, 속도 제한 개입49.44%. 중복되는 값이므로 두 비율을 더하지 않는다.
- 요청과 실행 target의 RMSE0.144rad, 실제 관절과 target의 RMSE0.388rad.

따라서 **이번 플레이에 여전히 필터 개입은 있지만, 실제 관절 한계 위반이 관측된 것은 아니다.** target 변화율과 실제 관절 속도도 다른 양이다.

학습 전체까지 실제 초과0이라고 확대하지 않는다.5초 이상 계측된 학습61,359episode 중21개에는 `guarded_actual>0` 기록이 있었다. 마지막1000회 구간에서는3982개 중1개다. 이 통계는10초 이상인 staged 성공 계측 집단과 분모가 다르다. 관련 원본: [guarded_training_diagnostics.json](guarded_training_diagnostics.json).

## 자동승급이 없었던 이유

[staged_training_progress.png](staged_training_progress.png), [staged_review.json](staged_review.json).

학습75,656episode에서211개 성공(0.279%)이었다. 모든 환경의 첫 완료episode만 받는 완전 집단 규칙에 채택된 결과는47,278개/성공163개이며, 나머지는 반복환경·초기 적응 등의 집계 제외다. 종료·실패 기록 자체는 모두 보존된다.

- 승급 판단372회 모두 저장된 core와episode로 정확 재현했다. episode별 채택 여부도 모두 재현했다.
- 완전 집단184회. 최고 성공률9/256=**3.515625%**, 마지막 완전 집단0%, 마지막10집단 평균0.15625%.
- 승급 기준80%를 두 집단 연속 충족한 적이 없어0단계 유지가 정상 판정이다.
- 마지막 판단의1.149%(2/174)는 미완성 집단의 중간값이며 마지막 완전 집단 성공률0%와 구분한다.
- 최고/최종 목표를 속도3m/s·80cm라고 설정한 것과 실제 그 단계에서 학습한 것은 다르다. 기록된 모든episode와 최종 파일은0단계다.

충분한 학생 계측10초 이상인61,263episode를 따로 보면:

|gate|통과율|
|---|---:|
|접촉 패턴|100.00%|
|공중기 포함 주기 조건|99.81%|
|앞다리 하중/ROM/참여|각100.00%|
|몸체 자세|99.93%|
|거리 조건, 현18cm|85.15%|
|발 높이|74.08%|
|정밀 속도·yaw 결합 추종|68.12%|
|안정 착지 조건|14.42%|
|앞뒤 발폭 결합 조건|**3.61%**|
|전체 성공|0.344%|

**가장 뚜렷한 병목은 학습 분포에서의 발폭과 안정 착지 조건**이다. 한 번의deterministic-action play에서 착지81%가 나온 것과 randomized/on-policy training 집단에서 착지 gate 통과가14%인 것은 모순이 아니다. 전자는 한episode의 착지 사건 비율이고 후자는 그 기준을 통과한 episode 비율이다.

학습구간의 event-weighted 평균 도약은 첫1000회18.20cm→마지막500회19.67cm로 완만하게 늘었다. 반면 발폭·전체 성공은 안정적으로 상승하지 않았다. 횟수를 더 늘리면 곧80cm가 될 것이라는 근거는 이번 로그에서 얻지 못했다.

## 이전 문헌조사의 phase profile 문제와의 관계

[문헌조사 중 구현 점검](../20261110_animal_gait_reward_literature/implementation_audit.md)에 기록된 보상 참조 profile 누락은 단계index4(30cm)부터 실제 참조 차이가 생긴다. **이번 학습은 모든episode가stage0(alpha0)이므로 그 누락이 이번0단계 정체를 설명하지는 않는다.** 향후30cm 이상 실험 전에 해결해야 할 별도 문제다. 이번 수집에서는 학습 코드를 수정하지 않았다.

## 다음 작업의 근거

- 현재 정책은 전진·공중기·도약을 유지하므로 보행 전환에 영구 정체됐던34k 실패 정책과 같은 붕괴로 분류하지 않는다.
- 그러나 전체 조건·승급·80cm 달성에는 실패했다. 최신46k를 기본 부모로 자동 채택하거나 폐기하지 않고 보존한다.
- 다음 수정은 폭 조건, 앞발 높이, 착지 보상 간 균형과 학습/플레이 분포 차이를 먼저 점검할 근거가 강하다. 관절 물리 제한을 풀어야 한다는 근거는 없다.
- 중간checkpoint가 더 나은지는 동일 조건의 여러seed 동결 평가가 필요하다. 학습 중 일부 구간 성공률이 더 높다는 이유만으로 특정 중간모델이 최선이라고 선택하지 않았다.
- 독립 다중seed 평가, 새로운MuJoCo Sim2Sim, 실기는 이번에 실행하지 않았다.

## 산출물

- [종합 데이터](summary.json), [단계 재현·부모 비교·1000회 구간 통계](staged_review.json).
- [학습곡선](training_curves.png), [단계/병목 추이](staged_training_progress.png), [접촉](play_contacts.png), [이륙·착지 사건](play_flight_events.png), [몸체·발폭](heading_motion_play.png).
- [각 도약 사건CSV](play_flight_events.csv), [관절별CSV](guarded_joint_diagnostics.csv), [episode CSV](train_episodes.csv), [TensorBoard CSV](tensorboard_scalars.csv).
- [원본 증거 해시](evidence_manifest.json), [수집 후 무결성 검증](verification.json), [수집 목록](collection_manifest.json).
- 기본 collector의 `leap_curves.png` 붉은80cm선은 최종설정 참고선이다. 현재18cm 활성 목표는 `staged_training_progress.png`에 별도로 표시했다.

재현:

```bash
cd /home/cvr/GH/go2_animalization/gallop
MPLCONFIGDIR=/tmp/go2-gallop-analysis-matplotlib \
/home/cvr/miniconda3/envs/kine2go/bin/python \
  -m scripts.analysis.collect_staged_gallop_review \
  --report reports/20261111_gallop_staged_46000_review \
  --parent-play runs/isaac/play/unitree_go2_gallop/20261110_215639_62b575
```

기본 수집/heading/flight/guarded 분석은 해당 report의로그와source에 기록했다. 자동계산 산출물은 원본의 파생자료이며 기존run을 덮어쓰지 않는다.

