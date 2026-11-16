# 3m/s·실제 도약거리 보상 추가 학습 결과 — 2026-10-29

**3.0m/s 명령에 실제 평균 2.956m/s로 전진하고 관절 움직임도 커졌지만, 최종 플레이에서 목표 도약은 확인되지 않았다.** 5ms 계측의 완료 도약 이벤트와 실제 도약 보상은 모두 0회/0점이다. 학습 중에는 거리12cm·시간40ms를 함께 충족한 이벤트가 5회 있었으나, 최종 정책에서 반복 가능한 도약으로 이어지지 않았다. 전체 Gallop 스타일 성공 조건도 미달이다.

사용자가 완료한 학습·플레이를 오프라인 수집했다. 이번 수집에서 PPO 업데이트, 보상·학습 설정 변경, 새 시뮬레이션 실행은 하지 않았다. 원본 실행 자료와 모델은 보존했다.

## 모델과 실행 자료

| 항목 | 기록 |
|---|---|
| 학습 실행 | `20261029_140301_dbb121` |
| 최종 모델 | [model_final.pt](../../runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt) |
| 부모 모델 | `20261029_123734_c563b2/model_final.pt` — 2m/s·좌우 쌍 패널티 |
| 추가 학습 | 2,000 iteration, 256환경, 12,288,000 transition |
| 계보 누적 학습 | 14,000 iteration — 이전 단계12,000 + 이번2,000 |
| 완료 시각 / 학습 소요 | 2026-10-29 14:25:29 KST / 1,344.98초, 약22분25초 |
| 정책 구조 | actor47 / critic62 / action12, action scale0.5 |
| 초기화 | 부모 actor·critic·noise 복원, 새 Adam, 새 단계 iteration0부터 시작 |
| 명령 | vx=3.0m/s, vy=yaw=0 고정 |
| 플레이 실행 | `20261029_143102_f0e828`, 1환경·1,500step·30초 trace |
| 영상 | [플레이 영상 29.98초](../../runs/isaac/play/unitree_go2_gallop/20261029_143102_f0e828/videos/gallop-play-step-0.mp4), 640×480·50fps |
| 학습 영상 | 0개 — 이번 학습 실행에는 영상이 저장되지 않음 |

모델 SHA256: `7f1b6f63aeb66ca84bacf5ddca28b6c476f57cb6f33cf29981776019dd0a554d`. 플레이 manifest의 체크포인트 해시와 일치한다. 학습 당시 저장한 core/env로 재생했고 정책 상태 불변 및 `policy_updates=0`이 기록됐다.

## 최종 플레이에서 확인한 동작

첫 episode는 20초까지 진행하고 자연 timeout으로 종료했다(`terminated=0`, `timed_out=1`). 주요 속도·자세·접촉 통계는 초기1초를 제외한 **1~20초의951개 저장 표본**을 사용했다. 20초 reset 뒤 구간은 아래 주요 평가에서 제외했다. 따라서 단일 환경의 한 연속 구간 결과이며 여러 seed에 대한 성공률은 아니다.

| 지표 | 측정 | 판단 |
|---|---:|---|
| 평균 실제 전진속도 | **2.95632m/s** | 3.0명령 추종 |
| 전진속도 RMSE | **0.10822m/s** | 추종 조건 통과 |
| 횡속도 / yaw RMSE | 0.09490m/s / 0.15917rad/s | 각각0.15 / 0.3 기준 이내 |
| 분석 구간 순수평변위 | 55.06484m | 전진 확인 |
| 완전한 참조주기당 전진거리 | **0.94811m**, 58주기 | 지지·스윙을 모두 포함한 주기 이동거리 |
| 앞다리 전체 하중 분담 | **43.10%** | 25% 기준 통과 |
| 각 앞다리 하중 분담 | FL20.43% / FR22.67% | 각각8% 기준 통과 |
| 앞다리 ROM / 스윙·지지 참여 | 목표 대비98.47% / 100% | 앞다리 조건 통과 |
| roll / pitch RMS | 0.05835 / 0.05482rad | 자세 기준 통과 |
| 몸 높이 범위 | 0.29529~0.34216m | 유지 |
| 목표 관절각 clipping 평균 | **16.956%** | 허용10% 초과, stable 실패 |
| 기존 공중기 포함 주기 비율 | **0%** | 60% 기준 미달 |
| 새 5ms 완료 도약 / 기준 충족 / 보상 | **0회 / 0회 / 0점** | 목표 도약 미확인 |

주기 이동거리는 몸통 전진속도를 해당 참조주기에 적분한 값이다. **0.948m 전진을 공중 도약거리로 해석하지 않는다.** 앞다리 하중은 첫 연속 episode의 수직력 표본 합으로 근사하며, 다른 속도·ROM 통계의 warmup 제외 구간과 집계 범위가 다르다. episode 기록의 속도 RMSE0.10764와 저장 trace 재계산0.10822는 집계 시점 차이로 소폭 다르다.

### 도약 계측이 실제로 수행됐는지

자연 종료 episode에 `leap_physics_samples=4000`이 기록됐고, 첫 연속 구간의 각20ms 정책 step마다 표본 counter가 정확히4씩 증가했다. 5ms 물리 계측이 수행됐다는 기록이다. 그 구간의 누적 `leap_events`, `leap_qualified_events`, `leap_reward_points_total`은 모두0이다. 20ms 접촉 trace에서도 warmup 이후 동시 비접촉 표본은0개였다.

따라서 이번 결론은 20ms 영상/저장 표본에서 짧은 공중기를 놓쳤다는 주장에만 의존하지 않는다. **현재8N 접촉 시작/3N 접촉 해제 hysteresis와5ms 계측으로, warmup 이후 유효한 이륙→착지 이벤트가 없었다.** `leap_distance_mean=0`은 이벤트가 없을 때의 기록값이며, 길이0인 도약을 측정했다는 뜻이 아니다. 접촉력 임계값과5ms 해상도보다 짧은 물리 현상에 대한 별도 판정은 하지 않는다.

원본 [gallop_trace.npz](../../runs/isaac/play/unitree_go2_gallop/20261029_143102_f0e828/gallop_trace.npz)와 [전체 도약·좌우 쌍 trace CSV](play_leap_trace.csv)에 위치·시간·누적 counter·실제 보상·물리 표본 수를 보관했다. CSV는20ms snapshot이며 내부 계측 간격은5ms다.

## 학습 중 도약 보상이 얼마나 나왔는지

총12,684개 episode 기록 중 초기 부분 episode256개를 제외한 **12,428개 적격 episode**를 집계했다. 마지막에 진행 중인 episode는 이 완료 episode 합계에 포함되지 않는다. 아래 거리·시간 평균은 이벤트 수로 가중했으며, 이벤트가 없는 episode의 기록값0을 섞지 않았다.

| 지표 | 전체 적격 episode | 마지막1,000 iteration에서 종료한 적격 episode |
|---|---:|---:|
| episode 수 | 12,428 | 6,217 |
| 계측된 완료 도약 이벤트 | 4,535 | 1,435 |
| 거리12cm·시간40ms 동시 충족 | **5회** | **2회** |
| 이벤트별 충족 비율 | **0.1103%** | 0.1394% |
| 이벤트 가중 평균 도약거리 | **3.162cm** | **3.154cm** |
| 이벤트 가중 평균 공중시간 | **12.205ms** | **11.446ms** |
| 실제 도약 보상 합계 | 117.21794점 | 36.75033점 |
| 보상을 받은 episode 수 | 2,623 | 1,004 |

거리·시간 동시 충족은5개 episode에서 각각1회씩 나왔다. 원자료 행은 [qualified_training_episodes.csv](qualified_training_episodes.csv)에 보관했다. 이 중1개 episode의 도약 보상은0이었다. `qualified_events`는 거리·시간만 판정하고, 실제 보상은 직전 완전 주기의 앞다리 자격·높이·착지 step 종료·주기별 예산도 적용하므로 두 지표는 다르다. episode 합계만으로 그1회의 차단 원인을 특정하지 않았다.

동일한 마지막100 iteration의 TensorBoard `Episode_Reward/*` 평균을 비교하면 도약 성분은 **0.00023205**, 보상 성분의 순합은5.95190으로 도약 성분 비중은 약 **0.0039%**다. [성분별 기록](reward_contributions_last100.json)을 저장했다. 이 값은 같은 정규화의 성분끼리 비교한 것이며 `Train/mean_reward`와 직접 나눈 비율이 아니다. 관측된 도약 신호가 매우 드물고 작다는 근거다. 이것만으로 보상 가중치가 유일한 실패 원인이라고 확정할 수는 없다.

학습은 확률적 행동·256환경의 결과이고 플레이는 `act_inference`의 결정론적 행동·1환경 결과다. 드문 학습 성공5회가 최종 정책의 반복 도약 성능을 뜻하지 않는다. TensorBoard `Gallop/last_*`도 마지막 기록 환경의 값으로 전체환경 합계가 아니다. 전체 도약 횟수는 episode JSONL/CSV에서 합산했다.

## 속도와 좌우 다리 움직임 비교

| 플레이 지표 | 이전1m/s 정책 | 부모2m/s 정책 | 이번3m/s 정책 |
|---|---:|---:|---:|
| 실제 평균 vx | 1.01086 | 1.93505 | **2.95632m/s** |
| 완전 참조주기당 전진거리 | 0.42419 | 0.71590 | **0.94811m** |
| FL thigh 실제 주기 ROM | 36.24° | 59.25° | **91.99°** |
| FR thigh 실제 주기 ROM | 42.06° | 58.35° | **85.19°** |
| RL / RR hip 실제 주기 ROM | 9.90° / 12.11° | 9.29° / 19.21° | **7.11° / 25.75°** |
| 목표각 clipping | 14.73% | 16.46% | **16.96%** |
| 위상 보정 좌우 관절 비용 | 기존 정책에 미기록 | 0.05552 | **0.23851** |

2m/s 대비 주기 전진거리는 약32.4% 늘었다. 속도·보상·참조주기가 서로 다른 실행의 비교이므로 속도 증가나 특정 패널티의 단독 인과 효과로 해석하지 않는다. 이전 두 정책에는 새5ms 도약 tracker가 없으므로 이번 새 도약 지표를 소급 비교하지 않는다. [비교 원자료](policy_play_comparison.json).

이번 플레이의 ROM/load/stance 쌍 비용은 모두0이지만, 각 비용에는 허용폭이 있고 ROM 비용은 thigh/calf만 사용한다. **RR hip25.75°와 RL hip7.11° 차이가 남아 있으므로 한쪽 다리 움직임 편중이 해결됐다고 볼 수 없다.** 앞다리를 충분히 사용하는 조건은 통과했다. 하중 분담은 FL/FR/RL/RR=20.43/22.67/30.06/26.83%다.

학습 첫200회→마지막200회에서도 좌우 관절 비용은0.15953→0.23075로 약44.6% 증가했다. ROM/load/stance 비용은 감소했지만 관절 잔차의 좌우 차이는 함께 개선되지 않았다. clipping은 실제 관절 한계 위반 측정이 아니라 **명령한 목표각이 soft-limit에서 잘린 비율**이다.

## 전체 조건과 다음 검토 지점

마지막1,000 iteration의 적격6,217개 episode에서 속도 추종 통과96.03%, 전체 ROM100%, 앞다리 하중100%, 앞다리 ROM99.57%, 앞발 지지/스윙99.61%였다. 반면 공중기 gate 통과는0.145%, stable 통과0%, 전체 성공0%다. 6,106개 episode는 stable의 자세·생존 항목을 만족했지만 clipping만으로 실패했다. 자연 timeout 비율은98.21%였으므로 `stable=false`를 모두 낙상으로 해석하지 않는다.

이번 설정은 **처음부터3m/s 고정**이다. 48회 판단에서 상한3.0을 유지하고 승급이0회인 것은 설정대로이며, 성공률 때문에 속도가3에 못 올라간 상황이 아니다. 착지 순서는 필수 gate에서 제외했으므로 낮은 순서 정확도를 현재 실패의 원인으로 삼지 않는다.

다음 변경을 검토할 때는 실제 이륙을 유도할 보상 신호의 빈도·크기, 목표각 clipping, RR hip 편중을 함께 확인해야 한다. 현재 수집 단계에서는 기준이나 보상을 바꾸지 않았다. 새 보상 변경의 효과를 검증하려면 이후 별도 학습과 결정론적 평가가 필요하다. 이번 결과에 MuJoCo Sim2Sim·실기 검증은 포함되지 않는다.

## 보관 파일과 재생

- [요약·학습/플레이 상태](summary.json), [학습 세부 통계](training_diagnostics.json), [플레이 통계](play_analysis.json), [도약·접촉·쌍 진단](play_diagnostics.json)
- [학습 episode CSV](train_episodes.csv), [플레이 episode CSV](play_episodes.csv), [TensorBoard scalar CSV](tensorboard_scalars.csv), [판단48회](curriculum_decisions.csv)
- [기본 상태 trace](play_state_trace.csv), [도약·쌍 trace](play_leap_trace.csv), [영상 색인](videos.csv)
- [학습 곡선](training_curves.png), [조건 통과율](criterion_pass_rates.png), [도약 곡선](leap_curves.png), [좌우 쌍 보상](pair_reward_curves.png)
- [속도·자세](play_tracking.png), [발 접촉](play_contacts.png), [관절 운동](joint_motion.png), [영상 연속 프레임](play_keyframes.png)
- [학습 설정](gallop_config.json), [원본 자료 해시](evidence_manifest.json), [수집 무결성 검사](verification.json)

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt \
  --video --video-length 1500
```

위 명령은 재생용이다. 새로운 재생 결과는 기존 폴더를 덮어쓰지 않고 `runs/isaac/play/unitree_go2_gallop/<실행ID>/`에 저장한다. 학습은 사용자가 직접 진행한다.
