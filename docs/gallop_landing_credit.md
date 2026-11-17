# 62k 이후: 안전한 접촉과 안정 착지의 보상 분리

**2026-11-17 완료:** 아래62k→66k 추가4,000회는 사용자가 실행 완료했다. [66k 수집 결과](gallop_66k_results.md)를 우선 확인한다. 아래 명령은 완료 학습 재현용이며 다음 추가 학습 계획이 아니다.

2026-11-17 수신한 62k 결과에서 최종 플레이의 거리·착지 보상은 0이었다. 착지 후보 56회 중 안정 착지는 0회, 취소 55회, 마지막 1회는 회복 진행 중이었다. 학습 초반/후반 1,000회 구간의 안정 착지/착지 후보 비율은 15.6%→7.4%였다. 서로 다른 학습 episode 집계이므로 동일 조건 평가에서 정책이 퇴보했다고 단정하지 않는다.

## 변경

새 설정은 `configs/gallop_2mps_50cm_landing_credit.json`이다. 기존 velocity-guard 설정과 차이는 아래 네 항목뿐이다.

| 항목 | 기존 | 새 설정 |
|---|---:|---:|
| 목표 미달 도약의 부분 점수 | 0.5×(거리비×체공비)² | 0.5×거리비×체공비 |
| 안전한 첫 접촉에서 지급하는 거리 점수 비율 | 0% | 25% |
| 안정 착지 보너스에 거리·체공비를 추가로 곱하기 | 적용 | 해제 |
| 이륙 보상 가중치 | 2 | 0.5 |

거리비·체공비는 각각 실제 거리/0.5m, 실제 체공시간/0.24초를 0~1로 제한한 값이다. 목표 둘 다 충족하면 원점수 1이며, 미달이면 위 부분 점수를 쓴다. 기준 주기별 최고점의 증가분만 지급하는 예산과 거리 보상 가중치 24는 유지한다.

첫 접촉 보상은 유효한 실제 이륙, 앞다리 참여, 전진, 최소 체공 20ms, 높이·자세·회전·옆속도·충격·접촉한 발의 속도 조건을 모두 만족해야 지급한다. 25%에는 접촉 순간 품질을 곱하고, 나머지 75%는 기존 120ms 연속 안정 회복과 그 구간 최저 품질을 적용한다. 완전히 같은 안전 착지라면 두 번 나눠 지급하는 총 거리 예산은 늘어나지 않는다.

이후 재이륙하거나 회복 조건을 어기면 나머지 75%와 착지 보너스는 지급하지 않는다. 이전 제어 스텝에서 받은 안전 접촉 보상 25%는 되돌리지 않는다. 종료한 제어 스텝의 모든 양의 사건 보상은 기존 방식으로 취소한다. 최초 낙하·후진·앞다리 미참여·너무 짧은 접촉 해제는 새 접촉 보상 자격이 없다.

안정 착지 보너스는 짧은 유효 도약에도 회복 품질에 따라 지급한다. 성공 판정은 그대로여서 짧은 도약으로 목표 달성을 기록하지 않는다. **2m/s·50cm·0.24초, 관절 각도·변화율·실제 속도·토크 한도, 제동, 종료, 착지 안전 조건, 120ms 회복 시간은 그대로다.**

이전 설정의 `flight_touchdown_distance_fraction` 기본값은 0이므로 기존의 전액 회복 후 지급 방식은 유지한다. 과거 실행의 소스와 체크포인트도 수정하지 않는다.

## 착지 진단

착지 조건은 5ms 물리 스텝마다 취소 사유를 별도 집계하고, episode 종료에 따른 취소는 기존 20ms 제어 경계에서 기록한다. `flight_abort_*`에는 재이륙, 접촉 상실, 높이, 기울기, 회전 속도, 옆속도, 접촉 힘, 하강 속도, 접촉 발의 수평 속도, episode 종료가 포함된다. 여러 조건이 동시에 위반될 수 있어 사유별 합이 취소 횟수보다 클 수 있다. episode JSONL·TensorBoard에 집계된다.

플레이 NPZ에는 `flight_last_abort_reasons`와 해당 순서(`manifest.json`의 `flight_design.abort_reason_order`), 마지막 취소 시점의 힘·속도·자세·발의 월드 속도 `flight_abort_sample_*`를 저장한다. 사유별 카운터는 모든 취소를 유지하지만 같은 20ms 안에서 여러 번 취소하면 상세 표본은 마지막 것만 남긴다. 현재 제어 경계의 값은 `flight_sample_*`다. 접촉 발 속도의 최대값은 `flight_landing_peak_foot_slide`다.

거리 지급은 trace의 `flight_step_touchdown_distance_score`, `flight_step_recovery_distance_score`로 나눠 확인한다. 합계는 기존 `flight_step_confirmed_distance_score`(현재는 두 단계 지급 총합), `leap_reward_score` 경로를 쓴다. 최종 품질·가중치·종료 마스크는 기존 환경에서 적용한다. 수집기의 사건별 거리 보상도 두 단계 전체를 합산한다.

## 서버에서 확인 후 학습

현재 코드 편집용 PC에서는 Isaac 실행이나 PPO 학습을 하지 않는다. 서버 저장소에 수정본을 반영한 뒤 기존 Isaac 환경에서 실행한다. 아래 부모는 서버에서 완료된 62k 모델이며 Release 최소 복원 묶음에는 아직 포함되지 않았다. 58k로 자동 대체하지 않는다.

비학습 검사(가중치는 복원하지만 optimizer 업데이트 없음):

```bash
CUDA_VISIBLE_DEVICES=0 ./isaac.sh gallop check --headless --device cuda:0 --num-envs 16 --steps 1500 --finetune-from runs/isaac/train/unitree_go2_gallop/20261117_160234_3e2363/model_final.pt --config configs/gallop_2mps_50cm_landing_credit.json
```

사용자가 진행할 추가 4,000회 학습, 영상 없음:

```bash
CUDA_VISIBLE_DEVICES=0 ./isaac.sh gallop train --headless --device cuda:0 --num-envs 256 --iterations 4000 --save-interval 500 --finetune-from runs/isaac/train/unitree_go2_gallop/20261117_160234_3e2363/model_final.pt --config configs/gallop_2mps_50cm_landing_credit.json
```

62k의 actor/critic/std를 복원하고 새 optimizer로 추가 학습한다. 완료 시 누적 66k다. 목표 승급은 없고 `--reset-curriculum`은 필요 없다. 원본 62k와 이전 모든 run을 보존하고 새 run에 저장한다. 보상 구조 변경은 성능 개선의 보장이 아니다. 이 수정본의 실제 Isaac 검사·학습·동일 조건 비교는 아직 서버에서 실행하지 않았다.

CPU 단위검사는 시뮬레이터 없이 보상 분할·안전 접촉 차단·중복 예산·종료 취소·초기화·5ms 취소 사유 기록과 기존 보호 기능을 확인한다. 학습 환경 Python에서 다음과 같이 재현할 수 있다.

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_gallop_landing_credit tests.test_gallop_flight tests.test_gallop_distance_focus tests.test_gallop_velocity_guard tests.test_gallop_leap tests.test_gallop_leap_env
```
