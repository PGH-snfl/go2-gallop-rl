# 최신 guarded 모델에서 3.0m/s·22cm 도약으로 추가 학습

**사용자 학습 완료 결과:** [26,000회 보고서·영상·부모 비교](../reports/20261103_gallop_guarded_3mps_leap_review/README.md). `20261103_165217_19801a/model_final.pt`에서 평균 도약18.57cm·최대21.38cm/실제2.44m/s를 확인했다. 앞발 지지폭21.8%·안정 착지65.4%·필터 개입50.0%로 전체 성공0이며 승급하지 못해, 실제 명령/도약 기준은2.5m/s·14cm였다. 최종3.0m/s·22cm 달성 결과가 아니다. 아래 학습 명령은 완료한 설정의 재현용이며 새 학습은 자동 실행하지 않는다.

최종 속도 목표를 **3.0m/s**, 최소 도약 거리 목표를 **22cm**로 높였다. 이전 거리 기준14cm보다57% 크고, 최신 플레이에서 측정한 평균17.4cm보다 약26% 크다. 새 설정은 [gallop_guarded_3mps_leap.json](../configs/gallop_guarded_3mps_leap.json)이다. 이 수치는 학습 목표이며 새 정책의 달성 결과가 아니다.

최신 완료 모델 `20261103_151318_e95d61/model_final.pt`의 실제 속도는2.49m/s였지만, 안정 착지68.3%·앞발 지지폭 유지71.7%·필터 개입53.5%로 전체 성공은 미달이었다. [수집 결과](../reports/20261103_gallop_guarded_2p5_review/README.md)를 바탕으로, 안정성 조건을 통과하면 속도와 도약 목표를 함께 올리는 경로를 적용했다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261103_151318_e95d61/model_final.pt \
  --config configs/gallop_guarded_3mps_leap.json \
  --video --video-length 1500 --video-interval 10000
```

Isaac Lab/RSL-RL에서 사용자가 실행한다. 부모 SHA256은 `70ca957fb986b4c04538eea1a78e053db0f1782746ca8ba3311eb7b5578aee92`다. actor49/critic64/action12·scale0.5·기존 delta 출력과 noise를 정확히 복원하고 새 Adam/LR1e-4로 추가2,000회를 학습한다. 완료하면 guarded24,000회에서 이어진 계보26,000회다. 새 모방 단계는 없다.

## 성공률에 따른 속도·도약 거리

정지 안정화→동결 일반 보행1m/s→추종 안정 확인→갤롭 전환을 유지한다. 갤롭은2.5m/s에서 시작한다.

| 갤롭 명령 | 최소 도약 거리 목표 |
|---|---|
| 2.5m/s | 14.0cm |
| 2.6m/s | 15.6cm |
| 2.7m/s | 17.2cm |
| 2.8m/s | 18.8cm |
| 2.9m/s | 20.4cm |
| 3.0m/s | 22.0cm |

1,000 제어 step마다 판단하되, 같은 단계의 적격 episode128개 이상에서 전체 성공률80% 이상을 두 번 연속 확인해야0.1m/s 승급한다. 성공률50% 미만이면 한 단계 내리며 최저2.5m/s다. 시간만으로 올리지 않는다. 환경별 reset에서 속도와 거리 목표를 함께 고정하므로 진행 중인 비행의 목표가 바뀌지 않는다.

기존 추종·공중기·앞다리 참여·좌우 균형 조건에 더해, 충분한 갤롭 계측5초, 앞/뒤 독립 지지폭24~30cm 유지율 각각80%, 안정 착지80%, 거리·60ms 공중시간 충족 비율60%, 관절 필터 개입35% 이하와 발 높이 기준을 모두 요구한다. 착지/도약은 각각10회 이상, 양발 지지는 앞/뒤 각각10표본 이상 필요하다. 엄밀한 개별 발 착지 순서는 여전히 필수가 아니다.

**현재 부모는 이 조건을 통과하지 못했으므로 새 학습도2.5m/s에 머물 수 있다.** 2,000회 안에3.0m/s에 도달한다고 보장하지 않는다. `trained_speed_cap`은 실제 학습한 단계, `highest_confirmed_speed`는 반복 성공으로 확인한 최고 단계이며 구분해서 저장한다.

## 도약과 안정성을 함께 조정한 내용

| 항목 | 이전 설정 | 새 설정 |
|---|---|---|
| 최종 최소 도약 거리 | 14cm | 22cm |
| 도약 보상 weight | 3 | 4 |
| 기준 미달 부분 점수 상한 | 1 | 0.5; 기준 충족 시1 |
| 빠른 기준 보행 주기 | 0.32초 | 0.38초 |
| 안정 착지 보상 weight | 1 | 2 |
| 몸체 균형 비용 weight | 1.5 | 2 |
| 착지 충격·미끄러짐 비용 weight | 1 | 1.5 |
| 앞/뒤 발폭 보상 합계 weight | 4 | 6; 각 쌍3 |
| 관절 요청-실행 차이 비용 weight | 4 | 6 |

빠른 기준 주기를 늘려3.0m/s에서 참조 공중 구간을64→76ms로 확보했다. 명령 속도를 그대로 유지한다는 운동학적 계산에서는 `3.0×0.076=0.228m`로22cm 목표를 포함한다. 이는 실제 도약 가능성의 동역학 검증이 아니다. 실제 판정은5ms 물리 표본에서 마지막 발이 떨어진 몸통 위치부터 첫 착지 위치까지 이륙 방향으로 투영한 거리와 공중시간60ms 이상을 사용한다.

거리 점수는 첫 착지에서 바로 지급하지 않는다. 앞다리 참여·유효 이륙을 만족하고, 착지 후80ms 동안 접지·몸체 균형·충격·미끄러짐 기준을 유지해야 지급한다. 기준 미달 도약도 안정 회복한 경우에는 최대 절반의 부분 점수를 받는다. 넘어지거나 회복 전에 재도약하면 보류한 거리·착지 점수를 취소한다. 착지 판정 기준을 느슨하게 만들지 않았다.

yaw0 목표, roll/pitch 진동과 자세 편향 보상, 앞/뒤 독립 목표 발폭26.98cm, 진행 방향 보상은 유지한다. 발폭은 두 발의 동시 움직임을 강제하지 않는다. 좌우 중심선 패널티를 새로 넣지 않았다.

## 관절·발 높이 제한 유지

- 실제 물리 관절 범위는 원래 모델 중앙90%, 목표각은 그 양끝5%를 더 비운 범위다.
- 실행 목표 변화 속도는 hip6rad/s, thigh/calf16rad/s다.
- 앞발16cm·뒷발10cm 초과 비용, 스윙 참조6~10cm를 유지한다. 더 긴 도약을 위해 발 높이나 관절 범위를 풀지 않았다.
- 실제 관절 초과 비용과0.01rad 초과60ms 연속 시 종료를 유지한다. 수치 오차가 있는 실제 관절 위치를 강제로 잘라 숨기지 않는다.

기존 v4 제어와 Isaac/MuJoCo가 공유하는 curriculum 경로를 사용한다. 이전2.5m/s 설정과 저장된 source로 재생하는 기존 모델의 동작은 보존한다. MuJoCo는 정책 평가·Sim2Sim 검증용이며 학습은 Isaac Lab에서만 수행한다.

## 결과와 플레이

새 모델·설정·manifest·로그·영상은 `runs/isaac/train/unitree_go2_gallop/<새 실행ID>/`에 함께 저장한다. 콘솔 iteration/주요 reward·loss/시간 표를 유지하고, 상세 보상과 성공 조건은 TensorBoard·`episodes.jsonl`·`progress.jsonl`·`curriculum_summary.json`으로 확인한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

플레이는 저장된 실제 속도 단계와 거리 목표를 복원하고 curriculum을 동결한다. 아직 승급하지 못한 모델에3.0m/s를 학습했다고 표시하지 않는다.

## 비학습 검증

새 curriculum의 승급/하향·환경별 목표 고정·기존 성공 판정·안정 회복 이후 지급을 합성 표본으로 검사했다. 실제24,000회 모델의 가중치/noise 정확 복원, 빈 optimizer, 기존 물리 범위 보존과 저장 명령 재처리 등 [20항목 검증](../runs/isaac/check/unitree_go2_gallop/20261103_164053_f0b2c8/guarded_verification.json)을 통과했다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_guarded \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261103_151318_e95d61/model_final.pt \
  --config configs/gallop_guarded_3mps_leap.json \
  --trace runs/isaac/play/unitree_go2_gallop/20261103_160608_cf586d/gallop_trace.npz
```

PPO/모방 optimizer 갱신0, 새 주행0이다. 현재 세션은 CUDA를 사용할 수 없어 새 Isaac 주행·3.0m/s 추종·22cm 도약 달성은 학습 후 플레이에서 확인해야 한다.
