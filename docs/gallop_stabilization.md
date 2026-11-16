# Gallop 안정화 추가 학습 — 2026-11-02

2026-11-03 이후 추가 학습에는 [관절 범위 출력·동작 모방 설계와 명령](gallop_bounded.md)을 사용한다. 이 문서는 이전 안정화 단계의 재현 기록이다.

20,000회 부모에서 자세·앞발 지지폭·yaw 흔들림·관절 목표각 초과를 보정하는 [gallop_stabilization.json](../configs/gallop_stabilization.json) 설정이다. **사용자가 추가 2,000회 학습을 완료했고 2026-11-03 결과를 수집했다.** 최신 모델은 `20261102_161522_fe6276/model_final.pt`(계보 22,000회)이며 [결과·영상·판정](../reports/20261103_gallop_stabilization_review/README.md)을 먼저 확인한다. 이번 수집은 기존 파일 분석만 수행했다.

30초 플레이에서 roll 평균 3.83° / 앞발 지지폭 25.42cm / yaw 각속도 RMS 0.182rad/s / 안정 착지 57/59회로 개선됐다. clipping 22.48%와 뒷발 지지폭 유지율 78.13%가 남아 전체 성공은 미달이다. 학습 중 반복 성공도 0이어서 cap 2.2m/s·거리 14cm를 유지했고 `highest_confirmed_speed=null`이다. 다음 명령은 이 설정을 학습했던 원래 명령이며 자동으로 재학습하지 않는다. 아래 검증 절의 미학습·미검증 표현은 구현 당시 기록이다. 새 모델 MuJoCo는 아직 미검증이다.

## 새 보상과 제어

| 항목 | 적용 내용 |
|---|---|
| roll/pitch 안정성 | 기존 양의 진동 보상에 자세 점수를 곱한다. roll±3°·pitch±5° 안은 감점하지 않고, 밖에서는 각각6°·8° 척도로 점수를 부드럽게 낮춘다. |
| 앞·뒤 발 간격 | 각26.98cm 목표를 독립적으로 유지. Gaussian 대신 `1/(1+오차²)` 점수로 멀리 벗어나도 개선 신호가 남게 했다. 오차 정규화 척도8cm, 지지 허용폭±0.5cm, 스윙±0.75cm. |
| 지지/공중 우선순위 | 발 쌍 중 한 발이라도 지지 중이면 폭 점수 가중치1, 둘 다 공중이면0.3. 앞뒤 각각 최대2점/초. 동시 착지나 동일 순간 관절각을 강제하지 않는다. |
| yaw 감쇠 | yaw0 각도 벌점은 유지하고, 실제 몸체 z각속도에 별도 Huber 비용을 추가했다. 정규화0.3rad/s, 가중치1. |
| 목표각 한계 초과 | soft-limit으로 잘리기 **전** 목표각 초과량에 Huber 비용을 적용. 척도0.35rad, calf 비용2배, 전체 가중치2. 최종 목표각 clipping과 물리 한계는 그대로 유지한다. |
| calf 출력 | Gallop actor의 calf action만 전환 비율에 따라1.0→0.85로 부드럽게 줄인다. 동결된 일반 보행 teacher는 변경하지 않는다. |

자세 점수는 `exp(−Σ(max(|각도|−허용폭,0)/척도)²)`다. 따라서11° 기울어진 채 흔들리지 않는 동작도 만점을 받지 못한다. 음의 roll/pitch 절대각 벌점을 다시 추가한 방식은 아니다. 첫 진동 표본은 이전 자세가 없으므로 기존처럼 보상/통계에서 제외한다.

calf 축소율은 전환 비율 μ에 대해 `1−0.15μ`다. 축소한 student action과 teacher action을 다시 μ로 혼합한다. μ=0인 일반 보행은 teacher가 그대로 제어하고, μ=1인 완전한 Gallop에서만 calf가0.85배가 된다. 나머지 관절의 action scale0.5와 모터PD·토크/속도 한계는 유지한다. PPO에는 원래 sampled action을 보관하고, 이전 action 관측과 실제 제어에는 축소·혼합된 action을 쓴다.

목표각 초과 비용은 Gallop 제어 비율에 맞춰 적용하고, 초과량/실제 clipping 계측은 별도로 유지한다. 단순히 clipping 수치를 숨기거나 soft-limit을 넓히지 않았다. 발 쌍 중심선을 고정하는 벌점은 추가하지 않았고 실제 목표 방향 이동 보상을 유지했다.

## 성공률 기반 속도·도약거리

| 속도 단계 | 실제 도약거리 목표 | 공중시간 기준 |
|---|---:|---:|
| 2.2m/s | 14.00cm | 60ms |
| 2.4m/s | 15.33cm | 60ms |
| 2.6m/s | 16.67cm | 60ms |
| 2.8m/s | 18.00cm | 60ms |

처음에는2.2m/s·14cm에서 보정한다. 목표는 episode reset 때 해당 환경에 고정한다. 다른 환경의 성공으로 전역 단계가 바뀌어도 진행 중인 episode의 속도·도약 목표는 바뀌지 않는다. 특히 공중에 있는 동안 거리 기준이 바뀌지 않게 했다. reset 뒤에는 기존 정지→1m/s 일반 보행→안정 전환 과정을 거친다.

매1,000 policy step마다 조건을 확인하되, **같은 단계의 적격 episode128개 이상에서 성공률80% 이상을 두 구간 연속 확인해야**0.2m/s 승급한다. 성공률50% 미만이면 한 단계 낮추며 최소2.2m/s를 유지한다. 시간이 지났다는 이유만으로 승급하지 않는다. 조건을 만족하지 못하면 안정화 단계에 머문다.

성공 episode에는 기존 전진/횡속도·yaw각속도 추종, 패턴·공중기·ROM·앞다리 참여, 생존·clipping 조건과 함께 아래 조건을 모두 요구한다.

| 추가 조건 | 기준 |
|---|---|
| 실제 Gallop 계측 | warmup 뒤 최소5초 |
| roll / pitch RMS | 각각5° /8° 이하 |
| 앞·뒤 발 폭 | 각 쌍의 양발 지지 표본10개 이상, 그중80% 이상이24~30cm |
| 착지 안정성 | 완료된 회복10회 이상, 안정 착지80% 이상 |
| 거리·시간 동시 충족 | 도약10회 이상, 현재 단계 거리·60ms 충족 비율60% 이상 |

기존 몸체 z각속도 RMS0.3rad/s, 목표각 clipping10% 한계도 유지한다. 계측이 비어 있으면 추종·안정 판정을 통과하지 못하도록 했다. 양발 지지 표본은 한 episode 전체에서 확보하면 되므로 매 주기의 동시 착지를 요구하지 않는다.

`trained_speed_cap`은 학습 종료 당시의 현재 단계이고, `highest_confirmed_speed`는 승급 판정을 실제로 통과해 본 최고 속도다. 마지막에2.8로 승급했다고2.8의 성공도 이미 검증됐다는 뜻은 아니다. 재생에서는 저장된 단계를 복원하고 curriculum을 고정한다. `stabilization_leap_target`에 실제 episode별 거리 기준을 저장하며, 설정의 `leap_min_distance=0.18`은 최종 목표다.

## 원래 학습 명령 — 사용자 실행 완료

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261102_145730_26810f/model_final.pt \
  --walk-checkpoint runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --config configs/gallop_stabilization.json \
  --video --video-length 500 --video-interval 10000
```

actor49/critic64/관절12의 학습 tensor와 noise를 정확히 복원하고 새 Adam·iteration0으로2,000회를 추가한다. 완료 시 계보22,000회다. 새 모델·설정·source·teacher·로그·영상은 `runs/isaac/train/unitree_go2_gallop/<새실행ID>/`에 보관한다. 초기 네트워크 출력이 같아도 제어에서는 새 calf 축소와 낮은 명령을 적용하므로 기존과 동일한 주행을 보장하는 것은 아니다.

완료된 모델 재생은 다음과 같다. saved source와manifest/teacher를 함께 보관해야 한다. 이전 모델은 학습 당시 source로 재생되므로 이번 보상이 소급 적용되지 않는다. Isaac과 MuJoCo 모두 저장된 단계·거리 목표·calf 제어를 복원한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/model_final.pt \
  --video --video-length 1500
```

## 계측과 검증

`episodes.jsonl`, TensorBoard의 `Gallop/last_stabilization_*`, trace에 실제 목표 속도/거리, 앞뒤 지지 표본과 폭 충족률, 자세·착지·거리 gate, 목표각 초과량을 저장한다. `gallop_yaw_damping`, `gallop_target_excess` 보상을 추가했고 콘솔 iteration 표는 유지했다. 수집기도 새 gate와 단계별 도약 목표를 구분하게 수정했다.

- 보상·제어·episode/reset·이벤트·승급/회귀·기존모델 호환성 **102개 검사 통과**.
- 실제20,000회 부모의 모든 학습 tensor·noise/초기 출력 보존, 새 optimizer,source·teacher 해시, 폭 가동범위 등 **13항목 통과**. [검증 결과](../runs/isaac/check/unitree_go2_gallop/20261102_161410_16d00a/posture_verification.json).
- 임시49차원 bundle에서 저장 단계2.6m/s와16.67cm 복원,calf0.85 실행과초과량 연결을 MuJoCo1step으로 검사했다. 임시 모델은 삭제했고 이 검사를 새 학습 정책의 보행 성능 검증으로 취급하지 않는다.
- 현재 세션 CUDA 인식 불가. 새 Isaac 주행·본학습·안정화 성공은 미검증이다. PPO 업데이트0이며 기존 모델/로그를 보존했다.

기존 플레이의 동일 상태·action에 새 수식만 대입한 [오프라인 비교](../runs/isaac/check/unitree_go2_gallop/20261102_161410_16d00a/saved_trace_reward_review.json)에서, 기울어진 자세의 quiet 점수는0.915→0.158, 앞발 양발지지 폭 점수는약0→0.117로 바뀐다. calf 축소만 대입한 clipping은24.24→21.88%로 여전히 크다. 이 수치는 개선 방향의 학습 신호를 확인하는 자료이며, 새 정책의 주행 성능 예측이나 성공 결과가 아니다. 실제 한계 초과를 줄이려면 추가 학습이 필요하다.

학습 없이 모델 호환성·기구학만 재확인하려면:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_posture \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_145730_26810f/model_final.pt \
  --config configs/gallop_stabilization.json
```
