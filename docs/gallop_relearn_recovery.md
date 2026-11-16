# 탐색 노이즈와 단계 평가를 수정한 추가 학습

**사용자 추가6,000회 학습 완료:** [결과·영상·플레이 종료 기록](../reports/20261105_gallop_recovery_review/README.md). 최신 모델 `20261105_140131_620578/model_final.pt`는 누적22,000회다. 일반보행1.5m/s까지 통과했고 첫 부분갤롭 단계에서 앞다리 스윙 참여 조건으로 반복 회귀했다. 아래 학습 명령은 이 완료 설정의 재현용이다.

2026-11-05 구현. 완료한 v5 일반보행 모델의 actor와 critic을 이어받아 추가 PPO 6,000회를 진행한다. 학습은 사용자가 Isaac Lab에서 실행한다. MuJoCo는 평가와 Sim2Sim 검증에 사용한다.

부모는 `runs/isaac/train/unitree_go2_gallop/20261104_144742_917f16/model_final.pt`이며 누적 16,000회, SHA256 `959457e283918bcc480e71c1d6ce026cffb63feb5a1b9e0834b78f60f09a8200`다. 부모는 보행 단계까지 학습됐고 갤롭 단계에는 아직 진입하지 못했다. [이전 결과와 병목 분석](../reports/20261105_gallop_relearn_review/README.md)을 참고한다.

## 실행

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 6000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261104_144742_917f16/model_final.pt \
  --config configs/gallop_relearn_recovery.json \
  --video --video-length 1500 --video-interval 20000
```

6,000회는 이번 추가분이며 완료하면 이 분기의 누적은 22,000회다. 새 Adam과 학습률 1e-4로 시작한다. 모방학습은 수행하지 않는다. 결과는 새 `runs/isaac/train/unitree_go2_gallop/<실행ID>/`에 저장되며 기존 모델과 결과는 보존한다.

## 변경한 내용

새 설정 [gallop_relearn_recovery.json](../configs/gallop_relearn_recovery.json)은 기존 설정에 recovery 활성화와 탐색 std 시작·끝값만 추가한다. 관측 49/64, 관절 행동 12, action scale 0.5를 유지한다.

탐색 분산을 학습 가능한 변수에서 단계별 고정값으로 바꿨다. actor와 critic은 계속 PPO로 학습한다. 행동을 사후 변경하지 않고 원래 Gaussian에서 표본과 log probability를 계산한다. std는 한 PPO 수집 구간 전체와 해당 구간의 update 동안 고정하고, 다음 수집 시작 시에만 바꾼다. entropy 계수는 0으로 설정한다.

| 요청각 초과 허용치 | 행동 단위 Gaussian std |
|---|---:|
| 0.06rad 이상 | 0.12 |
| 0.04rad | 0.10 |
| 0.02rad | 0.08 |
| 0.005~0.01rad | 0.07 |
| 0.002rad 이하 | 0.06 |

부모의 평균 std 0.2634를 초기 0.12로 낮춘다. action scale을 적용한 초기 목표각 탐색 std는 0.06rad, 약 3.44도다. 이 값은 상태 변화와 종료 확률을 낮추기 위한 새 실험 설정이며 수렴을 보장하는 값은 아니다. 단계가 회귀하면 그 단계의 std로 돌아간다. 플레이에서는 저장된 정책의 평균 행동을 사용한다.

단계는 기존 24개에서 31개로 나눴다. 부모의 숫자 단계 1은 새 숫자 단계 2로 옮기며, 이름·속도·갤롭 강도·종료 여유·지속시간이 같은 목표를 찾아 복원한다. 숫자만 복사하지 않는다. 시작 목표는 기존과 같은 1m/s 보행·0.06rad/40ms다.

보행 제약은 `0.06rad/40ms → 0.04rad/40ms → 0.02rad/40ms → 0.02rad/20ms`로 강화한다. 부분 갤롭에서도 먼저 갤롭 강도를 올려 적응한 뒤 요청각 여유를 줄이는 단계를 추가했다. 완전 갤롭 진입 후에도 0.005→0.002→1e-6rad로 나눈다. 최종 단계의 작업 범위 초과 한 제어 step 실패 종료와 요청각 초과 표본 0 조건은 유지한다.

빠른 실패만 먼저 모이는 평가를 줄이기 위해 환경마다 첫 완료 에피소드 한 개씩을 집계한다. 256환경 모두 결과를 낼 때까지 승급과 회귀를 기다린다. 먼저 실패한 환경이 반복해서 종료돼도 같은 평가 집단에서는 한 번만 반영한다. 모든 에피소드 원본은 계속 기록한다. 환경 수가 최소 표본 128개보다 적으면 완성된 평가 집단 여러 개를 누적한다. 기존 성공률 80% 두 구간 승급, 50% 미만 회귀, 세대가 다른 에피소드 제외 규칙은 유지한다.

물리 관절 범위 중앙 90%, 목표 작업 범위 중앙 81%, hip 6rad/s·thigh/calf 16rad/s, 실제 관절 초과 감시와 모터 토크·속도 모델을 유지한다. 발 간격·발 높이·안정성 보상과 최종 성공 판정도 유지한다. 최종 필터 개입 허용치는 기존 35%이므로 완전 갤롭 성공 판정만으로 필터 개입 0 또는 실기 배포 적합성을 주장하지 않는다.

## 로그와 다음 플레이

콘솔의 iteration·reward/loss·시간 표는 기존대로다. 단계 전환과 탐색 std 변경 시 짧은 상태 줄을 추가한다. `RELEARN_EXPLORATION`과 `GALLOP_CURRICULUM` 상세 기록은 `progress.jsonl`에, std와 entropy는 TensorBoard `Relearn/noise_std`, `Relearn/entropy_coef`에 저장한다. 평가 집단 크기·완료 환경 수·반복 제외 횟수도 curriculum 이벤트에 저장한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

플레이는 새 모델이 실제로 도달한 단계와 저장된 소스를 복원한다. 보행 단계 모델을 갤롭 완료 모델로 표시하지 않는다.

## 검증

Gallop 관련 145개 검사와 별도 checkpoint 저장 3개 검사가 통과했다. 새 검사에는 늦게 완료한 성공 에피소드 반영, 반복 실패 중복 집계 방지, 적은 환경 수의 평가 집단 누적, 기존 24단계 전체의 목표 일치, scalar/log std, PPO 수집 구간 내 분산 고정과 표본·log probability 일치가 포함된다.

[실제 부모 검증 26항목](../runs/isaac/check/unitree_go2_gallop/20261105_135002_4538a1/recovery_verification.json)도 통과했다. 실제 체크포인트에서 4,096개 입력의 평균 actor 출력·critic 출력 오차는 0이며 actor/critic 가중치 보존, std만 변경, 새 optimizer, 단계 목표 복원과 원본 무결성을 확인했다.

현재 세션 CUDA를 사용할 수 없어 새 Isaac 동역학·학습 수렴·갤롭 성능은 검증하지 않았다. 이번 구현에서 PPO·모방 갱신과 새 물리 주행은 수행하지 않았다. 학습 없이 실제 부모 검사를 반복하려면 다음 명령을 사용한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m scripts.validation.check_gallop_relearn_recovery
```
