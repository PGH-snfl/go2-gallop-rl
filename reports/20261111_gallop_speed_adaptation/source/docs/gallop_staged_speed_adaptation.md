# 갤롭 속도 적응 개선 이어 학습

2026-11-11. 최신48,000회 모델의1단계(2.75m/s·18cm)를 이어서 학습한다. 설정은 `configs/gallop_staged_speed_adaptation.json`이다. 원본48k 모델과 이전 설정은 보존한다.

## 학습 명령

우선2,000회 추가하며 완료 시누적50,000회다. 이번 구현에서는 실제 PPO/모방 업데이트를 실행하지 않았다.

```bash
cd /home/cvr/GH/go2_animalization/gallop

./isaac.sh gallop train \
  --headless \
  --num-envs 256 \
  --iterations 2000 \
  --save-interval 500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261111_141724_6a1661/model_final.pt \
  --config configs/gallop_staged_speed_adaptation.json
```

actor49/critic64/action12를 그대로 복원하고 새Adam으로 시작한다. 저장된1단계와 최고확인0단계를 복원하며, 평가 집단·적응 시간은 새로 시작한다. 설정의start_speed2.5는 계획의 첫 단계이며 이번 복원 단계의 실제 목표는2.75다.

## 적용한 수정

|항목|이전|새 설정|
|---|---|---|
|단계 적응 시간|60sim초,125PPO 상당|180sim초,375PPO 상당|
|행동 탐색 std|학습되는 평균약0.286|관절별 비율을 보존해 평균0.20까지500PPO 구간에 걸쳐 점진 축소|
|분산 학습|PPO가std도 학습|std는 스케줄이 관리; actor/critic은 계속 PPO|
|entropy 계수|0.01|0|
|속도 통과 조건|순간vx RMSE≤0.25m/s|유지|
|학습률|초기1e-4,adaptive/KL0.01|유지, 실제값·KL 추가 기록|
|보행·관절·모터·리워드|기존 core-first 조건/가중치|유지|

180초는 벽시계가 아닌 각 환경의sim시간이다. 시간이 지났다고 승급하지 않으며, 완전집단/80%두 구간·실패 집계·관절 보호는 유지한다.

std는 기존12개 관절의 상대 비율을 유지한다. 목표0.20은 관절별 최대값이 아니라 평균이다. 최초 수집 구간은 부모std를 그대로 사용하고500회 업데이트 후 시작하는 수집 구간부터 평균0.20을 사용한다. 부모의 평균std가이미0.20보다 낮으면 다시 늘리지 않는다.

std는 **완전한 PPO rollout 수집을 시작하기 직전**에만 바뀌며, 그 수집과 해당 PPO update 동안 고정된다. 샘플링된 행동을 사후 축소하거나 저장한 log-probability를 바꾸지 않는다. 스케줄은 stage 전환 때 높은 잡음으로 다시 시작하지 않는다.

새Adam과 함께 actor/critic을 복원하고 std만 고정한다. 모방학습·정책 평균 재초기화·관절 허용범위 확대는 없다.

## 새 진단 로그

episode/trace에 다음 항목을 추가했다. 승급 조건에는 사용하지 않는다.

- `gallop_mean_vx`, `gallop_mean_command_vx`, `gallop_mean_vx_error`: 학생 정식 구간의 평균 속도와 부족량.
- `gallop_vx_std`: 구간 내 속도 변동.
- `gallop_cycle_mean_vx_rmse`: 완전 계측된 참조 위상 주기의 평균속도 오차.
- `gallop_speed_cycles/samples`: 지표의 실제 표본 수. 계측0일 때 표시되는0을 성공으로 읽지 않는다.

초기·warmup·reset·불완전 주기를 주기 지표에서 제외한다. 여기의 주기는 독립적으로 분류한 동물 stride가 아니라 현재 기준 위상의 한 바퀴다.

TensorBoard와 `progress.jsonl`의 PPO_UPDATE.diagnostics에:

- `StagedAdapt/noise_std_mean`, `noise_ramp_fraction`, `entropy_coef`
- `StagedAdapt/learning_rate`, `learning_rate_at_floor`
- `StagedAdapt/rollout_kl_pre_update`, `rollout_kl_post_update`, `rollout_kl_samples`

KL은 저장된 전체 rollout의 Gaussian과 업데이트 전후 정책을 비교한다. **학습률 조절기가 각 미니배치에서 계산한 KL과는 다른 진단값**이다. 정책 평균 추론만 사용해 추가 난수를 소비하거나 optimizer/gradient/행동을 바꾸지 않는다. check/play에서는 PPO update가 없으므로 KL update 진단이 없다는 점을 구분한다.

콘솔은 기존 이터레이션·주요 평균값·시간 표를 유지한다. 탐색스케줄 시작/목표도달만 드물게 알리고 episode reward를 추가로 펼치지 않는다.

## 저장·재개·플레이

새run은 `runs/isaac/train/unitree_go2_gallop/<새ID>/`다. 새source에 `staged_adaptation.py`를 함께 저장하고 SHA를 검사한다. 각 체크포인트에는 기존 단계/목표/누적수와 별도로 탐색 스케줄 상태를 manifest에 기록한다.

이어 학습은 같은 설정과 원하는 체크포인트를 사용한다. 정밀optimizer/RNG 재개가 아니라 저장된actor/critic/std에서 새Adam과 새 집단을 시작한다. 중간ramp 모델이면 그 모델의std에서 다시 점진 축소하므로 과거 스케줄 위치의 정확 재개와는 다르다.

학습 후 재생은 새모델의 저장 설정/소스를 복원한다. inference에서는 std 스케줄을 다시 실행하지 않는다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새ID>/model_final.pt \
  --video --video-length 1500
```

[구현·검증 보고서](../reports/20261111_gallop_speed_adaptation/README.md). 현재 정책의 실제 추종 개선/승급·최종80cm 도달은 새 학습 후 확인해야 한다.

