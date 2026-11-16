# 관절 범위 안에서 출력하는 Gallop 추가 학습 — 2026-11-03

사용자가 이 설정으로24,000회 모델까지 학습을 완료했다. [최신 수집 결과·모션 변화 조사](../reports/20261103_gallop_bounded_review/README.md)를 먼저 읽는다. 목표각 제한은 작동했지만 빠른 명령·포화·실제 soft 범위 이탈이 남았다. [추종 가능한 명령 재학습 검토안](../reports/20261103_gallop_bounded_review/retraining_design.md)은 아직 실행 설정에 적용하지 않았다. 아래 명령은 완료된 기존 bounded 실험의 재현용이다.

최신 **22,000회 모델**의 동작을 새 출력 구조에 옮긴 뒤 PPO로 추가 학습한다. [설정](../configs/gallop_bounded.json), [학습 실행기](../scripts/training/gallop_baseline.py), [모방 단계](../isaac_task/gallop_adaptation.py). 기존 체크포인트·로그는 보존한다. 학습은 사용자가 실행하며, 구현 검증 중에는 PPO와 모방 학습 optimizer를 실행하지 않았다.

## 실행

**2026-11-03 저장 오류 수정:** 사용자 실행 `20261103_100059_9500d2`는 모방2단계/5,670회 supervised 갱신을 완료하고 목표각 RMSE0.04781rad/p95 0.08721rad로 오차 기준을 통과했다. `model_adapted.pt` 파일은 정상 저장됐지만, 직후 RSL-RL의 미초기화 `logger_type` 참조로 종료됐다. student 주행 보존 검사와 PPO는 아직 시작하지 않았다. `LoggedRunner` 생성 시 로깅 종류를 초기화하도록 수정했고 실제 후보 체크포인트 재저장·TensorBoard 초기화 전후 등 관련40검사를 통과했다. [수정 검증](../runs/isaac/check/unitree_go2_gallop/20261103_100813_9df8e2/prelearning_save_verification.json). 아래 명령을 다시 실행하면 새 폴더에서 모방부터 재진행한다. 실패한 원본 실행과 후보 파일은 보존했다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/model_final.pt \
  --config configs/gallop_bounded.json \
  --distill-steps 1500 --distill-epochs 30 --distill-rounds 2 \
  --adapt-check-steps 1500 \
  --video --video-length 1500 --video-interval 10000
```

이 명령은 모방과 주행 확인을 먼저 수행하고 통과하면 **PPO 2,000 iteration**을 추가한다. 모방 optimizer step은 PPO iteration과 별도로 센다. PPO 완료 시 계보는 24,000회다. 처음부터 무작위 정책을 학습하는 명령이 아니다.

## 실행 순서와 동작 보존

1. 부모 actor·critic·noise를 복원하고, 부모 actor 사본을 동결한다. 관측은 actor49/critic64, 관절 출력은12개다. 부모 SHA256은 `c2439fd522a98e5b782e5e08b749aabd01c64f50a73edb6c6ad020e6d2cdbf5a`다.
2. Isaac Lab에서 동결 부모를 1,500 step(30초) 실행한다. 이때 부모의 기존 calf0.85 변환, 혼합 후 clipping, clipping 전 previous-action 관측을 그대로 사용한다. 그 관측과 실제 실행 목표각에 맞는 새 student 목표각을 저장한다. 전환 중 새 구조로 정확히 표현할 수 없는 목표는 해당 관절 범위에 투영한다.
3. 새 actor가 해당 목표각을 모방하도록30 epoch 학습한다. 관절 경계의 목표도 유한한 역변환으로 학습하도록 `atanh` 입력을±0.999로 제한한다. 감독 손실은 latent Huber와 범위로 정규화한 목표각 오차다. critic과 동결 teacher는 이 단계에서 갱신하지 않는다.
4. 새 student를 다시30초 실행하면서, student가 방문한 관측에 부모 actor로 목표각 라벨을 붙인다. 기존 데이터와 합쳐30 epoch 더 모방한다. 출력 변경으로 previous-action 관측 분포가 달라지는 점을 반영하는 단계다. `--distill-rounds 2`는 부모 실행 데이터 학습1회와 student 상태 라벨 추가1회를 뜻한다.
5. 환경 ID의20%를 학습에서 제외해 검증한다. 인접 프레임을 무작위 분할하지 않는다. 검증 목표각 RMSE≤0.10rad, 절대오차95백분위≤0.25rad를 요구한다. 이는 초기 적응 검사 기준이며 실기 허용오차 인증값은 아니다.
6. 실제 새 출력·새 관측으로 결정론적 student를30초 확인한다. 환경의80% 이상이 Gallop 단계에 도달하고, 낙상을 경험한 환경은20% 이하여야 한다. 유한한 출력과 관절 범위도 확인한다. 부모/student 각각 최소 환경 수의50%에 해당하는5초 이상 Gallop 계측 episode가 필요하다. 패턴·공중기·관절운동·앞다리 하중/운동/주기 참여의 동시 통과율은 `max(20%, 부모 통과율−15%p)` 이상을 요구한다. 이 초기 보존 검사는 전체 성공 판정과 구분한다.
7. 검사를 통과하면 환경·episode 통계·curriculum을 초기화하고, 새 Adam으로 PPO2,000회를 시작한다. latent noise 표준편차는 새 단위에 맞춰0.1, PPO 학습률은1e-4다.

모방 오차나 주행 보존 검사에 실패하면 **PPO를 시작하지 않고 종료**한다. `status.json`, `console.log`, `adaptation/adaptation.json`에서 실패 단계와 실제 오차를 확인한다. 이전 모델로 자동 후퇴하거나 검사를 생략하지 않는다. 전체 모방·주행 과정은 사용자가 위 명령을 실행할 때 진행되며, 아직 새 정책의 주행 개선을 확인한 상태는 아니다.

## 관절 출력과 보상

관절마다 학습 USD의 기존 soft-limit `[lower, upper]`를 사용한다.

```text
q_student = (lower + upper)/2 + (upper − lower)/2 × tanh(latent)
```

soft-limit은 기존처럼 물리 범위의 중앙90%다. 범위를 넓히지 않았다. 정지 단계는 기본 자세, 일반 보행은 기존 동결45입력 teacher의 제한된 목표각을 사용하고, 전환 때는 범위 안의 teacher/student 목표각을 혼합한다. 새 latent에는 calf0.85배를 다시 적용하지 않는다. 기존 축소 결과가 모방 목표각에 이미 포함된다.

PPO에는 원래 Gaussian latent와 그 log probability를 저장한다. 관절 목표각이나 `tanh` 결과로 PPO sample을 덮어쓰지 않는다. 물리 제어와 previous-action 관측에는 `(실행 목표각−기본 자세)/0.5`를 기록한다. actor의 출력 의미가 바뀌므로 interface를 `unitree_go2_gallop_bounded_49_64_12_v3`로 구분하고, 이전 모델 재생은 저장된 이전 source를 사용한다.

새 비용은 관절 범위의 양 끝5% 안으로 들어갈수록 커지는 비용(calf2배, 전체weight1)과 목표각 변화율 Huber 비용(scale10rad/s,weight0.05)이다. 전환의 student 제어 비율에 따라 적용한다. 관절 운동 참조도 같은5% 여유를 두어 한계점으로 유도하지 않게 한다. reset 직후 변화율의 첫 표본은 제외한다.

기존 자세·발폭·앞다리 참여·느슨한 착지순서·안정 회복 조건과 **2.2m/s·14cm→성공률에 따른2.8m/s·18cm** 승급은 유지한다. 기존 clipping gate도 유지하며, 공식 Gallop 구간의 관절 명령 중 끝5% 영역에 머무는 비율이 전체 관절 표본 평균10% 이하여야 `bounded_limits`를 통과한다. 전체 성공과 속도 승급에 이 조건도 반영한다.

출력 구조상 clipping은 거의0이 되므로 clipping 감소만으로 성공을 주장하지 않는다. 목표각 한계 근접 비율, 실제 관절 추종 RMSE, 목표각 변화율과 토크 포화율을 함께 저장한다. 목표각이 범위 안에 있어도 실제 관절 추종과 안정적인 주행까지 보장되는 것은 아니다.

## 저장 위치와 로그

실행 시작 시 `runs/isaac/train/unitree_go2_gallop/<새 실행ID>/`를 표시한다.

| 파일 | 내용 |
|---|---|
| `source/gallop_teacher.pt` | 동결된22,000회 부모 사본과 manifest 해시 |
| `source/walking_teacher.pt` | 기존 일반 보행 teacher |
| `adaptation/teacher_dataset.pt` | 부모 관측·목표각·환경별 검증 마스크 |
| `adaptation/student_dataset_2.pt` | student가 방문한 상태와 부모 라벨 |
| `adaptation/*episodes*.jsonl` | 부모 실행, student 라벨 수집, 최종 보존 검사 각각의 episode |
| `adaptation/adaptation.json` | 모방 오차·업데이트 수·주행 보존·실패 원인 |
| `model_adapted.pt` | 모방 후 후보. PPO 전 모델이며 주행 보존 실패 시에도 남을 수 있음 |
| `model_*.pt`, `model_final.pt` | PPO 중간·최종 체크포인트 |
| `events.out.tfevents.*` | TensorBoard 리워드·새 관절 진단 |
| `episodes.jsonl` | PPO episode만 집계한 성공 판정·진단 |
| `videos/gallop-train-step-*.mp4` | PPO 단계부터 녹화한30초 클립 |

모방 단계에는 수집 진행률·epoch·목표각 오차를 표시한다. PPO 단계는 기존 iteration/주요 리워드/손실/시간 표를 유지한다. 주요 새 TensorBoard 항목은 `Episode_Reward/gallop_bounded_near_limit`, `Episode_Reward/gallop_bounded_target_rate`, `Gallop/last_bounded_near_limit_fraction`, `Gallop/last_bounded_tracking_rmse`, `Gallop/last_bounded_torque_saturation`이다. 마지막 항목의−1은 토크 계측 미제공을 뜻하며0으로 해석하지 않는다.

학습 완료 후 재생:

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

새 모델의 MuJoCo 검증 진입점도 같은 출력 변환을 지원한다. 실제 Sim2Sim 주행 검증은 새 모델 학습 후 별도로 진행한다.

## 구현 검증

- 관절 범위·단계 혼합·PPO latent/log probability 보존·비용·reset·성공 gate·기존 동작 회귀·MuJoCo 연결 등 **117검사 통과**.
- [실제 부모 tensor 복원·형상13항목 통과](../runs/isaac/check/unitree_go2_gallop/20261103_095153_71c45b/posture_verification.json).
- [실제 부모 trace 기반 범위·라벨·source15항목 통과](../runs/isaac/check/unitree_go2_gallop/20261103_095152_35e12c/bounded_verification.json).1,063표본의 수식상 목표각 재구성 오차는 최대0.00228rad/RMSE0.000526rad다. 원래 목표의 한계 근접 비율은26.47%로 여전히 크며, 재구성 가능성을 새 정책의 학습 성공으로 해석하지 않는다.
- [검증 로그·소스·해시](../runs/isaac/check/unitree_go2_gallop/20261103_095152_35e12c/implementation_validation.json). 본학습0/PPO0/모방 optimizer 갱신0/기존 모델 보존. 현재 세션에서 CUDA를 사용할 수 없어 새 Isaac 주행은 미검증이다. 테스트의 임시 MuJoCo 단일 step은 소프트웨어 연결 검사이며 새 정책의 Sim2Sim 성능 검증이 아니다.

학습 없이 재확인하려면:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_bounded \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/model_final.pt
```
