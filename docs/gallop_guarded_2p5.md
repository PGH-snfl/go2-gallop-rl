# 모방 전 모델에서2.5m/s·발 높이·관절 제한 추가 학습

**사용자 학습 완료 후 수집 결과:** [2026-11-03 보고서·영상·비교](../reports/20261103_gallop_guarded_2p5_review/README.md). 모델 `20261103_151318_e95d61/model_final.pt`에서 실제2.49m/s·30초낙상0, 발 높이 감소와 관절 목표 제한 준수를 확인했다. 착지68.3%·앞발 지지폭71.7%·필터 개입53.5%로 전체 성공은 미달이다. 아래 명령은 완료한 설정의 재현용이며 추가 학습은 자동 실행하지 않는다. 새 모델 MuJoCo는 아직 미검증이다.

사용자 요청대로 **모방 전22,000회 모델**에서 이어서 학습한다. 부모는 `20261102_161522_fe6276/model_final.pt`, SHA256 `c2439fd522a98e5b782e5e08b749aabd01c64f50a73edb6c6ad020e6d2cdbf5a`다. 이후의 모방 모델이나24,000회 bounded 최종 모델을 출발점으로 사용하지 않는다.

actor49/critic64/관절12·scale0.5의 가중치와 noise를 그대로 복원하고 새 Adam으로 추가2,000회를 학습한다. 기존 delta 출력 의미와 calf0.85배·정지→일반 보행→갤롭 전환은 유지한다. 새 모방 학습이나 전체 관절 범위를tanh로 다시 매핑하는 단계를 넣지 않았다. 바뀐 실행 제약은 `unitree_go2_gallop_guarded_49_64_12_v4`로 구분한다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/model_final.pt \
  --config configs/gallop_guarded_2p5.json \
  --video --video-length 1500 --video-interval 10000
```

설정은 [gallop_guarded_2p5.json](../configs/gallop_guarded_2p5.json)이다. 완료하면 부모22,000회에서 갈라진 새 계보24,000회가 된다. 현재24,000회 bounded 모델에2,000회를 추가하는 명령이 아니다. 학습은 사용자가 실행하며 이번 구현·검증에서는 optimizer 갱신을 하지 않았다.

## 적용 내용

| 항목 | 설정 |
|---|---|
| 갤롭 속도 | 2.5m/s 고정, 시간/성공률에 따른 속도 승급 없음 |
| 출발 | 정지 안정화→동결 일반 보행1m/s→안정 추종 확인→갤롭 전환 |
| 앞발 과도한 높이 | 발바닥 기준 지면 위16cm 초과 시 Huber 비용 |
| 뒷발 과도한 높이 | 지면 위10cm 초과 시 별도 Huber 비용 |
| 높이 비용 | 정규화4cm, 앞/뒤 각weight3, 두 발 평균을 독립 평가 |
| 스윙 높이 참조 | 느린 기준6cm/빠른 기준10cm. 초과 패널티와 목표 높이가 충돌하지 않게 조정 |
| 시뮬레이터 관절 제한 | 원래 모델 가동범위 중앙90%를 실제 물리 joint limit으로 설정 |
| 목표각 작업 범위 | 위 물리 제한의 양 끝5%를 더 비운 범위, 원래 모델 중앙81% |
| 명령 변화 속도 | hip6rad/s, thigh/calf16rad/s를 매20ms 실행 단계에서 강제 |
| 잘린 요청 비용 | 범위·변화 속도 제한 전후 차이에 Huber 비용, scale0.25rad/weight4/calf2배 |
| 실제 관절 초과 비용 | 새 물리 제한 밖 실제 각도 초과량², scale0.03rad/weight8 |
| 지속 초과 종료 | 허용오차0.01rad 초과가60ms 연속이면 실패 종료 |
| 도약 | 현재 부모가 학습하던14cm·60ms 목표 유지 |

앞·뒤 발 높이는 평지의 발바닥 높이이며 몸통에 대한 상대 높이가 아니다. 보상은 정지·일반 보행·전환·갤롭 모두에서 계산한다. 동결 일반 보행 actor의 가중치를 바꾸지는 않으며, 그 출력도 동일한 실행 제약을 거친다. 높은 발에 비용을 주지만 공중기·앞다리 참여·ROM 조건은 유지하므로 단순히 발을 끌거나 뒷다리만 쓰는 동작을 성공으로 판정하지 않는다.

사용자는 속도를2.5로 낮추도록 요청했다. 그 값은 과거2.8/3.0 목표보다 낮지만, 이번 부모가 실제로 학습하던 명령2.2m/s보다는 높다. 요청값을2.5로 명시적으로 고정했으며 안정성 개선을 확인한 결과로 취급하지 않는다.

## 강한 관절 제한의 구현

기존에는 목표각을 soft-limit 안으로 제한했지만 물리 시뮬레이터는 더 넓은 원래 hard-limit을 사용했다. 이번 새 실행에서는 `write_joint_position_limit_to_sim()`으로 물리 제한도 이전 soft 범위로 좁힌다. 원본 USD는 수정하지 않는다. `soft_joint_pos_limit_factor=1.0`을 사용해 물리 제한을 쓴 뒤 다시90%로 이중 축소하지 않는다. 실제 backend의 적용값을 읽어 설정과 일치하는지 시작 시 확인한다.

매 step의 실행 목표는 작업 범위와 `직전 실행 목표±속도 상한×dt`의 교집합으로 제한한다. 기본 자세가 작업 범위 안에서 시작하고 joint reset은 기본 자세/속도0으로 고정한다. 따라서 reset 직후부터 목표각·명령 변화 속도를 함께 제한할 수 있다. PPO에는 원래 sampled action과 log probability를 유지하고, 제어·previous-action 관측에는 실제 제한된 목표각을 scale0.5 단위로 변환한 값을 기록한다.

물리 solver에는 수치 오차와 접촉 영향이 있으므로 실제 관절이 단 한 번도 경계 밖으로 나가지 않는다는 성능 보장은 아직 하지 않는다. 실제 위치를 강제로 잘라 계측을 숨기지 않고, 초과량·지속시간을 기록하고 비용/실패 판정에 반영한다. 새 물리 제약과 목표 생성은 Isaac와 MuJoCo에 함께 연결했다.

## 로그·성공 판정·저장

`action_clip_fraction`은 제한 **전** 요청이 새 물리 범위를 넘은 비율이다. 최종 목표가 범위 안이라는 이유로 이 수치를0으로 덮어쓰지 않는다. `guarded_intervention`은 작업 범위/명령 속도 때문에 요청이 수정된 전체 관절 표본 비율이다.

기존 추종·패턴·공중기·앞다리 참여·착지·도약·자세·발폭 조건에 다음을 추가했다. 공식 갤롭 계측5초 이상에서 실제 관절 초과 표본이 없어야 하며, 필터 개입 비율35% 이하, 앞/뒤 각각 높이 기준 초과 비율10% 이하가 필요하다. 이는 초기 학습 설계값이며 성공을 이미 확인한 기준은 아니다. 속도 목표는 조건 통과 여부와 관계없이2.5를 유지하며, `highest_confirmed_speed`만 반복 성공 시 확인된다.

콘솔의 기존 iteration/주요 리워드/손실/시간 표는 유지한다. 상세 항목은 TensorBoard `Episode_Reward/gallop_guarded_*`, `Gallop/last_guarded_*`와 `episodes.jsonl`에 저장한다. 플레이 trace에는 요청/실행 목표각·명령 변화 속도·필터 개입·실제 초과량·발 높이가 포함된다.

새 결과는 `runs/isaac/train/unitree_go2_gallop/<실행ID>/`에 저장한다. `manifest.json`에 원래/물리/목표각 관절 제한을 실제 native joint 순서와 함께 저장하며 source·teacher·model·영상도 같은 실행 폴더에 보관한다. 기존 실행 폴더는 보존한다.

완료 후 플레이:

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

## 구현 검증

- 관절 범위·속도 교집합, reset, 실제 초과 계측/종료, 앞/뒤 높이 비용, 기존 모델 재생, PPO sample 보존, MuJoCo 임시 연결 등130검사 통과.
- 설치된 Isaac의 실제 `write_joint_position_limit_to_sim()` 코드를 tensor/backend stub으로 실행해 물리 제한 설정과 이중 축소 방지를 확인했다. 실제 GPU PhysX 주행 검증과 구분한다.
- [실제22,000회 모델 복원·저장 trace·범위17항목 통과](../runs/isaac/check/unitree_go2_gallop/20261103_130239_9a58cd/guarded_verification.json). [검증 로그·소스·해시](../runs/isaac/check/unitree_go2_gallop/20261103_130239_9a58cd/implementation_validation.json).
- 현재 세션CUDA 불가로 새 Isaac 주행/2.5m/s 안정성/학습 수렴은 미검증이다. 본학습/PPO/모방 optimizer 갱신0이며 원본22,000회 모델은 보존했다. 테스트의 임시 MuJoCo 한 step은 제어 연결 검사이고 학습 정책의 Sim2Sim 성능 검증이 아니다.

학습 없이 재확인:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_guarded \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276/model_final.pt
```
