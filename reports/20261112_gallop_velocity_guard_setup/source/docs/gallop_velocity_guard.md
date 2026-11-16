# 실제 관절속도 제동·단일 표본 종료 추가학습

2026-11-12. 사용자 지시로 속도 추종 완화·도약 집중 설정에 실제 관절속도 기반 제동과 강화된 제약을 추가했다. 목표는 2.0m/s·공중 전진거리 50cm·최소 공중시간 0.24초 고정이며 승급·회귀는 없다. 사용자가 학습하고 이번 작업에서는 PPO·모방·optimizer 업데이트를 수행하지 않았다.

## 학습 명령

최신 실행 폴더를 확인해 도약 집중 추가학습이 누적58k로 완료된 것을 발견했다. 그 모델의 가중치를 계승하고 새 Adam으로 추가4k를 실행한다. 부모58k가54k보다 성능이 우수하다는 평가를 한 것은 아니다. 이전 부모와 보호 기준은 보존한다.

```bash
cd /home/cvr/GH/go2_animalization/gallop

./isaac.sh gallop train \
  --headless --num-envs 256 --iterations 4000 --save-interval 500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/model_final.pt \
  --config configs/gallop_2mps_50cm_velocity_guard.json \
  --video --video-length 500 --video-interval 20000
```

이미 고정 목표 부모라 `--reset-curriculum`을 넣지 않는다. 완료 시 누적62k이며 성공 보장은 없다. 새 run의 위치는 `runs/isaac/train/unitree_go2_gallop/<새 ID>/`다. 아직 이 제약으로 학습한 새 정책은 없다.

## 적용한 값

| 항목 | 이전 도약 집중 설정 | 이번 |
|---|---:|---:|
| calf 목표각 변화율 | 12rad/s | 10rad/s |
| 실제 속도 기반 목표각 제동 | 없음 | 관절별 한도의 80%부터 |
| 속도 비용 시작 | 한도의 90% | 80% |
| 속도 비용 가중치 | 6 | 18 |
| 매 제어 주기 속도 비용 | physics 시간평균 | 시간평균 + physics 최고 비용 |
| 실제 속도/적용 effort 한도 초과 | 20ms 지속 시 종료 표시 | 물리 표본 한 번으로 종료 표시 |

calf는 12.56rad/s부터 제동을 요청한다. hip/thigh는 각각 24.08rad/s부터다. 실제 공식 감시 상한은 calf15.7·hip/thigh30.1rad/s 그대로이며 기존1e-3 수치 허용차도 유지한다. `edu_violation_seconds`는0.005초로 기록하고 `edu_terminate_on_exceedance=true`가 지속시간과 무관하게 한 표본의 실패를 latch한다. 따라서 MuJoCo1ms로 검증할 때도 첫 초과 표본에서 실패 표시된다.

Isaac는5ms physics 표본에서 종료 표시를 남기지만 환경 reset은20ms control-step 경계에서 실행된다. 초과가 발생한 물리 스텝을 되돌리거나 나머지 스텝의 실제 dq를 강제로 잘라내는 구현이 아니다. 초기 자세·teacher 보행·혼합·학생 구간 전체에서 감시한다.

## 선제 제동의 범위

액터의 원래 Gaussian 표본과 PPO 저장 확률은 바꾸지 않는다. 실행 목표각만 현재 q/dq를 이용해 투영하며, 실행 목표에 해당하는 이전 행동 관측을 Isaac·MuJoCo에서 동일하게 사용한다.

기존 actuator의 PD식 `tau=25*(target-q)-0.5*dq`를 이용한다. 실제 속도가 한도의80%를 넘으면 가속 방향 토크를 억제하고 초과분에 대해 `-sign(dq)*2*(abs(dq)-0.8*limit)` Nm의 제동을 요청하는 목표각 경계를 계산한다. 요청한 목표가 이미 더 강하게 제동하고 있으면 그대로 둔다. 설정의 PD계수가 실제 actuator와 다르면 실행을 거부한다.

이 목표는 기존 관절각·목표 변화율의 가능한 구간 안에서만 선택한다. 제동에 필요한 목표가 이 구간 밖이면 가장 가까운 가능한 목표를 선택하고 `guarded_brake_infeasible`에 기록한다. 실제 모터의 torque-speed 포화도 여전히 적용된다. 모터 곡선·Kp/Kd·실제 dq·관절의 물성을 임의로 바꾸지 않는다. 외력 아래 실제 속도 상한을 보장하는 제어 인증은 아니다.

## 로그와 확인

기존 콘솔 iteration·주요 학습 통계·시간 표는 유지한다. 상세 사항은 TensorBoard/episode/trace에 저장한다.

- `guarded_brake_active`: 실제 속도에 따른 제동 요청 활성 여부.
- `guarded_brake_intervention`: 일반 각도/변화율 투영 후 추가로 목표가 바뀐 비율.
- `guarded_brake_infeasible`: 각도/변화율 범위 때문에 요청한 제동을 달성하지 못한 관절.
- `guarded_brake_position/velocity/bound`: 제동 계산에 실제 사용한 상태와 경계.
- `edu_step_velocity_peak_cost`: 한 제어 주기 내 physics 최고 속도 비용.
- `edu_bad_physics_samples`, `edu_peak_velocity_ratio`, `edu_terminated`: 실제 초과와 종료.

episode의 `guarded_brake_*` 비율은 정식 학생 계측 구간의 집계다. startup을 포함한 모든 구간은 trace와 `edu_*` 감시기로 확인한다. 부분 기록과 전 구간 집계를 혼동하지 않는다.

관절 보호·모터 제동 한계 때문에 기존 부모를 동결한 검사에서는 조기 종료가 많다. 이것은 감시기 검증이며 안전한 모션이나50cm 학습 성공의 증거가 아니다. 새 제약 아래 학습한 정책을 별도로 평가해야 한다. 기존 공식 물성 MuJoCo teacher 출발·export/DDS 미완성 문제도 남아 있으며 실배포 보류다. [검증 자료](../reports/20261112_gallop_velocity_guard_setup/README.md).
