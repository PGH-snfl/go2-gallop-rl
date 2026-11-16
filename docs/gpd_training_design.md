# Isaac Lab 단독 Gallop 학습 설계

## 실행 구조

학습 경로는 `Isaac Lab/PhysX → IMU·관절·접촉 센서 → 공통 GPD-MPC → 관절 토크`다. RSL-RL PPO가 제어 참조를 보정한다. MuJoCo는 학습하지 않으며 내보낸 동일 정책과 동일 `config.json`을 이용한 Sim2Sim 및 배포 전 검증에만 사용한다.

로컬 Unitree Go2 USD를 사용하고, IMU offset은 MuJoCo 모델과 동일한 (-0.02557, 0, 0.04232) m다. 관절 이름을 명시적으로 FL/FR/RL/RR × hip/thigh/calf 순서로 조회한다. IMU의 wxyz, body angular velocity, 중력 바이어스를 포함한 가속도와 발의 수직 접촉력을 `LowState`에 전달한다. root ground truth는 보상·평가에만 쓰며 actor와 제어기에는 넣지 않는다.

PhysX 관절 액추에이터는 stiffness/damping=0인 토크 인터페이스다. 토크/속도 제한은 제어 설정과 맞추며, 마찰은 로봇과 바닥 모두 설정값을 적용한다. 이는 식별된 실제 EDU 모터 모델을 의미하지 않는다. USD와 기존 MPC의 URDF, MuJoCo MJCF 사이의 관성·접촉 차이는 Sim2Sim 검증 대상이다.

## 정책과 주기

| 항목 | 설정 |
|---|---|
| 학습 엔진 | Isaac Lab DirectRLEnv / PhysX GPU |
| 학습 알고리즘 | RSL-RL PPO |
| 관측 | 공통 제어기의 57차원 센서 기반 관측 |
| 행동 | 주기 ±0.04 s, 뒷발/앞발 간격 각각 ±0.035 phase, pitch ±0.06 rad, 높이 ±0.015 m |
| 물리/제어/정책 | 1 ms / 5 ms / 25 ms |
| 초기화 | 0.5초 서기; Trot 참조 및 전환 없음 |
| actor/critic | 각각 128×128 ELU; 같은 관측 사용 |
| PPO rollout | 환경당 32 step, mini-batch 4, epoch 5 |
| 초기 탐색 표준편차 | 0.15, log 파라미터화 |
| 권장 초기 환경 수 | 2~4; CPU QP를 유지하므로 대규모 벡터화 아님 |

보상은 속도 추종, 좌우 속도, 계획 접촉 일치, 자세, 기계적 파워, 토크 변화, residual 크기 및 넘어짐 페널티로 구성한다. 서기 초기화 중 residual과 양의 보상은 적용하지 않는다. 환경별 독립 reset에서 제어기·접촉 필터·위상·시계를 함께 초기화한다. 시뮬레이션 시간을 이용하므로 CPU 처리량이 느려도 제어 시간 간격이 바뀌지 않는다.

현재는 고정 평지/속도/lead 학습이다. 광범위한 domain randomization, 다양한 초기 상태, 속도 curriculum은 아직 포함하지 않았다. seed를 바꾸는 것만으로 물리 조건이 다양해지지 않는다.

## 접촉 및 성공 판정

기본 우측 lead 순서는 RL → RR → FL → FR이다. 실제 접촉력 필터로 검출한 착지 순서, 전체 실행 시간, 최소 8주기, 순서 일치율 80%, 속도 RMSE ≤0.35 m/s, 접촉 일치율 ≥70%, roll/pitch RMS ≤0.25 rad, 토크 클리핑 ≤10%를 검사한다. warmup 1.5초는 지표에서 제외한다.

기본 grounded 설정은 공중기가 필수가 아니다. `configs/paper_gallop.json`은 공중기 조건을 켠 논문형 템플릿이며 아직 안정성 검증을 통과한 설정이 아니다. 공중기 있는 Gallop을 성공했다고 하려면 주기별 공중기·실제 접촉·영상을 별도로 확인해야 한다.

학습 실행/체크포인트 저장 성공, 학습 정책의 보행 성공, MuJoCo Sim2Sim 성공, DDS 시간 제한 통과를 각각 구분한다. 학습 완료만으로 보행 또는 실물 배포 합격을 표시하지 않는다.

## 내보내기와 배포 검증

정책 네트워크와 관측 정규화를 함께 TorchScript로 내보낸다. 128개의 입력으로 RSL-RL 결정론 출력과 오차 1e-5 미만인지 검사한다. 정책과 제어 설정의 SHA256을 번들에 저장한다. `validate_mujoco.py`는 변조·혼용된 번들을 거부하고 저장된 설정을 복원한다.

1. Isaac에서 짧은 PPO 학습 및 reset 동작을 확인한다.
2. 본학습 후 Isaac에서 충분한 길이의 접촉 순서·생존 평가를 수행한다.
3. 같은 번들을 MuJoCo에서 30초 이상 실행한다.
4. 별도 C++ MuJoCo plant/DDS gateway의 지연·watchdog·착지·자세를 확인한다.
5. 실물 보정과 감독된 시험은 별도 단계다. 이 학습 코드에는 실제 로봇 연결이 없다.

## GPD 구현 출처

[Dynamic Gaits Locomotion for Quadrupedal Robots With Gait Phase Decomposition](https://doi.org/10.1109/TIE.2025.3613651)의 접촉 구간, 자세 QP와 힘 혼합을 토대로 작성한 구현이다. 저자 코드 재현을 주장하지 않는다. 기존 centroidal MPC에 world-frame 선형화, CasADi/OSQP, Coulomb cone 투영, stance impedance와 안정화용 duty 조정을 적용했다. 이전 MuJoCo/SB3 실험 결과는 역사적 진단 자료로만 남기며 현재 Isaac 학습 결과와 혼동하지 않는다.
