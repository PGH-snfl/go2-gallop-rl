# Isaac Lab 연결 검증 결과 — 2026-10-06

본학습은 사용자가 실행한다. 사용자 지시 이후 추가 학습·가중치 업데이트 없이 검증만 수행했다. 현재 실행 중인 본학습은 없다.

| 검증 | 결과 | 근거 |
|---|---|---|
| 실제 Isaac Lab / GPU PhysX 환경 생성 | 통과 | `results/isaac_environment_check/status.json` |
| 학습 입출력, 센서, 독립 reset | 2환경 × 440 step 실행, 5회 reset, nonfinite 없음 | 같은 폴더 |
| PPO 업데이트·저장 | 사용자 검증 전용 지시 전에 실시한 2환경 × 3회 시험에서 통과 | `results/isaac_train_verified/status.json` |
| RSL-RL → TorchScript 수치 비교 | 최대 절대오차 0, 통과 | `results/isaac_train_verified/export_validation.json` |
| 저장 체크포인트의 Isaac 추론 재개 | 1환경 × 64 step, 가중치 업데이트 없이 통과 | `results/isaac_checkpoint_check/status.json` |
| 자동 검사 | 12개 통과 | `results/isaac_contract_tests.log` |
| Isaac 정책 → MuJoCo 로딩·추론 | 실행 통과 | `results/gpd_isaac_transfer_check` |
| 현재 기준 제어기의 Isaac Gallop 성능 | 실패: 전체 생존/속도 등 기준 미달 | `results/isaac_environment_check/evaluation.json` |
| 시험 정책의 MuJoCo 보행 성능 | 실패 | `results/gpd_isaac_transfer_check/report.json` |

시험 정책은 수렴된 모델이 아니다. 기본 접촉 템플릿은 공중기를 요구하지 않는다. 코드 실행과 정책 이동 경로는 검증했지만, 공중기 있는 Gallop 및 Sim2Sim 보행 합격을 의미하지 않는다.

학습 설정은 PhysX와 PPO를 GPU에서 실행하고 기존 GPD-MPC QP를 CPU에서 환경별로 계산한다. 시험 처리량은 약 13~23 transition/s였으며 큰 환경 수가 선형 가속을 제공하지 않는다. 2~4환경으로 시작한다. 이전 MuJoCo/SB3 처리량으로 학습 시간을 추정하지 않는다.

다시 자동 검사만 실행하려면 `./gallop.sh -m unittest discover -s tests -v`를 사용한다. MuJoCo 검증 테스트가 포함되어 있어 이 명령은 MuJoCo 검증용 `.venv`에서 실행한다. Isaac 전용 `.isaacvenv`에는 MuJoCo가 설치되어 있지 않다.
