# 주기적 command curriculum 구현 검증 — 2026-10-28

사용자 요청에 따라 보행 실패 원인 중 curriculum 갱신 누락을 수정했다. 원본 task를 직접 수정하지 않고 별도 `baseline` variant를 만들었다. 실행 방법은 [학습 문서](../../docs/curriculum_fixed_baseline.md)를 따른다. 본학습/PPO 업데이트는 수행하지 않았다.

## 완료 정책 고정 재현

경로: `runs/isaac/check/periodic_curriculum_fixed_20261028/`.

이전 공식 모델의 가중치를 고정하고 256환경·초기 무작위 episode buffer·확률적 행동·2,200step으로 실행했다. 기존 원본 재현과 seed42/환경/모델이 같다. 원본과 수정본의 초기 episode offset 배열도 일치하는지 별도로 검사한다.

| 항목 | 원본 curriculum | 수정 curriculum |
|---|---:|---:|
| 초기 offset0 환경 수 | 0 | 0 |
| 2,000step에 종료된 환경 수 | 0 | 0 |
| 2,000step까지 명령 범위 확장 | 0 | **1** |
| 최종 vx 명령 상한 | 0.1m/s | **0.2m/s** |
| PPO 업데이트 | 0 | 0 |

수정본은1,000step에 초기 부분 episode256개를 제외하고 기다렸다. 2,000step에는 완료 episode258개의 평균 추종 보상 **1.4126904 > 1.2**로 범위를 확장했다. 이후 이전 단계에서 시작된 episode55개를 제외했다. reset이 정확히 갱신 시점에 있어야 하는 의존성이 제거됐다.

정책 state_dict와 원본 모델 파일 SHA256 불변, 공식 checkout clean을 확인했다. 모델 SHA256: `36aa7fe5d7dfc448ef93e0561a4669e2a933b200cb56d46d3985c08f24f7488c`.

이 검사는 curriculum 갱신 검증이다. 가중치를 바꾸지 않았으므로 기존 정지 정책의 보행이 개선됐다는 결과가 아니다. 새 가중치의 수렴/보행 성능은 사용자 학습 후 별도 평가한다.

## 실행 경로 연결 검사

`./isaac.sh baseline check --headless --num-envs 1 --steps 16 --video`를 실행했다. 결과는 `runs/isaac/check/curriculum_fixed_launcher_20261028/`다.

- `status=completed`, `policy_updates=0`, reset0회.
- 수정 variant와 별도 체크포인트 interface가 manifest/status에 기록됨.
- train과 같은 수정 환경 클래스를 사용하며 PPO 업데이트는 차단.
- params/source/manifest/progress/console/상태 trace/MP4 저장.
- 기존 원본 check와 env.yaml 전체 비교: 차이는 **`/curriculum/lin_vel_cmd_levels` 한 곳**. 보상 및 물리 설정은 동일.

## 회귀 검사

`tests/test_velocity_curriculum.py`의4개 검사 통과:

1. 종료 시점이 흩어져도 reset 없는 주기 경계에서 확장하며, 이전 단계 episode를 다음 단계 평가에서 제외한다.
2. 짧게 끝난 실패를 실제 짧은 길이로 나눠 완전 추종처럼 과대평가하지 않는다.
3. 최소 episode 수가 부족하면 통계를 다음 주기까지 유지한다.
4. 명령 상한 도달 및 같은 step의 반복 호출이 추가 승급을 만들지 않는다.

## 구현 파일

- `isaac_task/velocity_curriculum.py`: reset 전 통계 수집·실제 길이 계수·단계 분리·주기적 판단.
- `scripts/training/launch_isaac.py`: `baseline` 명령과 별도 저장 폴더.
- `scripts/training/official_baseline.py`: variant/체크포인트 구분, 원본 감사, 새 정책 학습 및 검증/플레이.
- `isaac_task/logging.py`: 주기별 간단한 curriculum 로그와 TensorBoard 지표.
- `scripts/validation/evaluate_unitree_baseline.py`: 새 모델 인터페이스 수치 평가 지원.

원본 실행은 `./isaac.sh official ...`로 유지한다. 기존 모델·가상환경·외부 공식 소스를 수정하지 않았다. 첫 실험에서는 curriculum 효과를 분리하며, 속도 추종/발 들기 보상 변경은 적용하지 않았다.
