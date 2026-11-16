# 2.0m/s·50cm 고정 목표 준비 검증

2026-11-11. 사용자 학습 실행 대기. 이번 작업의 PPO·모방·optimizer 업데이트는 0이다. [최종 학습 명령과 설계](../../docs/gallop_fixed_2mps_50cm_edu.md), [설정](../../configs/gallop_2mps_50cm_edu.json).

승급·회귀·집단·판정 창 없는 `FixedGallopGoal`을 연결했다. 목표속도 2.0m/s, 실제 공중 전진거리 0.50m를 고정했다. 출발의 일반보행/혼합, 보상, 안전 종료는 유지한다. 이전 staged 모델은 명시적인 `--reset-curriculum`으로 가중치만 계승한다. 공개 Go2 URDF 기반 관절별 실제 속도·effort 감시와 비용, calf 목표 변화율 12rad/s, 몸체 상승분을 분리한 발 높이 비용을 추가했다. 설정·코드·검사 스냅샷은 `source/`, 파일별 해시는 `source_sha256.json`이다.

## 검증 근거

| 검사 | 결과 / 원본 |
|---|---|
| Gallop 관련 전체 검사 | 214개 통과, `gallop_checks.log` |
| 마지막 계측·collector 수정 후 집중 검사 | 25개 통과, `final_focused_checks.log` |
| 실제 50k CPU 복원 | 21개 통과, `parent50k_verification.json` / `runs/isaac/check/unitree_go2_gallop/20261111_181600_291c49` |
| 실제 46k CPU 복원 | 21개 통과, `parent46k_verification.json` / `runs/isaac/check/unitree_go2_gallop/20261111_181717_3bc0fb` |
| Isaac 50k 동결 | 16환경×1,500step, `runs/isaac/check/unitree_go2_gallop/20261111_181501_bda33b` |
| Isaac 46k 동결 | 같은 조건, `runs/isaac/check/unitree_go2_gallop/20261111_181627_710366` |
| 마지막 환경 소스 Isaac 동결 | 16환경×600step, `runs/isaac/check/unitree_go2_gallop/20261111_182047_ffc1d5` |
| MuJoCo 어댑터 | 10 control / 40 physics step, `mujoco_adapter_check.json` |

부모별 actor·critic 4,096개 입력의 복원 오차 0, 초기 std 정확 복원, 새 Adam, 관리 탐색 후 평균 가중치 보존을 확인했다. 고정 목표와 빈 결정 목록, 공식 감시기·발 높이 원시값/몸체 상승 로그 연결을 검사했다. 마지막 Isaac 실행 후 런처의 상태 표시명을 `gallop_fixed_goal`로 정리했다. 해당 기존 check의 `status.json`에는 변경 전 표시명 `gallop_success_curriculum`이 남아 있으나 manifest/실제 제어는 고정 목표다. 원본 실행 기록은 소급 변경하지 않았다.

## 부모 선택과 성능 한계

추천은 최신 50k이며 [비교 수치](parent_comparison.json)를 보관했다. 46k·50k 모두 새 과제에서 첫 16 episode 전환 완료 0, 전체 56 episode 중 전환 완료 1씩이다. 정식 학생 계측 50k 0.32초·46k 0초라 성능 우열이나 50cm 가능성을 판정할 수 없다. 기존 관절 보호의 전환 실패가 다수다. 50k는 호환성과 최신 학습 계승에 따라 선택했으며 이전 모델로 돌아가야 할 증거는 없다. 4,000회는 첫 추가학습 확인 구간이고 달성 시점 보장이 아니다.

최신 실제 학습 완료는 여전히 50k다. 과거 30k 보호 기준과 모든 이전 run은 보존했으며 새 학습은 실행하지 않았다. 10step MuJoCo 검사는 연결 검사일 뿐 새 Sim2Sim 통과가 아니다. 기존 공식 물성 근사의 walking teacher 출발 실패·하드웨어 배포 보류 판정은 그대로다. 공개 모델의 URDF 경계와 EDU 개체의 실제 펌웨어/연속 운전 한계를 구분한다.

선택 메타데이터 변경 전후는 `selection_before.json`, `selection_after.json`; 검사 요약은 `verification.json`, 부모 원본 불변 확인은 `protected_checkpoint_sha256.json`에 기록한다.
