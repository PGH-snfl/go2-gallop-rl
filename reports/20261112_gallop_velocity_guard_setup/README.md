# 실제 관절속도 제동·강한 종료 조건 준비 검증

2026-11-12 사용자 실행 대기. [최종 명령·설계](../../docs/gallop_velocity_guard.md), [설정](../../configs/gallop_2mps_50cm_velocity_guard.json). 이번 PPO·모방·optimizer 업데이트0, 실기 명령0.

calf 목표변화율10rad/s·관절별 실제속도80%부터 PD 목표 제동·속도 비용 가중치18·시간평균에 physics 최고 비용을 더하는 항목·물리 한 표본의 실제 속도/적용 effort 초과 종료 latch를 구현했다. 물리 표본5ms/제어 경계20ms를 구분하며 실제 dq를 강제 수정하지 않는다. 속도 추종 완화·도약 집중 가중치와2m/s·50cm·0.24초 고정 목표, 승급 없음은 유지한다.

제동 목표가 기존 각도/변화율 제약과 충돌하면 가능한 가장 가까운 목표를 선택하고 불가능률을 기록한다. 제동 토크 포화·접지 외력에 대한 속도 보장은 아니다. 관측/실행 이전 행동을 맞추며 PPO latent 표본과 확률은 바꾸지 않는다. Isaac와MuJoCo에 같은 guard를 연결했고 실제 PD계수 일치 여부를 검사한다.

처음 확인한54k로 구현 검증을 진행한 뒤, 최신 폴더에서 사용자 완료58k `20261112_124757_47b3b7/model_final.pt`를 발견했다. 최종 추가학습 부모는58k/SHA `b870c94b016930c17c01b0de4fcebed12a3204ba158d2bc3f0b54b35aaffb8b1`, 추가4k/완료시62k다. 58k가54k보다 성능이 우수하다는 평가를 수행하지는 않았다. 기존 추천·보호30k 및 모든 원본은 보존했다.

## 검증 자료

- 최종 Gallop 관련230검사 통과: `final_gallop_tests.log`. 새8검사는 실행 목표 제동·제약 충돌 기록·관절별 경계·원시 행동 보존·한 표본 실패 latch·physics 최고 비용·reset 분리·잘못된 설정 거부를 확인한다. 초기에는 새 test fixture의 존재하지 않는 hip_ids 참조를 고쳤으며 최종 로그를 따른다.
- 실제54k CPU 복원21개: `runs/isaac/check/unitree_go2_gallop/20261112_150617_09b29f`, actor/critic4,096개 입력 오차0·초기std 정확 복원·새Adam.
- 실제54k Isaac16환경×1,000step: `20261112_150633_1963a0`. 기록된5,214개 첫 episode 목표를 독립 NumPy 수식으로 재현해 오차0rad. 제동 활성 관절 표본91개·불가능 표본1개. 실제 추가 목표 변경은 드물었으며 기존 목표가 이미 제동 방향인 상황을 더 강한 모터로 바꾸지는 않았다. 전체41episode 중34개 공식한도 종료·전환2개·정식 학생 계측0초. 상세 `isaac_guard_verification.json`.
- 같은 신규 source를 연결한54k MuJoCo10control/40physics step 어댑터 검사: `mujoco_adapter_verification.json`. 임시 평가 bundle은 검사 후 삭제했으며 원본 모델 불변. 새 학습 정책의 Sim2Sim 성능 검증이 아니다.
- 최종58k 부모의 별도 CPU복원·Isaac연결 및 목표 재계산은 `parent58k_restore_verification.json`, `parent58k_isaac_verification.json`에 보관한다.

최종58k도 CPU21복원 검사를 통과했다. Isaac 실행 `20261112_150948_22d0aa`는16환경×1,000step/PPO0·가중치불변으로 완료했으며,5,180개 첫 episode 목표의 독립 재계산 오차는0rad다. 전체42episode 중 공식한도 종료35개였다.64개 보호 원본 파일 해시가 모두 유지됐다. 최신58k의 새 제약 아래 학습 성공을 입증한 결과는 아니다.

가중치 미수정 상태의 잦은 종료는 강화한 제약이 동작한다는 사실만 보여준다. 초과가 제거되거나 새 제어가 더 좋은 동작을 만들었다는 결론은 아니다. 모델이 새 제약을 배우는 과정에서 생존/전환/도약과 실제 속도를 다시 확인해야 한다. 모든 모터 곡선과물리 파라미터는 그대로다.

준비 요약·부모 원본 불변은 `verification.json`, `parent58k_before_hashes.json`, 소스/설정은 `source/`와 `source_sha256.json`, 선택 계획 전후는 `selection_before.json`·`selection_after.json`에 보관한다. 원래54k의 과속 및 공식 물성 teacher 출발 실패·실배포 보류 판정은 유지하며58k 전체 학습/플레이 수집은 아직 별도 수행하지 않았다.
