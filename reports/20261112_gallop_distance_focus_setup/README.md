# 속도 추종 완화·50cm 도약 집중 추가학습 준비

2026-11-12. [설계·학습 명령](../../docs/gallop_distance_focus.md), [새 설정](../../configs/gallop_2mps_50cm_distance_focus.json). 사용자 학습 실행 대기이며 이번 PPO·모방·optimizer 업데이트 0이다.

목표는 2m/s·50cm로 고정하고 승급은 사용하지 않는다. 속도 Gaussian 보상 가중치 3→1.5·오차 척도 0.5→0.8, 전진 보상 가중치 2→1·속도 오차 척도 25→40%, 속도 RMSE 성공 상한 0.5→0.8m/s로 조정했다. 거리 보상 가중치 8→24·부분 점수 0.75×진행도→0.5×진행도², 착지 보너스는 거리·시간 진행도에 비례하도록 바꿨다. 이륙/공중 이격 보조 점수도 올렸다. 안전 종료·모터 모델·관절 제한·출발 teacher는 유지한다.

부모는 최신54k `20261111_182833_2552ed/model_final.pt`, SHA256 `2d0de1f15c57043af564e8800116c321df2befae02a3c87e497a976d815a78bc`. 49/64관측·12행동·actor/critic/std를 그대로 복원하고 Adam은 새로 시작한다. 추가4k는 첫 확인 구간이며 50cm 달성 보장이 아니다. 이미 고정 목표이므로 `--reset-curriculum` 없이 계승한다.

구현의 새 옵션은 모두 이전 값이 기본이다. 과거 checkpoint는 저장한 소스로 계속 재생하며 기존 모델 파일과 설정을 덮어쓰지 않았다. 검사 fixture 3개의 reward term에 실제 구조와 같은 `params` dictionary를 추가했다. 수정 전 fixture 관련 오류가 있었으나 최종 검사 기록은 `final_gallop_tests.log`를 따른다.

실제 부모 복원·검증은 `checkpoint_restore_verification.json`, Isaac 비학습 연결 검사는 `isaac_check_verification.json`, 전체 준비 요약은 `verification.json`에 기록한다. 6개의 신규 검사는 짧은 도약 점수 감소·목표 근처/완전 점수 증가, 거리/시간 독립 자격, 안정 회복 뒤 착지 보상과 거리 보상의 분리, 전진 목적 완화와 정지/후진 무보상, 목표/안전 유지, 잘못된 설정 거부를 확인한다.

최종 관련 검사 222개·실제 부모 CPU 복원 21개를 통과했다. Isaac 실행 `20261112_121220_74bd6c`에서 16환경×1,000step, PPO0·정책 불변·finite rollout·실제 저장 reward weight/std 및 빈 curriculum 이벤트를 확인했다. 별도 CPU 복원 실행은 `20261112_121158_fa5be4`이며 4,096개 입력에서 actor/critic 복원 오차 0이다. 사전 해시한 원본 학습물·보호 모델 35파일을 다시 검증했다.

현재 부모를 동결한 검사는 코드 연결과 복원만 입증한다. 수정한 보상으로 정책이 학습되거나 도약이 개선됐다는 증거가 아니다. 기존 실제 속도 초과 및 MuJoCo 실패/실배포 보류는 유지한다. 원본/소스 해시·설정 차이·점수 수치는 `verification.json`, `config_diff.json`, `distance_reward_examples.json`, `source_sha256.json`을 참조한다.
