# 66k 이후: 실제 체공 시간과 접촉 발 속도 개선

부모는 완료된66k `20261117_204032_022fd2/model_final.pt`이며 SHA256은 `ed2ac0758317a294574d1d5a3d6c25e6e076472a64372ad4f5d384eb573a0afe`다. [66k 결과](gallop_66k_results.md)에서 실제 전진속도는1.999m/s지만 평균/최대 체공55.45/75ms,최대 도약14.34cm였다. 착지 취소18회 모두 접촉한 발 중심의 수평 속도0.5m/s 기준 초과였다. 전체 훈련에서는 관절 한도 종료13.56%·순간 속도비 최고2.739가 기록돼 단일 플레이의 위반0회와 구분한다.

## 새 설정

`configs/gallop_2mps_50cm_airtime.json`은 기존 landing-credit 설정에서 다음 여섯 항목만 동작을 바꾼다. 수치는 새 학습을 위한 후보이며 성능 개선이 실증된 값은 아니다.

| 항목 | 66k 설정 | 새 설정 |
|---|---:|---:|
| 실제 체공 시간 사건 보상 가중치 | 0 | 12 |
| 발 이격 보상에 실제 전족 비행 요구 | 꺼짐 | 켜짐 |
| 접촉 발 수평 속도 비용 가중치 | 0 | 2 |
| 이륙 준비 상승 보상 가중치 | 4 | 8 |
| 거리와 별개인 안정 착지 보상 가중치 | 4 | 2 |
| 학습 탐색 표준편차 목표 평균 | 0.16 | 0.10 |

탐색 표준편차는 학습 중 무작위 행동의 폭이다. 기존 관리기를 사용해500회 업데이트 동안 줄이고,관절별 상대 비율은 보존한다. 한 rollout(정책 갱신 전에 모으는 데이터 묶음) 안에서는 폭을 바꾸지 않으며 행동 표본이나 저장 확률을 수정하지 않는다. 이것이 훈련 중 순간 속도 초과의 원인을 확정하거나 초과를 예방한다는 뜻은 아니다.

## 체공 보상과 접촉 비용

체공 원점수는 `clamp(실제 체공 / 0.24초,0,1)`이다. 실제 지면 접촉→전족 비행→재접촉을 완료하고,위로 이동하며 유효하게 이륙했으며,공중 평균 전진속도가 명령의 절반 이상이고,최소20ms 체공 및 기존 앞다리 참여·안전 착지 조건을 만족해야 한다. 초기 낙하·하강하며 발만 접은 동작·정지 수직 뛰기·후진·앞다리 미참여는 이 보상을 받지 못한다.

한 참조 주기에서 최고 점수 증가분만 인정한다. 짧은 도약을 반복해 한 주기에서 여러 번 점수를 쌓지 못한다. 점수의25%는 안전 접촉 순간에,75%는120ms 안정 회복 후에 해당 품질을 곱해 지급한다. 재이륙·회복 실패는 미지급분을 취소하고,종료된 제어 스텝의 양의 사건 보상도 취소한다. 기존 거리 보상과 성공 판정은 별도로 유지한다.

발 이격 보상은 실제로 네 발이 떠 있는 유효 비행에서만 지급한다. 이전의 `0.5×비접촉 발 비율 + 0.5×최소 발높이`는 일부 발만 들어도 점수를 받을 수 있었으므로 새 설정에서는 이 경로를 제거했다.

접촉 발 수평 속도 비용은 최대 접촉 발 속도 `v`에 대해 `clamp(((v−0.25)/0.25)₊²,0,4)`를 시간 적분한다. 즉0.25m/s부터 비용을 주며0.5m/s 착지 취소 기준은 그대로다. 접촉하지 않은 발은 이 비용에서 제외한다. 시작 보행 teacher에는 적용하지 않고 기존 학생 혼합 비율을 따른다. 발 중심 속도이며 실제 접촉점 마찰력을 측정한 값은 아니다.

2m/s·50cm·최소 체공240ms·앞다리 참여·주기·관측49/64·행동12·action scale·teacher·모터 곡선·관절 각/변화율/실제 속도/토크 한도·제동·단일 위반 종료·착지 한도·120ms 회복은 유지한다. 과거 설정의 새 항목 기본값은 비활성이라 기존 보상 방식은 유지하며,과거 모델 플레이는 저장된 원래 소스를 활성화한다.

## 진단 기록

플레이 trace에는 체공 점수 합계 `flight_step_airtime_score`,접촉/회복분 `flight_step_touchdown_airtime_score`·`flight_step_recovery_airtime_score`,미지급분 `flight_pending_airtime_score`가 추가된다. RewardManager가 제어 시간을 곱하므로 사건 점수를 환경에서 제어 시간으로 나눈다. 최종 체공 보상은 가중치12와 학생 강도를 적용한다.

착지 취소 시점의 실제5ms 접촉 마스크 `flight_abort_sample_contact`를 발 속도와 함께 저장해 어느 발이 접촉 중인지 직접 확인한다. 관절 속도 최고값에는 관절 인덱스·실제 속도·해당 한도·episode 물리 시간·유한값 여부를 episode/trace에 기록한다. 인덱스는 manifest의 `official_joint_limits.joint_order`로 이름에 대응한다. 비유한 표본의 실제 값은0 자리표시자와 finite=false로 구분하며 fault는 계속 유지한다.

## 서버 실행

편집 PC에서는 CPU 검사만 수행했다. 서버 프로젝트 폴더 `~/CVR-Work/PGH/go2-gallop-rl`에서 최신 코드를 받은 뒤 기존 Isaac Python 환경을 사용한다.

```bash
git pull --ff-only origin main
```

비학습 검사:

```bash
CUDA_VISIBLE_DEVICES=0 ./isaac.sh gallop check --headless --device cuda:0 --num-envs 16 --steps 1500 --finetune-from runs/isaac/train/unitree_go2_gallop/20261117_204032_022fd2/model_final.pt --config configs/gallop_2mps_50cm_airtime.json
```

오류 없이 완료되면 사용자가 추가4,000회 학습한다. GPU0·256환경·영상 없음·완료 시 누적70k다.

```bash
CUDA_VISIBLE_DEVICES=0 ./isaac.sh gallop train --headless --device cuda:0 --num-envs 256 --iterations 4000 --save-interval 500 --finetune-from runs/isaac/train/unitree_go2_gallop/20261117_204032_022fd2/model_final.pt --config configs/gallop_2mps_50cm_airtime.json
```

새 optimizer로 시작하며 고정 목표이므로 `--reset-curriculum`은 넣지 않는다. 새 run에 저장하고66k 및 이전 모든 원본을 보존한다. 새 설정의 실제 Isaac 검사·PPO 결과·50cm 성능은 아직 확인하지 않았으며 실기 배포는 보류다.

CPU 검증은100개 단위검사와 편집 검사,실제66k의모든 actor/critic/std 텐서 복원 일치,61개 보호 설정 불변을 확인했다. 가중치 복원 검사는 production 복원 함수를49/64→512/256/128 ELU 구조의 CPU probe에 적용한 것으로,실제 RSL/Isaac 실행 검사를 대신하지 않는다. 검사 중 학습·optimizer 업데이트·physics는0이다. 근거는 `reports/20261117_airtime_setup/validation.json` 및 `parent_validation.json`이다.
