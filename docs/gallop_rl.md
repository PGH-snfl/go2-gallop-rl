# Gallop 스타일 추가 학습: 좌우 쌍 협응과 성공률 승급

## 최신 구현: 일반 보행 후2.8m/s Gallop 전환

속도2.8m/s·도약18cm로 조금 높이고 정지→일반 보행→안정 상태 확인→Gallop 전환을 추가했다. 부모는 완료2.5m/s 모델이다. [학습 명령·설계·검증](gallop_walk_transition.md)을 따른다. 새 설정은 `configs/gallop_2p8mps_walk_transition.json`, CPU65검사/실제 부모 복원을 확인했고 PPO0/새 PhysX 미검증이다.

## 최신 결과: 2.5m/s 실제 도약 생성

사용자가도약·균형·착지조건으로추가2,000회학습을완료했다. 최신모델은 `runs/isaac/train/unitree_go2_gallop/20261029_151901_f393b4/model_final.pt`(계보누적16,000회), [결과·영상·재생 명령](../reports/20261029_gallop_2p5mps_flight_review/README.md)을확인한다. 실제평균vx2.526m/s,도약55회/평균14.16cm·59.27ms/최대16.86cm,16cm·60ms충족4회모두80ms안정착지통과다. 회복실패24회·앞다리하중편중·목표각clip15.24%가남아전체성공은미달이다. 수집중PPO0/보상·설정보존이다.

## 최신 구현: 2.5m/s 도약·균형·안정 착지

사용자 요청으로 전진 명령을 **2.5m/s 고정**으로 낮추고, 준비 단계의 상승 동작·하중 해제 보상, 공중 구간 접촉 패널티, 몸통 좌우 균형·hip ROM 편중 패널티, 안정 이륙·착지 보상을 추가했다. 거리 목표16cm/공중시간60ms, 착지 후80ms 안정 회복을 확인한 뒤 거리 보상을 확정한다. 설정 `configs/gallop_2p5mps_flight.json`, 부모는 아래 완료3m/s 모델이다. [학습 명령·보상·검증](gallop_flight_2p5.md)을 따른다. 구현당시CPU55개와실제부모복원을검증했고,이후사용자본학습·PhysX플레이완료결과는위최신항목을따른다. 학습은계속사용자가직접실행한다.

## 이전 결과: 3m/s·도약거리 보상 학습 완료

사용자가 아래3m/s 설정으로 추가2,000회 학습을 완료했다. 당시 모델은 `runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt`(계보누적14,000회), [결과 보고서·영상](../reports/20261029_gallop_3mps_leap_review/README.md)을 확인한다. 평균vx2.956m/s/참조주기당전진0.948m/앞다리thigh85~92°이지만 **최종플레이5ms계측의 완료도약0회·도약보상0점**으로 목표는미달이다. 학습의12cm·40ms충족은5/4,535이벤트뿐이다. clipping16.96%, RR hip25.75° vsRL7.11° 차이도남는다. 이프리셋은속도승급을기다리는설정이아닌고정3m/s다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt \
  --video --video-length 1500
```

학습 영상은없고 플레이영상1개를보관했다. 아래학습명령은완료한단계의재현용이며 수집중자동학습은하지않는다. 학습은계속사용자가직접실행한다.

## 구현과 학습 설정: 실제 도약거리 보상 + 3.0m/s 고정 명령

사용자 요청으로 **실제 공중 도약거리 보상**과 `configs/gallop_3mps_leap.json`을 추가했다. 시작·최대 상한3.0, `speed_band=0.0`이므로 학습 명령은 **vx=3.0m/s 고정**, vy=yaw=0이다. 2m/s 모델에서 이어 학습하며 구현당시에는본학습을실행하지않았다. 이후사용자가완료한결과는위최신항목을따른다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_123734_c563b2/model_final.pt \
  --config configs/gallop_3mps_leap.json \
  --video --video-length 500 --video-interval 10000
```

부모의 actor47/critic62/action12 가중치와 noise를 그대로 복원하며 action scale0.5도 유지한다. Adam은 새로 만들고 새 단계iteration0부터 센다. 부모 누적12,000회는 별도 기록한다. 기존1~3m/s 참조곡선을 유지하므로3m/s에서 주기0.32초·thigh±0.75rad·calf최대0.80rad·스윙 발 높이12cm다. 한 주기 전체 목표 이동0.96m와 아래의 공중 도약거리12cm는 다른 값이다.

| 항목 | 설정 |
|---|---|
| 수평 도약거리 기준 | **0.12m** |
| 공중시간 기준 | **0.04초** |
| 도약 계측 | 각 **5ms 물리 step**에서 네 발의 접촉력과 몸통 root 위치를 읽음 |
| 거리 정의 | 첫 동시 비접촉 표본의 몸통 위치 → 첫 발 착지 표본의 몸통 위치를 **이륙 순간 전진 방향에 투영** |
| 거리 미달 보상 | `0.25 × clamp(거리/0.12,0,1) × clamp(시간/0.04,0,1)` |
| 거리·시간 모두 충족 | 착지 때 score **1.0** |
| 보상 가중치 | `leap_reward_weight=1.0` |
| 보상 상한 | 한 참조주기에서 score 증가분 합계 최대1.0 |

12cm는 `3m/s × 0.04초`에 해당하는 초기 공학적 기준이다. 현재 참조의 공동 공중 구간0.048초와3m/s를 곱하면0.144m다. **보상은 속도×시간 추정값이 아니라 실제 위치 차이로 계산**한다. 이륙 방향을 고정해 옆으로 움직이거나 공중에서 몸 방향만 바꾸는 것을 전진거리로 잘못 보상하지 않는다. 뒤로 이동하면 양의 거리 보상이 없다.

보상은 실제 착지가 관측됐을 때 부여한다. 초기 낙하, 초기1초 이전에 시작한 비행, reset으로 잘린 비행, 착지 전 종료는 보상하지 않는다. 주기 경계에서는 reward budget만 초기화하고 **진행 중인 공중 구간의 이륙 위치·시간은 유지**한다. 짧은 점프를 여러 번 반복해도 한 주기 score1.0을 넘지 않는다. 거리·시간 기준에 도달하면 추가 길이에 대한 무제한 보상도 없다.

`leap_require_forelimbs=true`로, **직전 완전한 참조주기의 앞다리 하중·ROM·지지/스윙 조건이 충족된 경우**에만 새 보상 자격이 생긴다. 기존 앞/뒤 좌우 쌍 패널티도 유지한다. 몸 높이 범위를 벗어난 착지와 해당 policy step에서 물리 종료가 판정된 경우도 새 보상을 차단한다. 이는 뒷다리만 쓰는 도약이나 넘어진 착지로 새 점수를 얻는 경향을 줄이기 위한 구성이다.

착지 score는 `step_dt`로 나눠 reward term에 전달한다. Isaac RewardManager가 다시dt를 곱하므로 정상 완료한 도약 하나의 최대 새 보상은 가중치1.0 기준1점이다. 계측5ms와 정책20ms의 차이 때문에 보너스 크기가 달라지지 않는다. CPU 검사에서 정책10ms/20ms 모두 같은1점을 확인했다.

새 보상은 계획된 공중 위상 밖에서 발생한 실제 비행도 측정한다. 기존 `gallop_flight` 가중치0.75, 40ms/60%의 과거 주기 판정, clipping·자세·앞다리 조건은 유지한다. **새 거리 보상을 전체 success에 추가 hard gate로 넣지는 않았다.** 이번 프리셋은3.0m/s 고정이므로 속도 승급을 기다리지 않는다. 과거 `flight_fraction`은20ms 주기 집계이며 새5ms 도약 이벤트와 집계 방식이 다르다.

TensorBoard의 `Episode_Reward/gallop_leap_distance`, `Gallop/last_leap_distance_mean/max`, `last_leap_events`, `last_leap_qualified_events`, `last_leap_qualified_fraction`, `last_leap_reward_points`를 확인한다. episode JSONL/CSV에 같은 계측과 물리 표본 수를 기록한다. `leap_qualified_events`는 거리·시간만 충족한 관측 이벤트 수이고, 앞다리 자격·종료·주기별 예산을 적용한 실제 보상은 `leap_reward_points`로 확인한다. 이벤트가0개이면 평균거리0은 미관측을 나타낸다.

20ms trace에는 이륙·착지 위치, 마지막 도약거리·시간, 해당step 착지 이벤트, 진행 중 거리, 누적 횟수·거리·시간·보상도 저장한다. 내부 계측은5ms이고 trace snapshot은20ms임을 구분한다. 생략된 여러 짧은 이벤트의 합계는 누적 counter와 episode 기록에 남는다. 콘솔은 주요 iteration 표와 최초 설정 한 줄을 유지한다.

계측은 원래 `scene.update`를 호출한 뒤 물리 루프의4개 update에만 연결한다. reset/render의 추가 update는 기록하지 않는다. 보상을 계산할 때 표본 수가decimation과 다르면 예외로 중단한다. 구현당시에는CPU40개 회귀와 실제 부모 모델 복원을확인했고 CUDA불가로새PhysX rollout은미검증/PPO0회였다. 당시 검증 자료는 `runs/isaac/check/unitree_go2_gallop/leap_3mps_offline_20261029_final/`에 보관한다. 이후사용자의본학습과PhysX플레이가완료됐고 새수집에서step당4표본/episode4000표본을확인했다.

GPU 머신에서 새 조건을 업데이트 없이 점검하려면:

```bash
./isaac.sh gallop check --headless --num-envs 16 --steps 1200 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_123734_c563b2/model_final.pt \
  --config configs/gallop_3mps_leap.json --video --video-length 500
```

**이전2m/s 완료 모델:** `runs/isaac/train/unitree_go2_gallop/20261029_123734_c563b2/model_final.pt`. 아래2m/s·쌍패널티 추가 학습을사용자가2,000회완료했고,당시플레이에서평균1.935m/s/주기이동0.716m를확인했다. [수집 보고서·영상·플레이 명령](../reports/20261029_gallop_2mps_pair_review/README.md). 아래의기존6,000회모델출발명령은완료한이전학습단계의재현용이다.

현재1.0m/s 전진 정책을 이식해 더 큰 다리 운동·앞뒤 다리 리듬·공중기가 있는 보행을 학습한다. 사용자 지시에 따라 **개별 발 착지 순서는 매우 느슨하게 취급**한다. 따라서 이 실험의 목표는 엄밀한 순서 분류를 통과한 순수 Gallop이 아니라 **Gallop 스타일 모션**이다.

## 현재 Gallop 모델에서 2m/s·좌우 쌍 패널티 추가 학습

2026-10-29 사용자 요청에 따라 **FL/FR와 RL/RR를 각각 묶어 움직임과 지지의 지속적인 편중을 줄이는 패널티**를 추가했다. 완료된 6,000회 모델은 원래 가중치 그대로 보관하며, 새 패널티는 아래 명령으로 수행하는 추가 학습부터 적용된다. 현재 코드 수정만으로 이전 모델의 동작이 바뀌는 것은 아니다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261028_181029_379d1c/model_final.pt \
  --config configs/gallop_2mps_pair.json \
  --video --video-length 500 --video-interval 10000
```

설정 `gallop_2mps_pair.json`은 **1.8~2.0m/s부터 시작**, 성공률로 상한을 2.25/2.50/.../3.0까지 올린다. 이 실험의 최저 상한은 2.0이다. `reference_start_speed=1.0`과 `max_speed=3.0`으로 기존 동작 참조곡선을 유지하므로, 2.0m/s의 참조 주기는 0.37초·thigh 진폭은 ±0.60rad·스윙 발 높이는 9.25cm다. `start_speed=2.0`이 참조를 저속 값으로 되돌리지 않는다. 1m/s에서 시작하는 기존 프리셋은 `configs/gallop_rl.json`이며 여기에도 좌우 쌍 패널티가 활성화되어 있다.

Actor 47 / critic 62 / action 12 구조와 action scale 0.5를 유지해 **현재 Gallop의 actor·critic·행동 noise 파라미터를 그대로 복원**한다. 달라진 속도·보상에 맞춰 Adam은 새로 만들고 새 단계 iteration은 0부터 2,000회를 센다. 부모 누적 10,000회는 `manifest.json/fine_tuning/parent_cumulative_iterations`에 별도로 기록한다. 2m/s 프리셋의 `initial_gait_weight=1.0`으로 학습된 모션 참조를 처음부터 온전히 사용한다.

## 좌우 쌍 패널티

| 항목 | 가중치 | 계산과 허용 범위 |
|---|---:|---|
| `gallop_pair_joint` | 0.6 | 각 다리 자신의 위상 참조를 뺀 관절 오차를 좌우 비교. hip은 거울 부호를 적용. 오차 차이 0.20rad까지 허용하고 초과분을 0.40rad로 정규화 |
| `gallop_pair_rom` | 0.5 | 한 주기의 실제 thigh/calf peak-to-peak 불균형 `abs(L-R)/(L+R)`가 0.25를 넘으면 패널티 |
| `gallop_pair_load` | 0.5 | 한 주기의 양의 수직 충격량 불균형이 0.30을 넘으면 패널티. 쌍 안에서 약 35:65까지 허용 |
| `gallop_pair_stance` | 0.3 | 한 주기 지지시간 비율 차이가 템플릿의 차이 + 0.15를 넘으면 패널티. 현재 허용 차이는 0.25 |

각 초과분은 제곱해 0~1 비용으로 제한하고 앞/뒤 쌍을 평균한다. 실제 reward component는 이 비용의 음수이며 위 양의 가중치를 곱한다. 수치는 초기 공학적 설정값이다.

순간적인 좌우 관절각·접촉 상태를 같게 강제하지 않는다. **갤롭의 앞발/뒷발 사이 lead/lag는 각 다리 참조에 포함**하고, 한쪽만 다르게 흔들리거나 움직이지 않는 잔차와 주기별 편중을 줄인다. 한쪽 다리만 계속 지지하고 다른 쪽은 공중에서 움직이는 경우도 하중과 지지시간 비용으로 검출한다. 모든 발이 공중인 상태의 0/0은 비용 0으로 처리하며, 전체 관절 운동과 앞다리 참여 조건은 별도로 유지한다.

관절 비용은 초기 1초를 제외하고 매 step 계산한다. 주기 비용은 warmup 이후 **완전한 주기**에서만 갱신하고 다음 주기 동안 유지한다. 초기 불완전 주기와 reset으로 잘린 주기는 제외하며 환경별로 기록을 지운다. 속도를 낮춰 패널티를 회피하지 않도록 전진 속도 gate를 곱하지 않는다. 보상 강도는 기존 gait strength를 따른다. 이번 추가는 **soft penalty**이며 속도 승급에 새 좌우 동기화 hard gate를 추가하지 않았다.

TensorBoard에는 `Episode_Reward/gallop_pair_*`와 `Gallop/last_pair_*_cost`가 저장된다. `episodes.jsonl`에는 네 비용 평균과 `pair_valid_cycles`, `action_clip_fraction`, `terminated`, `timed_out`을 추가했다. trace에는 네 비용과 **앞/뒤 쌍 각각의 ROM·하중·지지시간 차이**를 저장한다. 콘솔에는 iteration·계산량·noise·loss·평균 reward·episode 길이·시간·ETA 표를 표시하며, 세부 episode scalar는 TensorBoard에 기록한다. 실제 RSL logger를 사용해 콘솔 축약 후에도 scalar가 보존되는 것을 회귀 검사했다.

검증: CPU 회귀 22개 통과(정상 위상차, hip 거울 대칭, 한쪽 다리 정지/하중 독점/지지시간 편중, warmup/reset 격리, 실제 `GallopState`의 합성 입력 연결, 가중치 복원, 이전 source 재생 경로). 실제 6,000회 모델의 모든 policy tensor가 정확히 복원됐으며 부모 파일 SHA256도 같다. 기존 1m/s trace를 새 비용으로 다시 계산한 결과는 `runs/isaac/check/unitree_go2_gallop/pair_offline_20261029_final/verification.json`에 있다. 이는 **오프라인 검사**이며 새 PhysX rollout·2m/s 성능 검증은 아니다. 현재 세션 CUDA 사용 불가. PPO 업데이트와 본학습은 실행하지 않았다.

기존/새 모델의 `play --checkpoint`는 해당 학습 폴더 `source/`에서 SHA256이 검증된 core/env를 불러온다. 보상 코드를 수정해도 이전 모델은 자신의 원래 task로 재생한다. 학습은 항상 현재 코드를 사용하며 `--finetune-from`은 전진 45/60 이식과 Gallop 47/62 복원을 모두 지원한다.

GPU를 사용할 수 있는 머신에서 추가 학습 전 새 조건을 비학습 점검하려면:

```bash
./isaac.sh gallop check --headless --num-envs 16 --steps 1200 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261028_181029_379d1c/model_final.pt \
  --config configs/gallop_2mps_pair.json --video --video-length 500
```

이 명령은 현재 정책을 새 속도·보상 조건에서 평가하고 PPO 업데이트를 금지한다. 이 GPU 점검은 이번 세션에서 실행하지 않았다.

## 기존 전진 정책으로 시작하는 경우

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 6000 \
  --finetune-from runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --video --video-length 500 --video-interval 10000
```

설정: `configs/gallop_rl.json`. 결과는 새 `runs/isaac/train/unitree_go2_gallop/<실행ID>/`에 저장한다. 기존 전진 정책은 보존한다. 학습은 사용자가 실행하며, 이번 구현 검증에서는 PPO 업데이트를 하지 않았다.6,000회는 학습 예산이며 수렴 보장이 아니다.

지형은 원본의 작은 flat tile 대신 넓은 ground plane을 사용하고 지형 curriculum/사용하지 않는 height scanner는 끈다. 고속 이동 중 tile 경계 이탈이 보행 실패로 섞이는 것을 피한다. 기존 로봇·모터·접촉 재질과 무작위화/밀기 이벤트는 유지한다.

## 속도는 성공률로만 증가

| 항목 | 설정 |
|---|---|
| 시작 명령 범위 | 기존 프리셋 **0.8~1.0m/s**, 2m/s 추가 학습 프리셋 **1.8~2.0m/s** |
| 목표 상한 | **3.0m/s** |
| 승급 간격 | 상한을0.25m/s씩 증가 |
| 승급 조건 | 완료 episode 최소128개인 통계 구간에서 **성공률80% 이상을2구간 연속** |
| 판단 주기 |1,000 policy step마다; 그 순간 reset이 없어도 판단 |
| 유지/하향 | 기준 미달이면 유지, 성공률50% 미만이면 한 단계 하향(최저 상한은 프리셋의 `start_speed`) |
| 명령 방향 | 양의vx만,vy=yaw0,정지 비율0 |

예를 들어 상한1.25에서는1.05~1.25m/s를 학습한다. 시간/iteration이 지났다는 이유만으로 속도를 올리지 않는다. 자료가 부족하면 통계를 모아두고 기다린다. 초기 무작위 episode buffer 때문에 짧아진 첫 episode와 속도 단계 변경을 가로지른 episode는 승급 통계에서 제외한다. 그 밖의 조기 실패는 실패로 반영한다.

**3.0m/s는 Go2용 고속 주행 학습 목표**다. 명령값을 실제 속도로 간주하지 않으며 동물 종마다 다른 실제 Gallop 속도와 같다고 주장하지 않는다.

## 성공 episode의 필수 조건

단순 생존/평균 리워드만으로 승급하지 않는다. 자연 timeout까지 살아남고 다음을 함께 충족해야 한다.

- 전진 RMSE≤명령의25% 또는0.15m/s 중 큰 값. 횡속도 RMSE≤0.15m/s,yaw≤0.3rad/s.
- 좌우 교환을 허용한 지지/스윙 일치율≥60%,최소8주기.
- 공중기가 관측된 주기가60% 이상. 공중기 판정은 네 발 접촉력이 모두 off인 상태가 최소0.04초 연속 유지될 때다.
- 실제 전체 관절 가동 범위가 목표의70% 이상. 명령각의 크기만 보지 않는다.
- 아래 앞다리 필수 조건 충족.
- roll/pitch RMS≤0.25rad,목표각 soft-limit clipping 평균≤10%,자세/바닥 접촉/높이 종료 없음.

착지 순서 정확도는 기록하지만 **승급 필수 항목이 아니다**. `require_touchdown_order=false`, 순서 보상 가중치0.1이며 잘못된 순서에 대한 별도 음의 순서 보상은 없다. RL→RR→FL→FR는 약한 참고 템플릿이다. 좌우 앞발/뒷발의 교환4가지를 허용해 가장 가까운 템플릿으로 보상한다. 앞뒤 그룹 리듬과 공중기 조건은 유지한다.

## 뒷다리만 사용하는 모션 방지

| 조건 | 초기 시험값 |
|---|---|
| 앞다리 전체 수직 충격량 분담 | 전체의 **25% 이상** |
| 각 앞다리 분담 | 각각 전체의 **8% 이상** |
| 뒷다리 전체 분담 | 전체의25% 이상 |
| 앞다리 ROM | 목표 대비 평균70% 이상,좌우 각각 최소50% |
| 앞발별 지지·스윙 참여 | 각 앞발이 주기의70% 이상에서 아래 조건 충족 |
| 한 주기의 앞발 참여 | 지지≥0.06초 + 발 높이≥3.5cm인 스윙≥0.04초 |

수직 충격량은 policy step의 접촉력 표본으로 `Σ max(Fz,0)·dt`를 근사한다. 고해상도 접촉 충격량 계측과 동일하지 않다. 앞발을 바닥에 계속 붙여 끌면 스윙 참여 조건에서,몸이 뜰 때 앞발도 따라 올라갈 뿐 관절을 거의 쓰지 않으면 ROM에서,앞다리를 공중에서만 흔들면 하중 분담에서 실패한다. 한쪽 앞다리만 쓰는 것도 통과하지 못한다.

앞다리에 뒷다리와 같은 추진력을 강제하지 않는다. 두 앞다리의 하중·관절 운동·지지/회수를 확인한다. 위 수치는 공학적 초기 시험값이며 `GallopSettings`/JSON에서 조정할 수 있다.

## 큰 관절 운동과 보상

- 주기 목표:0.42초→0.32초(현재 명령 속도에 따라).
- thigh 목표 진폭:±0.45→±0.75rad. 고속 끝점의 명목 peak-to-peak는1.5rad(약86도).
- calf 스윙 굽힘:최대0.45→0.8rad. 발 높이 목표6.5→12cm.
- Action scale:0.25→0.5. **모터 토크/속도 한계는 올리지 않고**, 목표 관절각을 기존 soft joint limits로 제한한다.
- 속도 추종·접촉 패턴·스윙 높이·위상 관절 참조·실제 ROM·공중기·앞다리 참여를 보상한다.
- 기존0.5초 발 공중시간 항/좌우 공중시간 동일화/정적 관절 자세 항은 교체한다. roll은 억제하고 제한된 주기 pitch를 허용한다.

위 숫자는 목표 참조다. 실제 관절 움직임은 학습 후 trace로 확인한다. 보상 가중치와 목표 진폭은 첫500iteration에 걸쳐 부드럽게 켜지만, **속도 승급은 여전히 실제 성공률만 사용**한다. 가중치가 충분히 켜지기 전에는 승급 성공으로 세지 않는다. 각 episode는 처음부터 이 모션을 목표로 하며 Trot→Gallop 전환 기능을 학습하는 단계는 아니다.

## 기존 정책 이식

Actor45→47,critic60→62에 위상sin/cos2개를 뒤에 붙이고 출력12개는 유지한다. 새 위상 입력 열은0으로 시작한다. Action scale 변경에 맞춰 actor 마지막 출력/표준편차를0.5배,이전 action 입력 열을2배 보정한다. 따라서 제한각 clipping 전의 초기 물리 목표각/행동 분포를 보존한다. 이 불변성과 critic 출력은 회귀 검사로 확인했다.

입력 행렬과 목적 함수가 바뀌므로 **Adam은 새로 초기화**한다. 이전 동일 구조의 속도 확장과 다른 처리다. 부모 누적4,000iteration은 `manifest.json/migration`에 기록하고, 새 학습은0부터 센다.

## 로그와 결과

기존 iteration/reward/시간/ETA 표를 유지한다. 속도 판단 때만 짧은 `GALLOP_CURRICULUM` 한 줄이 나온다. 자세한 실패 원인은 `episodes.jsonl`, `curriculum_summary.json`,TensorBoard의 `Gallop/*`에 남는다.

`front_load/front_motion/front_cycle`은 앞다리 조건이고 `pattern/flight/motion/tracking/stable/mature`도 구분해서 기록한다. 보상/몸체 상태가 좋아 보여도 어느 필수 항목이 부족해 승급하지 못했는지 확인할 수 있다.

## 학습 후 플레이

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 --video --video-length 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/실행ID/model_final.pt
```

기본 플레이 명령은 실제 학습이 도달한 `trained_speed_cap`을 읽는다. 상한3.0을 목표로 설정했다는 이유로 플레이에3.0을 강제하지 않는다. 특정 양의 속도는 `--command 1.5 0 0`처럼 지정할 수 있다. 영상·`gallop_trace.npz`·episode/설정/출처를 새 play 폴더에 함께 저장한다.

Trace는 실제/목표 접촉,위상,관절각/참조각,발 높이/참조높이,속도와 reset을 포함한다. `valid`는 첫 연속 구간만 포함하고,reset 행은 자동 reset 이전 terminal snapshot으로 대체했다. 이후 구간을 회복 성공으로 세지 않는다.

이 모델의 인터페이스는 `unitree_go2_gallop_47_62_12_v1`이다. 기존 GPD57/5 및 기본 전진45/60 정책과 혼용하지 않는다. MuJoCo는 이후 같은 위상·관측·action 변환을 구현해 Sim2Sim 검증에만 사용하며, 현재 그 검증이나 실물 배포 성공을 주장하지 않는다.
