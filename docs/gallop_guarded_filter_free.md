# 필터 개입률 승급 제한을 푼 추가학습

2026-11-05 사용자 요청. 원래 갤롭 v4 28,000회 모델을 이어받고 전체 필터 개입률이 성공·승급을 막지 않도록 허용치를35%에서100%로 변경했다. 학습은 사용자가 Isaac Lab에서 실행한다. 이번 준비 작업에서 PPO·모방·새 물리 주행은 수행하지 않았다.

## 추가학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261103_191405_bea09f/model_final.pt \
  --config configs/gallop_guarded_filter_free.json \
  --video --video-length 1500 --video-interval 10000
```

추가2,000회 완료 시 이 분기의 누적은30,000회다. 실제 완료 모델이 생기기 전까지 현재 최종 모델은28,000회다. 결과와 영상은 새 `runs/isaac/train/unitree_go2_gallop/<실행ID>/`에 함께 저장한다.

## 이번 변경 범위

| 항목 | 기존28k 설정 | 이번 추가학습 |
|---|---:|---:|
| `guarded_max_intervention` | 0.35 | **1.0** |

[gallop_guarded_filter_free.json](../configs/gallop_guarded_filter_free.json)은 [기존 설정](../configs/gallop_guarded_stability.json)과 위 항목 하나만 다르다. 개입률은0~1이므로100%까지 허용하면 빈도 자체로 실패하지 않는다. 개입률·변화율 개입·요청과 실행 차이 크기는 계속 기록한다.

실행 목표각 필터, 물리 관절 범위 중앙90%·목표 작업 범위 중앙81%, hip6·thigh/calf16rad/s, 실제 관절 초과 감시와 모터 토크·속도 모델을 유지한다. 실제 관절이 설정 범위를0.01rad 넘어서60ms 지속하면 종료한다.

필터 보정량 비용 가중치8도 유지한다. 이번 요청은 개입률에 따른 승급 제한 해제이므로 보상이나 모터 제한을 바꾸지 않았다. 별도의 원시 요청각 물리 범위 초과율10% 조건인 `max_action_clipping=0.1`도 유지한다. 전체 개입률50.89%와 원시 요청각 초과율6.32%는 서로 다른 지표다.

앞/뒤 지지폭24~30cm 유지율80%, 안정 착지80%, 앞다리 참여·공중기·속도 추종 등의 나머지 성공 조건도 유지한다. 이전 플레이를 저장된 수치로 재판정하면 관절 필터 gate와 `stable`은 통과하지만, 앞발 폭과 착지 gate가 남아 전체 성공은 여전히0이다. 따라서 이번 변경만으로 즉시 승급한다고 보장하지 않는다.

## 이어받는 최종 모델은 어떻게 학습됐는가

현재 부모는 `20261103_191405_bea09f/model_final.pt`, SHA256 `dc129d38e78654a6559657740b7a46865830ca3cefcf06c908f49cdcef53fea7`다. 일반보행에서 다시 시작해 보관한 v5 22,000회 모델과 별도 분기다.

| 항목 | 기존28k 모델의 실제 설정 |
|---|---|
| 학습 엔진 | Isaac Lab / RSL-RL PPO |
| 마지막 학습 | 26k 모델에서 추가2,000회, 256환경·12,288,000 transition |
| 전체 계보 | 전진 일반보행4k → 첫 갤롭 추가6k/누적10k → 2k씩 반복 개선 → 28k |
| 네트워크 | actor/critic 각각512→256→128, ELU |
| 입출력 | actor49·critic64·관절 행동12, 관측 정규화 꺼짐 |
| 행동 | 기본 자세 + 0.5×행동, 필터를 거친 목표각을 물리 시뮬레이션에서 실행 |
| 제어 / 물리 주기 | 20ms / 5ms |
| PPO 수집 | 환경마다24step 수집, epoch5·mini-batch4 |
| 학습률 | 단계 시작1e-4·adaptive KL, 저장된 마지막 optimizer LR3.375e-5 |
| 탐색 | 학습 가능한 Gaussian std, 마지막 평균0.28786·entropy 계수0.01 |
| 출발 | 정지 → 동결1m/s 일반보행 → 3초 혼합 → 학생 갤롭 |
| 명령과 거리 | 2.5m/s·14cm부터, 성공에 따라3m/s·22cm까지 |
| 승급 | 적격128episode 이상, 성공률80%를 두 구간 연속 확인하면0.1m/s씩 증가 |
| 보상 | 속도·진행 방향·yaw·roll/pitch 안정·위상/관절/발 스윙·공중기/도약·앞다리 참여·착지 회복·발폭/높이·요청 보정량 |
| 개별 착지 순서 | 필수 성공 조건에서 제외, 약한 보상만 사용 |

중간20k·22k에서 방향·자세 조건을 적용했고,24k에서는 모방학습 전22k 모델로 돌아와 guarded 관절 필터와 발 높이 비용을 적용했다.26k에서 속도·도약 확대를,28k에서 안정성 관련 가중치를 강화했다. 선택한 이 계보에는 이후 폐기한 모방학습 분기를 부모로 섞지 않았다. [12개 실행의 계보 기록](../runs/isaac/check/unitree_go2_gallop/20261105_203949_97f96e/parent_lineage.json)을 보관했다.

28k의 마지막2,000회에서는 착지 보상2→4, 균형 비용2→3, 충격·미끄러짐 비용1.5→2.5, roll/pitch 안정 보상2→3, 앞/뒤 발폭 보상 합계6→7, 높이 비용3→3.5, 요청 보정량 비용6→8을 적용했다. 실제 저장 결과는2.455m/s·평균 도약17.51cm·안정 착지66.04%·앞발 폭 유지64.10%였다. 전체 성공0으로 학습 목표는2.5m/s·14cm에 머물렀고 최종3m/s·22cm에는 도달하지 않았다.

## 이번 추가학습의 초기화와 목표

actor·critic·학습된 std를28k 체크포인트에서 정확히 복원하고 새 Adam/LR1e-4로 시작한다. actor49/critic64 및 행동 스케일을 유지하며 새 모방학습, v5 단계 재학습, std 고정 관리, entropy0 설정은 사용하지 않는다. 기존 Gaussian 탐색과 entropy0.01을 이어간다.

초기 속도2.5m/s·거리14cm에서 나머지 조건을 통과하면2.6/15.6cm,2.7/17.2cm,2.8/18.8cm,2.9/20.4cm,3.0/22cm로 진행한다. 목표는 환경 reset에서 고정하며 시간만으로 올리지 않는다. 새 조건에서도 앞발 폭·착지 개선 여부를 확인해야 한다.

## 검증

- 실제28k 체크포인트 복원·저장된 요청1,500개 재필터·각도/변화율 제한·부모/asset 무결성 등 [20항목](../runs/isaac/check/unitree_go2_gallop/20261105_203949_97f96e/guarded_verification.json) 통과.
- 설정 차이가1개뿐인지, 기존 플레이의 gate 재판정과 다른 조건 유지 등 [8항목](../runs/isaac/check/unitree_go2_gallop/20261105_203949_97f96e/filter_free_verification.json) 통과.
- 실제 episode 성공 계산에서50.89%·100% 개입을 허용하고, 실제 관절·착지·폭·원시 요청각 조건 위반 시 성공·승급을 막는지 등 승급·curriculum9개 검사 통과.

새 동역학·수렴·MuJoCo Sim2Sim 성능은 검증하지 않았다. 기존 가중치·노이즈 복원과 제어·성공 판정 로직을 검사한 결과다. 학습 후 플레이는 새 실행의 체크포인트를 명시한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```
