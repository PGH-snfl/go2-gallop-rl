# 일반 보행 출발 후 2.8m/s Gallop 전환 — 2026-11-02

이후 추가한 yaw0·roll/pitch 진동·어깨 기준 발 간격·목표 방향 보상은 [최신 설정과 추가 학습 명령](gallop_heading_motion.md)을 따른다. 아래는 그 부모가 되는 보행 전환 설정과 완료 결과다.

속도 목표를2.5→**2.8m/s**, 실제 도약거리 목표를16→**18cm**로 올린다. 정지 상태에서 곧바로 갤롭 명령을 주던 방식을 **정지 유지→일반 보행→안정 상태 확인→부드러운 Gallop 전환**으로 바꿨다. 설정은 [gallop_2p8mps_walk_transition.json](../configs/gallop_2p8mps_walk_transition.json)이다.

## 완료 결과와 MuJoCo 검증

사용자가 추가2,000회 학습을 완료했다. 최신 체크포인트는 `runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt`(계보 누적18,000회)다. 사용자 Isaac 플레이는8.34초에Gallop 전환을 완료했고, 공식계측도약63회/평균18.53cm·66.03ms/거리·시간충족47회다. 추종gate와목표각clipping21.02%로전체success는0이다.

MuJoCo nominal 평지30초1회에서는낙상없이끝났으나혼합65.76%에서진행이멈췄다. 마지막10초 명령2.18363m/s에실제평균3.09026m/s로속도오차0.25m/s기준을계속벗어났다. 공식Gallop계측구간에진입하지않아원본episode의0값과tracking1/stable1은유효성능판정으로사용하지않는다. [보고서·영상·한계·독립접촉분석](../reports/20261102_gallop_mujoco_review/README.md).

정책가중치·상태기기준·학습설정을보존하고PPO업데이트없이검사했다. 재실행은아래와같다. 기본값은관측노이즈/무작위화없는단일nominal조건이다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/go2-gallop-mujoco-matplotlib MUJOCO_GL=egl \
.venv/bin/python -m scripts.validation.sim2sim_gallop \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --isaac-play runs/isaac/play/unitree_go2_gallop/20261102_140231_b1e215 \
  --seconds 30 --video
```

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_151901_f393b4/model_final.pt \
  --walk-checkpoint runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --config configs/gallop_2p8mps_walk_transition.json \
  --video --video-length 500 --video-interval 10000
```

학습은 사용자가 직접 실행한다. 위 명령은 이번 완료 학습을 처음 시작할 때 사용한 명령이다. Gallop 부모는 실제 도약이 확인된2.5m/s 완료 정책(계보 누적16,000회)이다. actor47/critic62/action12와scale0.5, actor·critic·noise를 그대로 복원하고 Adam/단계iteration을 새로 시작했다. 추가2,000회를 완료해 계보 누적18,000회가 됐다.

## 출발과 전환

| 단계 | 명령·제어 | 다음 단계 조건 |
|---|---|---|
| 0 정지 유지 | vx=0, 기본 관절 자세를 PD로 유지. 초기 몸통·관절 속도0 | 최소0.5초 및 안정 상태0.5초 유지 |
| 1 일반 보행 | 완료된1m/s 전진 정책 사용. vx를0→1m/s로1.5초 동안 smoothstep 증가 | 최소3초 보행 구간＋자세·실제 속도 추종0.5초 유지 |
| 2 Gallop 전환 | vx1→2.8m/s와 실행 관절 명령을3초 동안 smoothstep 혼합 | 자세와 속도 추종이 정상일 때만 전환 진행 |
| 3 Gallop | vx=2.8m/s, Gallop 정책이 전부 제어 | episode 종료까지 유지 |

전환 준비 조건은 roll/pitch 각각0.18rad 이내, 몸통 각속도 norm1.5rad/s 이내, 횡속도0.15m/s 이내, 기존 높이 범위0.18~0.60m다. 정지 단계는 수평속도 norm0.15m/s 이내를, 보행·전환 단계는 실제 전진속도와 현재 명령의 오차0.25m/s 이내를 확인한다. 보행 여부의 자동 판정은 이 자세·속도 조건이며 별도의 엄밀한 gait 분류기는 아니다.

전환 도중 조건을 벗어나면 현재 속도 명령과 혼합 비율을 유지하고 전환 진행을 멈춘다. 정책을 갑자기 되돌리거나 목표 속도로 강제 점프하지 않는다. 물리 종료/reset 후에는 각 환경이 독립적으로 단계0부터 시작한다. 이상적인 입력의 전환 시점은 [합성 명령 그림](../runs/isaac/check/unitree_go2_gallop/startup_2p8_offline_20261102/startup_command_schedule.png)에 기록한다. 실제 로봇 동역학에서 확인한 시간은 아니다.

episode는30초로 늘리고, 초기 episode 길이 무작위화를 끈다. 초기 몸통 속도는 기존처럼0으로 유지하며, 남아 있던 관절 초기 속도±1rad/s도0으로 바꿨다. 그 밖의 물리·모터·관절 한계와 perturbation 설정은 유지한다.

## 보행 정책과 Gallop 정책의 연결

이 구성은 **동결된 일반 보행 actor＋추가 학습하는 Gallop actor＋전환 상태기**를 사용한다. 단일 actor에 일반 보행을 새로 학습했다고 주장하지 않는다. 정지 단계에서는 기본 관절 자세, 일반 보행 단계에서는 보행 actor, 전환 단계에서는 두 actor의 관절 명령을 연속 혼합하고, 최종 단계에서는 Gallop actor만 실행한다.

보행 actor의 관측45개와action scale0.25를 Gallop의관측47개/scale0.5에 맞춘다. 이전 실행 action의 입력 단위와 출력 단위를 각각 보정했다. 관측의 마지막 위상2개는 정지 시 크기0, 보행 시0.5, 전환 중0.5→1, Gallop 시1로 표시한다. 관측 차원을 늘리지 않아 기존 Gallop 정책 가중치를 그대로 가져온다.

PPO에는 원래 샘플링한 Gallop action을 보관한다. 물리 실행·이전 action 관측·action-rate 보상은 실제 혼합 action을 사용한다. 보행 actor는 gradient를 끄고 PPO optimizer에도 포함하지 않는다. 따라서 출발 구간의 안정성을 기존 보행 actor로 제공하며 Gallop actor는 전환 이후 제어를 학습한다.

보행 모델은 새 실행 폴더의 `source/walking_teacher.pt`에 함께 저장하고 manifest에 SHA256을 기록한다. 새 체크포인트 재생은 동봉된 보행 모델을 읽고 해시를 확인한다. 체크포인트를 옮길 때는 `model_final.pt`와manifest/source 폴더를 함께 보관한다. actor 하나만 따로 내보낸 결과에는 이 전환 절차가 포함되지 않는다.

## 도약·균형·집계

실제거리 목표18cm·공중시간60ms·착지 안정 확인80ms를 사용한다. 2.8m/s의 참조주기는0.33초, 공통 공중구간은약66ms로 목표 전진거리 약18.5cm에 대응한다. 이 값은 참조 설정이며 실제 거리는 기존5ms 몸통 위치 계측으로 계산한다.

기존 몸통 균형·좌우 hip/하중 편중·clipping 패널티는 출발과 전환에서도 유지한다. 갤롭 위상 참조 보상과 공중 구간 지지 패널티는 전환 비율에 맞춰 적용하고, 일반 보행에 공중기를 강제하지 않는다. 실제 도약·착지 이벤트와 성공 gate의 속도/접촉/ROM/공중기/하중 집계는 **전환 완료 후1초가 지난 Gallop 구간**에 한정한다. 그래야 일반 보행 구간 때문에 Gallop의 공중기 비율이 낮아지는 문제를 피할 수 있다.

episode JSONL에는 `startup_stage`, `startup_completed`, 단계별 시간과 `gallop_metric_seconds`를 추가했다. trace에는 실제 적용 단계·전환 비율·실행 action·Gallop 집계 mask를 저장한다. 콘솔은 기존 iteration 표와 초기 설정 요약을 유지하고 상세 값은 TensorBoard/episode/trace에서 확인한다.

## 재생과 검사

학습 완료 뒤 새 실행ID로 재생한다. `--command`를 지정하면 출발 순서를 우회하므로 이 설정에서는 허용하지 않는다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --video --video-length 1500
```

업데이트 없이 GPU 연결 검사만 하려면:

```bash
./isaac.sh gallop check --headless --num-envs 16 --steps 1500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_151901_f393b4/model_final.pt \
  --config configs/gallop_2p8mps_walk_transition.json --video --video-length 500
```

CPU65개 검사와 실제 Gallop 부모 복원을 확인했다. 실제 일반 보행 actor를 원래 RSL actor와 비교해 관측·action 단위 보정 후 출력이 일치하는 것도 확인했다. 정지/보행/전환 순서, 불안정 시 진행 정지, 환경별 reset, 실제 실행 action 관측, 보행 구간의 Gallop 집계 제외, 모델 bundle 해시를 검사했다. [오프라인 검증](../runs/isaac/check/unitree_go2_gallop/startup_2p8_offline_20261102/startup_verification.json).

구현 당시에는 CUDA 사용 불가로 새 PhysX 출발·전환 동작이 미검증이었다. 이후 사용자 본학습·Isaac 플레이와 이번 MuJoCo 검증은 위 완료 결과를 따른다. 에이전트는 PPO 업데이트나 본학습을 실행하지 않았다. 기존2.5m/s 모델은 자체 학습 source로 재생되므로 새 출발 절차는 새 설정으로 추가 학습한 모델부터 적용된다.
