# 2.8m/s 보행→Gallop 정책 MuJoCo 검증 — 2026-11-02

**30초 평지 실행은 낙상 없이 완료했지만, 갤롭 전환 검증은 통과하지 못했다.** 일반 보행 이후 갤롭 정책 비율65.76%에서 멈췄다. 고정된 명령2.18363m/s에 마지막10초 실제 평균3.09026m/s로 과속했고, 전진속도 오차0.25m/s 이내라는 전환 조건을 계속 벗어났다.

정책·전환 조건·보상 가중치는 그대로 사용했다. 이번 작업에서 PPO 학습이나 전환 조건 완화를 하지 않았다. 30초 단일 nominal 실행이며 물리 로봇/DDS는 사용하지 않았다.

## 실행 자료

- 체크포인트: [20261102_133218_92ee54/model_final.pt](../../runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt), 사용자 추가2,000회 완료·계보 누적18,000회
- MuJoCo 실행: [20261102_141502_01c472](../../runs/mujoco/evaluate/unitree_go2_gallop/20261102_141502_01c472/), MuJoCo3.1.6,5ms physics/20ms control,1,500정책step·6,000물리step
- [MuJoCo 영상30초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261102_141502_01c472/mujoco_gallop.mp4),640×480·50fps·1,500프레임
- [Isaac/MuJoCo 전환 비교 그림](transition_comparison.png), [영상 미리보기](video_preview.png), [전환 trace CSV](transition_trace.csv)
- [원본 실행 결과](../../runs/mujoco/evaluate/unitree_go2_gallop/20261102_141502_01c472/report.json), [모델·제어 계약](../../runs/mujoco/evaluate/unitree_go2_gallop/20261102_141502_01c472/model_contract.json), [진단 요약](diagnostics.json), [무결성 검사](verification.json)
- [증거 파일 SHA256 목록](evidence_manifest.json), [실행 코드·계약 사본 및 재검사 코드](source/). 무결성24개 항목이 통과했고, 재검사는 저장된 자료만 읽으며 물리 실행이나 학습을 하지 않는다.

## 전환이 멈춘 이유

| 단계 / 지표 | Isaac Lab 기존 플레이 | MuJoCo 이번 실행 |
|---|---:|---:|
| 정지 유지 종료 | 약0.94초 | 약1.26초 |
| 일반 보행 시작 | 0.96초 | 1.28초 |
| 보행 구간 평균 실제 vx | 0.614m/s | 0.618m/s |
| 혼합 전환 시작 | 3.98초 | 4.30초 |
| 완전한 Gallop 시작 | **8.34초** | **30초까지 도달하지 못함** |
| 최종 갤롭 정책 비율 | 100% | **65.757%** |
| 마지막 속도 명령 | 2.8m/s | **2.18363m/s** |
| 마지막10초 실제 vx | 아래 순수 Gallop 통계와 집계 구간이 다름 | **3.09026m/s** |

MuJoCo의 마지막10초501개 정책 표본 **전부**가 속도 오차 허용값0.25m/s를 초과했다. 같은 구간의 속도 RMSE는0.91783m/s다. 전환 전체1,286표본에서는 속도 오차 초과1,188개, 각속도 초과406개, 횡속도 초과239개, 기울기 초과34개였다. 조건은 중복될 수 있다.

현재 상태기는 불안정하거나 추종이 나쁘면 **속도 명령과 정책 혼합 진행을 함께 멈추는 구조**다. 이 실행에서는 명령보다 빠르게 움직이는 상태가 지속돼 혼합이65.76%에 머물렀다. 과속은 전환 정지의 직접적인 조건이며, 물성·관측 노이즈·접촉 모델 차이가 과속에 기여한 정도는 이번 단일 실행으로 분리하지 않았다.

낙상 종료·몸통 접촉 종료·높이 종료·자세 종료는 발생하지 않았다. 몸 높이는 전체0.23014~0.40178m였다. 마지막10초 roll/pitch RMS는0.08776/0.06538rad, yaw속도 RMS는0.34963rad/s, 목표 관절각 clipping은14.16%였다. 같은 구간의 수직 하중 분담은FL/FR/RL/RR=20.62/28.13/29.35/21.90%다.

## 도약은 어떻게 측정했는지

**공식 Gallop 지표는 미계측 상태다.** 저장된 정책의 계측은 전환 완료 후1초부터 켜지는데, 이번에는 단계3에 도달하지 못했다. 따라서 원본 episode의 `leap_events=0`, `vx_rmse=0`, `attitude_rms=0`, `action_clip_fraction=0` 등은 미계측 기록값이다. 원본의 `tracking=1`, `stable=1`도 유효 표본이 없는 집계에서 나온 값이므로 추종·안정성 통과로 취급하지 않는다.

실제 공중 구간을 놓치지 않도록5ms 물리 접촉·몸통 위치 기록을 별도로 분석했다. 초기1초 이전 이륙과 초기 낙하는 제외하고, 지지→네 발 동시 비접촉→첫 착지를 연결했다. [물리 이벤트 CSV](physical_flight_events.csv)에 원자료를 남겼다.

| 독립 물리 계측 | 결과 |
|---|---:|
| 네 발 동시 비접촉 완료 이벤트 | 123회 |
| 그중20ms 이상·전진 변위 양수인 이벤트 | **52회** |
| 위52회 평균 거리 / 공중시간 | **12.56cm / 43.65ms** |
| 최대 거리 | 20.31cm |
| 18cm·60ms 동시 충족 | **8회** |
| 독립80ms 착지 회복 검사 | 완료51회 모두 기준 미달,1회 진행 중 종료 |

나머지71회는 짧거나 전진 변위가 없는 접촉 변화다. 이를 모두 유효한 도약으로 세지 않는다. 회복 기준 미달51회 중 발 링크 중심 수평속도 한계 초과가49회, 각속도 초과17회, 지지 재해제10회, 힘 한계 초과1회에 나타났다. 사유는 중복 집계다.

이 독립 계측은 **혼합 제어 단계의 물리 기록**이며, 정책의 앞다리·위상 보상 자격과 예산을 적용한 공식 성공 판정은 아니다. 18cm·60ms 이벤트8회를 순수 Gallop 전환 통과로 해석하지 않는다. 발 속도는 링크 중심 속도로 근사하며 실제 접촉점 미끄러짐과 동일하지 않다.

비교 대상 Isaac 플레이는 완전한 Gallop을20.66초 계측했고, 평균vx2.93862m/s, 도약63회·평균18.53cm·66.03ms,18cm·60ms충족47회, 안정 착지25회가 기록됐다. 그러나 그쪽도 추종 gate와clipping21.02% 때문에 전체 성공 gate는 미달이었다. 두 엔진의 이번 수치를 모두 합격한 정책 사이의 비교로 취급하지 않는다.

## 검증 경로와 차이

기존 `scripts/validation/sim2sim.py`는 GPD57관측/5행동용이다. 새 [sim2sim_gallop.py](../../scripts/validation/sim2sim_gallop.py)와 [MuJoCo 어댑터](../../gpd/rl_mujoco.py)를 추가해 이번47관측/12관절 정책과 동봉된 일반 보행 actor를 실행했다. 체크포인트의 SHA 검증된 core와 `GallopState`를 불러와 위상·상태기·이벤트·성공 집계 조건을 재사용했다.

관절은 PhysX의hip4→thigh4→calf4 순서와 MuJoCo의관절/actuator 이름을 명시적으로 연결했다. 학습 당시PD25/0.5,action scale0.5,기본 관절 자세·높이0.4m,soft-limit factor0.9,Go2HV 토크-속도 곡선(Y1=20.2,Y2=23.4,X1=13.5,X2=30)을 적용했다. actor 출력에 임의의±1 clip을 넣지 않았고, 이전 action은 실제 혼합 실행값을 사용했다. MuJoCo COM Jacobian과 비교해 몸통COM 속도와 각속도 좌표계도 확인했다.

접촉력은 contact frame에서 world로 회전한 뒤 각 geom의 작용 방향을 적용했다. 이는 [MuJoCo contactForce API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)와 [접촉 프레임 방향 규약](https://mujoco.readthedocs.io/en/3.2.6/XMLreference.html#contact-pair)을 따른다.

소프트웨어 계약 검사5개가 통과했다: 이름 기반 관절/모터 매핑,COM 속도·각속도 프레임,관측·이전action 단위,토크 제한,초기 정지/정책 불변. 새 시뮬레이션은 위30초1회이며 재학습·조건 조정으로 재실행하지 않았다.

물리 모델은 별도의 공식 Unitree MuJoCo Go2 XML이다. 명목 총질량은15.20641kg로 학습 USD의16.08700kg보다 약5.47% 작다. 충돌·관성 모델이 다르고, 수동 damping/armature는0,frictionloss는설정된joint friction을0.01Nm로 근사했다. 바닥·접촉마찰은1,관측노이즈·랜덤질량/마찰/모터gain·push·상태추정오차·전송지연은 넣지 않았다. Isaac 플레이는 학습 환경의 노이즈와randomization을 유지하므로 엔진만 바꾼 완전 동일 조건 대조는 아니다.

## 재실행

```bash
cd /home/cvr/GH/go2_animalization/gallop
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/go2-gallop-mujoco-matplotlib MUJOCO_GL=egl \
  .venv/bin/python -m scripts.validation.sim2sim_gallop \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --isaac-play runs/isaac/play/unitree_go2_gallop/20261102_140231_b1e215 \
  --seconds 30 --video
```

매번 새 `runs/mujoco/evaluate/unitree_go2_gallop/<실행ID>/`에 결과를 보관한다. 다음 개선의 우선 대상은 혼합 단계의 과속 추종과 전환 진행 정지의 상호작용, 착지 시 발 속도다. 이번 검증에서는 학습 설정·정책·상태기 기준을 바꾸지 않았다.
