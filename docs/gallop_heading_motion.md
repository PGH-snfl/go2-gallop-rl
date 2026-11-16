# 2.8m/s Gallop: yaw·자세 진동·어깨 기준 발 간격·목표 방향 이동

이후 사용자 요청으로 적용한 안정화 보상·성공률 기반2.2→2.8m/s 설정은 [최신 설계와 명령](gallop_stabilization.md)을 따른다. 아래는 그 부모가 되는20,000회 모델의 학습 설정과 결과다.

2026-11-02 사용자 수정 요청을 반영한 설정은 [gallop_2p8mps_heading_motion.json](../configs/gallop_2p8mps_heading_motion.json)이다. 이전 수직축/발쌍 중심선 고정 보상 대신 아래 항목을 사용한다. 사용자가2,000회 추가 학습을 완료했으며 [실제 결과·영상](../reports/20261102_gallop_heading_motion_review/README.md)을 수집했다.

최신 완료 모델은 `runs/isaac/train/unitree_go2_gallop/20261102_145730_26810f/model_final.pt`(계보20,000회)다. 30초 낙상 없이 전환했지만 앞발 간격·고정된 roll 기울기·yaw각속도·clipping 문제로 전체 성공은 미달이다. 아래 명령은 이번 학습을 시작할 때 사용한 부모18,000회 기준이며, 구현 당시 새 주행 미검증 기록과 이후 완료 결과를 구분한다.

## 보상 정의

| 항목 | 계산 | 가중치 |
|---|---|---:|
| `gallop_yaw_heading` | `−min(4, (wrap(yaw−0)/0.35)²)` | 2 |
| `gallop_roll_pitch_quiet` | `exp(−mean((각도 변화율/1rad/s)²))` | 2 |
| `gallop_front_width` | 앞발의 signed 좌우 간격이 앞 어깨 폭×0.95에 가까우면0~1점 | 1 |
| `gallop_hind_width` | 뒷발의 signed 좌우 간격이 뒤 어깨 폭×0.95에 가까우면0~1점 | 1 |
| `gallop_forward_progress` | 실제 월드 속도의 목표방향 전진량·이동각 오차·명령속도 오차를 함께 평가,0~1점 | 2 |

yaw 목표0은 **월드 +X**다. 목표 방향을 보고 출발하도록 이 설정의 reset yaw도0으로 고정했다. 기존 초기 yaw 무작위−π~π 출발은 사용하지 않는다. 목표 yaw는 설정의 `target_yaw`에서 변경할 수 있다.

roll·pitch의 진동은 현재와 이전 각도의 wrapped 차이를 policy dt로 나눈 **각도 변화율**로 측정한다. 차분 주기는20ms다. 변화율이 작을수록 양의 보상을 받으며, 절대 roll/pitch 각도를0으로 고정하는 항목은 아니다. reset 직후에는 이전 자세가 없으므로 첫 표본의 진동 보상을0으로 두고 통계에서도 제외한다. 환경별로 이력을 따로 초기화한다.

이 설정에서 기존 `gallop_attitude`의 가중치는0이고 `gallop_upright`, `gallop_stance_center`는 등록하지 않는다. 비행 balance 항목의 절대 roll/pitch 비용도 제거했다. 기존 낙상 종료, 착지 회복의 자세·충격 한계와 각속도 안정 조건은 유지한다.

## 앞·뒤 발 간격

학습에 사용하는 실제 USD의 FL/FR thigh pitch축 간격은 **28.4000009cm**, RL/RR도 **28.4000009cm**다. [USD 측정값과 SHA256](../assets/go2_shoulder_widths.json)을 보관했다. 여기서 어깨 폭은 상부 다리의 pitch 관절 축 간격이며, 몸통 외장 폭이나9.3cm인 hip abduction 축 간격과 구분한다.

앞·뒤 기준을 각각 계산해 목표 발 중심 간격을 **28.4cm×0.95=26.98cm**로 설정했다. 현재 Go2 모델의 앞뒤 형상이 대칭이라 두 숫자가 같지만, 보상과 설정은 독립적이다. 앞발 폭이 틀렸다고 뒷발 점수를 함께 깎지 않는다.

발 위치를 몸체 yaw에 맞춘 수평 좌표계로 변환하고 `왼발y−오른발y`를 측정한다. `e=max(|실제폭−목표폭|−허용폭,0)`에 대해 `exp(−(e/0.04m)²)`를 각 쌍의 점수로 쓴다. 양발 지지 시 허용폭±0.5cm, 한 발 이상 스윙·공중이면±0.75cm다. 어깨 폭 그대로보다 약간 좁은 목표 폭이 더 높은 점수를 받는다. signed 폭을 사용해 교차한 다리는 높은 점수를 얻지 못한다. 앞뒤 스윙과 발 높이는 자유다.

발쌍의 중심이 몸체 중심선에서 벗어났다는 이유로 별도 벌점을 주던 항목은 제거했다. 대신 실제 목표 방향 이동 점수를 사용한다.

## 목표 방향 이동 점수

실제 몸통 월드 속도를 목표방향으로 투영한 속도를 `v∥`, 횡방향을 `v⊥`, 현재 전진 명령을 `c`라고 한다. 다음 점수를 사용한다.

```text
angle = atan2(v⊥, v∥)
scale = max(0.25 m/s, 0.25*c)
score = clamp(v∥/max(c,0.05), 0, 1)
        * exp(-((v∥-c)/scale)^2 - (angle/0.15rad)^2)
```

`c≤0.05m/s` 또는 `v∥≤0`이면0점이다. 정지·후진·옆으로만 이동하는 동작은 이 보상을 얻지 못한다. 목표보다 과속해도 점수가 감소한다. 몸체가 앞을 보고 있다는 이유만으로 이동 점수를 주지 않고, 실제 월드 속도를 평가한다. 기존 몸체 좌표계 속도 추종 보상도 유지한다.

## 관측과 기존 모델 이식

기존47차원 actor 관측에는 절대 yaw가 없었다. yaw0 목표를 구분할 수 있게 위상 뒤에 `sin(target−yaw)`, `cos(target−yaw)−1` 두 열을 추가했다. actor **47→49**, critic **62→64**, 관절 출력12와action scale0.5는 유지한다. 인터페이스는 `unitree_go2_gallop_heading_49_64_12_v2`다.

이식 시 actor/critic 첫 층의 기존 열과 나머지 모든 학습 tensor·noise는 그대로 복원하고 새 두 열만0으로 초기화한다. 새 관측이 들어와도 최초 출력은 기존 actor와 같다. 동결된 일반 보행 teacher는 앞45개 관측만 읽으므로 입력 구조와 단위를 유지한다. 새 Adam·새 단계iteration0부터 추가 학습한다. 기존 모델 replay는 계속 저장된 source를 사용한다.

새 항목은 정지·보행·전환과 지지·이륙·공중·착지 전체에서 계산한다. 정지 자세/동결 teacher 자체는 수정하지 않으며 Gallop actor가 관여하는 혼합 이후 행동을 학습한다. 속도2.8m/s·도약18cm/60ms·착지80ms·느슨한 발 순서·앞다리 참여와 기존 전환 추종 기준은 유지한다.

## 추가 학습 명령

최신 완료18,000회 모델에서2,000회를 추가한다. 학습은 사용자가 실행한다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --walk-checkpoint runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --config configs/gallop_2p8mps_heading_motion.json \
  --video --video-length 500 --video-interval 10000
```

저장 위치는 `runs/isaac/train/unitree_go2_gallop/<새실행ID>/`다. 완료 후49차원 새 모델을 그 실행ID의source/manifest와 함께 재생한다. MuJoCo 어댑터와 결과 수집기도 새 인터페이스를 지원하도록 수정했다.

## 검증과 기록

- 회귀·새 보상·보상 연결·reset·10/20ms 차분·MuJoCo 관측 연결 **90개 검사 통과**. 이후 최종 허용폭 설정과 어깨 폭 점수 검사를 포함한 관련11개도 재확인했다.
- 실제 부모 모델의 기존 tensor 보존, 새 열0 초기화, 초기 actor 출력 일치, teacher/source SHA, USD 어깨 폭, 목표 발 폭의 관절 가동범위 등 **14개 항목 통과**.
- [최종 모델·형상 검증](../runs/isaac/check/unitree_go2_gallop/20261102_144451_a5c367/posture_verification.json). 로그와 코드 사본도 같은 폴더에 보관했다.
- 현재 세션 CUDA 인식 불가. 새 정책의 Isaac/MuJoCo 주행 episode와 추가 학습은 실행하지 않았다. 기존 정책으로 수행한 MuJoCo 단위 검사1step은 새 보상의 주행 검증으로 취급하지 않는다. 임시49차원 bundle은 최적화/물리 적분 없이 관측·모델 연결만 검사하고 삭제했다.

TensorBoard의 `Episode_Reward/gallop_*` 다섯 항목과 `Gallop/last_posture_*`에서 확인한다. episode/trace에 yaw오차, roll/pitch 각도 변화율, 앞뒤 폭·독립 점수, 목표방향 속도·이동점수를 기록한다. 통계는 전환 완료 여부와 무관한 전체 구간이며 진동 통계만 첫 표본을 제외한다. 콘솔 iteration 표는 유지한다.

학습 없이 호환성·형상만 재확인하는 명령:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_posture \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --config configs/gallop_2p8mps_heading_motion.json
```
