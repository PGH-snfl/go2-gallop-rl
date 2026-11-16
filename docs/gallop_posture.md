# 2.8m/s Gallop: 몸체 수직 정렬과 좌우 발 간격 — 2026-11-02

이 문서는 이전 설계다. 이후 사용자 수정 요청에 따른 **yaw0 벌점·roll/pitch 진동 보상·어깨보다 작은 독립 발 폭·목표방향 이동 보상**은 [최신 설계와 명령](gallop_heading_motion.md)을 따른다. 아래 설정은 과거 재현용으로 보존했다.

몸체의 +Z축을 지면 법선(+Z)에 맞추는 벌점과 앞·뒷발 쌍의 좌우 간격 보상을 추가했다. 새 설정은 [gallop_2p8mps_posture.json](../configs/gallop_2p8mps_posture.json)이다. 기존2.8m/s 모델에서 추가 학습하는 별도 설정이며 기존 체크포인트 재생에는 소급 적용되지 않는다.

## 추가 보상

| 항목 | 계산과 범위 | 가중치 |
|---|---|---:|
| `gallop_upright` | 몸체 수직축과 지면 법선 사이 각도 θ. `−min(4, (max(θ−0.05,0)/0.2)²)` | 6 |
| `gallop_stance_width` | 앞·뒤 각각 `왼발 y−오른발 y`의 목표30cm 오차 비용을 계산. 쌍별 비용을 평균해 `1−비용`, 범위−3~1 | 2 |
| `gallop_stance_center` | 앞·뒤 발 쌍의 평균y가 몸체 중심선에서 벗어나면 벌점. 쌍별 비용 평균, 범위−4~0 | 1 |

기존 `attitude` 보상은 위상에 따른 pitch 진동을 목표로 했다. 새 설정은 **pitch_amplitude=0**으로 바꾸고 기존 자세 비용도 수평을 향하게 했다. 새 upright 벌점은 yaw와 무관한 전체 기울기를 사용하며 좌/우·앞/뒤 기울기를 같은 방식으로 처리한다. 뒤집힌 자세도 벌점을 받는다. 0.05rad(약2.86°)까지는 새 항목에 허용폭을 두고, 기존 약한 자세 벌점은 그대로 작동한다.

발 위치는 몸체의 yaw 방향만 반영한 **수평 좌표계**로 옮긴다. 그래서 몸이 기울어졌다고 발 간격 기준도 함께 기울어지지 않는다. 앞발FL/FR, 뒷발RL/RR을 각각 계산한다. 폭을 절댓값으로 바꾸지 않아 다리를 교차해도 목표 폭을 만족한 것으로 처리하지 않는다. 두 발이 같은 쪽으로 함께 움직여 폭만 맞추는 동작은 중심선 벌점으로 제한한다.

양발이 지지 중이면 폭 허용오차±2.5cm, 쌍 중심 허용오차±1.5cm다. 한 발이라도 스윙/공중 상태이면 각각1.5배(±3.75cm, ±2.25cm)로 완화한다. 폭 오차 비용은 `min(4, (max(|폭−목표|−허용폭,0)/0.06)²)`, 중심 비용은 같은 형태에 척도0.04m를 쓴다. 발의 앞뒤 위치·높이와 착지 타이밍은 이 보상에서 제한하지 않는다.

목표30cm는 **발 중심 사이 간격의 초기 설계값**이다. 외장 몸통 폭을 실제 측정한 값이라고 주장하지 않는다. 공식 로컬 MuJoCo XML의 hip 횡오프셋4.65cm와 thigh 횡오프셋9.55cm를 합한 좌우 다리 축 폭은28.4cm다. 그 폭에 여유를 둬30cm를 정했다. 같은 모델에서 학습 기본 관절 자세의 발 간격은 앞34.52cm/뒤34.33cm다. thigh/calf를 그대로 두고 hip 각도만 소폭 줄여30cm에 도달할 수 있고 soft limit 안이라는 순기구학 검사를 통과했다. 이는 실제 주행 안정성 검증과 구분한다.

새 세 항목은 속도·gait강도·warmup·Gallop 집계 mask로 끄지 않는다. 정지·보행·전환과 지지·이륙·공중·착지 모두에서 계산한다. 다만 정지 단계의 고정 자세와 일반 보행의 동결 teacher는 학습 대상이 아니므로 그 actor 자체가 이번 추가 학습으로 수정되는 것은 아니다. Gallop actor가 관여하는 혼합 이후부터 새 행동을 학습한다.

속도2.8m/s, 도약18cm·60ms, 착지80ms 회복, 앞다리 참여, 느슨한 착지 순서, 정지→1m/s 보행→갤롭 전환은 유지한다. MuJoCo에서 관찰한 과속과 전환 정지의 직접 조건을 이번 수정으로 해제하지 않았다. 자세·발 간격 보상이 실제 방향 추종과 전환을 개선하는지는 추가 학습과 재검증이 필요하다.

## 추가 학습

학습은 사용자가 실행한다. 최신 완료 모델의 actor/critic/noise를 정확히 복원하고, 새 Adam·iteration0으로2,000회를 추가한다. 완료 시 계보는18,000→20,000회가 된다. 관측47/62·행동12·scale0.5는 유지한다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --walk-checkpoint runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --config configs/gallop_2p8mps_posture.json \
  --video --video-length 500 --video-interval 10000
```

저장 위치는 `runs/isaac/train/unitree_go2_gallop/<새실행ID>/`다. 새 모델 재생도 해당 새 실행ID를 지정한다. 이전 모델에 새 config를 강제로 넣어 play하지 않는다.

## 기록과 검증

TensorBoard의 `Episode_Reward/gallop_upright`, `gallop_stance_width`, `gallop_stance_center`에서 보상 성분을 확인한다. `Gallop/last_posture_*`와 `episodes.jsonl`에는 전체 구간 기울기RMS, 앞·뒤 폭 평균/목표 오차RMS/중심선 오차RMS/비용/계측 시간을 저장한다. 이 통계는 전환에 실패해도 유효하도록 기존 순수Gallop 성공 지표와 별도 분모를 사용한다. `gallop_trace.npz`에는 `posture_tilt`, `posture_width`, `posture_center`와 비용을 저장한다. 콘솔 iteration 표는 유지한다.

- 새 보상·실제 보상 연결·reset·10/20ms 적분·기존 전환/도약/복원 회귀 **74개 검사 통과**.
- 실제 최신 체크포인트의 모든 actor/critic/noise tensor 정확복원, 새 optimizer, 기존 source/teacher SHA,30cm 자세의 가동범위 등 **10개 항목 통과**.
- [검증 결과](../runs/isaac/check/unitree_go2_gallop/20261102_143300_d8d961/posture_verification.json), [검사 로그](../runs/isaac/check/unitree_go2_gallop/20261102_143300_d8d961/tests.log). 코드/설정 사본도 같은 폴더에 보관했다.
- 현재 세션 CUDA 인식 불가. 새 Isaac 물리 주행·학습·보행 개선은 미검증이며 이번 작업의 PPO 업데이트와 새 물리 rollout은0이다.

추가 학습 없이 모델 호환성과 순기구학을 다시 확인하려면:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m scripts.validation.check_gallop_posture \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt \
  --config configs/gallop_2p8mps_posture.json
```
