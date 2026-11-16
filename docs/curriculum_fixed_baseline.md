# 기본 보행 재학습: 주기적 command curriculum

기본 보행의 속도 범위가 확장되지 않는 문제를 수정한 **별도 실험 환경**이다. 공식 원본 경로 `./isaac.sh official ...`와 기존 모델은 유지한다. 수정본은 `./isaac.sh baseline ...`으로 실행하며 공식 task와 동일하다고 부르지 않는다.

## 변경 내용

- 원본의 `reset 시점 && global_step % 1000 == 0` 조건을 제거한다. reset 직전 완료 episode의 선형 추종 보상을 모으고 **매1,000policy step**에 평가한다. 해당 step에 reset이 없어도 평가한다.
- 256환경에서는 최소32개 완료 episode가 있어야 판단한다. 부족하면 모아둔 통계를 다음 주기까지 유지한다.
- 실제로 수집한 step을 별도로 센다. 초기 무작위 episode buffer 때문에 짧아진 첫 episode를 제외한다. 이후의 짧은 실패 episode는 포함하고, 원본과 같이 최대 episode20초로 정규화해 조기 실패가 높은 점수를 얻지 못하게 한다.
- 명령 범위가 바뀌기 전 시작된 episode는 다음 단계 점수에서 제외한다. 이전 저속 명령의 높은 점수로 연속 승급하는 것을 막는다.
- 평균 선형 추종 보상>1.2이면 원본과 같이 vx/vy 범위를±0.1씩 확장한다. 원본 상한 vx±1.0,vy±0.4를 유지한다.
- **보상·초기 명령 범위·무작위 episode 시작·지형·모터·PPO는 유지한다.** 첫 실험은 curriculum 수정 효과를 분리한다. 전진0.2~0.4로 시작하거나 feet_air_time을 조정하는 후속 보상 변경은 아직 적용하지 않았다.

구현: `isaac_task/velocity_curriculum.py`. 기본 학습은 원본처럼 새 정책에서 시작한다. train에서 `--checkpoint`를 거부하므로 이전 정지 정책을 자동으로 이어 학습하지 않는다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh baseline train --headless --num-envs 256 --iterations 2000 \
  --video --video-length 500 --video-interval 10000
```

학습은 사용자가 실행한다. 이 명령의2,000iteration은 비교 예산이며 보행 수렴을 보장하지 않는다. iteration 표/평균 reward/시간/ETA 출력과 TensorBoard를 유지한다.

결과: `runs/isaac/train/unitree_go2_curriculum_fixed/<실행ID>/`

- `model_1999.pt`, `model_final.pt`: 학습 체크포인트.
- `videos/`: 학습 중 영상.
- `console.log`, `progress.jsonl`, `status.json`: 실행과 진행 기록.
- `manifest.json`, `params/`, `source/`: 원본 커밋·수정 variant·수정 모듈 SHA256/사본과 적용 설정.
- `curriculum_summary.json`: 정상 완료 시 갱신 결과. 중단 시에도 주기별 기록은 progress/console에 남는다.

주기적 로그 예시:

```text
[VELOCITY_CURRICULUM] step=2000 | vx 상한=0.20 m/s | episode=256 | 추종=1.42/1.20 | expanded
```

수치는 예시다. `waiting_for_episodes`는 실제 완료 episode가 부족하다는 뜻이고 `reward_below_threshold`는 추종 보상이 기준에 못 미쳤다는 뜻이다. 초기 부분 episode를 제외하므로 첫1,000step에서 기다리는 것은 정상이다. 무조건 일정 시간이 지나면 속도를 올리는 방식은 아니다.

TensorBoard에는 `Curriculum/lin_vel_cmd_levels`, `eligible_episodes`, `window_track_reward`, `expansions`, `discarded_initial`, `discarded_cross_stage`를 기록한다.

## 학습 후 고정 전진 명령 플레이

아래 `실행ID`를 실제 학습 폴더명으로 바꾼다.

```bash
./isaac.sh baseline play --num-envs 1 --steps 1500 --video --command 0.4 0 0 \
  --checkpoint runs/isaac/train/unitree_go2_curriculum_fixed/실행ID/model_final.pt
```

학습 범위가0.4m/s까지 확장됐는지 먼저 로그에서 확인한다. play에서는 curriculum을 끄고 고정 명령만 평가하며 PPO 업데이트를 차단한다. `play_trace.npz/csv`와 MP4는 새 `runs/isaac/play/unitree_go2_curriculum_fixed/<실행ID>/`에 저장된다. trace의 reset=True 행은 reset 이후 상태다.

공식 원본 모델은 `official play`, 수정 모델은 `baseline play`로 구분한다. 다른 variant의 모델을 넣으면 실행을 거부한다.

수치 평가도 수정본 인터페이스를 지원한다:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.isaacvenv/bin/python -m scripts.validation.evaluate_unitree_baseline \
  --headless --profile nominal --case-suite standard --video \
  --checkpoint runs/isaac/train/unitree_go2_curriculum_fixed/실행ID/model_final.pt
```

고정0.2/0.4m/s 및 회전에서 실제 속도 오차·생존·접촉을 평가한다. 아직 해당 속도 범위를 학습하지 않았다면 결과 해석에서 구분한다. 기본 보행이 확인된 뒤 Gallop 조건을 추가한다.

## 비학습 검사

```bash
.isaacvenv/bin/python -m unittest tests.test_velocity_curriculum -v
./isaac.sh baseline check --headless --num-envs 1 --steps 16 --video
```

check는 PPO 업데이트를 차단하며 신경망/환경/로그/영상 연결만 확인한다. 체크포인트가 없으면 무작위 초기 가중치를 사용하므로 보행 성능 검사가 아니다.

2026-10-28 검증: [검증 보고서](../reports/20261028_curriculum_fixed_verification/README.md). 회귀 검사4개 통과. frozen256환경/2,200step에서 reset0개인2,000step에0.1→0.2 확장. `baseline check` 1환경16step과 MP4/trace/로그 저장 완료. 원본 실행과 env.yaml 전체를 비교해 변경이 `curriculum.lin_vel_cmd_levels` 한 곳뿐임을 확인했다. 본학습은 실행하지 않았다.
