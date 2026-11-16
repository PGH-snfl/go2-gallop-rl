# 전진 명령만 학습하는 기본 보행 실험

다음 단계: 기존 성공 체크포인트에서 [전진 상한1.0m/s 추가 학습](forward_1ms_finetune.md). `--forward-speed-range 0.2 1.0 --finetune-from ...`을 사용하며, 아래 기본 명령은0.2~0.4m/s 새 학습용으로 유지한다.

사용자 지시에 따라 학습 명령 분포에서 정지·후진·횡이동·회전을 제거했다. 기존 curriculum 수정본의 최종 정책도0.4m/s 전진 명령에 거의 정지했다. [기존 결과](../reports/20261028_curriculum_fixed_review/README.md)를 보존하고, 명령 분포를 바꾼 별도 실험을 새 가중치에서 시작한다.

| 항목 | 새 설정 |
|---|---|
| 전진 vx | **+0.2~+0.4m/s** |
| 횡이동 vy | 0 |
| yaw 속도 | 0 |
| 정지 명령 비율 | **0%** |
| 명령 재추출 | 10초마다 |
| 명령 curriculum | 비활성, 위 범위 고정 |

기존의 대칭 curriculum을 유지하면 vx 하한이 음수로 확장되고 vy 범위도 생기므로 이 실험에서는 끈다. 보상/지형/모터/PPO는 이전과 같다. 물리 외란으로 실제 속도가 음수가 될 수는 있지만 actor에 주는 명령은 양의 전진뿐이다.

## 학습

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh forward train --headless --num-envs 256 --iterations 2000 \
  --video --video-length 500 --video-interval 10000
```

학습은 사용자가 실행한다. 기존 정지 정책을 자동으로 불러오지 않는다. 저장은 `runs/isaac/train/unitree_go2_forward_only/<실행ID>/`이며, 최종 체크포인트는 `model_final.pt`다. 원본/기존 baseline과 다른 interface/manifest/출력 폴더를 사용한다. 입력 명령 분포 수정과 실제 전진 보행 성공은 구분하며, 학습 후 실제 속도를 확인한다.

## 학습 후 전진 플레이

```bash
./isaac.sh forward play --num-envs 1 --steps 1500 --video --video-length 1500 --command 0.3 0 0 \
  --checkpoint runs/isaac/train/unitree_go2_forward_only/실행ID/model_final.pt
```

위 명령은30초 플레이 전체를 영상으로 저장한다. 플레이 영상과 명령/속도/접촉 trace는 `runs/isaac/play/unitree_go2_forward_only/<실행ID>/`에 저장한다. forward 모드의 명시적 명령은 체크포인트 또는 CLI 설정 범위의 양의vx,vy=yaw=0만 허용한다. 기존 저속 체크포인트의 범위는0.2~0.4다.

## 비학습 검증

```bash
./isaac.sh forward check --headless --num-envs 16 --steps 600
```

2026-10-28 실제 실행: `runs/isaac/check/forward_only_commands_20261028/`. actor 입력을 매 step 검사하며 정지/후진/횡이동/회전이 들어오면 즉시 예외를 낸다. 무작위 초기 정책으로16환경×600step을 검사했다. 이는 명령 분포/실행 연결 검증이며, 전진 추종 성능 검사는 아니다. PPO 업데이트는0회다.

구현: `isaac_task/forward_only.py`, 공통 실행기 `scripts/training/official_baseline.py`. 공식 원본 checkout과 기존 학습 결과는 유지한다.

## 최신 완료 모델과 실제 결과

사용자 학습2,000회 완료 모델: `runs/isaac/train/unitree_go2_forward_only/20261028_150349_3efb1b/model_final.pt`.

[결과 수집 보고서](../reports/20261028_forward_only_review/README.md): nominal평지0.2/0.3/0.4m/s에서평균실제0.210/0.291/0.377m/s로세조건통과. 플레이0.3명령에서도17초동안5.03m이동했다. 학습5개/플레이1개/평가1개영상및수치보관. Gallop/Sim2Sim/실기는아직평가하지않았다.

정지 명령을 넣지 않는 수치 평가:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.isaacvenv/bin/python -m scripts.validation.evaluate_unitree_baseline \
  --headless --profile nominal --case-suite forward-only --video \
  --checkpoint runs/isaac/train/unitree_go2_forward_only/20261028_150349_3efb1b/model_final.pt
```

이 프로토콜은 각속도20초 동안 전진만 주며 초기정지/감속/최종정지를 넣지 않는다. 첫1초를 제외한19초의추종오차와생존/자세/접촉을평가한다. 정지 능력은 검사하지 않는다.
