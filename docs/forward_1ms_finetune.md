# 전진 상한1.0m/s 추가 학습

0.2/0.3/0.4m/s 평지 추종을 통과한 기존 전진 정책을 불러와 **전진 명령 범위를0.2~1.0m/s**로 넓힌다. 정지/후진/횡이동/회전 명령은 계속 제외한다. 보상·물리·PPO 구조와 명령 재추출 주기는 유지한다. command curriculum 없이 이 범위에서 고정 학습한다.

## 사용자 실행 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh forward train --headless --num-envs 256 --iterations 2000 \
  --forward-speed-range 0.2 1.0 \
  --finetune-from runs/isaac/train/unitree_go2_forward_only/20261028_150349_3efb1b/model_final.pt \
  --video --video-length 500 --video-interval 10000
```

이번 단계에서 **추가2,000iteration**을 실행한다. 새 무작위 정책으로 시작하지 않는다. 기존 actor/critic/행동 분포와 Adam moment/step/LR를 복원한다. Adam의 LR와 PPO adaptive scheduler의 내부 LR도 일치시킨다. 기존 모델의 복원 LR는 약1e-5이며 이후 원본 adaptive 규칙으로 조정된다.

새 단계의 iteration은0부터 다시 세므로 로그는0~1,999다. 이전2,000회는 `manifest.json`의 `fine_tuning.parent_cumulative_iterations=2000`에 기록한다. 이번 단계가 완료되면 누적4,000회이며, 서로 다른 명령 분포에서 수행한 두 단계라는 점을 구분한다. 이번 단계 경험은256×24×2,000=12,288,000transition이다.

저장은 **새** `runs/isaac/train/unitree_go2_forward_only/<새 실행ID>/`에 한다. 이전 성공 정책/영상/보고서는 유지한다. `--checkpoint`는 추론 전용이고, 추가 학습에는 `--finetune-from`을 사용한다. 두 옵션은 함께 사용할 수 없다.

## 저장되는 복원 이력

- `manifest.json`: 요청한 속도 범위, 부모 모델 경로/SHA256, 부모 iteration, 복원 optimizer 항목 수/LR, 새 단계 계획, 누적 이력.
- `source/parent_manifest.json`, `source/finetune.py`, `source/forward_only.py`: 부모 출처와 적용 코드.
- `progress.jsonl`의 `FINETUNE_RESTORED`: 실제 복원 상태.
- `status.json`: 정상 완료 시 이번 policy update 수와 누적 iteration.
- 나머지 체크포인트/영상/TensorBoard/console은 기존 forward 실행과 동일하다.

## 학습 후1.0m/s 플레이

아래 `새실행ID`를 새 학습 폴더명으로 바꾼다. 속도 범위는 체크포인트 manifest에서 자동으로 읽는다.

```bash
./isaac.sh forward play --num-envs 1 --steps 1500 --video --video-length 1500 --command 1.0 0 0 \
  --checkpoint runs/isaac/train/unitree_go2_forward_only/새실행ID/model_final.pt
```

기존0.4m/s 모델을1.0m/s 정책으로 간주하지 않는다. 추가 학습 후1.0m/s뿐 아니라 기존 저속 추종도 함께 평가한다. 현재 일반 `forward-only` 수치 평가의 기본 명령은0.2/0.3/0.4이므로, 그 평가만으로 새1.0m/s 목표를 통과했다고 판단하면 안 된다.

## 수행한 비학습 검증

- `tests/test_forward_finetune.py`:3개 검사 통과. actor/critic과 Adam moment/step의 정확한 복원, adaptive LR 동기화, 새로운 stage counter0, 비학습 checkpoint 거부, 양의 속도 범위 검증.
- `runs/isaac/check/forward_finetune_1ms_20261028/`: 실제 부모 정책을 복원해16환경×600step에서 고정 `[1,0,0]`을 전달했다. 입력9,600개, 정책 state_dict 불변, 부모 파일 해시 불변, PPO update0, 공식 checkout clean을 확인한다.
- 이 검사에서 기존 저속 정책의 reset4회가 있었으며, 이는1.0m/s 성능 합격 시험이 아니다. 새 학습 전 설정/복원/명령 연결 검사로 기록한다. 새1.0m/s 정책은 아직 학습하지 않았다.

학습은 사용자가 실행한다. 이번 작업에서는 추가 학습 코드를 준비하고 검증만 수행했다.
