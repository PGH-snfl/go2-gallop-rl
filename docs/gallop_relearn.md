# 일반보행 모델에서 관절 제약을 먼저 학습하는 새 갤롭 분기

**사용자12,000회 학습 완료 결과:** [수집·경향·영상·병목 분석](../reports/20261105_gallop_relearn_review/README.md). `20261104_144742_917f16/model_final.pt`는 누적16,000회로 완료됐다. 1m/s 플레이에서 요청각 클리핑0%·필터8.52%·앞뒤 폭 유지100%를 확인했지만, 학습 중1↔2단계를46회 반복해 갤롭 단계에 진입하지 못했다. 종료 조건 강화와 탐색 분산, 빠른 회귀 평가를 재검토해야 한다. 아래 명령은 완료한 설정의 재현용이며 새 학습은 자동 실행하지 않는다.

검증된 전진1m/s 일반보행 체크포인트에서 출발해, 동일한 관절 제약 아래 보행 적응→갤롭 형성→속도·도약 확대를 한 번의 학습으로 진행한다. 설정은 [gallop_relearn.json](../configs/gallop_relearn.json)이다. 학습은 사용자가 Isaac Lab/RSL-RL에서 실행하며, 이번 구현 중 PPO·모방 optimizer 갱신은0회다.

부모는 `runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt`, 누적4,000회, SHA256 `d4bd406a2e0cc97b1370610a3791bf115c31cac8b718056322d95df2f9d192bf`다. 새12,000회 완료 시 이 분기의 누적은16,000회다. 기존28,000회 모델과 분석 자료는 보존한다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 12000 \
  --finetune-from runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt \
  --config configs/gallop_relearn.json \
  --video --video-length 1500 --video-interval 20000
```

새 결과는 `runs/isaac/train/unitree_go2_gallop/<새 실행ID>/`에 모델·설정·manifest·로그·영상과 함께 저장한다. 콘솔의 iteration/주요 reward·loss/시간 표는 기존대로다. 단계별 상세 정보는 TensorBoard `Relearn/*`, `Gallop/last_relearn_*`, `episodes.jsonl`, `progress.jsonl`, `curriculum_summary.json`에 저장한다.

## 단계와 성공 기준

총24단계이며 적격 episode128개 이상에서 단계 성공률80% 이상을 두 구간 연속 확인하면 한 단계 올린다. 50% 미만이면 한 단계 내린다. 시간만으로 승급하지 않는다. 단계·속도·도약거리·요청각 종료 여유는 환경별 reset에서 고정한다. 다른 단계에서 시작한 episode는 현재 단계 승급에 사용하지 않으며, 회귀 후 같은 단계에 돌아와도 이전 세대의 episode를 재사용하지 않는다.

| 단계 | 학생 정책 목표 | 요청각 초과 종료 |
|---|---|---|
| 0 | 일반보행1.0m/s·관절 제약 적응 | 작업 범위 초과0.12rad가60ms 연속 |
| 1 | 일반보행1.0m/s·요청 정밀화 | 초과0.06rad가40ms 연속 |
| 2 | 일반보행1.0m/s·요청 정밀화 | 초과0.02rad가20ms 연속 |
| 3~4 | 일반보행1.25→1.5m/s | 초과0.02rad가20ms 연속 |
| 5~7 | 1.5m/s에서 갤롭 강도0.25→0.50→0.75 | 초과0.02→0.01→0.005rad,20ms |
| 8~23 | 완전한 갤롭1.5→3.0m/s,0.1씩 | 초과1e-6rad,20ms 한 제어 step |

일반보행 단계는 충분한 학생 제어 계측5초·episode 생존·속도 추종·몸체 자세·앞/뒤 하중·실제 관절 제약·발 높이와 요청각 준수를 요구한다. 요청각 초과 관절 표본 비율 허용치는 초기5%→3%→1%로 줄인다. 이때 갤롭 공중기나 착지 회복을 선행 요구하지 않는다.

부분 갤롭은 앞다리 운동·참여와 위상 패턴을 추가하고, 강도0.5부터 공중기 포함 주기30%, 강도0.75부터 기존 공중기60%·거리 기준·10회 이상 회복 평가에서 안정 착지60%를 요구한다. 마지막 완전 갤롭 단계와 고속 단계는 기존 최종 조건을 모두 유지한다. 앞/뒤 폭24~30cm 유지80%, 안정 착지80%, 필터 개입35% 이하 등과 **요청각 초과 표본0**을 함께 요구한다.

단계 진행용 `relearn_stage_success`와 최종 갤롭 `success`는 별도다. 보행 적응을 통과해도 갤롭 성공으로 표시하지 않는다. `highest_confirmed_speed`는 완전한 갤롭 단계의 반복 성공에서만 갱신한다. 예산12,000회 안에 최종 단계까지 도달한다고 보장하지 않는다.

도약 최소거리 목표는 속도1.0m/s에서8cm→3.0m/s에서22cm로 연속 증가한다. 1.5m/s의 초기 갤롭 목표는11.5cm다. 최소 공중시간은60ms, 최종3m/s의 참조 공중 구간은76ms다. 보행 단계에서는 갤롭·도약 양의 보상을 끄고, 부분 갤롭에서는 강도에 따라 켠다.

## 보행 출력 보존과 실제 학습 구간

기존 actor45/critic60 입력을49/64로 확장하고 위상·방향에 해당하는 새4열을0으로 초기화한다. action scale0.25→0.5에 맞춰 actor 출력·noise와 previous-action 입력을 보정해 물리 명령을 보존한다. 새 Adam/LR1e-4를 사용한다. 새 모방 학습이나 tanh 출력 재매핑은 없다. 인터페이스는 `unitree_go2_gallop_relearn_49_64_12_v5`다.

매 episode는 정지→동결 일반보행0.8m/s→안정 추종 확인→학생 정책으로 전환한다. 전환 뒤 보행 적응 단계는 **학생 정책이 직접 보행을 제어하고 학습**한다. 보행 단계의 위상 관측은0이며 calf0.85 축소도 적용하지 않는다. 갤롭 강도를 올리면 위상 관측과 calf 축소가 점진적으로 활성화된다.

## 관절 제한과 실패 종료

원래 기계 범위의 중앙90%를 실제 물리 제한, 그 양끝5%를 추가로 비운 중앙81%를 목표각 작업 범위로 사용한다. hip6rad/s·thigh/calf16rad/s 목표 변화 속도, 앞16cm·뒤10cm 높이 비용, 발폭 가중치7·높이 비용3.5와 직전 안정성 보상을 유지한다.

**필터 전 요청 목표각**이 작업 범위를 초과하면 위 표의 여유·지속시간에 따라 실패 종료한다. 일반 timeout으로 처리하지 않는다. 종료를 판정하는 step에도 기존 각도·변화 속도 필터를 거친 명령을 실행한다. 초기부터 이 종료 규칙을 사용하고, 성공에 따라 엄격하게 만든다. 기존 요청-실행 차이 등의 보조 비용도 학습 신호로 유지한다.

동결 보행 정책이 제어하는 출발·전환 중에는 요청각 종료를 면제한다. 학생이 제어할 수 없는 출발 동작 때문에 학습이 계속 끊기는 것을 방지하기 위해서다. 이 구간에도 실행 필터·실제 관절 초과 감시는 작동한다. 실제 관절각의0.01rad 초과가60ms 지속되면 기존대로 실패 종료한다. 요청각 종료와 실제 관절각 감시는 별도다.

명령 변화 속도 때문에 필터가 개입하는 경우는 요청각 초과 종료와 구분한다. 종료 조건만으로 전체 필터 개입이 구조적으로0이 되는 것은 아니다. 초기 조기 실패도 첫 episode부터 승급 통계에 포함하고, 요청 초과 종료 사유·최대 초과량·지속시간을 기록한다.

## 중간 체크포인트 재개와 플레이

각 체크포인트 저장 시 파일별 SHA256·학습 단계·속도·확인된 최고속도·누적 iteration을 manifest의 `relearn_checkpoints`에 기록한다. 오래된 중간 모델을 불러올 때 최신 단계가 잘못 적용되지 않도록 파일별 기록을 사용한다. 재개는 새 run/새 Adam으로 시작하며 단계 집계 구간은 새로 센다.

```bash
./isaac.sh gallop train --headless --num-envs 256 --iterations 6000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/<실행ID>/model_XXXX.pt \
  --config configs/gallop_relearn.json \
  --video --video-length 1500 --video-interval 20000
```

완료 또는 중간 모델 플레이:

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

저장된 실제 단계와 목표를 복원해 curriculum을 동결한다. 보행 단계에 머문 모델은 보행을 재생하며, 설정의 최종 목표3m/s·22cm를 달성했다고 표시하지 않는다. MuJoCo도 같은 저장 단계·위상·요청각 종료를 복원하며 정책 평가와 Sim2Sim용으로만 사용한다.

## 검증과 한계

- 이전 정책 재생·이식·종료·단계 승급/회귀·목표 고정·중간 체크포인트 복원·MuJoCo 제어 연결 등149검사 통과.
- [실제4,000회 부모의 CPU 이식20항목](../runs/isaac/check/unitree_go2_gallop/20261104_140428_831db6/relearn_verification.json) 통과. 4,096개 입력에서 최대 물리 명령 오차2.38e-7rad, critic 오차0, noise 보존을 확인했다.
- [검증 로그·소스·해시](../runs/isaac/check/unitree_go2_gallop/20261104_140428_831db6/implementation_validation.json)를 보관한다.
- 현재 세션CUDA 불가로 새 Isaac 동역학·학습 수렴·갤롭 성능은 미검증이다. 임시 MuJoCo 한 제어 step 검사는 소프트웨어 연결 검사이며 새 학습 정책의 Sim2Sim 성능 평가가 아니다. PPO/모방 갱신0, 새 Isaac 주행0이다.

학습 없이 이식 검사를 다시 실행할 수 있다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m scripts.validation.check_gallop_relearn
```
