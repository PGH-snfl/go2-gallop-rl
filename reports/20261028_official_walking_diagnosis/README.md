# 공식 기본 보행이 정지한 이유 — 2026-10-28

후속 구현: [curriculum 수정본과 학습 명령](../../docs/curriculum_fixed_baseline.md). 원본 비교 환경은 보존하며, 수정본 비학습 검증에서 reset이 없는 갱신 시점에도 범위 확장이 작동했다. [검증 기록](../20261028_curriculum_fixed_verification/README.md).

완료된 정책은 명령을 받아도 발을 교대로 떼며 전진하는 행동을 만들지 못한다. **저속 정지에도 높은 보상을 주는 구성과, 명령 범위 curriculum의 갱신 시점을 놓치는 문제가 함께 확인됐다.** 단순히 플레이 명령을 빠뜨렸거나 영상의 카메라 때문에 전진이 보이지 않는 문제가 아니다.

대상: `runs/isaac/train/unitree_go2_official/20261028_131510_3b7c2b/model_final.pt`. 기존 수정본 정책과 구분한다. 원본 출처와 학습 완료 증거는 [수집 보고서](../20261028_unitree_official_review/README.md)를 참조한다.

## 실제 현상

- actor 입력에 고정 `vx=0.1, vy=0, yaw=0`이 매 step 들어가는 것을 확인했다.
- 19초 평가 구간 평균 실제 vx는 **0.0000567m/s**, 네 발 접촉력 모두5N 초과인 비율은 **100%**였다.
- 따라서 실제 관측 결과는 정지다. 학습 reward29.44와 episode19.88초는 걷기 성공을 뜻하지 않는다.
- 학습 전체2,000iteration에서 전진 command curriculum 상한은 **0.1m/s로 고정**됐다.

## 1. curriculum은 보상이 충분해도 갱신 기회를 놓친다

원본 `mdp/curriculums.py:lin_vel_cmd_levels`는 다음을 동시에 요구한다.

1. 환경이 종료되어 `_reset_idx`에서 curriculum 함수가 호출될 것.
2. `common_step_counter % max_episode_length == 0`일 것. 이 환경에서는 전체 policy step이 정확히1,000의 배수여야 한다.
3. 그 시점에 끝난 환경들의 `track_lin_vel_xy` 누적 보상/20초가 **1.2보다 클 것**.

반면 원본 학습 호출은 `init_at_random_ep_len=True`로 각 환경의 초기 episode buffer를0~999 사이로 흩어 놓는다. 예를 들어 처음172step에 timeout한 환경은 이후1,172/2,172step에 timeout하므로, 안정적으로 살아남을수록1,000/2,000step 갱신 시점을 계속 비껴갈 수 있다. 중간 낙상은 이 위상을 바꾸지만 올바른 갱신 기회를 보장하지 않는다.

### 완료 정책을 고정한 비학습 재현

원본 함수를 감싸 인자와 반환값만 기록했다. 함수 본문·보상·명령 분포·외란은 원본을 유지했다. 256환경, seed42, 각2,200step, 학습 때와 같은 확률적 행동 샘플링을 사용했다. `PPO.update`는 예외를 던지도록 차단했다.

무작위 초기 episode 위상 시험에서는:

- curriculum 호출490번 중 **306번이 보상 기준을 통과**했다.
- 이306번은 모두 global step 조건에 걸려 확장되지 않았다.
- 1,000/2,000step에는 종료된 환경이 각각0개라 curriculum 함수 자체가 호출되지 않았다.
- 최종 명령 상한은0.1m/s였다.

초기 episode buffer를0으로 맞춘 대조 시험에서는1,000step에250개 환경이 함께 종료됐고, 같은 완료 정책의 평균 추종 보상1.433으로 기준1.2를 넘어 **원본 함수가 범위를0.1→0.2m/s로 확장**했다. 2,000step에도224개 환경이 종료되고 평균 추종 보상1.350으로 **0.2→0.3m/s로 다시 확장**했다. 이는 재학습으로 정책을 개선한 결과가 아니라, 갱신 호출 시점을 확보한 결과다.

| 초기 episode 위상 | 1,000step 종료 환경 | 2,000step 종료 환경 | 범위 확장 횟수 | 최종 vx 상한 |
|---|---:|---:|---:|---:|
| 원본 방식의 무작위 buffer | 0 | 0 | 0 | 0.1m/s |
| buffer0 대조 | 250 | 224 | 2 | 0.3m/s |

원본 함수에 동일한 합성 보상1.45를 넣는 별도 검사에서도999/1,001step에서는 확장하지 않고1,000step에서만 확장했다. 보상 계산의 dt 중복 정규화 문제는 발견되지 않았다. RewardManager는 가중 보상×dt를 누적하고 curriculum이 최대 episode20초로 나눈다.

**확정 범위:** 이 갱신 누락 메커니즘과 완료 정책에서의 재현은 확인했다. 과거 본학습의 reset별 callback/RNG 상태는 저장돼 있지 않아, 과거48,000step에서 어떤 callback이 차단됐는지를 완전히 복원한 것은 아니다. 과거 로그의0.1 고정 현상을 설명하는 강한 원인 후보다. 동기화 대조는 같은 가중치·설정·seed를 쓰지만 reset에 따른 난수 소비가 달라 동일 물리 궤적 비교는 아니다.

원본 기본 환경 수는4,096이고 이번에는256으로 줄였다. 초기 위상을 독립 균등 샘플링한다고 가정할 때 offset0이 하나도 없을 확률은256환경에서77.4%,4,096환경에서1.66%다. 이는 축소 실행이 갱신 기회를 줄일 수 있다는 설명용 계산이며, 과거 실행의 실제 확률 추정이나 원본의 성공 보장은 아니다.

## 2. ±0.1m/s에서는 정지도 높은 추종 보상을 받는다

원본 선형 속도 보상은 `1.5 × exp(-||v_cmd-v||² / 0.25)`다. `vy_cmd=0`일 때:

| vx 명령 | 실제 vx=0인 정지의 보상 | 최대1.5 대비 |
|---|---:|---:|
| 0.05m/s | 1.4851 | 99.0% |
| 0.1m/s | 1.4412 | 96.1% |
| 0.4m/s | 0.7909 | 52.7% |

초기 vx/vy가 각각±0.1인 범위에서는 속도 오차를 줄여 얻는 추가 보상이 작다. 반면 움직이면 관절 자세 이탈·행동 변화·토크·에너지 등의 비용이 붙는다. curriculum이 저속에서 멈추면 정지에 머물기 쉬운 보상 구조다. 이는 보상 항의 성질과 실제 정지 결과를 연결한 해석이며, 정지가 모든 가능한 보행보다 총보상이 높다는 수학적 증명은 아니다.

`joint_position_penalty`의5배 항을 “느리게 움직이면 항상 적용된다”고 해석하면 안 된다. 실제 코드는 **전체 속도 명령이0이고 실제 평면 속도도 임계값 이하일 때만**5배를 적용한다. 비영 속도 명령에는 기본−0.7 가중치가 적용된다.

## 3. 발 들기 보상도 초기 명령 대부분에서 비활성화된다

설치된 Isaac Lab의 원본 `feet_air_time`은 `norm(command[:2]) > 0.1`일 때만 보상을 준다. 정확히0.1m/s 전진은 이 엄격한 부등식에서 비활성화된다. 대각 명령은 norm이0.1을 넘을 수 있으므로 전부 꺼진다는 뜻은 아니다.

vx/vy 독립 균등±0.1과 정지 명령10%를 적용하면 활성 비율은 이론적으로 **19.31%**다. 나머지약80.69%에서는 이 보상 항이0이다. 또한 이 항은 착지 시 `last_air_time−0.5초`를 쓰므로, 활성화돼도0.5초보다 짧은 공중 시간은 양의 보상을 주지 않는다. 현재 구성에 보행 주기나 교차 접촉 순서를 직접 요구하는 항은 없다.

## 다음 수정의 우선순위

1. 별도 실험 환경에서 curriculum을 reset/global-step의 우연한 일치에 의존하지 않도록 수정한다. 완료 episode 통계를 모아 정해진 주기에 평가하고, 초기 무작위 buffer 때문에 실제 수집 시간이 짧은 첫 episode를 구분한다. 원본 비교용 checkout은 유지한다.
2. 수정 후 명령 범위가 실제로 확장되는지를 먼저 로그로 검증한다. 그 다음에도 정지에 머무르면 명령 크기·추종 보상 폭·발 들기 보상의 활성 조건을 한 가지씩 대조한다.
3. 추가 학습의 성공은 평균 reward보다 고정 속도 명령의 실제 속도/접촉/영상으로 판단한다. 기본 보행이 확보된 뒤 Gallop을 추가한다.

단순히2000iteration이 짧았다는 이유만으로 같은 설정을 더 오래 돌리는 것은 우선 해법으로 삼기 어렵다. 이번에는 갱신 누락을 먼저 확인했고, 더 많은 학습이 이 문제를 자동으로 해결한다고 보장할 수 없다.

## 저장 자료와 재현

- [종합 수치](diagnosis.json): 원본 함수 검사, 보상 계산, 재현 비교.
- `runs/isaac/check/official_curriculum_random_20261028/`: 무작위 초기 위상.
- `runs/isaac/check/official_curriculum_zero_20261028/`: 초기 buffer0 대조.
- 각 폴더의 `curriculum_calls.csv/json`: 호출 step·보상·각 gate·변경 전후 범위·종료 환경 ID.
- `summary.json`, `initial_episode_offsets.npy`, `status.json`, `diagnostic_source.py`, `console.log`.
- 비학습 진단: `scripts/validation/diagnose_official_curriculum.py`.
- 오프라인 집계: `scripts/analysis/analyze_official_walking_failure.py`.

```bash
cd /home/cvr/GH/go2_animalization/gallop
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.isaacvenv/bin/python -m scripts.validation.diagnose_official_curriculum \
  --headless --initial-phase random --steps 2200 \
  --checkpoint runs/isaac/train/unitree_go2_official/20261028_131510_3b7c2b/model_final.pt
```

대조 시험은 `--initial-phase zero`를 쓴다. 모든 진단에서 PPO 업데이트0회, 정책 state_dict 불변, 원본 모델 파일 SHA256 불변 및 공식 checkout clean을 확인한다. 이번 작업은 진단 코드/보고서만 추가했으며 학습 설정이나 기존 정책을 수정하지 않았다.
