# 공식 원본 Go2 기본 보행 결과 — 2026-10-28

**학습은 완료됐지만 실제로는 전진·회전 명령을 제대로 수행하지 못한다.** 학습 범위 내0.05/0.1 m/s를 명시적으로 넣어도 거의 제자리에 서 있다. Gallop 조건을 추가하기 전에 기본 속도 추종부터 해결해야 한다.

후속 원인 조사: [curriculum 갱신 누락 비학습 재현과 저속 정지 보상 분석](../20261028_official_walking_diagnosis/README.md). 무작위 초기 episode 위상에서는 확장0회였고, buffer0 대조에서는 같은 완료 정책으로0.1→0.3m/s 확장됐다. 아래의 추가 조사 항목은 이 후속 보고서로 갱신한다.

## 모델과 원본 확인

- 학습 폴더: `runs/isaac/train/unitree_go2_official/20261028_131510_3b7c2b/`
- 최종 정책: 위 폴더의 `model_final.pt` 및 `model_1999.pt`.
- 공식 저장소: `/home/cvr/GH/unitree/unitree_rl_lab_official`, commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`.
- 학습 전후 tracked 소스 clean, 환경 source SHA256와 원본 task audit가 manifest에 저장돼 있다. 기존 수정본 환경의 결과와 구분한다.
- 원본 보상·지형·초기 자세·action scale을 유지했다. 로컬 USD 경로와 현재 RSL-RL의 obs_groups/normalization=False 호환만 처리했다. 과거의 전체 dependency stack까지 그대로 복원한 실험은 아니다.
- actor45/critic60/관절 행동12, 512×256×128 ELU, normalization=False.

## 학습 지표

| 항목 | 결과 |
|---|---:|
| 환경 수 / rollout | 256 / 환경당24step |
| 학습 완료 | 2,000 iteration = **12,288,000 transition** |
| 마지막 평균 리워드 | 29.4383 |
| 최근100회 평균 리워드 | 29.1937 |
| 마지막 평균 episode 길이 | 993.93step = **19.88초 / 목표20초** |
| 최근100회 시간 제한 종료 통계 | 98.14% |
| 최근100회 자세 제한 종료 통계 | 1.86% |
| 전진 명령 curriculum 상한 | **처음부터 끝까지0.1m/s** |
| 마지막 전진 추종 보상 | 1.4057 / 최대 가중치1.5 |

긴 생존 시간과 높은 리워드가 걷기 성공을 의미하지 않는다. 로그의 종료 통계는 학습 환경들의 최근 episode 상태이며 독립 평가의 성공률이 아니다. 기존 수정본 실험과는 보상 함수가 달라 리워드 크기를 직접 비교하면 안 된다.

현재 초기 명령 범위에서 vx 명령0.1m/s, 실제 속도0이면 선형 속도 추종 보상은 `1.5*exp(-0.1²/0.25)≈1.4412`로 최대값의 약96.1%다. 작은 명령에서 정지 해법도 높은 보상을 받는 구조는 정지에 머문 원인 후보다. 이것만으로 근본 원인을 확정하지는 않는다. 공식 curriculum은 reset 시점과 `common_step_counter % max_episode_length` 조건을 함께 사용하므로, 실제 curriculum 갱신 기회가 있었는지도 추가 조사 대상이다.

## 기존 플레이에서 확인 가능한 것

플레이: `runs/isaac/play/unitree_go2_official/20261028_134908_a53718/`.

- 500step(10초), reset0회, 영상 저장 완료.
- 영상: 위 폴더의 `videos/official-play-step-0.mp4`.
- 이 실행은 고정 전진 명령이 아니라 원본 초기 범위인 vx/vy 각각−0.1~0.1m/s, yaw−1~1rad/s의 무작위 명령을 사용했다. 정지 명령 비중도10%다.
- 이 과거 플레이에는 명령/실제 속도 trace가 없으므로 당시 정확히 어떤 값을 뽑았는지는 영상만으로 알 수 없다. 기본 플레이가 항상 앞으로 가는 시험은 아니었다.

## 새로 수행한 명시적 명령 평가

평가 폴더: `runs/isaac/check/unitree_official_commands_20261028/`.

원본 공식 환경·모터·정책을 읽고, 평가용으로 평지·고정 초기 상태·잡음/외란 없는 nominal 조건을 사용했다. 물리 소스는 수정하지 않았다. 정지2초→ramp1초→명령20초→감속1초→정지4초의28초 프로토콜이다. 유지 구간 첫1초를 제외한19초를 평가했다.

**매 step 요청 명령과 실제 actor 입력 명령을 대조했으며 오차는0이었다.** 첫 실패 상태는 자동 reset 전에 따로 저장하고, 이후 reset된 상태는 해당 trial의 성능에서 제외했다. PPO 업데이트를 차단했고 정책 파일은 변경하지 않았다.

| 명령 | 평균 실제 vx | vx RMSE | 판정 |
|---|---:|---:|---|
| 정지 | 0.000006m/s | 0.00010m/s | 정지 유지 통과 |
| 전진0.05m/s | 0.000029m/s | 0.04997m/s | 실패 |
| 전진0.1m/s | **0.000057m/s** | **0.09994m/s** | 실패 |
| 전진0.05 + 좌회전0.5rad/s | −0.000201m/s | 0.05668m/s | 전진·회전 실패 |
| 전진0.05 + 우회전−0.5rad/s | 0.000877m/s | 0.04984m/s | 전진·회전 실패 |
| 참고: 전진0.2m/s | 0.000090m/s | 0.19991m/s | 실패 |
| 참고: 전진0.4m/s | 0.000195m/s | 0.39981m/s | 실패 |
| 참고: 전진0.8m/s | 평가 구간 전 종료 | — | 3.88초 자세 제한 종료 |

저속 vx 허용오차는 명령의25% 또는0.015m/s 중 큰 값으로 사전 설정했다. 고정0.1m/s 허용오차를 사용하면 아예 움직이지 않아도 저속 시험이 통과할 수 있어 이 기준을 구분했다. yaw RMSE 허용값은0.25rad/s였다. 좌/우 회전의 실제 yaw 추종 RMSE는 각각0.911/0.914rad/s로 실패했다.

전진0.1 시험의19초 평가 구간에서 네 발 모두 접촉력5N을 초과한 비율은 **100%**다. 화면의 카메라 추종 때문에 전진이 안 보인 것이 아니라, 실제 root 속도와 접촉 상태가 정지를 가리킨다. 0.2m/s 이상은 학습 명령 범위보다 높은 참고 시험으로 기본 합격 판단과 구분한다.

- 고정0.1m/s 영상: 평가 폴더의 `videos/forward_01.mp4` (28초,50fps,640×480).
- 전체 추종 그래프: `tracking.png`.
- 발 접촉/속도: `forward_01_contacts.png`.
- 원시 상태/명령/정책 행동: `trace.npz`, `state_trace.csv`.
- 판정: `report.json`, `trials.csv`, 실행 상태: `status.json`.

## 보관한 수집 자료

- `summary.json`: 학습·플레이·고정 명령 평가 요약과 원본 manifest.
- `tensorboard_scalars.csv`: 전체 TensorBoard scalar.
- `training_curves.png`: 리워드·episode 길이·종료 통계·추종 오류·curriculum 차트.
- `videos.csv`: 학습5개 및 기존 플레이1개 영상의 메타데이터.
- `play_frames.png`: 기존 플레이 영상 프레임.
- `params/`: 학습 설정 사본.
- `evidence_manifest.json`: 원본/평가 파일의 크기 및 SHA256.

원본 체크포인트·학습 로그는 보존했다. 이번 작업은 결과 수집과 추론 평가만 수행했으며 추가 학습은0회다.

## 이후 재현 및 조사

이제 공식 play/check는 `play_trace.npz`, `play_trace.csv`에 실제 명령·속도·자세·접촉·행동과 reset 표시를 자동 저장한다. reset=True인 상태 행은 reset 이후 값이므로 넘어지기 직전 상태로 해석하지 않는다. 고정 명령을 주려면:

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh official play --num-envs 1 --steps 500 --video --command 0.1 0 0 \
  --checkpoint runs/isaac/train/unitree_go2_official/20261028_131510_3b7c2b/model_final.pt
```

이 명령은 정책을 고치는 것이 아니라 전진 명령을 명확히 주는 재현 방법이다. `--command`는 평가 전용으로 train에서 거부한다.

다음에는 curriculum이0.1에서 확장되지 않은 이유와 정지해도 높은 보상을 받는 구성을 먼저 조사한다. 필요시 학습 시간/명령 범위/보상 조정안을 마련한 뒤 사용자가 추가 학습을 실행한다. 현재 정책을 정상적인 보행 기준선으로 간주하고 Gallop 조건을 덧붙이지 않는다.
