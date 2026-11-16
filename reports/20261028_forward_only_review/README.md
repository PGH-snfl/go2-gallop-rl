# 전진 전용 학습 결과 수집 — 2026-10-28

**전진 전용 정책이 실제로 이동하며, 외란 없는 평지의0.2/0.3/0.4m/s 추종 평가3조건을 모두 통과했다.** 이전 curriculum 수정본의 정지 해법과 구분한다. 이번에 확인한 것은 제한된 속도 범위의 평지 전진 기준선이며 Gallop, Sim2Sim, 실제 로봇 검증은 아직 하지 않았다.

## 모델과 학습 완료

- 학습 폴더: `runs/isaac/train/unitree_go2_forward_only/20261028_150349_3efb1b/`.
- 최종 체크포인트: `model_final.pt`, `model_1999.pt`. 체크포인트22개.
- 완료: **2,000iteration / 12,288,000transition** (256환경×rollout24). 체크포인트 iteration, TensorBoard2000개 update, completed 상태를 대조했다.
- source: 공식 커밋4960b847을 사용하는 별도 `forward_only` variant. actor45/critic60/관절 행동12. 학습 설정과 snapshot 해시를 확인했다.
- 명령: **vx+0.2~+0.4m/s, vy=yaw0, 정지 비율0%, command curriculum 비활성**. 보상/모터는 이전 실험과 동일하다.
- scalar 기록:15:04:09.72~15:32:33.25 KST, 약28분24초.

| 학습 지표 | 마지막 | 최근100회 평균 |
|---|---:|---:|
| 평균 reward | 33.3992 | 33.7422 |
| 평균 episode 길이 | 19.45초/20초 | 19.76초/20초 |
| timeout 종료 통계 | 96.88% | 97.30% |
| 자세 제한 종료 통계 | 3.13% | 2.70% |
| 선형 속도 추종 보상 | 1.4574/1.5 | 1.4194/1.5 |

학습 종료 통계는 독립적인 성공률이 아니다. 보상 증가만으로 성공을 판정하지 않고 아래 실제 속도와 접촉을 측정했다. command curriculum은 의도적으로 껐으므로 해당 TensorBoard 곡선이 없는 것이 정상이다.

## 사용자 플레이:0.3m/s

`runs/isaac/play/unitree_go2_forward_only/20261028_153742_c7f3d7/`에서1환경/1,500step,30초 trace가 저장됐다. 고정 명령 `[0.3,0,0]`과 실제 actor 입력 trace가 일치했다. 학습 때의 잡음/외란/초기 상태 무작위화를 유지한 플레이다.

초기3초 및 첫 reset 이후를 제외한 **3~20초,17초**에서:

- 평균 실제 vx **0.296568m/s**, vx RMSE **0.055876m/s**.
- 평면 순변위 **5.034m**.
- vy RMSE0.037683m/s, yaw RMSE0.056818rad/s.
- 네 발 모두 접촉력5N 초과인 비율 **46.0%**. 발 접촉이 교대로 변하며 이전의 네 발 고정 정지와 다르다.
- roll/pitch RMS0.00677/0.05718rad, 최소 높이0.3236m.

reset1회는20.0초에 발생했다. play trace는 reset 후 상태를 저장하고 종료 이유는 기록하지 않으므로, 이를 낙상이라고 단정하지 않는다. reset 행과 이후 episode는 위 평가에서 제외했다. 영상은 약30초(메타데이터29.98초), trace는30초 전체다.

## 정지 명령 없는 평지 전진 검증

평가: `runs/isaac/check/forward_only_nominal_20261028/`.

새 `--case-suite forward-only`는 **처음부터 끝까지 양의 전진 명령만** 준다. 초기 정지·ramp·감속·최종 정지·회전 조건은 넣지 않았다.0.2/0.3/0.4m/s 각20초를 동시에 실행하고, 첫1초를 제외한19초를 평가했다. 잡음·밀기·질량/마찰/초기 상태 무작위화는 제거하고 원본 모터와 관절 제어는 유지했다. 모든 step에서 actor 입력 명령 일치 및 양의vx/나머지0을 검사했다.

| vx 명령 | 평균 실제 vx | vx RMSE | 허용 RMSE | 네 발 지지 비율 | 판정 |
|---|---:|---:|---:|---:|---|
| 0.2m/s | **0.209969** | 0.020643 | 0.05 | 55.05% | 통과 |
| 0.3m/s | **0.291376** | 0.018125 | 0.075 | 46.42% | 통과 |
| 0.4m/s | **0.377424** | 0.031408 | 0.10 | 41.26% | 통과 |

세 조건 모두20초 동안 종료/낙상 없이 생존했다. vy RMSE0.0169~0.0224m/s, yaw RMSE0.0171~0.0229rad/s, 최소 높이0.3232m 이상으로 다른 공학적 평가 기준도 통과했다. 접촉 중 발 링크 수평 속도 RMS는0.0994~0.1413m/s로 기준0.2 이하였으며, 이 값은 접촉면의 정확한 미끄럼 측정과 다르다.

접촉력5N 기준 공중기 비율은 세 조건 모두0%였다. 이 결과만으로 trot/gallop 등 보행 패턴을 분류하지 않는다. 특정 접촉 순서/lead/주기 조건을 검사한 Gallop 평가는 아니다.

이 평가는 단일 seed42/조건별1회다. 정지·후진·회전·거친 지형/외란 강건성은 새 정책의 검증 범위에 포함하지 않았다. vx 허용오차는 명령의25%이며 나머지 문턱값은 `evaluation_report.json`에 저장했다. 공식 인증 기준은 아니다.

## 이전 모델과 비교

같은 nominal 평지 프로파일에서 이전 curriculum 수정본은0.2/0.4 명령에 평균 실제vx0.000005/0.000032m/s로 거의 정지했다. 새 전진 전용 정책은0.210/0.377m/s로 이동했다.

전진 범위, 정지 비율, 횡/회전 명령, curriculum을 함께 바꿨으므로 개선 원인을 **정지 명령 제거 하나만으로 분리해 입증한 실험은 아니다**. 전진 전용 명령 분포로 집중 학습한 결과로 기록한다. 기존 모델과 학습 데이터를 덮어쓰지 않았다.

## 저장 자료와 영상

- `summary.json`: 출처·학습·플레이·평지 평가·이전 모델 비교 통합.
- `tensorboard_scalars.csv`, `training_curves.png`: 전체 학습 지표.
- `play_analysis.json`, `play_state_trace.csv`, `play_tracking.png`, `play_frames.png`: 실제 플레이 분석.
- `evaluation_report.json`, `evaluation_trials.csv`, `evaluation_protocol.json`, `evaluation_tracking.png`, `forward_04_contacts.png`: 전진3조건 평가.
- `videos.csv`: 학습5개(각10초), 플레이1개(약30초), 평가1개(20초) 영상 색인.
- `params/`, `evidence_manifest.json`: 학습 설정 사본 및 원본·평가 파일 SHA256.
- 원시 trace/terminal 상태 처리/CSV는 각 원본 play/check 폴더에 보관한다.

플레이 영상: `runs/isaac/play/unitree_go2_forward_only/20261028_153742_c7f3d7/videos/official-play-step-0.mp4`.

0.4m/s 평지 평가 영상: `runs/isaac/check/forward_only_nominal_20261028/videos/forward_04.mp4` (**20초/1,000frame/50fps/640×480**).

최종 모델 SHA256: `f9a79323e9848f58bc4cc0f281b2f2d5f74e451568cf5b3590468a3220450302`.

이번 수집 중 본학습/PPO 업데이트는0회다. 비학습 평가 이후에도 모델 해시와 공식 checkout clean을 확인했다. 다음에는 이 전진 기준선을 보존하면서 Gallop 추가 학습 설계와 Sim2Sim을 별도 작업으로 진행한다.
