# 2.5m/s 도약·좌우 균형·착지 안정화 추가 학습

**완료 결과:** 사용자가아래설정으로2,000회추가학습을완료했다. 모델 `runs/isaac/train/unitree_go2_gallop/20261029_151901_f393b4/model_final.pt`, [수집 보고서·영상](../reports/20261029_gallop_2p5mps_flight_review/README.md). 실제도약55회/평균14.16cm·59.27ms/최대16.86cm,16cm·60ms충족4회모두80ms안정착지통과를확인했다. 회복실패24회와앞다리하중편중/clipping15.24%는남는다. 아래학습명령은완료단계재현용이다.

2026-10-29 요청: 실제 도약을 만들고 거리를 늘리며, 몸통 좌우 균형과 이륙·착지를 안정화한다. 전진 명령은 **2.5m/s 고정**이다. 설정은 [gallop_2p5mps_flight.json](../configs/gallop_2p5mps_flight.json), 부모는 완료된3m/s 모델이다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt \
  --config configs/gallop_2p5mps_flight.json \
  --video --video-length 500 --video-interval 10000
```

학습은 사용자가 직접 실행한다. actor47 / critic62 / action12와 action scale0.5를 유지하고, 부모 actor·critic·noise를 그대로 가져온다. Adam과 새 단계 iteration은 초기화한다. 부모의 계보 누적14,000회는 manifest에 별도 보관한다. 위2,000회 완료 시 계보 누적16,000회가 된다.

결과는 새 `runs/isaac/train/unitree_go2_gallop/<실행ID>/`에 저장한다. `--video`로 같은 폴더의 `videos/`에 학습 영상도 저장한다. 플레이는 새 모델을 지정해 실행한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/새실행ID/model_final.pt \
  --video --video-length 1500
```

## 이륙 전부터 신호가 생기는 보상

이전 최종 플레이는 공중 예정 구간118개 표본 모두 FL＋RR로 지지했고 실제 도약은0회였다. 도약을 완료해야만 주어지는 보상에 더해 **도약 준비→이륙→공중→착지 후 회복**을 연결했다.

| 구간 / 항목 | 이번 구성 |
|---|---|
| 명령과 참조 | vx=2.5, vy=yaw=0. 참조 속도 anchor1~3m/s를 분리해 유지 |
| 주기 | 2.5m/s에서0.345초, thigh 진폭±0.675rad |
| 도약 준비 | 공동 공중 구간 직전80ms. 지지 중 수직 상승속도0.4m/s에 접근하면 연속 보상 |
| 공동 공중 구간 | 위상0.80~1.00, 약69ms. FR duty0.35→0.30으로 확보 |
| 공중 구간 하중 해제 | 비접촉 발 비율과 네 발의 최소 지면 여유 높이2.5cm에 연속 보상 |
| 공중 구간 접촉 유지 | 접촉 발 비율에 가중치8의 음의 보상 |
| 기존 gait 양의 보상 | 해당 공중 구간에서 접촉 비율에 따라 줄임. 네 발 지지 시20%, 두 발 지지 시60%, 완전 비접촉 시100% |
| 전진 추종 | 계속 보상해 학습 초기의 전진 동작을 유지하도록 함 |
| 실제 도약거리 목표 | **16cm**, 기존12cm보다 증가 |
| 실제 공중시간 목표 | **60ms**, 기존40ms보다 증가 |

2.5m/s에서16cm에는 약64ms가 필요하며, 설정한 공통 공중 구간69ms와 맞춘 초기 목표다. 주기 전체의 전진거리와 공중 도약거리는 별도로 기록한다. 수치들은 이번 학습의 공학적 초기 설정이며 달성 결과가 아니다.

도약거리·시간의 부분 score를 `clamp(거리/0.16) × clamp(시간/0.06)`으로 연속화했다. 기존25% 부분 credit을100%로 바꿔 짧은 실제 도약에서도 신호를 얻는다. 기준 이상은 score1로 제한하고, 관측 score 증가 예산은 주기당 최대1이다. 거리 보상 가중치는1→3으로 올렸다. 최초 동시 비접촉 몸통 위치부터 첫 착지 몸통 위치까지 실제 전진방향 변위를5ms마다 계측한다.

## 몸통과 좌우 균형

- roll/pitch 기울기, 몸통 roll 각속도, 횡속도에 연속 패널티를 준다. 제자리나 저속으로 회피할 수 있도록 속도 gate를 곱하지 않는다.
- 기존 위상 보정 좌우 관절 오차 비용의 가중치를0.6→1.0, 허용 잔차를0.20→0.15rad로 조정했다.
- 실제 한 주기의 좌우 **hip ROM 차이** 비용을 추가했다(가중치0.75, 비대칭 허용0.20). 기존 thigh/calf ROM 비용이 놓친 RR hip 편중을 대상으로 한다. 작은 hip 움직임의 수치 잡음을 줄이기 위해 분모 하한0.15rad를 둔다.
- 좌우 누적 하중 비용은 가중치0.75, 비대칭 허용0.20으로 조정했다. 실제 ROM/하중/지지시간은 완전한 주기로 평가한다.
- 목표 관절각 clipping 비율에도 별도 패널티를 준다. 모터·관절 한계는 유지한다.

정상적인 좌우 lead/lag는 유지하며 같은 시각의 관절각이나 동시 이륙·동시 착지를 강제하지 않는다. 개별 발 착지 순서도 기존처럼 필수 조건에서 제외한다. 기존 앞다리 하중·관절 ROM·스윙 참여 조건을 유지해 뒷다리만 사용하는 동작을 억제한다.

## 이륙과 착지 안정화

안정적인 몸통 자세로 준비/공중 예정 구간에서 실제 이륙하면 상승속도·자세 품질에 따른 이륙 보상을 준다(주기당 최대0.5점). 정지, 뒤로 이동, 초기 낙하, 앞다리 자격이 없는 이륙에는 이 이벤트 보상을 주지 않는다. 준비 단계의 연속 보상은 앞다리 자격이 아직 없어도 학습 신호를 제공한다.

실제 비행 뒤 첫 착지에서 거리 score를 바로 지급하지 않고 보관한다. **20ms 이상 비행하고 전진한 착지**에 대해 다음80ms 동안 지지를 유지하며 아래 조건을 모두 만족하면 착지 보상과 보관한 거리 score를 확정한다. 16cm·60ms 미달의 실제 비행도 부분 score로 안정적인 착지를 배울 수 있다.

| 착지 안정 조건 | 초기 한계 |
|---|---:|
| 각 순간 roll/pitch 절댓값 | 각각0.30rad 이내 |
| 몸통 각속도 norm | 2.0rad/s 이내 |
| 몸통 횡속도 절댓값 | 0.35m/s 이내 |
| 몸통 하강속도 | 1.2m/s 이내 |
| 네 발 양의 수직 접촉력 합 | 해당 환경의 로봇 무게3배 이내 |
| 접촉 중인 발 링크 중심의 수평속도 | 0.5m/s 이내 |
| 몸 높이 | 기존0.18~0.60m 범위 유지 |

힘은 초기화 시 읽은 해당 환경의 실제 PhysX 질량×중력으로 정규화한다. 하강속도는 충돌 후 감속값으로 숨겨지지 않도록 **직전5ms 표본의 수직속도까지 포함**한다. 발 미끄러짐은 발 링크 중심 속도로 근사하며, 실제 접촉점 속도 계측과 동일하지 않다. 충격·미끄러짐에는 별도의 연속 패널티도 적용한다.

확정 보상은 회복 구간에서 관측한 최저 자세/속도 품질을 곱한다. 착지 이벤트 보상은 주기당 최대1점, 거리 보상은 원래 주기의 관측 예산 기준 최대3점이다. 회복 중 넘어짐·임계값 초과·재이륙·reset이 발생하면 보관한 score는 취소한다. 같은 policy step에서 물리 종료가 발생하면 확정한 양의 이벤트 보상도 취소한다. 참조 위상 경계에서는 진행 중 비행과 회복 기록을 유지한다.

## 기록과 검증

콘솔은 기존 iteration 요약 표를 유지한다. TensorBoard에는 `Episode_Reward/gallop_flight_{drive,clearance,support,balance,takeoff,landing,impact,clip}`, `gallop_pair_hip`, `gallop_leap_distance`가 추가된다. episode JSONL과 trace에는 자격 이륙/착지/안정 착지/회복 실패 횟수, 이륙 수직속도, 착지 peak힘/하강속도, 몸통 균형·공중기 지지 비용, 위상, 회복 중 보관 score를 기록한다. `leap_qualified_events`는 거리·시간 판정이며 `flight_stable_landings`와 실제 `leap_reward_points`를 함께 확인한다. `Gallop/last_*`는 마지막 기록 환경 값이다.

CPU 회귀55개가 통과했다. 실제3m/s 부모 모델의 모든 tensor·noise 복원, 새 Adam/iteration0, 원본 모델 해시 보존도 확인했다. 검증 자료: [flight_2p5_offline_20261029](../runs/isaac/check/unitree_go2_gallop/flight_2p5_offline_20261029/verification.json). 합성 입력 검사에는 준비 단계 신호, 남은 접촉 패널티, 안정 회복 지연, 충격/기울기/미끄러짐·직전 하강속도, 재이륙/종료/reset 취소, hip 편중, 정책10/20ms 보상 일치가 포함된다.

구현당시에는세션CUDA사용불가로새PhysX rollout은미검증/PPO0회였다. 이후사용자가2.5m/s본학습과PhysX플레이를완료했으며실제성능은맨위수집보고서를따른다. 구현당시의합성보상검사와완료정책성능은구분한다. 기존 모델의 `play --checkpoint`는 학습 당시 source를 읽으므로 새 보상은 새 추가 학습부터 적용된다.

GPU에서 학습 없이 연결만 확인하려면:

```bash
./isaac.sh gallop check --headless --num-envs 16 --steps 1500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt \
  --config configs/gallop_2p5mps_flight.json --video --video-length 500
```
