# 최신26,000회 모델에서 안정성 가중치 강화

**사용자 학습 완료 결과:** [28,000회 보고서·영상·부모 비교](../reports/20261103_gallop_guarded_stability_review/README.md). `20261103_191405_bea09f/model_final.pt`에서 실제2.455m/s/앞폭 유지64.10%로 개선됐으나, 안정 착지66.04%·필터 개입50.89%로 전체 성공0이었다. 평균 도약17.51cm/최대20.72cm이며 승급0으로 명령2.5m/s·도약 기준14cm다. 아래 명령은 완료한 설정의 재현용이며 새 학습은 자동 실행하지 않는다.

사용자 요청대로 현재 모델에서 이어 학습하며 **보상 가중치7개만** 조정했다. 발 간격과 발 높이는 조금만 올리라는 후속 지시를 반영했다. 새 설정은 [gallop_guarded_stability.json](../configs/gallop_guarded_stability.json)이다.

출발점은 `20261103_165217_19801a/model_final.pt`, SHA256 `57607c19b9635ef21567e7a90fdbe75acce1cab532b9f78a44637a937696939d`다. [직전 결과](../reports/20261103_gallop_guarded_3mps_leap_review/README.md)는 실제2.44m/s/평균도약18.57cm였지만, 안정 착지65.38%·앞발 폭 유지21.78%·필터 개입50.01%로 전체 성공에 미달했다. 앞발 최대높이도22.66cm로 증가했다.

## 조정한 가중치

| 항목 | 이전 | 추가 학습 |
|---|---:|---:|
| 안정 착지 보상 | 2 | 4 |
| 몸체 균형 비용 | 2 | 3 |
| 착지 충격·미끄러짐 비용 | 1.5 | 2.5 |
| roll/pitch 진동 억제 보상 | 2 | 3 |
| 앞/뒤 발 간격 보상 합계 | 6 | **7**, 약16.7% 증가 |
| 발 높이 초과 비용, 앞/뒤 각각 | 3 | **3.5**, 약16.7% 증가 |
| 관절 요청-실행 차이 비용 | 6 | 8 |

발 간격은 기존 목표폭에 가까울수록 받는 양의 보상이며 앞/뒤 각 쌍의 가중치는3→3.5다. 새 중심선 패널티나 발 간격 오차 공식을 추가하지 않았다. 높이는 앞16cm/뒤10cm 초과량에 대한 기존 비용의 가중치만 소폭 올린다.

정지→보행1m/s→갤롭 전환, 속도2.5→3.0m/s·도약14→22cm 성공률 curriculum, 빠른 참조주기0.38초와 도약 보상4는 유지한다. 안정 착지는 기존80ms 회복 확인 후 지급한다. 관절의 물리/목표각 범위와 명령 속도 제한, 실제 초과 감시도 유지한다.

성공 판정은 기존대로이므로 안정성·발폭·관절 필터 적응 등을 충족하기 전에는2.5m/s·14cm에 머물 수 있다. 발 높이 판정은 여전히 앞/뒤 각 쌍 평균이므로 개별FL 발 높이도 별도로 확인해야 한다. 이번 변경을 안정성 개선이 이미 확인된 결과로 취급하지 않는다.

## 학습 명령

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh gallop train --headless --num-envs 256 --iterations 2000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261103_165217_19801a/model_final.pt \
  --config configs/gallop_guarded_stability.json \
  --video --video-length 1500 --video-interval 10000
```

학습은 사용자가 Isaac Lab/RSL-RL에서 실행한다. v4 actor49/critic64/action12·scale0.5 가중치와 noise를 정확히 복원하고 새 Adam/LR1e-4로 추가2,000회를 학습한다. 완료하면 계보28,000회다. 새 모방 학습은 없다.

모델·설정·로그·영상은 `runs/isaac/train/unitree_go2_gallop/<새 실행ID>/`에 함께 저장한다. 콘솔의 기존 iteration/주요 reward·loss/시간 표를 유지한다.

완료 후 플레이:

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새 실행ID>/model_final.pt \
  --video --video-length 1500
```

## 비학습 검증

실제 부모 체크포인트의 actor/critic/noise 정확 복원·빈 optimizer·기존 물리 범위·저장 명령 제약 등20항목을 검사했다. 실제 `configure_gallop()`을 API 컨테이너와 함께 실행해 가중치 등록을 확인하고, 부모와의 차이가 가중치7개뿐인지 등26항목을 추가 확인했다. 같은 저장 통계의 가중치만 바꿔 비용·보상의 방향도 확인했다. 이는 새 정책 성능이나 새 주행 검증이 아니다.

[최종 설정 복원·제약20항목](../runs/isaac/check/unitree_go2_gallop/20261103_175418_66c2d9/guarded_verification.json), [보상 등록·가중치26항목](../runs/isaac/check/unitree_go2_gallop/20261103_175418_66c2d9/stability_reward_verification.json), [검증 소스·해시](../runs/isaac/check/unitree_go2_gallop/20261103_175418_66c2d9/implementation_validation.json)를 보관했다.

이번 구현에서 PPO/모방 optimizer 갱신과 새 물리 주행은0회다. 현재 세션은 CUDA를 사용할 수 없어 실제 안정성 개선·속도 승급은 사용자 학습 후 플레이에서 확인해야 한다. MuJoCo는 정책 평가·Sim2Sim 검증용으로 유지한다.
