# 핵심 조건 우선 자동승급 — 46,000회 모델 이어 학습

2026-11-11. 설정: `configs/gallop_staged_core_first.json`. 사용자 요청에 따라 앞발 간격 같은 스타일 기준이 초기 속도/거리 학습을 막지 않도록 변경했다. 기존 설정은 보존하며 새 설정만 `staged_core_first_enabled=true`로 활성화한다.

## 학습 명령

최신 누적46,000회 모델에서 우선2,000회 추가 학습한다. 완료 시 누적48,000회이며80cm 달성을 보장하는 횟수는 아니다.

```bash
cd /home/cvr/GH/go2_animalization/gallop

./isaac.sh gallop train \
  --headless \
  --num-envs 256 \
  --iterations 2000 \
  --save-interval 500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261110_223314_cef16b/model_final.pt \
  --config configs/gallop_staged_core_first.json
```

actor49/critic64/action12·관절 출력·std를 그대로 복원하며 새Adam·기존 초기LR1e-4로 시작한다. 모방학습은 없다. 기존 stage0을 복원하고 새60sim초 적응/새 완전 집단으로 집계한다. 에이전트는 이번 작업에서 PPO를 실행하지 않았다.

## 단계별 필수 조건

속도·거리9단계는 유지한다.0:2.5m/s·18cm,1:2.75·18cm,2:3·18cm,3:3·20cm,4:3·30cm,5:3·40cm,6:3·50cm,7:3·65cm,8:3·80cm.

|단계|착지율|roll RMS|pitch RMS|yaw RMS|발폭·발높이의 승급 필수 여부|
|---|---:|---:|---:|---:|---|
|0–3|60%|10°|12°|8°|제외, 리워드 유지|
|4–5|65%|8°|10°|5°|제외, 리워드 유지|
|6|70%|8°|10°|5°|제외, 리워드 유지|
|7|75%|6°|9°|4°|제외, 리워드 유지|
|8 최종|80%|5°|8°|3°|기존 엄격 조건 복원|

최종 발폭: 앞뒤 각각24~30cm 비율80% 이상/각 최소10개 지지표본. 최종 발높이: 앞16cm·뒤10cm 기준+1mm 초과율 각쌍 평균10% 이하. 단계마다 reset에 고정된 자기 단계를 적용하므로 이전 단계 episode가 다음 단계 기준으로 재판정되지 않는다.

## 전 단계에서 계속 필수인 핵심 조건

- 실패 없이30초 생존, 일반보행→학생 전환, 학생 단독10초 계측.
- 전진 RMSE≤0.25m/s, 횡속도RMSE≤0.15m/s, yaw각속도RMS≤0.3rad/s.
- 접촉 일치60%/최소8주기, 충분한 공중기가 포함된 주기60%.
- 전체 ROM70%, 앞다리 평균70%/각50%, 앞다리 하중합25%/각8%·뒷다리합25%.
- 앞다리 각70%주기에서 stance60ms 및 높이3.5cm 이상 swing40ms.
- 해당 단계 거리와 공중시간을 만족한 도약60% 이상/최소10회.
- raw요청각 초과율10% 이하, 실제 관절 허용오차.01rad 초과계측0.
- 전체 필터 개입100%허용, 관절 각도/변화율 및 모터 토크·속도 제한 유지.
- 큰 자세 붕괴를 막는 기존 attitude RMS≤0.25rad도 유지. 단계별 자세 기준이 이보다 엄격하다.

착지 검증의 물리 기준은 유지한다. 이륙/공중의 안전 판정, 접촉 후80→120ms 연속 회복, 충격3BW·하강1.2→1.8m/s·slip0.5m/s 등은 원래 phase profile을 따른다. 변경한 것은 이 안정 착지를 성공시키는 사건 비율의 초기 문턱이지 개별 착지의 안전 판정을 삭제한 것이 아니다.

## 승급과 회귀

- 모든 환경에서 첫 완료episode 하나씩, 최소128개, 성공률80% 이상인 완전 집단2회 연속이면 승급한다.
- 초기 실패·전환 실패도 집계한다. 중복 환경/이전 세대/초기 적응의 제외 규칙을 유지한다.
- 회귀는 기본 운동·관절 보호·큰 자세·방향·착지60%를 유지한 episode 비율이50% 미만인 집단2회 연속일 때 한 단계 뒤로 간다.
- 최종 단계에서 발폭·발높이 같은 세부 품질만 미달하고 핵심 보행이 유지되면 `refine_quality`로 최종 단계에 머문다. 핵심 보행이 나빠지면 회귀한다. 최종 안정화를 마치기도 전에 스타일 미달로 반복 후퇴하는 것을 피한다.
- 최종80cm 인증은 기존 엄격 조건을 모두 요구하며 완화한 초기 판정으로 대신할 수 없다.

## 리워드와 기록 해석

**모든 reward 가중치는 이전 학습과 동일하다.** 발폭은 앞/뒤+3.5씩, 발높이 비용은 앞/뒤−3.5씩 계속 적용한다. 이번 변경은 승급의 필수 판정 우선순위다. 핵심 조건을 reward 점수와 상관없이 필수로 유지한다.

- `success`: 현재 단계의 승급 성공. 초기에는 발폭/발높이가 미달이어도 핵심을 만족하면true일 수 있다.
- `staged_strict_success`: 기존 엄격 조건에 따른 성공.
- `staged_final_success`: 마지막80cm 단계에서 엄격 조건까지 통과한 성공.
- `staged_core/pose/heading/landing/details`: 새 승급의 병목별 진단.
- `staged_regression`: 회귀를 막을 핵심 보행·기본 안전 조건.
- manifest의 `promotion_policy=core_first_v1`에 정책 의미와 단계별 문턱을 저장한다.

초기 승급을 최종 실배포 성공으로 표시하지 않는다. 안정성이 최종 단계에서 저절로 해결된다는 보장도 없다. 물리 안전은 계속 유지하고 세부 품질은 최종에 집중시키는 구성이다.

## 함께 수정한 참조 일치

30cm 이상 단계에서 `update_measurements`도 reset-latched profile로 접촉·관절·발높이 참조를 계산한다. pair stance 허용차도 각 episode의 duty에 맞춘다. 과거 모델 재생은 기존 run의 저장된 source를 사용하며 원본을 덮어쓰지 않았다.

[검증 보고서](../reports/20261111_gallop_core_first/README.md). 새 결과는 `runs/isaac/train/unitree_go2_gallop/<새ID>/`, 플레이는 `runs/isaac/play/unitree_go2_gallop/<새ID>/`에 저장한다.

```bash
./isaac.sh gallop play --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/<새ID>/model_final.pt \
  --video --video-length 1500
```

