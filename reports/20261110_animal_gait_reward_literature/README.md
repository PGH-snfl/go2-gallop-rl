# Go2 animalization: 동물보행 조건을 직접 reward로 만드는 선행연구

조사일: 2026-11-10. 비교 기준: `configs/gallop_staged_80cm.json`, `isaac_task/gallop_env.py`, `isaac_task/gallop_core.py`의 현재 작업본. 학습 성능 실험이 아닌 문헌·구현 대조다.

## 결론

**있다.** 동물보행을 발 접촉·위상·지지/스윙·다리 협응·발 궤적·몸체 운동 같은 조건으로 풀고, 그 조건을 명시적인 reward/cost로 만드는 연구가 여럿 있다. 특히 **Go2에서 gallop을 포함한 보행을 위상/대칭 보상으로 구현한 연구 [P01]**, **여러 허용 범위를 barrier reward로 바꾸는 연구 [P02]**, **동물형 발 접촉 패턴을 직접 보상하는 연구 [P03]**가 가깝다.

**우리 조건 전체와 수치를 그대로 사용한 논문은 이번 범위에서 확인하지 못했다.** 3m/s·공중 이동80cm·앞다리 총하중25%/각8%·발폭26.98cm·관절 필터·128episode/80% 자동승급을 하나의 검증된 생물학적 gallop 공식으로 인용할 수는 없다. 이는 생물학적 보행 특징, task 목표, 로봇 제약, 학습 알고리즘을 조합한 우리 설계다.

또한 **조건을 정의한 것**, **그 조건을 매 step 보상하는 것**, **조건을 만족해야 거리 보상을 주는 것**, **승급을 막는 조건**, **출력 필터/종료 조건**은 다르다. 현재 프로젝트에는 이들이 함께 존재한다. “조건을 모두 그대로 reward에 넣었다”라는 설명은 정확하지 않다.

## 조사 범위와 신뢰도

- Exa 검색 30회 × 요청 결과10건 = **중복 포함 검색 결과300건**, 원시 URL205개. 300편의 논문을 읽었다는 뜻이 아니다.
- 원문/출판 페이지/공개 구현64개 문서 레코드, URL63개를 열었다. 버전·preprint/출판본·공개 코드가 섞여 있다.
- 서지 항목 **50건**을 선별했다. **48건은 원문의 핵심 방법을 확인**, 2건[P15, P31]은 초록/서지 위주라 상세 보상 검증으로 사용하지 않았다. 일부 긴 원문은 추출 길이 한도가 있으므로 전체 논문을 줄 단위로 전부 검증했다고 주장하지 않는다.
- 정식 체계적 문헌고찰이나 모든 데이터베이스 전수검색은 아니다. “같은 조합을 못 찾음”은 절대적인 부재·신규성 증명이 아니다.
- 연구 방법별로 직접 보상, 모방 보상, 제약/CPG, 비RL 이론을 분리했다. 같은 저자 계열과 후속 논문을 독립 재현으로 과대 계산하지 않았다.
- 2026년 preprint는 출판 완료·실기 검증과 구분했다. 제목이나 초록에 gallop이 있다는 이유만으로 방법/엄밀 보행 분류가 검증됐다고 보지 않았다.

재현 자료: [검색 기록](search_log.json), [검색 결과 메타데이터](search_results.json), [열어본 문서 목록](source_inventory.json), [50건 CSV](papers.csv), [50건 JSON](papers.json), [항목별 상세 목록](paper_notes.md).

## 먼저 읽을 핵심 논문 10건

### 1. Go2에서 위상·대칭을 reward에 넣어 gallop: Ding et al.

[Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds](https://arxiv.org/html/2403.10723v4), 2024 최초 공개 / 2026 v4. [P01]

- **대상:** Unitree Go2, Isaac Gym PPO, 시뮬레이션과 실기. EDU 모델인지 별도 확인하지 못했다.
- **직접 보상:** swing 구간에 지면반력이 있으면 비용, stance 구간에 발이 움직이면 비용. 보행 phase·duty·leg offset이 보상의 기준을 만든다.
- **대칭:** 같은 phase offset을 갖는 다리만 관절 차이를 비용화한다. gallop처럼 위상이 다른 좌우를 무조건 동시 운동시키는 식이 아니다.
- 속도·yaw **각속도**·몸체 높이·스윙 클리어런스·토크 변화도 보상에 포함한다.
- **우리와 가까운 부분:** phase-based 접촉 보상과 다리 협응으로 목표 gait를 정의한다.
- **차이:** 우리 absolute yaw=0, 발폭26.98cm, 앞다리 최소 하중, 공중80cm를 사용한 것은 아니다. time-reversal은 별도 scalar reward가 아니라 위상 재매핑으로 구현한다.

확인 위치: III-A Table I, III-D Eq(4)–(9)/Table II, IV-D.

서지 주의: [2510.10455](https://arxiv.org/abs/2510.10455)는 철회되어 2403.10723의 대체 버전에 통합된 기록이다. 두 개의 독립 연구로 세지 않았고 현재 v4를 기준으로 읽었다. 구버전 Bittle 결과와 최신 Go2 결과도 섞지 않았다.

### 2. “이 구간을 만족하라”를 직접 reward로: Kim, Lee & Park

[A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards](https://arxiv.org/html/2409.15780v4), ICRA2025. [P02]

- 접촉 보행 변수, 발 클리어런스, 관절 위치/속도, 전후 몸체 높이, 명령 오차, 몸체 운동에 **상·하한을 정하고 relaxed logarithmic barrier reward**로 바꾼다.
- 조건 안에서는 유연성을 주고 위반 방향으로 갈수록 비용을 크게 만든다. 우리처럼 조건을 여러 개 조합하는 접근에 가장 직접적인 설계 참고다.
- HOUND 계열의 여러 locomotion 모드와 gallop을 보고한다.
- **차이:** reward가 soft이므로 물리적인 관절/모터 제한을 보장하지 않는다. 논문의 관절 범위·발 높이를 Go2에 복사할 근거는 없다.

확인 위치: III-A/B/C, 특히 Table I. 출판 DOI: [ICRA2025](https://doi.org/10.1109/icra55743.2025.11128517).

### 3. 동물형 접촉 패턴을 그대로 보상: Yu et al.

[Discovery of skill-switching criteria for learning agile quadruped locomotion](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1697159/full), Frontiers2026. 식은 [preprint](https://arxiv.org/html/2502.06676) Table1–2도 대조. [P03]

- 동물형 기준 발 접촉 패턴 일치를 보상하고, 속도·자세·높이·토크·관절 속도·발 배치·스윙/스탠스 등을 함께 사용한다.
- A1의 trot/bound/gallop 스킬 및 전환을 다룬다.
- **우리와 가까움:** 접촉 패턴과 locomotion task를 한꺼번에 최적화한다.
- **주의:** 논문의 symmetric foot placement는 네 발 평균의 몸체 중심 정렬이다. 우리 “앞/뒤 각 쌍의 좌우 폭” 보상과 같지 않다. gait 정책에서 가중치0인 항목을 실제 적용한 보상으로 인용하지 않았다.

### 4. 발폭·스윙 높이·접촉까지 조절: Margolis & Agrawal

[Walk These Ways](https://arxiv.org/html/2212.03238), CoRL2022/PMLR2023. [P04]

- 발 힘/속도를 phase에 맞춰 보상하고 stance width·스윙 높이·몸체 pitch/height·빈도 등을 명령으로 넣는다.
- Go1 실기와 [공개 구현](https://github.com/Improbable-AI/walk-these-ways)이 있다. 공개 reward/config 코드까지 확인했다.
- 발폭은 단순 좌우 거리 고정이 아니라 **Raibert heuristic에 따른 상태·명령 의존 발 배치**로 반영한다. 논문은 일정 거리만 보상하면 빠른 회전에 불리하다고 설명한다.
- **우리와 가까움:** 발 간격/높이까지 명시 reward로 만드는 검증된 구현.
- **한계:** 주요 보행 표현은 two-beat 패턴이다. 엄밀한 비대칭 gallop을 검증한 직접 증거로 확장하면 안 된다.

확인 위치: §3.1, Table1, Appendix; 공개 코드 `corl_rewards.py`의 contact force/velocity·Raibert·clearance 보상.

### 5. 몸체 진동·발 궤적·생체역학 보행 선택: Humphreys & Zhou

[Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion](https://www.nature.com/articles/s42256-025-01065-z), Nature Machine Intelligence2025. [P05]

- Unitree A1에서 목표 접촉 상태 오차 + 발 Cartesian 참조 오차를 보상한다.
- 접지 발 slip, roll/pitch 각속도, 몸체 orientation/height, hip motion, torque/jerk 등으로 안정성과 효율을 다룬다.
- **우리 roll/pitch 진동 억제**와 발 궤적/접촉 조건에 매우 가깝다.
- 생체역학 지표로 보행을 선택하는 상위 정책까지 있다. 다만 CoT·발 접촉력·기계적 일·stride 변동을 모두 하위 정책의 같은 reward라고 해석하면 틀린다.

확인 위치: Methods → Locomotion policy Eq(5)–(9), Biomechanics gait transition metrics, Gait selection policy. [코드](https://github.com/ihcr/learning_to_adapt).

### 6. 두세 다리만 쓰는 문제를 직접 분석: Zhang et al.

[Learning Emergent Gaits with Decentralized Phase Oscillators: on the role of Observations, Rewards, and Feedback](https://arxiv.org/html/2402.08662v2), ICRA2024. [P06]

- 보행 phase 관측(O), phase별 reward(R), oscillator coupling(C)를 나누어 비교한다.
- **위상 관측과 위상 보상을 함께 쓴 설정**에서 네 다리 하중이 더 일관되고 균형 있게 분포한다.
- 그렇지 않은 설정에서는 발을 끌거나 계속 들고 있는 2/3-leg policy가 자주 나타난다.
- **우리 개구리형 뒷다리 편중 문제에 가장 가까운 ablation 근거**다. 숫자 하중을 더하는 것뿐 아니라, 각 발의 swing/stance를 보상하고 그 기준 phase를 정책이 알게 해야 한다는 근거다.
- 앞다리25%·각8%를 논문이 검증한 것은 아니다. 대규모 통계는 Mini Cheetah 시뮬레이션 결과이고 초기 실기 시험은 별도로 다룬다.

확인 위치: Balanced Leg Use, Fig3; 각 구성10개 정책 ×50개 시작 상태.

### 7. 앞뒤 각 쌍의 “약간 어긋난 착지”만 보상: S-Cheetah

[S-Cheetah: A Novel Quadrupedal Robot with a 3-DOF Active Spine Learning Agile Locomotion](https://arxiv.org/html/2605.27909), 2026 preprint. [P07]

- 앞쌍·뒷쌍의 접촉 위상차가 목표 비영 offset에 가까우면 Gaussian reward를 주고 두 쌍의 점수를 곱한다.
- 전체 착지 순서, duty factor, 주기, 공중기 길이는 직접 고정하지 않는다. bound의 동시 착지와 구분할 최소 조건을 사용한다.
- 능동 척추의 위상·진폭·안전 범위도 명시적으로 보상한다.
- **우리의 느슨한 순서를 유지하면서 gallop 정체성을 남기는 참고**가 된다.
- **시뮬레이션 연구**이며 실기는 향후 과제다. Go2에는 능동 척추가 없으므로 척추 reward나 속도 성과를 이전할 수 없다.
- 원문의 ipsilateral 명칭과 front/rear pair 설명이 혼재한다. 실제 구현을 따라갈 때 다리 인덱스·목표 offset 정의를 반드시 확인해야 한다.

확인 위치: III-B, III-C1 Eq2, III-C2 Eq3–6, V.

### 8. 도약 거리·이륙·안정 착지까지 직접 보상: Atanassov et al.

[Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design](https://arxiv.org/html/2401.16337), 2024 preprint / [RAM 출판 DOI](https://doi.org/10.1109/mra.2024.3487325). [P16]

- phase별 jumping reward와 착지 위치/방향, 최대 높이, 이륙/점프 사건 등을 다룬다.
- **제자리 도약 → 거리 도약 → 장애물 도약** curriculum을 사용한다.
- 목표 지점까지 걷거나 기는 정책, 정지만 하는 정책 같은 local optimum을 명시적으로 다룬다.
- Go1에서 60cm 도약을 sim/hardware로 비교하고 90cm 전방 점프도 보고한다.
- **80cm 과제의 관련 실기 선행연구**지만, **주기적으로 gallop하며 매 공중기80cm를 이동하는 것과 단발 점프는 다른 과제**다. 착지 위치 목표와 우리 all-off→첫 touchdown 이동거리의 측정 정의도 맞춰야 한다.

확인 위치: III-A/B/C; IV-C Table I; V-B Figs6–8; V-E.

함께 볼 연구: [Impedance Matching](https://arxiv.org/pdf/2404.15096) [P17]. Mini Cheetah Vision의 running jump에서 force shaping과 liftoff velocity reward, walk/jump 혼합 학습을 다룬다. 출력각이 같아도 PD/impedance 모델 차이로 실기 동작이 달라지는 문제까지 다룬다.

### 9. 조건 문장 자체를 reward로 변환: Atasever et al.

[Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications](https://arxiv.org/html/2607.00442), 2026 preprint. [P10]

- 안전 범위, command tracking, gait synchronization, actuation bounds를 STL 시간 논리로 정의한다.
- 최근 trajectory 창에서 조건을 얼마나 만족/위반했는지의 smooth robustness를 계산해 dense reward로 쓴다.
- **“조건을 만족하라”를 보상으로 만드는 방식 자체가 우리 질문과 매우 직접적으로 일치**한다.
- 임계값은 expert rollout으로 보정한다. Barkour/MJX 시뮬레이션의 walk-trot/trot/bound 연구이며 Go2 gallop 실기는 아니다.
- [후속 LLM specification 논문](https://arxiv.org/html/2609.07111) [P11]도 확인했다. LLM은 기호 조건을 제안하고 데이터로 수치를 맞춘다. LLM이 작성한 조건이 곧 검증된 생물학 법칙은 아니다.
- 학습된 정책에 대한 formal satisfaction guarantee로 해석하면 안 된다.

확인 위치: III Methodology; IV benchmark; V Discussion.

### 10. 접촉 순서를 시간표보다 유연하게: Logical Rules / SayTap

[Learning Quadruped Locomotion Policies using Logical Rules](https://arxiv.org/html/2107.10969v3), ICAPS2024 [P09]는 발 접촉 사건을 Reward Machine으로 정의한다. 현재 진행 상태를 관측에 넣어 어떤 사건에 보상을 줄지 명시한다. 우리처럼 순서 판정을 후하게 하고 싶을 때 “정해진 시각”과 “허용된 다음 사건”을 분리하는 참고다. 논문의 검증 gait를 gallop으로 확대해서는 안 된다.

[SayTap](https://arxiv.org/html/2306.07580v1), 2023 [P08]은 목표 foot-contact pattern 위반 비용을 직접 사용한다. 언어모델은 그 패턴 명령을 생성하는 역할이다. 동물형 동작을 만들기 위해 반드시 full motion imitation이 필요한 것은 아니라는 구현 사례다.

## 우리 조건별 대응표

표의 “직접”은 **같은 종류의 상태/조건을 보상화했다**는 뜻이다. 숫자·식·로봇·학습 설정이 동일하다는 뜻이 아니다. CSV 원본: [condition_matrix.csv](condition_matrix.csv).

| 우리 현재 조건 | 구현상 역할 | 가장 가까운 선행연구 | 판정·차이 |
|---|---|---|---|
| 위상별 접촉, 지지/스윙 일치 | dense contact/swing/joint reward | P01·P03·P04·P05·P08 | 직접 대응. 힘/속도 기반인지 binary contact인지 차이 |
| 착지 개별 순서 완화, 좌우 교환 | 약한 order reward; 승급 필수 아님 | P07·P09·P12 | 느슨한 패턴/사건 접근과 유사. 패턴을 너무 지우면 bound도 통과 가능 |
| 완전 공중기, 최소0.06→0.26s, 공중 주기 비율 | phase reward + event 측정 + success gate | P01·P04·P07·P16 | swing/contact·jump phase 근거 존재. 우리 시간·60% 기준 직접 검증은 아님 |
| 공중 이동18→80cm, 착지 확인 뒤 지급 | event reward + 거리 gate | P16·P17·P19 | jump landing/distance 보상 관련. 연속 gallop의 공중 이동80cm와 동일 실험 없음 |
| 앞다리 총하중25%/각8%, 지지·스윙·ROM 참여 | fore_use dense + leap 자격 + 승급 gate | P06·P01·P03 | 네 다리 사용 근거 강함. 정확한 앞 하중 임계값은 우리 설계 |
| 큰 thigh/calf 운동 및 최소ROM | 참조 joint/motion reward + success gate | P12·P26·P34 | 참조 기반 운동량 보상 관련. 클수록 항상 좋은 자연 gait 조건은 아님 |
| 앞/뒤 각 쌍 좌우 잔차·ROM·하중·지지 균형 | pair costs | P01·P06·P45 | 대칭/균등 사용 근거. phase offset 보존 필요. 네 종류 전체 동일 식은 없음 |
| yaw=0, 목표 방향 전진 | heading cost + progress reward | P01·P05·P08·P16 | 방향 task의 선례. yaw rate0과 absolute yaw0은 다름 |
| roll/pitch 진동 적을수록 점수 | angular-rate/자세 품질 reward | P05·P02·P24 | 직접 대응. 정상적인 phase 운동까지 없애지 않도록 규모 조정 필요 |
| 앞/뒤 독립 발폭26.98cm, 통과폭24–30cm | stance width reward + success gate | P04·P05·P23 | 발 배치/style constraint 선례. 어깨95%·고정 폭 수치는 미검증 |
| 앞발16cm/뒷발10cm 높이 초과 비용 | soft height costs + 비율 gate | P02·P04·P23·P24 | clearance/height 조절 선례. 최소 clearance와 최대 높이는 다른 조건 |
| 이륙vz·공중 균형·착지 하강·충격·미끄럼 | phase/event rewards + landing confirmation | P16·P17·P19·P24 | 직접/밀접 대응.3BW·하강1.2→1.8·80%착지 수치는 별도 검증 필요 |
| 관절각/변화율·모터 제한, 실제 초과 종료 | 필터 + soft costs + hard failure | P02·P22·P23·P24·P25 | reward와 물리 제약을 분리해야 함. 필터 빈도100%허용이 모터 제한 해제는 아님 |
| 성공도에 따라 속도·거리 자동승급 | curriculum algorithm | P16·P20·P21·P42 | 선행연구 있음.128episode·80%두 구간·60s는 우리 운영 설계 |
| 정지→일반보행→gallop,12s 전환기한 | startup controller / failure | P03·P12·P17·P20·P39 | 전환·스킬 유지와 관련. 동일한 startup 설계의 문헌 검증 아님 |

## “그대로 적용”을 판단할 때 필요한 구분

### 직접 조건 보상

P01·P02·P03·P04·P05·P06·P07·P08·P10·P16 등이 해당한다. 사람이 정한 접촉·자세·위치·phase·허용 구간이 보상 계산에 직접 들어간다. 우리 접근은 이 계열에 속한다.

### 동물 동작 추종 / 학습된 style reward

- [Peng et al., RSS2020](https://www.roboticsproceedings.org/rss16/p064.pdf) [P26]: joint pose/velocity·발 위치·root pose/velocity를 동물 참조와 비교한다.
- [AMP locomotion, IROS2022](https://xbpeng.github.io/projects/AMP_Locomotion/AMP_Locomotion_2022.pdf) [P28]: 개 mocap에서 discriminator style reward를 학습한다. canter를 strict gallop이라고 동일시하지 않았다.
- [Generalized Animal Imitator](https://arxiv.org/html/2310.01408v2) [P29], [monocular-video animal locomotion](https://www.nature.com/articles/s44182-025-00048-x) [P33], [Spatio-Temporal Motion Retargeting](https://arxiv.org/html/2404.11557v1) [P34] 등.

이 연구들은 동물형 동작을 보상하지만, “앞다리 하중25%”처럼 해석 가능한 조건을 독립적으로 적는 방법과는 다르다. 자연스러운 모양을 보완하는 대안이며, 임계조건·안전 보장을 대체하지 않는다. Peng2020을 gallop 성공 근거로 인용할 만큼 해당 동작을 확인하지 못했다.

### 비디오에서 reward 코드를 생성

[SDS – See it, Do it, Sorted](https://proceedings.mlr.press/v305/stamatopoulou25a.html) [P37], [MA-ROESL](https://arxiv.org/html/2505.08367v1) [P38]은 비디오 해석을 executable reward 작성/개선에 사용한다. SDS의 검증 스킬은 trot/bound/pace/hop이며 기존 SDS-Trot 자료가 곧 strict gallop 자료는 아니다. “동물 영상을 보아 만든 reward”와 “생체역학 조건의 타당성 검증”도 구분해야 한다.

### 목표 gait를 보상 외 구조에 넣는 연구

[AllGaits](https://arxiv.org/html/2411.04787) [P39]는 CPG coupling으로 gait를 지정하며 **style reward를 사용하지 않는다고 명시**한다. 좋은 비교 논문이지만 “우리처럼 조건을 직접 reward화했다”는 근거는 아니다.

[EMBC2021 Trot–Gallop energetic analysis](https://www.paperhost.org/proceedings/embs/EMBC21/files/0505.pdf) [P46]는 제목과 달리 gait를 reward로 만드는 사례가 아니다. reward는 속도 오차/토크 중심이며 좌우 토크 복제로 gait mode를 제한한다. 좌우 위상 지연을 무시하므로 논문의 gallop 명칭을 엄밀한 비대칭 gallop 분류로 가져오면 안 된다.

[CaT](https://arxiv.org/html/2403.18765) [P22]는 제약을 확률적 종료로 처리한다. [Not Only Rewards But Also Constraints](https://arxiv.org/html/2308.12517) [P23]는 task reward와 물리/style constraint를 분리한다. 우리 관절 보호는 이 축으로 비교해야 한다.

## 우리 설계에 대한 판단

### 이미 선행연구가 충분한 부분

1. **발 접촉·스윙/지지·위상 기반 보상:** P01/P03/P04/P05/P08. 동물 모방 데이터가 없어도 설계할 수 있다.
2. **네 다리의 일관된 사용:** P06의 관측/보상 ablation과 P01의 phase-conditioned symmetry.
3. **몸체 각속도·미끄럼·발 배치 안정성:** P05/P04/P24.
4. **도약 사건·착지 목표·단계 curriculum:** P16/P17/P20.
5. **관절 제한은 reward만으로 해결하지 않고 제약으로 분리:** P22/P23/P24/P25.

즉 지금 방향 자체가 근거 없는 접근은 아니다. 하지만 해당 논문들을 합친다고 우리 전체 설계의 수렴이나80cm 달성이 자동 보장되지는 않는다.

### 논문이 정해주지 않은 우리 선택값

- 80cm 공중 이동, 속도3m/s.
- 앞다리 총수직충격량25%, 각8%, ROM/스윙 참여70%.
- 발폭은 어깨95%=26.98cm, 앞/뒤 같은 값, 허용24–30cm.
- 앞/뒤 발 높이16/10cm 비용 기준.
- 공중기 포함 주기60%, 착지성공80%.
- raw 요청각 초과10%, 전체 필터 개입100%허용.
- 관절 실제 초과.01rad/60ms, 관절 중심 범위90% 및 target margin.
-128episode·80%두 창·50%미만 회귀·60sim초 적응.

이는 robot-specific/task-specific engineering choices다. 모터 스펙·물리 모델·측정 로그·ablation으로 정당화할 수는 있지만 “동물이 gallop하려면 꼭 만족해야 하는 수치”라고 쓰면 안 된다. 관절/모터의 공식 허용치는 설계 근거로 별도 확인해야 하며 논문 보상과 다르다.

### 조건끼리 충돌할 수 있는 부분

- **gallop과 과도한 대칭:** 좌우는 협응해야 하지만 일반적으로 같은 순간 같은 각도로 움직일 필요는 없다. 현재 `pair_joint_penalty`는 각 다리의 자기 phase 참조를 뺀 잔차를 비교하므로 이 점을 이미 고려한다. “현재 코드가 좌우 동시 움직임을 강제한다”는 진단은 부정확하다.
- **진동 억제와 추진:** roll/pitch 각속도를 과도하게 줄이면 안정적이지만 추진력이 약한 해를 선호할 수 있다. P05처럼 task·gait reference·stability를 함께 평가하고 phase에 필요한 움직임과 불필요한 진동을 구분할 필요가 있다.
- **폭과 공중 자세:** 접지 안정성과 공중 회수 자세가 항상 같은 발폭을 요구하는 것은 아니다. P04의 상태 의존 목표와 현재 air width 가중치0.3을 같이 검토할 근거가 있다.
- **스윙 최소 높이와 최대 높이 비용:** 발을 반드시 들어야 하는 보상과 앞/뒤 높이 제한은 동시에 충족 가능한 목표 궤적으로 정의해야 한다. clearance reward는 “높을수록 무조건 좋음”이 아니다.
- **거리와 gait 정체성:** 멀리 뛰고 빠르게 가도 bound·pronk·일회성 jump일 수 있다. 착지 순서를 완화하더라도 접촉 offset/전후 협응 같은 최소 특징과 사후 gait 분류는 남아 있어야 한다.
- **에너지/속도와 gallop:** [Fast and Efficient Locomotion](https://arxiv.org/html/2104.04644v2) [P40]은 A1에서 fly-trot가 더 효율적인 결과를 보여준다. 속도만 키우면 로봇이 반드시 gallop을 선택하는 것은 아니다.
- **hard gate와 reward:** 통과 조건을 더 많이 AND로 묶으면 아직 실패하는 조건을 알려주기는 쉽지만 보상 신호 자체를 늘려주지는 않는다. 특히 거리 reward 지급을 여러 사전 조건으로 잠그면 탐색 단계에서 유효 보상을 거의 못 볼 수 있다. 이는 현 프로젝트의 원인이라고 확정한 것이 아니라 P16의 local optimum 논의와 연결되는 검증 가설이다.

### 공중80cm 목표에 대한 해석

수평 속도가 공중에서 일정하게3m/s라면80cm에는 약0.267s가 필요하다. 이륙·착지의 COM 높이가 같고 중력을9.81m/s²로 단순화하면 이륙 수직속도 약1.31m/s, COM 상승 약8.7cm다. 이것은 **단순 탄도 일관성 계산**이며 모터·접촉력·착지 가능성 검증이 아니다. 발 높이와 COM 상승도 같은 양이 아니다.

[16 Ways to Gallop](https://arxiv.org/html/2503.13716) [P47]은 최적화 모델에서 공중기0/1/2회에 따른 gallop 계열을 분석한다.80cm는 보편적 gallop 정의가 아니라 우리가 원하는 도약 성능 목표다. 엄밀 보행 판정은 접촉 타이밍·위상·보행 주기를 별도로 봐야 한다.

## 후속 설계에 쓸 권고안

이번 조사는 설정이나 모델을 바꾸지 않았다. 다음 변경 시 아래 순서로 근거를 적용하는 것이 합리적이다.

1. **참조·관측·보상 위상을 먼저 일치시킨다.** P06은 위상 관측과 reward의 일치가 네 다리 사용에 영향을 준다는 직접 근거다. 현재 단계별 profile 전달 누락은 [구현 점검](implementation_audit.md)에서 별도 재현했다.
2. **gait 기준의 주축은 contact/force/foot velocity로 유지한다.** P01/P03/P04를 기준으로 발별 swing/stance 신호가 관절 참조·발 높이와 모순 없는지 확인한다.
3. **너무 엄밀한 순서 대신 최소한의 gait 정체성을 남긴다.** P07처럼 pair별 phase offset을 넓은 허용폭으로 주는 후보를 검토하되 실기·Go2 검증이 없는 최신 preprint라는 한계를 유지한다. 목표가 animal-like bound까지 허용하는지에 따라 판정 명칭을 구분한다.
4. **안전 범위와 스타일 선호를 분리한다.** 물리 관절/모터 한계는 지키고, 폭·진동·운동 진폭 같은 선호는 P02의 구간형 보상으로 과도한 경직을 줄이는 ablation 후보로 둔다.
5. **앞다리 편중 문제는 하중 하나보다 네 발의 반복 참여를 본다.** P06에 따라 phase 관측·swing/stance·contact를 확인하고, 현재의 load/ROM/participation을 서로 독립 로그로 평가한다.
6. **도약 성능은 이륙→공중→착지를 따로 채점하고 최종 거리 보상과 구분한다.** P16/P17처럼 필요한 추진과 안전 착지 신호를 유지한다. 실패/미착지 flight에 완성된 거리 성과를 부여해서는 안 된다.
7. **거리 단계와 속도 단계를 섞지 않고 검증한다.** P16/P20/P42가 curriculum 근거다. 현재9단계 설계는 이 방향과 맞지만, 문턱 수치·샘플 수·phase 변경의 적절성은 우리 데이터로 검증해야 한다.
8. **스타일이 여전히 동물과 다르면 모방 prior를 보완 축으로 고려한다.** P28/P29/P33/P34가 후보지만 현재 조건 보상과 별개의 접근이며 모방을 도입했다고 안전/거리 목표를 보장하지 않는다.

권장 ablation: 동일 부모·동일 예산·여러 seed에서 (A)현 reward, (B)접촉 보상 정리, (C)구간형 style, (D)최소 phase offset 추가를 한 번에 하나씩 비교한다. success뿐 아니라 공중거리 분포, 착지율, 접촉 위상, 네 발 impulse/참여, body yaw/rates, raw target/실제 관절 초과, filter 개입량을 함께 남긴다. 이 실험은 이번 조사에서 수행하지 않았다.

## 문서·출처 해석 주의

- 모든 숫자는 로봇 크기·질량·형태·PD/모터·지면·거리 측정 정의에 의존한다.
- 시뮬레이션에서 보상이 높아진 것과 실기 안전성은 다르다.
- 논문의 go2/Go1/A1/능동 척추 모델을 서로 같은 플랫폼으로 취급하지 않았다.
- barrier reward·constrained RL·termination·CPG coupling을 모두 “reward”라고 부르지 않았다.
- 학위 보고서[P13], 2족 기반[P14], 초록 한정[P15/P31], 비RL 분석[P47–P50]은 직접 Go2 gallop 실증과 따로 읽어야 한다.
- 구현 점검은 논문 결론이나 실제 학습 실패 인과 증거가 아니다. 현재 소스의 계산 경로 차이를 확인한 결과다.

