# 논문별 검토 기록

조사일2026-11-10. ID는 README·CSV·조건 대응표와 동일하다. 같은 종류의 보상을 사용했다는 것과 우리 수치/로봇/실험 전체가 동일하다는 것은 다르다. 초록 한정 자료는 상세 수식 근거로 사용하지 않는다.

## P01. Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds

- 시기/상태: 2024 최초 / v4 2026-02-13; arXiv
- 원문: [Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy Enables Free Gait Transitions at Varying Speeds](https://arxiv.org/html/2403.10723v4)
- 방법 분류: 직접 보상 + 위상 설계
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree Go2; 시뮬레이션·실기
- gallop 판정 범위: rotary/transverse gallop 포함
- 보상 또는 작동 원리: 스윙 중 지면반력, 지지 중 발 속도를 위상 마스크로 비용화; 같은 위상의 다리 관절 대칭성; 속도·높이·스윙 클리어런스·토크 변화
- 우리 조건과의 대응: 접촉/지지·스윙/좌우 협응/속도/몸체 높이/발 높이
- 차이/한계: Go2 EDU 명시 확인 못함; time reversal은 위상 재매핑; 서로 다른 위상의 양다리를 같은 각도로 묶는 보상이 아님; 80cm·앞 하중25%는 미제시; 철회된2510.10455와 중복
- 원문 위치: III-A Table I; III-D Eq(4)–(8), Table II; IV-D

## P02. A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards

- 시기/상태: 2024 preprint / ICRA 2025; v4 확인
- 원문: [A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards](https://arxiv.org/html/2409.15780v4)
- 방법 분류: 직접 보상
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: HOUND 계열; 시뮬레이션·실기
- gallop 판정 범위: gallop 실험 보고
- 보상 또는 작동 원리: 접촉 보행 변수·클리어런스·관절 위치/속도·전후 몸체 높이·명령 추종·몸체 운동의 허용 구간을 relaxed log barrier로 변환
- 우리 조건과의 대응: 접촉/발 높이/관절 범위/몸체 안정성/속도
- 차이/한계: 허용 구간의 soft reward로 하드웨어 제한 보장 아님; Go2 수치로 재보정 필요; 전체 조건을 정확히 복제한 것은 아님
- 원문 위치: III-A; III-B; III-C Table I

## P03. Discovery of skill-switching criteria for learning agile quadruped locomotion

- 시기/상태: 2025 preprint / Frontiers in Robotics and AI 2026-02-18
- 원문: [Discovery of skill-switching criteria for learning agile quadruped locomotion](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1697159/full)
- 방법 분류: 직접 보상 + 계층 스킬
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1; 시뮬레이션·실기
- gallop 판정 범위: trot/bound/gallop 스킬
- 보상 또는 작동 원리: 동물형 기준 발 접촉 패턴 일치, 발 배치 대칭성, 스윙/스탠스, 방향·높이·속도·토크·관절 속도; 상위 전환 기준 최적화
- 우리 조건과의 대응: 접촉/앞뒤 리듬/스윙/몸체/속도/보행 전환
- 차이/한계: symmetric foot placement는 네 발 평균의 몸체 중심 정렬로 우리 앞/뒤 발폭과 다름; 하중25%·각8%·80cm 아님; preprint와 출판본 동일 계열
- 원문 위치: 출판본 reward tables; preprint Table 1–2

## P04. Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

- 시기/상태: CoRL 2022 / PMLR 2023
- 원문: [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/html/2212.03238)
- 방법 분류: 직접 보상
- 확인 수준: 원문·공개 구현 확인
- 플랫폼/실증: Unitree Go1; 실기
- gallop 판정 범위: 주요 검증은 two-beat gaits; 엄밀한 비대칭 gallop 근거로 한정하지 않음
- 보상 또는 작동 원리: 위상별 접촉 힘/발 속도, Raibert 기반 발 배치와 stance width 명령, 스윙 높이·몸체 자세/높이·속도
- 우리 조건과의 대응: 접촉/발폭/발 높이/안정성/속도
- 차이/한계: 일정 좌우 거리만 보상하면 회전 시 충돌할 수 있어 상태 의존 목표 사용; 우리 26.98cm·앞뒤 독립 조건과 동일하지 않음
- 원문 위치: §3.1; Table 1; Appendix; corl_rewards.py
- 공개 자료: [링크](https://github.com/Improbable-AI/walk-these-ways)

## P05. Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion

- 시기/상태: Nature Machine Intelligence 2025
- 원문: [Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion](https://www.nature.com/articles/s42256-025-01065-z)
- 방법 분류: 직접 보상 + 생체역학 보행 선택
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1; 실기
- gallop 판정 범위: 이 검토에서 엄밀한 gallop 실증은 별도 확인하지 않음
- 보상 또는 작동 원리: 명령 접촉 상태 오차와 발 Cartesian 참조 오차, 접지 발 미끄럼, roll/pitch 각속도, 몸체 자세·높이·hip·토크/jerk; 상위 보행 선택에는 생체역학 지표
- 우리 조건과의 대응: 접촉/발 궤적/roll-pitch 진동 억제/안정성/전환
- 차이/한계: CoT·접촉력·기계적 일·보폭 변동 지표를 모두 하위 scalar reward라고 읽으면 안 됨; 보행 선택 목표와 구분
- 원문 위치: Methods: Locomotion policy Eq(5)–(9); Biomechanics gait transition metrics; Gait selection policy
- 공개 자료: [링크](https://github.com/ihcr/learning_to_adapt)

## P06. Learning Emergent Gaits with Decentralized Phase Oscillators: on the role of Observations, Rewards, and Feedback

- 시기/상태: ICRA 2024
- 원문: [Learning Emergent Gaits with Decentralized Phase Oscillators: on the role of Observations, Rewards, and Feedback](https://arxiv.org/html/2402.08662v2)
- 방법 분류: 직접 위상 보상 + ablation
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Mini Cheetah 계열; 이 비교는 시뮬레이션 ablation 중심
- gallop 판정 범위: 특정 엄밀 gallop을 강제하는 논문으로 분류하지 않음
- 보상 또는 작동 원리: 각 발의 위상에 따른 swing/stance 보상과 위상 관측·oscillator coupling의 역할 분리
- 우리 조건과의 대응: 네 다리 참여/편측 또는 두세 다리 사용 방지/위상 보상 관측 일치
- 차이/한계: 주요 근거는 phase observations와 rewards 조합의 balanced leg use; 앞 하중25%·각8%를 직접 보상한 것은 아님
- 원문 위치: OBSERVATION, REWARD, COUPLING ABLATION → Balanced Leg Use; Fig.3

## P07. S-Cheetah: A Novel Quadrupedal Robot with a 3-DOF Active Spine Learning Agile Locomotion

- 시기/상태: arXiv 2026-05
- 원문: [S-Cheetah: A Novel Quadrupedal Robot with a 3-DOF Active Spine Learning Agile Locomotion](https://arxiv.org/html/2605.27909)
- 방법 분류: 직접 보상
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 3-DOF 능동 척추 S-Cheetah; 시뮬레이션, 실기 향후 과제
- gallop 판정 범위: rotary G2 gallop 보고; 실기 증거 아님
- 보상 또는 작동 원리: 앞/뒤 각 쌍의 비영 접촉 위상차 Gaussian 보상; 전후 곱; 척추 운동의 위상·진폭·안전 초과 비용; 속도 curriculum
- 우리 조건과의 대응: 느슨한 순서에서 bound와 구별할 최소 위상차/큰 동작/단계 속도
- 차이/한계: 앞뒤 그룹 간 관계나 공중기 길이를 직접 강제하지 않음; 척추 보상은 rigid Go2에 적용 불가; 본문의 ipsilateral 용어와 front/rear pair 표현 혼재 주의
- 원문 위치: III-B; III-C1 Eq(2); III-C2 Eq(3)–(6); V

## P08. SayTap: Language to Quadrupedal Locomotion

- 시기/상태: arXiv 2023
- 원문: [SayTap: Language to Quadrupedal Locomotion](https://arxiv.org/html/2306.07580v1)
- 방법 분류: 직접 보상 + 패턴 명령
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1; 실기
- gallop 판정 범위: 이 조사에서 gallop 실증은 확인하지 않음
- 보상 또는 작동 원리: 목표 접촉 binary pattern 위반 비용, 명령 속도·yaw rate·action rate·joint acceleration 등
- 우리 조건과의 대응: 느슨한 목표 접촉 패턴/발 참여/속도
- 차이/한계: LLM은 패턴 명령 생성; reward에는 명시적 접촉 오차 사용; 생체역학 전체 조건/거리 보상은 아님
- 원문 위치: Appendix B: Reward Functions

## P09. Learning Quadruped Locomotion Policies using Logical Rules

- 시기/상태: 2021 최초 / ICAPS 2024 v3
- 원문: [Learning Quadruped Locomotion Policies using Logical Rules](https://arxiv.org/html/2107.10969v3)
- 방법 분류: 논리 사건을 보상 기계로 변환
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 시뮬레이션
- gallop 판정 범위: 6종 보행; gallop 검증으로 단정하지 않음
- 보상 또는 작동 원리: 발 접촉 사건과 진행 단계를 Reward Machine으로 정의하고 automaton 상태를 관측에 추가
- 우리 조건과의 대응: 착지 순서/사건 기반 보상/시간표 완화
- 차이/한계: 특정 고정 타이밍 대신 논리적 진행을 쓰지만 시간·거리·하중 수치를 모두 검증한 것은 아님
- 원문 위치: Reward Machines; gait rules; experiments

## P10. Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications

- 시기/상태: arXiv 2026-07
- 원문: [Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications](https://arxiv.org/html/2607.00442)
- 방법 분류: 조건·시간 논리의 robustness를 직접 보상화
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Barkour vb; MJX 시뮬레이션
- gallop 판정 범위: walk-trot/trot/bound; gallop 아님
- 보상 또는 작동 원리: 안전·actuation·명령 추종·접촉 동기화·지지/공중 사건을 STL로 적고 30-step 창의 smooth robustness로 보상
- 우리 조건과의 대응: 조건 자체를 보상화/지속시간·접촉·안정성·제약
- 차이/한계: expert rollout으로 임계값 보정; 시뮬레이션·preprint; 조건 만족을 수학적으로 보장한 것 아님; Go2 gallop 검증 아님
- 원문 위치: III Methodology; IV benchmark; V Discussion

## P11. From LLM-Generated Specifications to Learned Quadruped Locomotion

- 시기/상태: arXiv 2026-09
- 원문: [From LLM-Generated Specifications to Learned Quadruped Locomotion](https://arxiv.org/html/2609.07111)
- 방법 분류: 생성된 시간 논리의 보상화
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Barkour 계열; 시뮬레이션
- gallop 판정 범위: walk/trot/bound 범위
- 보상 또는 작동 원리: LLM이 기호 STL 후보 작성, expert rollout으로 임계값 추정/후보 거부, robustness 기반 dense reward
- 우리 조건과의 대응: 조건의 수치화/시간 관계/실패 해석
- 차이/한계: P10의 후속 계열이며 독립 실증으로 과대 계산 금지; LLM이 정한 수치를 그대로 믿는 방식 아님; gallop/실기 검증 아님
- 원문 위치: III–IV; reference[34]

## P12. Learning Free Gait Transition for Quadruped Robots via Phase-Guided Controller

- 시기/상태: RA-L 2022 (accepted 2021)
- 원문: [Learning Free Gait Transition for Quadruped Robots via Phase-Guided Controller](https://arxiv.org/pdf/2201.00206)
- 방법 분류: 설계된 위상 참조의 모방 보상
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: Black Panther; 실기
- gallop 판정 범위: walk/trot/pace/bound; gallop 실증으로 단정하지 않음
- 보상 또는 작동 원리: 독립 네 발 phase를 관측·보행 생성기에 연결해 참조 발 운동을 추종하는 RL
- 우리 조건과의 대응: 위상 참조/발 궤적/전환/관측
- 차이/한계: 동물 실측 motion 자체가 아닌 설계 참조; 보상만으로 전부 자발 발견하는 방식 아님
- 원문 위치: Phase-Guided Controller; reward/phase-gait training sections

## P13. Quadrupedal Locomotion via Periodic Reward Composition and Deep Reinforcement Learning

- 시기/상태: UC Berkeley MS report, Spring 2021; 비학술지
- 원문: [Quadrupedal Locomotion via Periodic Reward Composition and Deep Reinforcement Learning](https://john-s-li.github.io/docs/Johnathon%20Li%20-%20MS%20Report%20Final%20Draft-compressed.pdf)
- 방법 분류: 직접 주기 보상
- 확인 수준: 원문 방법·결과 확인
- 플랫폼/실증: 주요 성공 결과는 planar Half-Cheetah; 3D/실기 향후 과제
- gallop 판정 범위: 보고서가 gallop이라 칭하는 planar 결과
- 보상 또는 작동 원리: 주기별 foot force/velocity, 목표 전진·높이·자세·토크·가속/행동 변화
- 우리 조건과의 대응: 주기 보상/공중·지지 구분
- 차이/한계: 3D Go2 gallop 실증 아님; gallop 20M step/2–3일 수치는 당시 PC·환경 기준으로 우리 iteration 시간 추정에 사용 불가
- 원문 위치: §5–8; Table 2; Figs8–11

## P14. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition

- 시기/상태: ICRA 2021
- 원문: [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/pdf/2011.01387v2)
- 방법 분류: 직접 주기 보상; 2족 기초 연구
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Cassie; 2족 실기
- gallop 판정 범위: 4족 gallop 증거 아님
- 보상 또는 작동 원리: stance/swing 위상별 발 힘과 속도, 확률적 phase boundary
- 우리 조건과의 대응: 위상 마스크 보상 설계의 기초
- 차이/한계: 4족으로 이전할 논리 근거이며 Go2 전체 조건 실증 아님
- 원문 위치: periodic reward composition formulation

## P15. Footstep reward for energy-efficient quadruped gait generation and transition through deep reinforcement learning

- 시기/상태: online 2024-12-24 / Advanced Robotics 2025 issue
- 원문: [Footstep reward for energy-efficient quadruped gait generation and transition through deep reinforcement learning](https://doi.org/10.1080/01691864.2024.2442718)
- 방법 분류: 직접 발걸음 보상 후보
- 확인 수준: 초록·서지 확인; 전체 방법 미확인
- 플랫폼/실증: 4족 로봇; 상세 실증 범위 이 조사에서 미확인
- gallop 판정 범위: 제목·초록만으로 엄밀 gallop 성공 판정 보류
- 보상 또는 작동 원리: footstep reward와 energy-efficient gait generation/transition
- 우리 조건과의 대응: 발걸음 보상의 추가 문헌 후보
- 차이/한계: 수식·reward exact definition·실기 범위 미검증; 핵심 결론 근거로 사용하지 않음
- 원문 위치: publisher abstract; DOI metadata

## P16. Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design

- 시기/상태: arXiv 2024; IEEE Robotics & Automation Magazine DOI 확인
- 원문: [Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design](https://arxiv.org/html/2401.16337)
- 방법 분류: 직접 도약·착지 보상 + curriculum
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree Go1; 시뮬레이션·실기 (60cm 비교 및 90cm 전방 점프 보고)
- gallop 판정 범위: 단발/목표 지점 도약; 주기 gallop 아님
- 보상 또는 작동 원리: 서기/비행/착지 phase별 높이·이륙·착지 위치/방향, 점프 사건; 제자리→거리→장애물 curriculum
- 우리 조건과의 대응: 이륙/공중/도약 거리/안정 착지/점진 목표
- 차이/한계: 우리 80cm 주기 gallop의 실현 가능성을 입증하지 않음; 걷거나 기어서 목표로 가는 local optimum을 별도 지적
- 원문 위치: III-A/B/C; IV-C; Table I; V-E

## P17. Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot

- 시기/상태: arXiv 2024
- 원문: [Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot](https://arxiv.org/pdf/2404.15096)
- 방법 분류: 직접 이륙 보상 + 단계 학습
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: Mini Cheetah Vision; 실기
- gallop 판정 범위: running jump, 연속 gallop 아님
- 보상 또는 작동 원리: force 관련 dense jump shaping과 목표 liftoff velocity sparse reward; walk/jump 섞어 망각 완화
- 우리 조건과의 대응: 이륙 추진/힘 분포/보행에서 도약/안정성과 실기
- 차이/한계: 표 추출 부호가 모호한 힘 std 식을 그대로 코드에 옮기지 말 것; PD/impedance mismatch가 별도 핵심
- 원문 위치: jump reward table; training stages; impedance analysis

## P18. Robust Quadruped Jumping via Deep Reinforcement Learning

- 시기/상태: 2020 preprint / IROS 2021
- 원문: [Robust Quadruped Jumping via Deep Reinforcement Learning](https://arxiv.org/html/2011.07089v3)
- 방법 분류: 최적화 참조 + residual RL
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 로봇; sim-to-real jump
- gallop 판정 범위: jumping; gallop 아님
- 보상 또는 작동 원리: trajectory optimization reference tracking과 residual action, robust landing
- 우리 조건과의 대응: 도약/착지/제약/참조 가능성
- 차이/한계: reference-free animal condition reward와 다름; 단발 점프 결과를 연속 gallop으로 전용 불가
- 원문 위치: method/trajectory optimization and RL controller

## P19. Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots

- 시기/상태: 2025 preprint / v3 2026
- 원문: [Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots](https://arxiv.org/html/2507.16481v3)
- 방법 분류: 모델 기반 guidance + 직접 jump 보상/제약
- 확인 수준: 원문 방법 확인; 말미 일부 추출 한도
- 플랫폼/실증: 4족 3D 점프; 자세한 하드웨어 수치 채택 안 함
- gallop 판정 범위: 전방·측방·회전 점프; gallop 아님
- 보상 또는 작동 원리: 착지 목표·이륙 twist·자세, 관절 q/dq/torque·마찰·unilateral feasibility
- 우리 조건과의 대응: 이륙/착지/거리/방향/관절 모터 제약
- 차이/한계: model guidance가 있어 동일 학습 조건 아님; 논문 식을 hard guarantee로 해석 금지
- 원문 위치: guided policy/optimization constraints/reward sections

## P20. Learning Agile Motor Skills on Quadrupedal Robots using Curriculum Learning

- 시기/상태: 2021 저자 공개 원문
- 원문: [Learning Agile Motor Skills on Quadrupedal Robots using Curriculum Learning](https://faculty.cc.gatech.edu/~sha9/projects/tang2021learning.pdf)
- 방법 분류: 직접 task 보상 + curriculum
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 locomotion/jumping
- gallop 판정 범위: gallop 전용 증거 아님
- 보상 또는 작동 원리: task·environment 난이도 curriculum, 초기 탐색 후 landing safety penalty 도입, 성취도 기준 난이도
- 우리 조건과의 대응: 단계 학습/도약/안정성 보상 충돌 관리
- 차이/한계: 보상 구성과 제약은 단계별로 달라질 수 있음; 15,500회 내80cm 보장 근거 아님
- 원문 위치: curriculum learning/reward design/experiments

## P21. Learning agility and adaptive legged locomotion via curricular hindsight reinforcement learning

- 시기/상태: Scientific Reports 14:28089, 2024-11-15
- 원문: [Learning agility and adaptive legged locomotion via curricular hindsight reinforcement learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/)
- 방법 분류: curriculum + hindsight RL
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 locomotion; 실기 연구
- gallop 판정 범위: gallop 조건 전체 실증으로 채택 안 함
- 보상 또는 작동 원리: 명령/난이도 curriculum과 hindsight command relabeling
- 우리 조건과의 대응: 속도·난이도 적응/실패 표본 활용
- 차이/한계: Scientific Reports 논문; Nature Communications와 혼동 금지; 우리 자동승급 숫자 근거 아님
- 원문 위치: CHRL method/curriculum sections

## P22. CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning

- 시기/상태: CoRL 2024
- 원문: [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://arxiv.org/html/2403.18765)
- 방법 분류: 제약을 확률적 종료로 처리
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 locomotion; 실기 연구
- gallop 판정 범위: gallop 자체를 정의하는 보상 아님
- 보상 또는 작동 원리: 제약 위반도에 비례한 termination probability로 feasible motion 유도
- 우리 조건과의 대응: 관절/동역학/접촉 제한
- 차이/한계: 우리 .01rad·60ms 결정론 종료와 다름; 생물학적 조건을 직접 dense reward로 넣은 사례와 별도
- 원문 위치: constraint formulation; soft termination probability

## P23. Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion

- 시기/상태: 2023 preprint / 2024 version
- 원문: [Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion](https://arxiv.org/html/2308.12517)
- 방법 분류: constrained RL
- 확인 수준: 원문 방법 확인; 말미 일부 추출 한도
- 플랫폼/실증: 4족 locomotion; 실기 연구
- gallop 판정 범위: gallop 전용 근거 아님
- 보상 또는 작동 원리: 관절 위치/속도/토크·몸체 접촉 물리 제약, 발 높이/클리어런스/대칭 style constraints; IPO
- 우리 조건과의 대응: 관절/발 높이/좌우/안전 조건
- 차이/한계: 제약 비용과 task reward를 분리; 모든 조건을 하나의 가중합 reward로 더하는 방식과 다름
- 원문 위치: physical constraints/style constraints/constrained optimization sections

## P24. Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion

- 시기/상태: 2023 preprint / 2024 연구
- 원문: [Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion](https://arxiv.org/html/2312.17507v1)
- 방법 분류: 직접 안정성 보상 + 모터 operating constraint
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 고속 4족 플랫폼; 실기
- gallop 판정 범위: reward table gait는 diagonal trot; gallop 증거 아님
- 보상 또는 작동 원리: torque-speed operating region, foot landing speed/slip/clearance, 몸체 orientation, 관절·행동 smoothness
- 우리 조건과의 대응: 모터 제약/착지/미끄럼/안정성
- 차이/한계: Go2와 모터 스펙 다름; 출력각 제한만으로 torque-speed/접촉 안정성 보장 불가
- 원문 위치: actuator constraints; reward table

## P25. Guided Constrained Policy Optimization for Dynamic Quadrupedal Robot Locomotion

- 시기/상태: RA-L 2020
- 원문: [Guided Constrained Policy Optimization for Dynamic Quadrupedal Robot Locomotion](https://www.robots.ox.ac.uk/~mobile/drs/Papers/2020RAL_gangapurwala.pdf)
- 방법 분류: guided constrained RL
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: 동적 4족 locomotion; sim-to-real 연구
- gallop 판정 범위: gallop 전용 실증으로 채택 안 함
- 보상 또는 작동 원리: guidance와 constrained policy optimization 결합
- 우리 조건과의 대응: 참조 가이드/실제 동역학 제한
- 차이/한계: 우리 조건 그대로 dense reward 사례가 아닌 안전한 학습 구조 비교
- 원문 위치: guided constrained optimization formulation

## P26. Learning Agile Robotic Locomotion Skills by Imitating Animals

- 시기/상태: RSS 2020
- 원문: [Learning Agile Robotic Locomotion Skills by Imitating Animals](https://www.roboticsproceedings.org/rss16/p064.pdf)
- 방법 분류: 동물 참조 추종 보상
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Laikago; 실기
- gallop 판정 범위: 본문 gallop 확인 못함; pace/backward trot/hop/turn 등
- 보상 또는 작동 원리: 관절 pose/velocity, 발 위치, root pose/velocity reference imitation + task
- 우리 조건과의 대응: 자연스러운 관절/발 궤적/몸체/리듬
- 차이/한계: 동물 조건을 각자 hand-crafted threshold로 적은 방식과 다름; gallop 성공 논문이라고 확대 금지
- 원문 위치: imitation reward and motion retargeting sections

## P27. DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills

- 시기/상태: ACM TOG 2018
- 원문: [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/html/1804.02717v3)
- 방법 분류: 참조 동작 추종 보상
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 물리 기반 character simulation
- gallop 판정 범위: quadruped 동작 포함; Go2 실기 아님
- 보상 또는 작동 원리: pose/velocity/end-effector/COM reference reward + task
- 우리 조건과의 대응: 관절·발·중심 운동의 모방
- 차이/한계: 생물학 조건을 독립 threshold로 검증한 논문 아님; sim character 연구
- 원문 위치: imitation objective/reward formulation

## P28. Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions

- 시기/상태: IROS 2022
- 원문: [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://xbpeng.github.io/projects/AMP_Locomotion/AMP_Locomotion_2022.pdf)
- 방법 분류: 학습된 adversarial style reward
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1; 실기
- gallop 판정 범위: canter 등; strict gallop으로 동일시하지 않음
- 보상 또는 작동 원리: 개의 mocap에서 discriminator style reward 학습 + 간단한 task reward
- 우리 조건과의 대응: 동물 리듬/자연스러운 전신 모션
- 차이/한계: 명시적 발폭/하중/80cm 조건의 논리적 만족 보장 아님; 하드웨어 안전 조건 별도 필요
- 원문 위치: motion prior/style reward; experimental gaits
- 공개 자료: [링크](https://xbpeng.github.io/projects/AMP_Locomotion/)

## P29. Generalized Animal Imitator: Agile Locomotion with Versatile Motion Prior

- 시기/상태: 2023 preprint / CoRL 2024, PMLR 2025
- 원문: [Generalized Animal Imitator: Agile Locomotion with Versatile Motion Prior](https://arxiv.org/html/2310.01408v2)
- 방법 분류: versatile motion prior + task
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 animal imitation; 실기 연구
- gallop 판정 범위: 이 보고서에서 개별 엄밀 gallop 판정 보류
- 보상 또는 작동 원리: 다양한 reference style prior와 task 기능을 결합·스케줄링
- 우리 조건과의 대응: 다양한 동물 모션/속도 task와 스타일 양립
- 차이/한계: 스타일 reward가 암묵적이라 모든 명시 조건 충족 근거 아님
- 원문 위치: versatile motion prior/reward scheduling sections

## P30. Learning Highly Dynamic Behaviors for Quadrupedal Robots

- 시기/상태: arXiv 2024
- 원문: [Learning Highly Dynamic Behaviors for Quadrupedal Robots](https://arxiv.org/html/2402.13473v1)
- 방법 분류: 동작 모방 + latent skill
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Tencent quadruped 연구
- gallop 판정 범위: 이 조사에서 gallop 명시 확인 못함
- 보상 또는 작동 원리: imitation, discrete latent prior, early termination, prioritized sampling
- 우리 조건과의 대응: 동적 동작/모방/실패 억제
- 차이/한계: Lifelike 계열과 관련되므로 완전히 독립적인 전체 조건 실증으로 세지 않음
- 원문 위치: motion imitation/latent prior/training sections

## P31. Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models

- 시기/상태: Nature Machine Intelligence 2024
- 원문: [Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models](https://www.nature.com/articles/s42256-024-00861-3)
- 방법 분류: generative prior + RL
- 확인 수준: 초록·공식 소개 확인; 전체 Methods 미확인
- 플랫폼/실증: Tencent quadruped; 실기 연구
- gallop 판정 범위: 개별 gallop 조건·수식 확인 보류
- 보상 또는 작동 원리: 학습한 동작 prior와 RL로 다양한 동적 행동
- 우리 조건과의 대응: 전신 자연스러움/기술 재사용
- 차이/한계: 출판 페이지 추출이 abstract/reference 위주여서 exact reward 근거로 사용 안 함
- 원문 위치: abstract; official code availability
- 공개 자료: [링크](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play)

## P32. Imitation and Adaptation Based on Consistency: A Quadruped Robot Imitates Animals from Videos Using Deep Reinforcement Learning

- 시기/상태: arXiv 2022
- 원문: [Imitation and Adaptation Based on Consistency: A Quadruped Robot Imitates Animals from Videos Using Deep Reinforcement Learning](https://arxiv.org/html/2203.05973)
- 방법 분류: 영상 모방 + 명시 consistency 보상
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 robot imitation
- gallop 판정 범위: 이 검토에서 strict gallop 실기 판정 보류
- 보상 또는 작동 원리: 영상 기반 contact/stamp consistency와 motion direction consistency; reference adaptation
- 우리 조건과의 대응: 접촉/방향 추종/동물 스타일의 부분 조건화
- 차이/한계: mocap 완전 추종도 순수 조건 보상도 아닌 혼합; 우리 임계값 근거 아님
- 원문 위치: consistency/reward/adaptation sections

## P33. Learning aggressive animal locomotion skills for quadrupedal robots solely from monocular videos

- 시기/상태: npj Robotics 2025
- 원문: [Learning aggressive animal locomotion skills for quadrupedal robots solely from monocular videos](https://www.nature.com/articles/s44182-025-00048-x)
- 방법 분류: 영상 기반 AMP + task + regularization
- 확인 수준: 원문 Methods 확인
- 플랫폼/실증: 4족 sim-to-real 동물 모방
- gallop 판정 범위: 세부 엄밀 gallop 판정은 이 비교에서 채택하지 않음
- 보상 또는 작동 원리: 단안 영상에서 motion prior 구성, adversarial style/task/regularization rewards
- 우리 조건과의 대응: 자연스러운 운동/관절·몸체/동물 데이터 활용
- 차이/한계: 생체역학 임계조건을 전부 직접 써넣는 방식과 다름
- 원문 위치: Methods reward Eq(24)
- 공개 자료: [링크](https://github.com/arclab-hku/Imitation_from_video)

## P34. Spatio-Temporal Motion Retargeting for Quadruped Robots

- 시기/상태: arXiv 2024
- 원문: [Spatio-Temporal Motion Retargeting for Quadruped Robots](https://arxiv.org/html/2404.11557v1)
- 방법 분류: 공간·시간 retargeting + 추종 RL
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree Go1 / AlienGo; 실기 포함
- gallop 판정 범위: gallop 조건의 직접 보상 증거로 쓰지 않음
- 보상 또는 작동 원리: reference 시간/공간을 물리 타당성에 맞게 조절 후 추종
- 우리 조건과의 대응: 공중기/몸체 크기/관절 가능 범위/동작 이전
- 차이/한계: 동물 stride time·jump distance를 같은 숫자로 복사하면 안 되는 근거; task별 검증 필요
- 원문 위치: spatio-temporal retargeting; policy learning

## P35. Learning and Reusing Quadruped Robot Movement Skills from Biological Dogs for Higher-Level Tasks

- 시기/상태: Sensors 24(1):28; online 2023 / issue 2024
- 원문: [Learning and Reusing Quadruped Robot Movement Skills from Biological Dogs for Higher-Level Tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC10780440/)
- 방법 분류: 개 mocap 모방 + latent reuse
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 skill imitation
- gallop 판정 범위: 이 조사에서 strict gallop 실기 판정 보류
- 보상 또는 작동 원리: 관절·몸체 mocap reference reward, VAE latent representation
- 우리 조건과의 대응: 동물 관절/몸체/기술 재사용
- 차이/한계: 공중80cm·앞다리 최소하중 등 명시 조건 세트가 아님
- 원문 위치: §2.6 rewards; latent skill reuse

## P36. Learning Robust, Agile, Natural Legged Locomotion Skills in the Wild

- 시기/상태: arXiv 2023 v3
- 원문: [Learning Robust, Agile, Natural Legged Locomotion Skills in the Wild](https://arxiv.org/html/2304.10888v3)
- 방법 분류: motion prior + terrain task RL
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 자연스러운 야외 locomotion
- gallop 판정 범위: 개별 엄밀 gallop 판정 보류
- 보상 또는 작동 원리: adversarial style와 terrain/task objective 결합
- 우리 조건과의 대응: 자연스러운 gait와 환경 강건성
- 차이/한계: 특정 생체역학 조건을 독립 scalar reward로 모두 적은 사례 아님
- 원문 위치: style prior/task reward/terrain training sections

## P37. SDS – See it, Do it, Sorted: Quadruped Skill Synthesis from Single Video Demonstration

- 시기/상태: CoRL 2025 / PMLR 305
- 원문: [SDS – See it, Do it, Sorted: Quadruped Skill Synthesis from Single Video Demonstration](https://proceedings.mlr.press/v305/stamatopoulou25a.html)
- 방법 분류: 영상에서 executable reward 합성
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: Go1/ANYmal; 네 보행 스킬 연구
- gallop 판정 범위: trot/bound/pace/hop; gallop 증거 없음
- 보상 또는 작동 원리: VLM이 영상에서 reward code 작성, 학습 영상 feedback으로 반복 개선
- 우리 조건과의 대응: 동물 모양/접촉/속도 조건을 코드 reward로 바꾸는 자동화
- 차이/한계: 정해진 gallop 조건을 과학적으로 검증하는 논문과 다름; 기존 SDS-Trot 자료를 gallop이라고 분류하면 안 됨
- 원문 위치: method loop; reward generation; gait experiments

## P38. MA-ROESL: Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos

- 시기/상태: arXiv 2025
- 원문: [MA-ROESL: Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos](https://arxiv.org/html/2505.08367v1)
- 방법 분류: 영상 기반 reward 생성·최적화
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: robot skill learning; 이 비교에서 하드웨어 수치 채택 안 함
- gallop 판정 범위: strict gallop 검증으로 채택하지 않음
- 보상 또는 작동 원리: motion-aware reward 생성·offline 평가·online fine-tuning
- 우리 조건과의 대응: 비디오에서 모션 특성을 reward로 자동 변환
- 차이/한계: 생물학적 조건·보상 수치의 타당성 보장 아님; 검증/ablation 필요
- 원문 위치: motion-aware reward optimization method

## P39. AllGaits: Learning All Quadruped Gaits and Transitions

- 시기/상태: arXiv 2024
- 원문: [AllGaits: Learning All Quadruped Gaits and Transitions](https://arxiv.org/html/2411.04787)
- 방법 분류: CPG coupling/행동 매개변수화; style reward 없음
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree Go1; 실기
- gallop 판정 범위: 9종 보행 및 전환
- 보상 또는 작동 원리: gait는 CPG coupling에; reward는 velocity tracking, 불필요한 몸체 운동·power 등 task 중심
- 우리 조건과의 대응: 전환/위상/부족한 리듬의 구조적 가이드 대안
- 차이/한계: 본문이 style reward를 쓰지 않는다고 명시; 조건을 그대로 reward로 넣은 긍정 사례로 집계 불가
- 원문 위치: CPG-RL framework/reward function; experiments

## P40. Fast and Efficient Locomotion via Learned Gait Transitions

- 시기/상태: CoRL 2021 / PMLR 2022
- 원문: [Fast and Efficient Locomotion via Learned Gait Transitions](https://arxiv.org/html/2104.04644v2)
- 방법 분류: 속도·에너지 기반 emergent gait
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1; 실기
- gallop 판정 범위: 여러 gait 비교; fly-trot가 자발적·효율적
- 보상 또는 작동 원리: task speed/energy/survival과 gait parameter·WBC 계층
- 우리 조건과의 대응: 속도 curriculum/에너지/모션 가능성 판단
- 차이/한계: 속도만 높이면 gallop이 된다는 근거가 아님; 해당 로봇/실험에서 gallop 강제가 더 비효율적
- 원문 위치: gait transitions; energy comparison; gallop discussion

## P41. Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots

- 시기/상태: CoRL 2021 / PMLR 2022
- 원문: [Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots](https://proceedings.mlr.press/v164/fu22a/fu22a.pdf)
- 방법 분류: 에너지 최적화 emergent gait
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: legged robot locomotion
- gallop 판정 범위: 특정 gallop 조건 세트 강제 아님
- 보상 또는 작동 원리: 에너지 비용과 locomotion task로 gait가 출현
- 우리 조건과의 대응: 효율/속도/자발 보행
- 차이/한계: 명시적 gallop 스타일 보상과 다른 계열; 결과를 Go2 80cm 증거로 전용 불가
- 원문 위치: energy reward; gait emergence experiments

## P42. Rapid Locomotion via Reinforcement Learning

- 시기/상태: RSS 2022
- 원문: [Rapid Locomotion via Reinforcement Learning](https://arxiv.org/html/2205.02824v1)
- 방법 분류: 속도 task + adaptive curriculum
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 고속 4족 locomotion; 실기
- gallop 판정 범위: gallop 특정 조건을 보장하는 연구 아님
- 보상 또는 작동 원리: velocity reward와 Box/Grid adaptive command curriculum
- 우리 조건과의 대응: 성공도에 따른 속도 범위 확대
- 차이/한계: 성공률 문턱/환경 구성은 우리와 다름; 목표 gait 별도 정의 필요
- 원문 위치: command curriculum; reward; speed experiments

## P43. Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion

- 시기/상태: arXiv 2024
- 원문: [Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion](https://arxiv.org/html/2403.20001)
- 방법 분류: adaptive energy regularization
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 locomotion
- gallop 판정 범위: 직접 gallop 템플릿 보상 아님
- 보상 또는 작동 원리: motion reward와 에너지 regularization 크기 적응
- 우리 조건과의 대응: 효율/명령 속도에 따른 전환
- 차이/한계: 조건을 명시 reward로 복사한 선행연구로 분류하지 않음
- 원문 위치: adaptive energy reward formulation

## P44. DeepTransition: Viability Leads to the Emergence of Gait Transitions in Learning Anticipatory Quadrupedal Locomotion Skills

- 시기/상태: 2023 preprint / Nature Communications 15:3073, 2024 계열
- 원문: [DeepTransition: Viability Leads to the Emergence of Gait Transitions in Learning Anticipatory Quadrupedal Locomotion Skills](https://arxiv.org/html/2306.07419)
- 방법 분류: CPG + viability task RL
- 확인 수준: 원문 방법 확인; 말미 일부 추출 한도
- 플랫폼/실증: 4족 locomotion/gap crossing
- gallop 판정 범위: walk-trot/trot-pronk 중심; strict gallop 증거 아님
- 보상 또는 작동 원리: viability/terrain task와 CPG 제어로 전환 출현
- 우리 조건과의 대응: 갭 도약/전환/몸체 생존성
- 차이/한계: 의도한 gallop contact를 직접 보상화한 사례와 다름; 여러 제목/버전 동일 계열
- 원문 위치: viability objective; transition and gap experiments

## P45. Leveraging Symmetry in RL-based Legged Locomotion Control

- 시기/상태: IROS 2024; arXiv v3 확인
- 원문: [Leveraging Symmetry in RL-based Legged Locomotion Control](https://arxiv.org/html/2403.17320v3)
- 방법 분류: 대칭 데이터 증강/regularization/구조
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: legged locomotion
- gallop 판정 범위: gallop footfall 증거로 쓰지 않음
- 보상 또는 작동 원리: 대칭 변환을 data augmentation·mirror loss·equivariant control에 사용
- 우리 조건과의 대응: 좌우 편중 완화/학습 효율
- 차이/한계: 대칭이 반드시 같은 시간 양다리 각도 일치를 뜻하지 않음; P01의 gait-specific reward와 구별
- 원문 위치: symmetry methods/experiments

## P46. Deep Reinforcement Learning with Gait Mode Specification for Quadrupedal Trot-Gallop Energetic Analysis

- 시기/상태: EMBC 2021
- 원문: [Deep Reinforcement Learning with Gait Mode Specification for Quadrupedal Trot-Gallop Energetic Analysis](https://www.paperhost.org/proceedings/embs/EMBC21/files/0505.pdf)
- 방법 분류: 출력 복제 제약; reward는 task/torque
- 확인 수준: 원문 PDF 방법 확인
- 플랫폼/실증: MuJoCo 4족 시뮬레이션
- gallop 판정 범위: 저자 gallop 명칭; 좌우 phase delay 무시한 대칭 모드
- 보상 또는 작동 원리: 속도 오차+torque squared; gallop mode는 좌우 토크를 같게 복사
- 우리 조건과의 대응: 모션 구조를 제한하는 다른 접근
- 차이/한계: gallop 조건을 reward로 넣은 논문이 아님; 비대칭 gallop과 bound의 구별을 그대로 따를 수 없음; 실기 아님
- 원문 위치: gait mode specification; reward formula; discussion of left-right delay

## P47. 16 Ways to Gallop: Energetics and Body Dynamics of High-Speed Quadrupedal Gaits

- 시기/상태: IROS 2025
- 원문: [16 Ways to Gallop: Energetics and Body Dynamics of High-Speed Quadrupedal Gaits](https://arxiv.org/html/2503.13716)
- 방법 분류: trajectory optimization; RL 아님
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: A1 동역학 모델/시뮬레이션
- gallop 판정 범위: gallop 유형·0/1/2 공중기 분류
- 보상 또는 작동 원리: 최적화로 gait energy/body dynamics 비교
- 우리 조건과의 대응: gallop 분류/공중기/모션 가능성
- 차이/한계: 80cm가 보편적인 gallop 필요조건이 아님; RL reward 실증으로 인용 불가
- 원문 위치: gallop classification/optimization/results

## P48. Breaking Symmetries Leads to Diverse Quadrupedal Gaits

- 시기/상태: RA-L 2024; arXiv v3
- 원문: [Breaking Symmetries Leads to Diverse Quadrupedal Gaits](https://arxiv.org/html/2303.04857v3)
- 방법 분류: 이론·hybrid dynamics/최적화; RL 아님
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: 4족 모델
- gallop 판정 범위: 대칭 깨짐과 여러 gait 분류
- 보상 또는 작동 원리: 대칭성을 보존·깨뜨려 주기 해의 가족 분석
- 우리 조건과의 대응: 비대칭 gallop/대칭 정규화의 범위
- 차이/한계: 직접 reward 논문 아님; P01의 이론 배경
- 원문 위치: symmetry framework/gait analysis

## P49. Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion

- 시기/상태: Scientific Reports 2025-01-13
- 원문: [Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion](https://www.nature.com/articles/s41598-024-84060-5)
- 방법 분류: bio-inspired neural/CPG 제어; RL reward 아님
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Unitree A1 model; 시뮬레이션
- gallop 판정 범위: trotting/pacing/galloping 재현
- 보상 또는 작동 원리: hip–knee coordination·foot trajectory·sensory feedback
- 우리 조건과의 대응: 발 궤적 모양/다리내 협응/모션 가능성
- 차이/한계: 소개에서 RL 언급하지만 제안법은 RL reward로 조건을 넣는 사례 아님
- 원문 위치: neural control model/trajectory experiments

## P50. A Quadruped Robot Exhibiting Spontaneous Gait Transitions from Walking to Trotting to Galloping

- 시기/상태: Scientific Reports 2017
- 원문: [A Quadruped Robot Exhibiting Spontaneous Gait Transitions from Walking to Trotting to Galloping](https://www.nature.com/articles/s41598-017-00348-9)
- 방법 분류: 힘 피드백 oscillator; RL 아님
- 확인 수준: 원문 방법 확인
- 플랫폼/실증: Oscillex3 계열 단순 4족 실험
- gallop 판정 범위: walk→trot→gallop 전환
- 보상 또는 작동 원리: local force feedback와 oscillators
- 우리 조건과의 대응: 동물 접촉 패턴/분산 협응의 기초
- 차이/한계: Go2의 RL·관절·거리·실기 갤롭 보상 검증으로 전용 불가
- 원문 위치: control system/gait transition experiments


