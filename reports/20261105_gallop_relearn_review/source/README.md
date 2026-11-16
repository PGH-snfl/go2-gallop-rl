# Go2 Gallop — Isaac Lab 학습 / MuJoCo Sim2Sim 검증

**최신 수집·경향 분석: v5 장기학습12,000회 완료(2026-11-05).** 새 분기 누적16,000회 `20261104_144742_917f16/model_final.pt`는1m/s 보행 단계에 머물렀다. 플레이30초낙상0·실제1.11m/s·요청각 클리핑0%·필터 개입8.52%·앞뒤 폭 유지100%다. 학습은1→2단계 승급과회귀를46회 반복했고 갤롭 단계 진입0회였다.1단계 성공률은 마지막1,000회90.94%로 개선됐지만2단계는11.48%였다. 탐색 std가 약2.5배 증가한 상태에서 요청 종료를0.06rad/40ms→0.02rad/20ms로 동시에 강화한 전환과 빠른 회귀 평가가 병목으로 보인다. [결과·영상·경향·수정 후보](reports/20261105_gallop_relearn_review/README.md). episode66,644개 판정·288회 단계 판단을 재현하고41검증 통과, 수집 중 정책·설정 변경/추가학습/새주행0. 아래 사용자 실행 대기 표현은 구현 당시 기록이다.

**최신 구현: 일반보행에서 관절 제약을 먼저 배우는 장기학습(2026-11-04, 사용자 실행 대기).** [학습·재개·플레이 명령](docs/gallop_relearn.md), [설정](configs/gallop_relearn.json). 전진1m/s·4,000회 모델에서 새 v5 분기를 시작한다. 학생 정책의 보행 적응→부분 갤롭→성공률에 따른3m/s·22cm 확대를24단계로 진행하고, 필터 전 요청각 초과를 초기부터 실패 종료로 처리해 점차 엄격하게 만든다. 보행 단계 통과와 최종 갤롭 성공은 구분한다. 중간 모델마다 단계·누적 학습량·해시를 저장한다.149검사와 실제 부모 이식20항목 통과/새PPO0·새Isaac주행0, 사용자12,000회 완료 시 이 분기 누적16,000회다. 기존28,000회 결과는 아래에 보존한다.

**최신 수집: 안정성 강화28,000회 학습 완료(2026-11-03).** `20261103_191405_bea09f/model_final.pt`가 추가2,000회를 완료했다. 플레이30초낙상0/실제2.455m/s, 앞발 폭 유지21.8→64.1%로 개선됐지만 안정 착지65.4→66.0%·필터 개입50.01→50.89%로 전체 성공은 미달이다. 평균 도약18.57→17.51cm/최대20.72cm, 앞발 최대높이22.66→23.83cm/요청각 초과5.36→6.32%다. 승급0으로 실제 명령2.5m/s·도약 기준14cm이며3.0m/s·22cm 달성 결과가 아니다. [결과·영상·비교](reports/20261103_gallop_guarded_stability_review/README.md). 수집40검증 통과/추가학습·새주행0/원본 보존, 새 정책MuJoCo 미검증. 아래 사용자 실행 대기 표현은 구현 당시 기록이다.

**최신 구현: 안정성 가중치 강화, 사용자 추가 학습 대기(2026-11-03).** 최신26,000회 `20261103_165217_19801a/model_final.pt`에서 이어가는 [새 학습 설정·명령](docs/gallop_guarded_stability.md)을 준비했다. 착지2→4·몸체 균형2→3·충격1.5→2.5·진동 억제2→3·필터 비용6→8, 사용자 후속 지시로 발 간격은6→7·높이 초과는3→3.5만 소폭 강화했다. 가중치7개만 변경했고 기존 속도·도약 curriculum과 제약은 유지한다. 모델 복원·제약20항목 및 보상 연결26항목 통과/추가학습·새주행0, 사용자가2,000회 학습하면 계보28,000회다.

**최신 수집: 3.0m/s·22cm 목표 추가 학습 완료(2026-11-03).** `20261103_165217_19801a/model_final.pt`가 추가2,000회/계보26,000회를 완료했다. 플레이30초낙상0/실제2.44m/s, 평균 도약17.4→18.6cm지만 최대21.38cm로22cm 충족0회다. 앞발 지지폭 유지71.7→21.8%, 안정 착지68.3→65.4%, 앞발 최대높이17.2→22.7cm로 악화됐다. 필터 개입53.5→50.0%로 소폭 개선됐지만 기준35% 미달이다. 전체 성공0/승급0으로 실제 학습 단계는2.5m/s·14cm이며3.0m/s·22cm 달성을 주장하지 않는다. [결과·비교·영상](reports/20261103_gallop_guarded_3mps_leap_review/README.md). 수집37검증 통과/추가학습·새주행0/원본 보존, 새 정책MuJoCo 미검증. 아래 사용자 실행 대기 표현은 구현 당시 기록이다.

**최신 구현: 성공률에 따라3.0m/s·22cm 도약으로 추가 학습(2026-11-03, 사용자 실행 대기).** 최신 guarded24,000회 `20261103_151318_e95d61/model_final.pt`에서 이어간다. 2.5→3.0m/s와14→22cm를 함께 승급하며, 착지·몸체 균형·발폭·요청-실행 차이 보상을 강화했다. 관절 범위·명령 속도·발 높이 제한을 유지하고 참조 공중 구간을76ms로 늘렸다. 현재 부모는 안정성 gate 미달이므로 조건 충족 전에는2.5m/s를 유지한다. [학습 명령·변경·검증](docs/gallop_guarded_3mps_leap.md), [설정](configs/gallop_guarded_3mps_leap.json). 실제 가중치 복원/오프라인20항목과136검사 통과, PPO0/새주행0이며 새 목표 달성은 미검증이다. 아래는 이전 완료 결과다.

**최신 수집: guarded2.5m/s 학습 완료(2026-11-03).** 모델 `20261103_151318_e95d61/model_final.pt`는 모방 전22,000회에서 추가2,000회/새 분기24,000회를 완료했다. 30초낙상0/실제2.49m/s, 앞발 최대32.7→17.2cm·뒷발11.6→10.4cm, 명령 각도·변화 속도 상한 준수다. 실제 새 물리 경계 초과 최대0.0574°는 설정 허용오차 이내였다. 다만 착지 안정68.3%·앞발 지지폭 유지71.7%·필터 개입53.5%로 전체 성공은 미달이다. [결과·영상·비교](reports/20261103_gallop_guarded_2p5_review/README.md). 검증35항목 통과/수집 중 학습·새물리0/원본 보존. 아래 학습 대기 표현은 구현 당시 기록이다.

**최신 구현: 모방 전 모델에서2.5m/s·발 높이·강한 관절 제한 추가 학습.** 사용자 요청으로22,000회 `20261102_161522_fe6276/model_final.pt`에서 직접 이어가는 새 설정을 준비했다. 추가 모방 없이 앞발16cm/뒷발10cm 초과 패널티, 물리 관절 제한 중앙90%·목표각 중앙81%·명령 속도 상한, 실제 초과 비용/지속 초과 종료를 적용한다. [학습 명령·기준·저장 위치](docs/gallop_guarded_2p5.md). 130검사+실제 모델17항목 통과, 본학습 미실행/새 Isaac 주행 미검증. 기존24,000회 bounded 실험은 아래에 보존했다.

**최신 결과 수집(2026-11-03): 관절 범위 정책24,000회 완료.** `20261103_100859_83b730/model_final.pt`, 사용자 플레이 `20261103_123515_36ff0c`를 수집했다. 목표각 soft/실제 hard 범위 이탈은0%지만 실제 soft 이탈1.81%, 한계 근접28.73%, 명령 변화 속도95백분위42.6rad/s, 관절 추종RMSE0.540rad다. 발폭·roll은 개선됐으나 최대 횡편차0.72→3.56m, 도약17.37→15.99cm로 악화된 항목도 있다. [결과·영상·관절 그래프](reports/20261103_gallop_bounded_review/README.md), [각도·속도 제약과 동작 보존 재학습 검토안](reports/20261103_gallop_bounded_review/retraining_design.md). 오프라인 검증30항목 통과, 새 학습/시뮬레이션0. 조정안은 검토 단계이며 실행 정책·보상·설정은 보존했다. 아래 '학습 대기'는 최초 구현 당시 기록이다.

**저장 오류 수정(2026-11-03):** 모방 완료 뒤 PPO 전에 저장할 때 발생한 `logger_type` 초기화 오류를 수정했다. 사용자 실행의 모방 목표각 RMSE는0.04781rad로 통과했으며, 주행 확인·PPO는 아직 미시작이다. 실제 후보 재저장 등40검사 통과. [같은 학습 명령으로 새 실행](docs/gallop_bounded.md)을 시작하면 모방부터 다시 진행한다.

**최신 구현(2026-11-03, 사용자 학습 대기): 관절 범위 안에서 출력하는 정책.** 최신22,000회 모델의 실행 목표각을 모방하고, student가 방문한 상태에도 부모 라벨을 추가한 뒤, 주행 보존 검사를 통과하면 PPO2,000회를 추가한다. 관절별tanh 출력, 한계 근접/목표각 변화율 비용과 승급 gate를 추가했다. [실행 명령·단계·저장 위치](docs/gallop_bounded.md). 117검사와 실제 모델 복원13항목·trace15항목 통과. 구현 검증 중 학습 optimizer는 실행하지 않았으며, 새Isaac 주행은 사용자 실행 시 확인한다. 아래22,000회 모델이 현재 완료된 최신 모델이다.

**최신 완료 결과(2026-11-03): 안정화 학습 수집.** 사용자 추가 2,000회 / 계보 22,000회 완료 모델은 `20261102_161522_fe6276/model_final.pt`다. 30초 플레이에서 낙상 없이 전환했고, 평균 roll 11.1→3.8°, 앞발 지지폭 4.2→25.4cm, yaw 각속도 0.478→0.182rad/s, 안정 착지 57/59회로 개선됐다. 다만 clipping 22.48%와 뒷발 지지폭 유지율 78.13%가 기준에 미달해 전체 성공은 0이며 속도 단계는 2.2m/s·14cm다. [결과·영상·비교](reports/20261103_gallop_stabilization_review/README.md). 기존 파일 분석과 37항목 검증을 완료했고, 이번 수집의 학습/새 시뮬레이션은 0이다. 새 모델 MuJoCo는 미검증이다.

**안정화 설계(2026-11-02, 이후 사용자 학습 완료):** 자세 범위와 진동의 결합 보상, 독립 발 폭 보상, yaw 감쇠와 목표각 초과량 비용, calf 출력 0.85배를 적용했다. 반복 성공에 따라 2.2m/s·14cm부터 2.8m/s·18cm까지 승급한다. [설계·원래 학습 명령](docs/gallop_stabilization.md). 아래 미학습·미검증 표현은 각 구현 당시 기록이며 현재 결과는 위 수집 보고서를 따른다.

**최신 완료 결과(2026-11-02): yaw·진동·발 폭 추가 학습 수집.** 최신49/64모델 `20261102_145730_26810f/model_final.pt`는추가2,000회/계보20,000회완료다. 30초낙상없이10.12초Gallop전환,실제2.92m/s,도약57회/평균19.39cm·69.30ms/안정착지45회/거리·시간·안정동시충족27회다. 다만평균roll11.1°와좁은앞발지지폭4.2cm가남았고,z각속도.478rad/s·목표각clip24.24%로전체성공미달이다. [결과·영상·이전정책비교](reports/20261102_gallop_heading_motion_review/README.md). 원본을보존하고수집PPO0/새시뮬레이션0,새모델MuJoCo는미검증이다.

**최신 보상 수정(2026-11-02): yaw0·roll/pitch 진동·발 간격·목표 방향 이동.** yaw는월드+X기준0오차를벌주고,roll/pitch는각도변화율이작을수록점수를준다. 앞뒤발쌍을독립적으로어깨pitch축폭28.4cm의95%=26.98cm에맞추면양의보상. 발쌍중심선벌점은실제목표방향이동보상으로교체했다. [최신설계·추가학습명령](docs/gallop_heading_motion.md). yaw관측2열을추가해49/64·관절12로이식하며기존학습tensor를보존하고추가열만0으로초기화한다. 구현검사90개·모델이식/형상14항목통과후사용자본학습과플레이를완료했으며맨위수집결과를따른다.

**이전 보상(2026-11-02): 몸체 수직 정렬·앞뒤 발 간격.** 주기pitch목표를0으로바꾸고전체기울기벌점,앞뒤각좌우발중심폭30cm보상,발쌍이한쪽으로치우치는벌점을추가했다. 지지/이륙/공중/착지와출발·전환전체에서계산한다. [설계·최신모델에서추가학습명령](docs/gallop_posture.md). CPU74검사와실제모델정확복원등10항목통과,CUDA인식불가로새주행은미검증이다. 새학습은사용자가진행하며기존완료모델은보존했다.

**이전47관측정책 MuJoCo 검증(2026-11-02): 30초 생존, Gallop 전환 미통과.** 최신2.8m/s 정책으로 정지→일반 보행까지 수행했지만 혼합 비율65.76%에서 멈췄다. 마지막10초 명령2.18m/s에 실제평균3.09m/s로 과속해 전환 조건을 계속 벗어났다. [결과·30초 영상·재실행 명령](reports/20261102_gallop_mujoco_review/README.md). PPO 업데이트0/가중치불변이며 학습·전환 조건은 변경하지 않았다. MuJoCo 실행은 `.venv/bin/python -m scripts.validation.sim2sim_gallop`을 사용한다.

**이전 학습 완료: 일반 보행 후2.8m/s Gallop 전환.** 사용자 추가2,000회/계보 누적18,000회 완료, 최신모델 `runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt`. 사용자 Isaac 플레이는8.34초에Gallop전환,도약63회/평균18.53cm·66.03ms/거리·시간충족47회다. 추종gate·clipping21.02% 때문에전체success0이며MuJoCo완전전환도미달이다. [제어·명령·검증](docs/gallop_walk_transition.md). 학습은 계속 사용자가 직접 실행한다.

**이전 결과: 2.5m/s에서 실제 도약 생성.** 사용자2,000회 추가학습 완료. 실제평균2.526m/s, 도약55회/평균14.16cm·59.27ms/최대16.86cm,16cm·60ms충족4회는모두80ms안정착지를통과했다. 회복실패24회와앞다리하중편중/목표각clip15.24%는남는다. [결과·영상·재생 명령](reports/20261029_gallop_2p5mps_flight_review/README.md). 부모모델 `runs/isaac/train/unitree_go2_gallop/20261029_151901_f393b4/model_final.pt`(계보누적16,000회).

**이전 학습 설정: 2.5m/s 도약·균형·안정 착지.** 준비 단계부터 상승·하중 해제 보상을 주고, 공중 구간 접촉과 몸통·hip 좌우 편중에 패널티를 추가했다. 도약 목표16cm·60ms, 착지 후80ms 안정 회복을 확인해 거리 보상을 확정한다. [학습 명령·보상·검증](docs/gallop_flight_2p5.md). 구현당시CPU55검사와 부모복원 후 사용자가본학습과PhysX플레이를완료했다. 이번수집에서는PPO업데이트와학습설정변경을하지않았다.

**이전3m/s 결과(2026-10-29): 실제 도약거리 보상 추가 학습2,000회와 플레이 수집 완료.** 실제평균2.956m/s, 주기전진0.948m, 앞다리thigh85~92°로 속도와 관절운동은 증가했다. 하지만5ms계측의 당시플레이 완료도약0회/보상0점으로 목표도약은미달이었다. clip16.96%와RR hip좌우차이도남았다. [당시 보고서·영상·재생 명령](reports/20261029_gallop_3mps_leap_review/README.md). 부모모델은 `runs/isaac/train/unitree_go2_gallop/20261029_140301_dbb121/model_final.pt`이며 계보누적14,000회다.

**이전3m/s 설정:** `configs/gallop_3mps_leap.json`은 몸통의 실제 공중 전진거리12cm·공중시간40ms 달성에 보상을 준다. 5ms계측/앞다리자격/주기별보상상한을 적용했다. [실행 명령·기준·검증](docs/gallop_rl.md). 사용자가본학습을완료했으며, 해당결과수집에서는PPO업데이트와설정변경을하지않았다.

**이전2m/s 결과: 좌우 쌍 추가 학습2,000회와 플레이 수집 완료.** 2.0명령에서 실제평균1.935m/s,주기전진0.716m(이전0.424m대비약69%증가),앞다리thigh약58~59°를확인했다. clipping16.46%와공중기기준미달로상한은2.0유지다. [당시 보고서·영상·재생 명령](reports/20261029_gallop_2mps_pair_review/README.md). 부모모델은 `unitree_go2_gallop/20261029_123734_c563b2/model_final.pt`다.

**좌우 쌍 패널티:** 각 쌍의 위상 보정 관절 오차·실제 ROM·누적 하중·지지시간 차이를 줄인다. [추가 학습 명령·패널티·검증](docs/gallop_rl.md). 구현 당시 CPU 회귀22개와 모델복원을 검사했고, 이후 위 사용자학습과플레이를수집했다. 학습은계속사용자가직접실행한다.

**이전1m/s 결과: Gallop 6,000회 학습 완료, 고속 목표 미달.** [수집 보고서·영상·수치](reports/20261029_gallop_review/README.md). 1.0m/s 명령에 실제 평균 1.0109m/s, 앞다리 하중 49.47%와 관절/스윙 참여 조건은 통과했다. 초기 1초 이후 동시비접촉저장표본0%, 목표각 clipping 14.73%로 필수 조건을 통과하지 못해 속도 상한은 1.0에 머물렀다. 학습 영상 15개와 플레이 영상 1개를 보관했다.

**초기 Gallop 학습 코드: Gallop 스타일 + 앞다리 참여 + 성공률 기반 속도 승급.** [학습 명령·조건·검증](docs/gallop_rl.md). 초기 설정은 기존 1.0m/s 정책에서 시작해 성공률로만 상한을 최대 3.0까지 올리는 설계다. 이후 고정 속도·도약·출발 전환 설정은 위 최신 문서를 따른다.

**이전1m/s 모델:** `runs/isaac/train/unitree_go2_gallop/20261028_181029_379d1c/model_final.pt` (actor 47 / critic 62 / action 12). [당시 플레이 영상](runs/isaac/play/unitree_go2_gallop/20261029_110531_367535/videos/gallop-play-step-0.mp4). 이전 GPD 57/5와 전진 45/60 정책의 실행 경로를 혼용하지 않는다. 이1m/s모델의MuJoCo와실기는미검증이며최신2.8m/s모델은위검증결과를따른다.

아래는 이전 전진 학습과 기본 보행 실험 기록이다. 현재 결과는 위 수집 보고서와 [인계 기록](docs/CONTINUE_HERE.md)을 기준으로 한다.

**이전 완료 단계: 기존 전진 정책에서 상한1.0m/s까지 추가 학습.** [체크포인트 복원·학습 명령](docs/forward_1ms_finetune.md). 사용자가 추가 2,000회를 완료한 `unitree_go2_forward_only/20261028_160258_54620c/model_final.pt`가 현재 Gallop의 부모다(부모 포함 누적 4,000회). 기존 모델은 보존한다.

**이전 결과: 전진 전용 정책의 평지 추종3조건 통과.** 0.2/0.3/0.4m/s 명령에 실제 평균0.210/0.291/0.377m/s. [수집 보고서·영상·수치](reports/20261028_forward_only_review/README.md). 이 전진 기준선의 Gallop/Sim2Sim은 검증하지 않았다.

**이전 실험: 전진 명령만 학습:** `./isaac.sh forward train --headless --num-envs 256 --iterations 2000 --video`. vx+0.2~+0.4,vy=yaw0,정지명령0%. [학습·플레이·검증](docs/forward_only_baseline.md). [직전 학습 결과](reports/20261028_curriculum_fixed_review/README.md)는 curriculum이0.4까지 확장됐지만 전진 평가에 실패했다.

**Curriculum 수정본으로 기본 보행 재학습:** `./isaac.sh baseline train --headless --num-envs 256 --iterations 2000 --video`. 보상은 유지하고 명령 범위 갱신 누락을 수정했다. [변경 내용·학습·플레이](docs/curriculum_fixed_baseline.md). 저장 위치는 `runs/isaac/train/unitree_go2_curriculum_fixed/<실행ID>/`다.

**공식 원본 기본 보행을 새로 학습하려면:** `./isaac.sh official train --headless --num-envs 256 --iterations 2000 --video`. 별도 공식 checkout을 사용하고 원본 task 설정을 감사한다. [원본 버전·호환 처리·저장 위치](docs/official_baseline.md)를 확인한다. 아래 기존 `./isaac.sh train`은 GPD 실험 경로다.

**공식 원본 모델의 현재 결과:** [2026-10-28 수집 및 고정 명령 평가](reports/20261028_unitree_official_review/README.md). 실제 학습은 완료됐지만 전진 명령을 추종하지 못했으며, 기본 보행 개선이 먼저 필요하다.

**최신 결과·다음 작업:** [인계 기록](docs/CONTINUE_HERE.md). 다음 검토 대상은 MuJoCo 혼합 단계의 과속과 전환 진행 정지, 착지 발 속도와 관절 목표 clipping이다. 이번 검증에서 학습 설정은 변경하지 않았다. 아래 GPD 실행 명령은 기존 57차원/5행동 실험 경로다.

**실제 보행 평가:** [2026-10-27 명령 추종 검증](reports/20261027_unitree_baseline_evaluation/README.md). 기본 정책은 서 있기는 하지만 직진·회전 명령을 추종하지 못했다. 학습 완료와 보행 성공을 구분한다.

**학습은 Isaac Lab의 GPU PhysX + RSL-RL PPO에서만 진행한다.** MuJoCo는 학습된 정책의 Sim2Sim·배포 전 검증용이다. Trot 전환은 포함하지 않는다.

## 이전 GPD-MPC 실험의 학습 실행

현재 머신에 전용 `.isaacvenv`를 구성했다. 기존 Isaac Lab/Gymnasium 버전을 변경하지 않고 CasADi만 추가했다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh train --headless --num-envs 4 --iterations 1000
```

처음부터 학습하면서 주기적으로 영상을 저장하려면:

```bash
./isaac.sh train --fresh --headless --num-envs 4 --iterations 1000 \
  --video --video-length 400 --video-interval 5000
```

`--fresh`는 무작위 초기화이며 `--checkpoint`와 함께 사용할 수 없다.
기본 정책 주기 25 ms 기준으로 5,000 step마다 환경 0의 10초 영상을 녹화한다.
영상은 각 학습 실행 폴더의 `videos/gallop-train-step-*.mp4`에 저장한다.
`--video`가 카메라 렌더링을 자동 활성화하므로 `--headless`에서도 녹화할 수 있다.
이 옵션은 녹화 기능이며 기존 실패 패턴의 개선을 보장하는 학습 설정 변경은 아니다.

`runs/isaac/train/<시간_ID>`에 다음을 저장한다.

- `model_*.pt`, `model_final.pt`: 재개 가능한 RSL-RL 체크포인트
- `actor.pt`: 관측 정규화를 포함한 TorchScript 배포 정책
- `config.json`, `manifest.json`: 동일 제어기 설정, 관절 순서, 모델 경로, 번들 해시
- `export_validation.json`: RSL-RL과 내보낸 정책의 수치 일치 검사
- `status.json`, TensorBoard 로그, 소스 스냅샷

실행하면 출력 폴더를 먼저 표시하고, 전체 콘솔을 `console.log`에 자동 저장한다.
`progress.jsonl`에는 시간·단계·진행률·속도·높이·자세·접촉 상태·reset 원인을 기록한다.
`status.json`의 `last_event`에서도 최신 진행 상황을 확인할 수 있다.

학습 콘솔은 RSL-RL 기본 표를 그대로 표시한다: Learning iteration, Computation,
Mean action noise std, value/surrogate/entropy loss, Mean reward, Mean episode length,
Total timesteps, Iteration time, Time elapsed, ETA. 완료 에피소드가 없으면 reward/episode length는 생략된다.
중복 한 줄 요약과 학습 중 step/reset 출력은 표시하지 않는다. 저장 완료·최종 결과·오류는 별도로 알린다.
상세 `PROGRESS`, `EPISODE_END`, 손실 정보는 `progress.jsonl`에 유지하고 텐서보드 기록도 유지한다.
접촉 배열 순서는 FL/FR/RL/RR이며, 1은 접촉이다. 상태 수치는 현재 환경들의 요약이며 Gallop 합격 판정은 `EVALUATION_RESULT`에서 별도로 확인한다.

기본 진행 로그는 50 policy step마다 또는 5초마다 출력한다. 간격을 줄이려면 `--log-every 25`를 추가한다.
다른 터미널에서는 `tail -f runs/isaac/train/학습폴더/console.log`로 확인한다.

처음 연결을 짧게 확인하려면:

```bash
./isaac.sh train --headless --num-envs 2 --iterations 3
./isaac.sh check --headless --num-envs 2 --steps 440
```

`check`는 residual=0인 GPD-MPC를 PhysX에서 실행하고 접촉 순서와 생존 결과를 기록한다. 실행이 끝난 것과 Gallop 평가 합격은 별개다. `status.json`의 `gait_accepted` 및 `evaluation.json`을 확인한다. 시험용 몇 회 업데이트는 학습 수렴을 의미하지 않는다.

## 학습 재개 및 Isaac Lab 재생

```bash
./isaac.sh train --headless --num-envs 4 --iterations 1000 \
  --checkpoint runs/isaac/train/학습폴더/model_final.pt
./isaac.sh play --num-envs 1 --steps 1200 \
  --checkpoint runs/isaac/train/학습폴더/model_final.pt
```

체크포인트의 저장된 제어 설정을 자동으로 복원한다. 기존 MuJoCo/SB3의 `policy.zip`은 호환되지 않으며 새 학습이 필요하다. `scripts/training/train.py`도 Isaac Lab 실행기로 변경했다. 이전의 `--steps 200000 --envs 4` 학습 명령은 사용하지 않는다.

학습 완료 후 **최종 정책의 30초 플레이 영상**을 저장하려면:

```bash
./isaac.sh play --headless --num-envs 1 --steps 1200 \
  --checkpoint runs/isaac/train/학습폴더/model_final.pt \
  --video --video-length 1200
```

새 플레이 폴더의 `videos/gallop-play-step-0.mp4`에 저장한다. 내부 episode reset도 영상에 포함되므로 영상 길이가 연속 보행 성공 시간을 뜻하지 않는다.
play/check는 시작부터 한 클립을 녹화하며, `--video-length`가 실행 step보다 길어도 종료 시 짧은 클립을 저장한다.
2026-10-27에는 기존 정책의 50-step 비학습 재생으로 640×480, 40 fps MP4 저장과 프레임 내용을 확인했다.

## MuJoCo Sim2Sim 검증

학습 폴더 자체가 배포 번들이다. 정책·제어 설정 해시와 export 검사를 확인한 뒤 MuJoCo에서 실행한다. 여기서는 학습이나 정책 업데이트를 하지 않는다.

```bash
./gallop.sh validate_mujoco.py runs/isaac/train/학습폴더 \
  --seconds 30 --output runs/mujoco/evaluate/actor_check --require-pass
./gallop.sh render.py runs/mujoco/evaluate/actor_check --video
./gallop.sh play.py --replay runs/mujoco/evaluate/actor_check/rollout.npz
```

별도 C++ MuJoCo plant와 DDS gateway까지 검증하려면:

```bash
./gallop.sh validate_mujoco.py runs/isaac/train/학습폴더 --dds \
  --binary deploy/build_316/gallop_dds --seconds 30 \
  --output runs/mujoco/dds/actor_check --require-pass
```

DDS domain 1과 loopback `lo`는 코드에 고정되어 있다. 실제 로봇에는 연결하지 않는다. 이 단계도 실패하면 종료 코드 2를 반환한다. 실물 배포 승인을 의미하지 않는다. C++ adapter는 Unitree SDK2를 사용한 프로젝트 구현이며, 공식 `unitree_mujoco` 실행 파일 자체가 아니다.

중간 체크포인트만 별도 export하려면:

```bash
./isaac.sh export --headless --num-envs 1 \
  --checkpoint runs/isaac/train/학습폴더/model_50.pt --output runs/isaac/export/model_50_bundle
```

## 구성과 현재 범위

- `isaac_task/env.py`: Isaac Lab DirectRLEnv, 평지·공식 로컬 Go2 USD·IMU·발 접촉 센서·토크 적용
- `scripts/training/run_isaac.py`: RSL-RL PPO, 학습/재개/평가/내보내기
- `gpd/controller.py`: Isaac Lab과 MuJoCo에서 공유하는 GPD-MPC
- `configs/flat_gallop.json`: 우측 lead, 목표 0.8 m/s, 주기 0.45초
- `scripts/validation/validate_mujoco.py`: Isaac 정책 번들의 Sim2Sim 검증 진입점

GPD-MPC의 Pinocchio/CasADi QP는 CPU에서 환경별로 계산한다. PhysX와 PPO는 GPU를 사용하지만 수천 환경용 GPU MPC 구현은 아니다. 먼저 2~4환경을 사용한다. 기본 관측은 57차원, 행동은 주기·앞/뒷발 위상 간격·pitch·높이의 5차원 보정이다. 0.5초 서기 초기화 후 Gallop 스케줄로 진입한다.

현재 기본 설정은 공중기 없는 transverse 접촉 패턴이다. 이전 MuJoCo 기준 제어기의 30초 성공 결과를 Isaac 학습 정책의 성공 결과로 간주하지 않는다. 공중기 있는 Gallop, 학습 수렴, DDS 안정성, 실제 Go2 EDU 배포는 각각 추가 검증 대상이다.

[학습 설계](docs/gpd_training_design.md)

[실행 검증 결과](docs/isaac_validation.md). 본학습은 자동으로 시작하지 않는다.

## 디렉토리와 저장 위치

```text
gallop/
├── isaac.sh / gallop.sh / setup_isaac.sh  # 실행 진입점
├── scripts/
│   ├── training/     # Isaac 학습·재생·export 실행기
│   ├── validation/   # MuJoCo 평가·재생·DDS Sim2Sim
│   ├── analysis/     # 접촉 차트·영상·Kine2Go 비교 보고서
│   ├── data/         # 참고 데이터 다운로드
│   └── experiments/  # 파라미터 비교
├── gpd/              # 공유 제어기 및 중앙 경로 규칙(paths.py)
├── isaac_task/       # Isaac 환경·정책 export·진행 로그
├── configs/          # 실험 설정
├── assets/           # scene/model 리소스
├── deploy/           # DDS C++ 소스와 빌드
├── data/             # 입력 참고 자료
├── runs/
│   ├── isaac/{train,check,play,export}/
│   ├── mujoco/{evaluate,play,dds}/
│   └── experiments/{sweep,tune}/
├── reports/kine2go/  # 공유용 비교 결과
├── archive/results_before_layout/  # 구조 정리 전 결과 원본
├── results -> archive/results_before_layout  # 이전 경로 호환
├── docs/ / tests/
└── AGENTS.md         # 이후 작업의 저장·실행 규칙
```

`--output`을 생략하면 실행 목적별 폴더에 시간과 고유 ID를 붙여 저장한다.
로그·체크포인트·영상은 각 실행 폴더 안에 모은다. `results/`는 과거 자료 접근용이며 새 실행에는 사용하지 않는다.
기존 `./isaac.sh ...`, `./gallop.sh evaluate.py ...` 명령은 그대로 동작한다.
Python을 직접 호출할 때는 루트에서 `python -m scripts.validation.evaluate`처럼 모듈 경로를 사용한다.
과거 체크포인트·로그의 내용은 수정하지 않았다.
