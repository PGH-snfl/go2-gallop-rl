# Gallop50k 플레이 수집·Go2 EDU 심투심/배포 적합성 검증 — 2026-11-11

**판정: Isaac 현재 단계 플레이는 성공했지만 MuJoCo 심투심은 미통과다. 실제 Go2 EDU 배포는 보류한다.** 무릎 실제 속도의 공식 URDF 경계 초과, 보행→갤롭 전환 실패, 현재 정책의 DDS/상태추정/내보내기 경로 미완성이 확인됐다.

사용자 요청에 따라 MuJoCo 창을 실제 디스플레이`:0`에 표시하고 녹화했다. 종료 이후 창의 반복 표시는 저장한qpos 재생이며 추가 동역학 시험이 아니다. PPO/모방 업데이트0, 실기 통신·모터 명령0. 학습 정책·리워드·보호 문턱은 변경하지 않았다.

[최종 판정·영상 목록](edu_sim2sim_verdict.json) · [원본 무결성](verification.json) · [학습·플레이 수집](summary.json) · [공식 자료·로컬 버전](repository_provenance.json) · [검증 코드 스냅샷](source/).

## 대상과 학습 결과

- 학습: `runs/isaac/train/unitree_go2_gallop/20261111_155012_ad6309/`.
- 사용자 추가PPO2000회/새12,288,000transition/누적50,000회.
- 최종 정책 SHA256: `93394343a7faf50f848e94ec4138c6a354641addb5b0d4e0b65db3f7ad4674a2`.
- 부모48k와 보호30k·46k·모든 이전 체크포인트를 보존했다.
- 저장단계1:2.75m/s·18cm. 최고 확인 통과단계0.3m/s·80cm에는 진입하지 않았다.
- 9684episode/완전집단19회/48판단을 저장된 소스로 재현했다. 승급0·회귀0, 최고 집단성공81.64%, 마지막78.52%.80%두 구간 연속을 달성하지 못했다.
- 탐색std 관리와180sim초 적응을 포함한 이번 학습과 과거의 성공률은 집계·노출 조건이 달라 직접적인 인과 비교로 해석하지 않는다.

단계 재현 수집기는 재개 부모의`highest_confirmed_stage=0`도 복원하도록 보완했다. 기존 분석기는 이 값을 빠뜨려 첫 이벤트 대조에 실패했다. 학습 또는 승급 로직 변경이 아니다. [재현 결과](staged_review.json).

## 방금 실행한 Isaac 플레이

플레이: `runs/isaac/play/unitree_go2_gallop/20261111_170803_4c378d/`.30초/자연timeout1/실패종료0, 정식 학생 계측9.34~30초/20.68초. 단일 저장 플레이이므로 강건성 검증은 아니다.

|항목|결과|
|---|---:|
|명령/평균 실제 전진속도|2.75/2.638m/s|
|계측 속도RMSE, episode/저장 표본 재계산|0.2035/0.2053m/s|
|도약 횟수/평균/최대|56회/21.10cm/25.39cm|
|평균 공중시간|85.18ms|
|안정 착지|48/54=88.89%|
|앞/뒤 양발지지 폭24~30cm 통과율|70.33%/93.48%|
|roll/pitch/yaw각 RMS|약1.73°/2.20°/2.02°|
|필터 전체 개입/원시 요청각 초과율|52.21%/2.34%|
|실행 목표각·목표변화율 위반|0|
|저장 실제각의 프로젝트 보호범위 허용오차 초과|0|
|현 단계/엄격/최종80cm 성공|성공/실패/아님|

엄격 기준 실패 항목은 앞발 지지폭이다. 현 단계에서는 발폭을 승급 필수에서 제외했다. 개별 착지 순서는 현재 필수가 아니므로 이 결과를 엄밀한 transverse gallop 인증이라고 부르지 않는다. 도약56회와 회복평가54회는 분모가 다른 이벤트다.

저장 Isaac trace에는 실제 관절속도·물리 스텝별 모터 토크가 없어 그것까지 공식 한도 안이라고 판정할 수 없다. 목표변화율16rad/s와 실제 관절속도는 다르다. [관절·필터 재계산](guarded_play_diagnostics.json).

[Isaac 원본 영상](../../runs/isaac/play/unitree_go2_gallop/20261111_170803_4c378d/videos/gallop-play-step-0.mp4) · [학생 구간 프레임](isaac_student_keyframes.png) · [접촉 패턴](play_contacts.png).

## MuJoCo 시험과 녹화

모두 동일50k 정책·동결 평균 행동·평지·seed42를 사용했다. 각 시험은30초를 요청했지만 보호 종료 시점에서 중단했다. 성공을 만들기 위한 보호조건 해제나 강제 갤롭 진입은 하지 않았다.

|조건|물리간격|실제 시험길이|결과|
|---|---:|---:|---|
|학습 모터·필터에 맞춘 모델|5ms|6.38초|전환중`joint_limit`|
|동일 조건 시간간격 민감도|1ms|6.84초|전환중`joint_limit`|
|공식 모델 물성+LowCmd PD 근사|5ms|12.02초|보행출발 실패`startup_deadline`|
|첫 조건의 화면 표시 재실행|5ms|6.38초|같은`joint_limit` 재현|

네 시험 모두 완전 학생 갤롭 전환 전 종료해 정식 갤롭 계측0초다. episode에 남는0도약/0보호초과 등의 미계측 기본값을 성공 또는 실제 발생0으로 해석하지 않는다. 첫 조건은 학생 혼합 약60.9%에서 끝났다. 공식 물성 근사에서는 마지막 명령1m/s에 실제 약0.00051m/s였다.

화면 표시 재실행의 원본 대비 최대 차이는q약8.22e-14rad/dq약3.09e-12rad/s로 수치 정밀도 수준이다. 독립적인 성공률 표본4개라는 의미가 아니다.1ms에서도 실패했으므로5ms 시간간격 하나만 원인으로 확정할 근거는 없다.

영상:

- [화면으로 본 학습 설정 시험6.38초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_172752_411b17/mujoco_gallop.mp4)
- [공식 물성·PD 근사12.02초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_172551_71ad44/mujoco_gallop.mp4)
- [1ms 시험6.84초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_172630_32a6ca/mujoco_gallop.mp4)
- [최초 비표시 시험6.38초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_172349_7dc524/mujoco_gallop.mp4)

각 실행 폴더에`report.json`, `physics_trace.npz`, `gallop_trace.npz`, `joint_limits.csv`, 모델 계약과 상태 파일을 저장했다. 녹화4개는 첫·마지막 프레임까지 읽기 확인했다. 초기 공식 모델 복원 준비 오류 실행`20261111_172405_821feb`는 물리시험 결과에서 제외하고 오류 기록을 보존했다.

## 공식 Go2 한도와 실제 측정

공식 자료는 [Unitree Go2 제품표](https://www.unitree.com/go2/), [공식 Go2 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf), [공식 MuJoCo 모델](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/go2/go2.xml), [공식 RL actuator 코드](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py)다.2026-11-11에 내용을 가져와[sources](sources/)에 Git blob SHA와 함께 보관했다.

**URDF/MJCF는 공식 일반 Go2 소프트웨어 모델이다. 실제 보유한 EDU의 시리얼·모터 버전·펌웨어별 연속운전 한도 확인서는 아니다.** 제품표의약45Nm는 모든 관절의 연속 토크를 뜻하지 않는다.

|관절|공식 URDF 위치,rad|effort,Nm|velocity,rad/s|
|---|---|---:|---:|
|모든 hip|−1.0472~1.0472|23.7|30.1|
|앞 thigh|−1.5708~3.4907|23.7|30.1|
|뒤 thigh|−0.5236~4.5379|23.7|30.1|
|모든 calf|−2.7227~−0.83776|45.43|15.7|

제품표에는body±48°,thigh−200~90°,shank−156~−48°가 표시돼 있다. URDF와 축 부호·영점·앞뒤 구분이 다르므로 thigh 숫자를 그대로 서로 대입하지 않는다. 보수적인hip±48°도 별도 확인했으며 Isaac 저장 최대31.66°, MuJoCo 시험 최대19.35°로 그 경계 안이다. 다른 좌표계의 일치와 실기 보호한도를 확인한 것으로 확대하지 않는다.

### 위치: 공식 모델 범위 안, 프로젝트 보호범위는 초과

모든 MuJoCo 시험에서 실제각·실행 목표각은 공식 URDF 위치 범위 안이었다(1e-4rad 수치 허용). 하지만 학습 설정 시험의왼뒷 calf는 중앙90%로 좁힌 프로젝트 보호범위를 최대0.07061rad(약4.05°) 넘어 종료했다.1ms에서는 최대0.08887rad였다. 이 초과를 원래 공식 기계각도 한계 초과와 혼동하지 않는다.

목표 필터가 각도를 제한해도 실제 관절은 관성·접촉력·유한한 모터 응답 때문에 그 목표를 벗어날 수 있다. MuJoCo의 부드러운 관절 정지점은 목표 필터와도 다르다. 종료 조건을 풀어 성공으로 처리하지 않았다.

### 실제 속도: 공식 URDF 기준 미통과

학습 설정5ms 시험에서:

|무릎|실제 최대 절대속도|15.7rad/s 초과 누적시간|
|---|---:|---:|
|FL calf|15.62rad/s|0|
|FR calf|19.29rad/s|40ms|
|RL calf|20.65rad/s|35ms|
|RR calf|18.84rad/s|10ms|

1ms에서도 최대19.54rad/s이며 네 calf 모두 경계 초과 표본이 있었다. 공식 물성 근사는 움직임 자체가 거의 없어 속도 경계는 통과했지만 동작 성공이 아니다.

학습의공식RL `Go2HV` 곡선은모든관절X1=13.5/X2=30rad/s,Y1=20.2/Y2=23.4Nm다. 이것과URDF의calf15.7rad/s·45.43Nm는 같지 않다. **현재 설정의 목표변화율16rad/s도 calf15.7보다 크며, 실제 속도는 목표변화율과 별도로 검증해야 한다.** 시뮬레이터에서20.65가 나왔다는 것이 실기도 반드시 그 속도로 움직인다는 뜻은 아니다. 실기에서는 모터 한계로 추종이 깨져 다른 동작이 나올 수도 있다.

[속도 한도 그림](mujoco_velocity_limits.png) · [관절별 수치](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_172349_7dc524/joint_limits.csv).

### 토크: 적용값은 소프트웨어 한도 안, 요구 토크·연속운전은 별개

각 시험의 적용 토크는 URDF effort 경계 안이었다. 학습 설정의 최대 적용23.4Nm, 제한 전 PD 요구 최대28.89Nm로 포화가 존재한다. 무릎의공식45.43Nm보다 낮은23.4Nm를 사용한다고 모터 전체가 실물과 일치하는 것은 아니다. URDF effort·velocity는 각각의 최대 경계이며 두 최대값을 동시에 낼 수 있다는 보장이 아니다.

RMS토크·양의 기계적 출력도 파일에 기록했지만 연속 허용 토크·전류·온도·배터리 부하 모델이 없으므로 연속운전/열 안전은 미검증이다. `qfrc_constraint`에는 접촉과 관절 제약 반력이 포함돼 모터 명령 토크와 구분한다.

### 착지력: 기록 완료, 제조사 허용 하중 판정 불가

학습 설정5ms에서 개별 발 최대 합력FL190.50/FR119.95/RL133.77/RR169.65N, 전체 수직반력 최대337.82N/2.26BW였다.1ms에서는전체수직반력2.41BW, 공식 물성 근사는2.58BW였다. 물리 모델이 다른 결과의 크기를 실물 충격의 정확한 예측으로 해석하지 않는다.

프로젝트의3BW는 자체 착지 평가 기준이다. 공식 foot impact/구조 강도 허용치가 아니다. 제품표 적재량이나45Nm를 이 값의 허용 기준으로 바꿔 쓸 수 없다. 제조사의동적발하중·충격허용치가 확인되지 않아 **공식 하중 안전 판정은UNKNOWN**이다.

## 공식 물성·PD 근사의 정확한 범위

`trained`는 저장된PD25/.5, Go2HV 토크속도 제한, passive damping/armature0, frictionloss.01, 접촉마찰1, 중앙90% 물리 정지점을 사용한다.

`sdk_proxy`는 소스MJCF의원래정지점·damping.1·armature.01·frictionloss.2·로봇접촉설정과 정적모터범위23.7/45.43을복원하고 평지마찰.4를사용한다.저장목표각/변화율필터·teacher·전환구조는유지한다. 힘 계산은공식bridge와같은 `tau_ff+kp*(q_des-q)+kd*(dq_des-dq)`이며여기서는tau_ff=0/dq_des=0이다.

이 근사는 실제 EDU의식별된물성·모터곡선이나공식DDS제어기시험이아니다. 여러 물성·접촉·토크 조건을 함께 바꿨으므로 출발 실패를마찰또는모터하나의원인으로확정하지않는다. 로컬공식모델은upstream내용과파일끝개행만다르다. 외부Unitree checkout은수정하지않았다.

## 실제 배포 코드 감사

|항목|판정|
|---|---|
|저장정책49입력/12출력·이름기반MuJoCo매핑|검사 완료|
|actor관측의IMU·q·dq·내부위상|실기 구성 가능한 항목, 실제 센서 경로 동등성 미검증|
|출발/전환 supervisor의선속도·높이|시뮬레이터정답 사용; 검증된실기추정기미연결|
|보행teacher·전환·calf보정·필터·이전실행action|실배포에도전체동일하게필요|
|기존DDS/C++Gallop연결|5행동GPD제어용; 현재49/12정책과직접호환안됨|
|현재GallopCLI의export|지원모드없음|
|일반exporter|rawaction을±1로자름; 현재정책과계약불일치|
|정식C++/SDK2/DDS동일스택심투심|이번정책미검증|
|추론시간|CPU추론계측; DDS/센서나이/루프deadline보장은아님|
|통신단절·수신측timeout·긴급정지|이번정책의실배포경로미검증|

원본Isaac행동은최대절대값4.186, 전체행동원소37.26%가±1밖이다. 기존`isaac_task/export.py`의clamp를거치면정책이바뀐다. 해당일반export명령을이모델에사용하지않는다. 정책의unbounded행동을그대로보존하고저장된실행변환을재현해야한다.

공식[MuJoCo README](https://github.com/unitreerobotics/unitree_mujoco/blob/main/readme.md)는실기내장제어서비스를끄면SportModeState를읽을수없다고명시한다. 현재actor49입력에선속도/높이가직접없다는사실만으로 supervisor까지 실기입력이해결된것은아니다.

## 다음 수정의 우선순위

1. 관절별공식모델과실기확인자료를대조해calf속도·모터곡선·관성/접촉모델을정렬한다. 관절보호범위를넓히거나실제속도를강제클립해통과시키지않는다.
2. 실제dq를기준으로한보호·로그를Isaac에도추가하고,허용범위내에서추종·도약을함께확인한다. 목표각필터만으로충분하다고가정하지않는다.
3. 현재정책의teacher/전환/위상/필터를포함한배포runtime과unbounded export parity를구현한다.
4. 실기에서구할수있는속도·높이추정입력을연결하고동일C++/DDS스택으로재검증한다.
5. 그이후지연·노이즈·물성편차·연속부하·실기보호한도를검증한다.

이번에는수집·검증까지만실행했으며새학습계획/보상변경/실기실행을자동진행하지않았다. 보호30k를현재50k로자동교체하지않았다.

## 재실행

프로젝트루트에서다음명령은GUI·녹화·최대30초검증후90초기록재생을수행한다. 학습은하지않는다.

```bash
cd /home/cvr/GH/go2_animalization/gallop
MUJOCO_GL=glfw .venv/bin/python -m scripts.validation.validate_gallop_edu \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261111_155012_ad6309/model_final.pt \
  --official-urdf reports/20261111_gallop_50000_sim2sim_review/sources/official_go2_description.urdf \
  --profile trained --seconds 30 --viewer --video --replay-seconds 90
```

공식물성·PD근사는`--profile sdk_proxy`, 물리간격검사는`--physics-dt .001`을사용한다. GUI표시를끄려면`--viewer --replay-seconds 90`을제외한다. 실행결과는새`runs/mujoco/evaluate/unitree_go2_gallop/<ID>/`에저장된다.

검증: MuJoCo기존계약8검사·EDU감사2검사통과,48개원본학습/플레이파일해시불변,동일정책4시험·녹화4개확인. 숫자·종료원인·원본SHA는[verification.json](verification.json),[edu_sim2sim_verdict.json](edu_sim2sim_verdict.json)에보관했다.
