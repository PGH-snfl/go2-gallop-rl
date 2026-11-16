# 갤롭 속도 적응 개선 구현·검증 — 2026-11-11

**구현 완료.** 새 속도 체류/적응을180sim초로 늘리고,48k 부모의 평균 탐색std약0.286을0.20까지 점진 축소한다. 현재 속도·보행·관절 안전 기준은 유지한다. 실제 추종 개선과 승급은 사용자가 새로 학습한 뒤 확인해야 한다.

[학습 명령과 동작 설명](../../docs/gallop_staged_speed_adaptation.md) · [종합 검증](verification.json) · [실제 부모 복원 검사](parent_verification.json) · [변경 diff](implementation.diff).

## 해결 대상으로 삼은 구조

이전48k는0↔1단계를 왕복했고,1단계 핵심 실패999개 중875개가속도RMSE만 실패했다. 새속도에서각약292PPO후회귀했고, action std는0.286정도, 실제adaptive학습률은60.6%구간에서1e-5였다. 이것이 단독 원인이라는 확정은 아니므로 학습률을 임의로 올리지 않았다.

|변경|효과와 범위|
|---|---|
|적응60→180sim초|새 단계에서375PPO 상당을 학습한 뒤 집단 판정 시작|
|평균std0.286→0.20,500회 ramp|관절별 상대 탐색크기 보존, 갑작스러운 분산 변경 방지|
|std스케줄관리·entropy0|actor/critic은 계속 PPO, variance만 별도 관리|
|속도 평균/주기 진단|평균 부족과 주기내 변동을 구분|
|실제LR·정책KL진단|학습률 바닥 현상을 관측, optimizer/scheduler 자체는 유지|

std는 완전한 PPO수집 구간 사이에서만 바꾼다. 이미 샘플링한 행동에 배수를 곱하거나 저장log-probability를 고치지 않는다. 동일rollout과그PPOupdate에서std는고정이다. per-joint 비율을 유지하고 부모평균std가이미0.20보다낮으면올리지않는다.

실제학습률의adaptive/KL0.01, 초기1e-4, 기존PPO설정은 유지한다. entropy계수만0.01→0이며fixedvariance에서불필요한variance확대목표를제거한다. 낮은std는같은평균변화의KL을더크게만들수도있으므로, 새KL로그로 확인하며학습률을무작정상향하지않는다.

이번에적응/노이즈를함께바꿨으므로학습후효과가나오더라도두요인의기여를개별인과로분리한결과는아니다.

## 유지한 조건

- 순간 전진속도RMSE≤0.25m/s,횡/회전 추종,공중기,앞다리 참여,ROM,단계별거리.
- 낙상/실제 관절 감시·모터 토크/속도·실행각/변화율·raw요청각10%.
- 완전집단·성공80%두구간·초기실패집계·회귀판정·최종80cm 엄격인증.
- 기존reward가중치,초기발폭/발높이의soft보상,최종엄격조건복원.
- 관측49/64·행동12·action scale0.5·정책평균·critic·동결일반보행teacher.

새속도평균/주기 지표는 진단 전용이며 승급을 대신하지 않는다. 시간만으로 승급시키거나48k를새성공모델로재표시하지않았다. 원본체크포인트는 보존했다.

## 검사1: 실제48k 복원과 샘플링

CPU검사 `runs/isaac/check/unitree_go2_gallop/20261111_153021_dbaba7/`,27개통과.

- 저장된48,000회·1단계(2.75m/s·18cm)와SHA확인.
- actualRSL의actor/critic출력최대차0,초기std복원정확.
- 새Adam·iteration0,actor/critic학습가능,std만고정.
- 실제RSL샘플·저장sigma·log-probability일치/rollout안분산불변.
- annealing끝평균std0.20·관절별비율보존.
- 속도문턱·물리제한·리워드보존,180초이전episode집계제외.
- PPO/update/optimizer함수는금지해업데이트0,physics0.

출력: [parent_verification.json](parent_verification.json).

## 검사2: 단위/회귀·수치·로그

[단위검사204개](unit_tests.log), [runner체크포인트검사3개](runner_tests.log) 통과.

- std변경은rollout경계만/샘플·Gaussian확률일치/관절별비율보존/저장된낮은분산재확대금지.
- KL진단이난수·정책·storage행동/확률을변경하지않고원래update반환값유지.
- KL알고리즘은해석해가있는Gaussian분포로검사.
- rollout을읽는진단경로는no-op update로검사해실제PPO업데이트를실행하지않음.
- 실제RSL모듈의샘플링후actor/criticbackward수치호환검사,optimizer.step없이가중치불변.
- 불완전/초기/reset주기제외,평균속도·변동·주기오차의분리.
- 기존핵심/최종gate·위상·관절/필터·checkpoint회귀검사.
- 새보조소스누락/변조시replay거부.
- KL/학습률지표가TensorBoard와progress의diagnostics에남고콘솔표를늘리지않는지확인.

**새KL은전체rollout의업데이트전후KL**이며RSL내부scheduler의미니배치KL과다르다. 이둘을같은기록으로표현하지않았다.

## 검사3: 실제Isaac Lab 연결

실행 `runs/isaac/check/unitree_go2_gallop/20261111_153025_d3c9dd/`,16환경×1600step,policy_updates0·정책불변·정상종료.

- 현재48k평균정책을새환경/계측코드와연결했다.
- 첫16개중일반보행→학생전환14개,현단계성공13개,엄격성공0개,최종80cm성공0개,실패종료2개.
- 평균속도/변동/주기평균지표와표본수/주기수를episode와trace에저장.
- 정상계측episode의순간RMSE²≈평균오차²+속도분산관계를float32누적오차범위에서확인.
- 수집모드에서는managed분산을진행시키지않고평균행동으로검사하므로,이13/16을새노이즈설정의성능개선으로해석하지않는다.
- 전체 초기 구간을 포함한 실제 관절의 최대 경계 차이는0.00208rad로 허용오차0.01rad 이내였다. 실행target 각도/변화율 범위도 검사했다.
- 새설정이학습중실제로승급/회귀를개선하는지는아직미검증.

## 검사4: MuJoCo 어댑터 호환

[10step 연결검사](mujoco_adapter_verification.json).새source·설정에48k가중치를복사한비학습fixture를사용했다.1단계복원/유한상태/정책불변확인. 새진단class가저장source기반MuJoCoAST복원과호환되는지본것이며,새Sim2Sim성능·2.75m/s추종·80cm성공을검증한것은아니다.

## 로그와 체크포인트

TensorBoard와PPO_UPDATE.diagnostics에noise평균/ramp/entropy/실제LR/floor여부·rollout전후KL/표본수를남긴다. 콘솔은기존표유지,스케줄시작/목표도달만별도알림.

`staged_adaptation.py`를체크포인트source에동봉하고SHA검사한다.각중간/최종파일의noise상태를manifest에기록한다. 정확optimizer/RNG재개가아닌이어서학습에서는저장std에서새ramp/새Adam/새집단을시작한다.

## 다음 실행

부모 `20261111_141724_6a1661/model_final.pt`(48k),새설정 `gallop_staged_speed_adaptation.json`,우선추가2k/완료시50k.새학습run에보존하며최신48k·부모46k·보호30k를덮어쓰지않는다.

본학습PPO/모방0.물리검사는Isaac1600step과MuJoCo어댑터10step을실행했다.새속도추종·승급·최종80cm개선은아직확인한결과가아니다.

