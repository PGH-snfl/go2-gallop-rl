# 2.5m/s·46k Gallop MuJoCo/Go2 EDU 검증 — 2026-11-11

**판정: 이번2.5m/s 정책도 심투심 미통과이며 실배포 보류다.** 학습 설정5ms·1ms 모두 갤롭 전환 중 관절 보호 종료, 공식 물성·PD 근사는 공통 기본 보행 단계에서 출발 실패였다. 실제 무릎속도가 공식 URDF 소프트웨어 경계15.7rad/s를 초과했다.

사용자가 직전 안내한 최신2.5m/s 영상의 모델로 같은 검증을 요청했다. 대상은 **46,000회 모델**이며50k 정책의 명령만2.5로 바꾼 시험이 아니다. 새학습/모방0·실기통신0, 정책·리워드·보호기준 변경0. 창 표시2회와 영상3개를 저장했다. 종료 후90초 표시한 것은 저장qpos의 반복 재생이다.

[최종 수치·영상 목록](edu_sim2sim_verdict.json) · [50k 비교](comparison_50000.json) · [무결성](verification.json) · [사용 코드](source/) · [검증 계획](validation_plan.json).

## 대상과 기존 Isaac 기준

- 모델: `runs/isaac/train/unitree_go2_gallop/20261110_223314_cef16b/model_final.pt`.
- SHA256: `32ac3e282cf014ec8ac342ca832a49ea809b1e9a158beb17b490c9fe6b22ecdb`.
- 누적46k, 저장0단계:2.5m/s·18cm.3m/s·80cm 달성 모델이 아니다.
- 연결된 Isaac 플레이: `20261111_134458_3dcc3c`.
- 기존 Isaac 기록:30초 낙상0, 실제평균2.529m/s, 평균도약20.24cm/최대23.80cm, 안정착지44/54=81.48%. 엄격 성공은 앞발폭73.63%와 앞발높이초과율10.40% 때문에 실패했다.
- [Isaac 원본 영상](../../runs/isaac/play/unitree_go2_gallop/20261111_134458_3dcc3c/videos/gallop-play-step-0.mp4).

기존 [46k 수집 보고서](../20261111_gallop_staged_46000_review/README.md)의 summary·원본해시·관절진단·단계진단을 새 폴더에 복사해 참고했다. 이번에 Isaac 주행이나 학습을 다시 실행한 것은 아니다. 해당 원본73개 파일의 해시를 다시 확인했다.

## 동일 프로토콜 시험

모든 시험은 동결 평균정책·seed42·평지·최대30초 요청이다. 보호조건을 풀거나 강제 학생 진입을 하지 않았다. 관절 한도·모터·접촉 설정은 이전50k 시험과 같은 검증기를 사용했다.

|조건|이번46k·2.5m/s|이전50k·2.75m/s|
|---|---|---|
|학습 모델·5ms|6.42초`joint_limit`|6.38초`joint_limit`|
|학습 모델·1ms|6.86초`joint_limit`|6.84초`joint_limit`|
|공식 물성+LowCmd PD 근사·5ms|12.02초`startup_deadline`|12.02초`startup_deadline`|

학습 모델5ms의 마지막 학생 혼합은60.92%,1ms는62.87%였다. **세 시험 모두 정식 학생 갤롭 계측0초**이므로 갤롭 완성 또는 지속보행 성공을 주장하지 않는다. episode의미계측0을 실제0도약·0위반으로 해석하지 않는다.

정책과 학습 이력이 다르므로 이 비교는 속도 하나만 변경한 인과실험이 아니다. 이번 결과로 확인되는 것은2.5m/s 정책에도 전이 문제가 남았다는 점이다.1ms에서도 실패해5ms 적분 간격 하나만 원인으로 볼 근거는 없다.

영상과 기록:

- [화면 표시·학습 설정5ms 영상6.42초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_173835_b82ae8/mujoco_gallop.mp4)
- [학습 설정1ms 영상6.86초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_173836_ad31cb/mujoco_gallop.mp4)
- [화면 표시·공식 물성/PD 근사12.02초](../../runs/mujoco/evaluate/unitree_go2_gallop/20261111_173913_ba9283/mujoco_gallop.mp4)

각 폴더에는 `report.json`, `physics_trace.npz`, `gallop_trace.npz`, `joint_limits.csv`, 모델계약·공식한도·상태파일이 함께 있다. MP4첫·마지막 프레임까지 확인했다. [속도 비교 그림](mujoco_velocity_limits.png).

## 관절 각도와 보호 종료

세 시험의 실제각·실행목표각은 공식URDF 위치경계 안이었다(1e-4rad 수치허용). 하지만 학습 모델에서는 프로젝트가 중앙90%로 축소한 물리/감시범위를 왼뒷 calf가 초과했다.

|항목|5ms|1ms|
|---|---:|---:|
|왼뒷calf의 프로젝트 보호범위 최대초과|0.07092rad/약4.06°|0.09907rad/약5.68°|
|첫 허용오차 초과 물리표본|6.38초|5.937초|
|허용오차 초과 물리표본 누적시간|45ms|74ms|
|종료시각|6.42초|6.86초|

이는 원래 공식 기계각도 범위 초과와 다르다. 종료 모니터는20ms 제어표본을 누적하므로 표의 물리표본 초과 누적시간과 종료판정의 시간은 동일한 측정값이 아니다. 명령 목표를 제한해도 실제각은 관성·접촉·추종오차 때문에 달라질 수 있다.

공식 제품표의 더 좁은hip±48°도 별도 확인했으며 MuJoCo 최대19.70°로 그 경계 안이다. 제품표와URDF의thigh 부호·영점·앞뒤 구분을 동일하다고 가정하지 않았다.

## 실제 관절속도

공식 [Go2 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf)의속도경계는hip/thigh30.1rad/s,calf15.7rad/s다.

|무릎|5ms 최대절대속도|1ms 최대절대속도|
|---|---:|---:|
|FL|13.82rad/s|16.45rad/s|
|FR|18.27rad/s|18.19rad/s|
|RL|18.22rad/s|20.41rad/s|
|RR|16.07rad/s|18.50rad/s|

5ms에서FR/RL/RR은15.7을넘었고,1ms에서는네무릎모두넘었다.5ms에서FR초과25ms/RL5ms/RR10ms다. **이번2.5m/s 모델도 공식 소프트웨어 속도 경계 검사에 실패한다.** 실제 EDU가 같은 속도로 움직인다는 뜻은 아니며 실제 구동 한계 때문에 추종이 깨질 가능성도 별도 검증해야 한다.

기존 학습모터`Go2HV`는모든관절에X1=13.5/X2=30rad/s,Y1=20.2/Y2=23.4Nm를사용한다. 이는URDF의calf15.7rad/s·45.43Nm와동일한계약이아니다. 목표변화율16rad/s 제한도 실제dq 한도를 보장하지 않는다.

## 토크·힘·실기 한도 판정

- 세시험의 적용토크는공식URDF effort(hip/thigh23.7Nm,calf45.43Nm) 경계안이다. 학습조건최대적용23.4Nm, 제한전PD요구최대44.76Nm로모터포화가존재한다. 요구토크와실제적용토크를구분한다.
- 학습조건5ms의개별발최대합력은FL182.53/FR124.03/RL140.24/RR132.80N. 전체수직반력최대337.82N/2.26BW다.1ms전체최대2.42BW, 공식물성근사2.58BW였다.
- 이힘은초기낙하·보행출발·전환을포함한전체시험의값이다. 정식갤롭착지력이라고부르지않는다.
- 공식동적발하중·충격허용치, 연속/RMS모터토크·전류·열·배터리한도는확인되지않아안전판정UNKNOWN이다. 프로젝트3BW나제품적재량/최대45Nm를공식착지하중허용치로대체하지않는다.
- 일반Go2공식URDF/MJCF는보유EDU의시리얼·펌웨어별하드웨어인증자료가아니다. 위각도/토크통과만으로실배포가능판정을하지않는다.

공식자료내용과Git blob SHA는[sources](sources/)에이전동일일자조사본을보존했다. 관련모델·배포제약의상세근거는[이전50k검증](../20261111_gallop_50000_sim2sim_review/README.md)과동일하다.

## 공식 물성 근사 실패는 공통 출발 경로

공식물성근사는기존소스MJCF정지점·damping/armature/frictionloss·로봇접촉설정·정적모터범위를복원하고, 평지마찰.4에서LowCmd와같은PD식을계산했다. 저장된목표필터와teacher/전환로직은유지했다. DDS실행이나실물식별모델은아니다.

이시험은마지막명령1m/s에실제약0.00051m/s였고학생혼합0%에서출발기한종료했다. 두모델의동결보행teacher SHA256은모두 `d4bd406a2e0cc97b1370610a3791bf115c31cac8b718056322d95df2f9d192bf`이며, 두공식근사시험의실제q/dq기록은완전히같다. 따라서이실패를두갤롭학생정책의차이로해석할수없다. **공통기본보행·구동/접촉환경경로부터확인해야한다.** 여러물성을함께바꿨으므로어느한파라미터의원인인지이번시험만으로확정하지않는다.

## 배포 판정과 다음 작업

실배포는보류한다. 현재49관측/12관절정책의teacher·위상·전환·필터가5행동GPD DDS제어기에연결돼있지않다. 전환supervisor는시뮬레이터의정확한선속도·높이를사용하며검증된실기추정기경로가없다. ±1로자르는일반exporter도현재unbounded행동계약에사용할수없다. 정식동일C++/DDS스택·수신timeout·센서나이·지연검증은남아있다.

다음수정은공통teacher출발의모터/접촉모델차이를분리하고, 관절별실제속도·구동한도를맞춘뒤 전환을검증하는순서가적절하다. 보호범위를넓혀이번종료를통과시키지않는다. 이번요청에서는보상·정책·필터·종료조건을변경하지않았다.

원래보호30k/46k/50k와원본학습·플레이를보존했다. 이번46k평가를이유로최신완료학습50k또는기본부모선택을자동교체하지않았다.

## 재실행

```bash
cd /home/cvr/GH/go2_animalization/gallop
MUJOCO_GL=glfw .venv/bin/python -m scripts.validation.validate_gallop_edu \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261110_223314_cef16b/model_final.pt \
  --official-urdf reports/20261111_gallop_46000_sim2sim_review/sources/official_go2_description.urdf \
  --profile trained --seconds 30 --viewer --video --replay-seconds 90
```

공식물성근사는`--profile sdk_proxy`,1ms검사는`--physics-dt .001`을사용한다. 실행마다새`runs/mujoco/evaluate/unitree_go2_gallop/<ID>/`에저장하며PPO·실기제어는실행하지않는다.
