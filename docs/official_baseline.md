# 공식 원본 Go2 환경으로 기본 보행 학습

현재 기본 보행 재학습은 [curriculum 수정본](curriculum_fixed_baseline.md)의 `./isaac.sh baseline train ...`을 사용한다. 이 문서의 `official` 명령은 원본 비교용으로 유지한다.

최신 결과(2026-10-28): 실제2,000회 학습은 완료됐지만 저속 전진 명령도 추종하지 못했다. [수집·명령 추종 평가 보고서](../reports/20261028_unitree_official_review/README.md)를 먼저 읽는다. 아래 실행 방법의 연결 검증과 보행 성능을 구분한다.

기존 수정본 대신 `/home/cvr/GH/unitree/unitree_rl_lab_official`을 사용한다. Unitree 공식 GitHub에서 복제하고 [공식 커밋 4960b847](https://github.com/unitreerobotics/unitree_rl_lab/commit/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)에 고정했다. 최신 main이라고 주장하는 것이 아니라, 기존 비교 기준과 같은 공식 공개 버전을 재현하는 설정이다.

공식 checkout의 tracked 파일은 수정하지 않았다. `official_baseline.py` 실행기는 매번 commit, origin과 tracked 변경 유무를 검사한다. 수정된 소스면 학습을 거부한다. 기존 `/home/cvr/GH/unitree/unitree_rl_lab`과 그 모델은 보존한다.

## 원본 그대로 유지하는 설정

| 항목 | 값 |
|---|---:|
| JointPositionAction scale | 0.25 |
| feet_air_time 보상 | 0.1 |
| joint position limits 페널티 | −10 |
| action_rate 페널티 | −0.1 |
| joint_pos 페널티 | −0.7 |
| undesired_contacts 페널티 | −1 |
| calf 초기각 | −1.5 rad |
| 외부 밀기 이벤트 | 활성 |
| 활성 지형 | 해당 커밋 원본의 flat만 활성(terrain generator 사용) |
| 물리 / 정책 주기 | 5 / 20 ms |
| PPO 네트워크 | actor/critic 512×256×128 ELU |
| 관측 / 행동 | actor45, critic60 / 관절12 |

실행 시 위 값들을 다시 검사한다. 보상·curriculum·종료 조건·actuator 모델을 추가 조정하지 않는다.

## 실행기가 처리하는 경로·버전 호환

원본 환경 클래스 `RobotEnvCfg`와 원본 `BasePPORunnerCfg`를 직접 읽는다. 원본 파일은 수정하지 않고 아래만 런타임 인스턴스에서 처리한다.

1. USD placeholder를 기존 로컬 Unitree Go2 USD의 절대 경로로 연결한다. 루트 USD 경로/해시를 manifest에 기록한다.
2. RSL-RL 3.x의 `obs_groups`를 policy/critic으로 명시한다.
3. 원본 `empirical_normalization=False`에 맞게 actor/critic normalization=False를 명시한다.
4. 명령행의 환경 수·iteration·seed·device를 적용한다. 영상 사용 시 해상도/카메라만 바꾼다.

이 작업은 원본 task의 사양을 변경하는 것과 구분한다. Isaac Lab/RSL-RL 설치 전체를 원본의 과거 dependency 버전으로 되돌린 것은 아니며, 현재 설치된 Isaac Lab/PhysX/RSL-RL을 사용한다.

## 처음부터 2,000회 학습

```bash
cd /home/cvr/GH/go2_animalization/gallop
./isaac.sh official train --headless --num-envs 256 --iterations 2000 \
  --video --video-length 500 --video-interval 10000
```

새 가중치에서 시작한다. train 모드는 `--checkpoint`를 거부한다. 예전 수정본이나 GPD 모델을 자동으로 불러오지 않는다.

저장 위치는 `runs/isaac/train/unitree_go2_official/<실행ID>/`다. `model_1999.pt`, `model_final.pt`, `params/`, `manifest.json`, `status.json`, `progress.jsonl`, `console.log`, `videos/`를 저장한다. RSL-RL 기본 iteration 표와 TensorBoard 기록을 유지한다. 요청한2,000회는 원본 기본값50,000회를 명령행으로 줄인 것이며 수렴을 보장하는 횟수는 아니다.

## 플레이와 검증

```bash
./isaac.sh official play --num-envs 1 --steps 500 --video \
  --checkpoint runs/isaac/train/unitree_go2_official/실행ID/model_final.pt
```

play/check는 PPO 업데이트를 차단한다. 영상은 새 `runs/isaac/play/unitree_go2_official/<실행ID>/videos/`에 저장한다. 실행 완료 후에도 실제 속도 추종·연속 보행 성능은 별도 수치 평가가 필요하다. `check`는 연결 확인용이며 보행 합격 판정이 아니다.

평가 전용 고정 명령은 `--command 0.1 0 0`처럼 지정한다. 기본값은 원본 환경의 무작위 저속 명령이며 항상 전진하는 재생이 아니다. 새 play/check는 `play_trace.npz`, `play_trace.csv`에 명령·실제 속도·자세·접촉·행동 및 reset 여부를 저장한다. `--command`는 train에서 거부하며 학습 환경의 명령 분포는 변경하지 않는다.

2026-10-28 검증: `runs/isaac/check/unitree_official_verified_20261028/`. 1환경16step 신경망 추론, 관측45/60, 행동12, MP4 저장, 정책 업데이트0회, 실행 후 공식 소스 clean을 확인했다. 본학습은 수행하지 않았다.
