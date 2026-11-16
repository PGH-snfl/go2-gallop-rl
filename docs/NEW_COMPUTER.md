# 다른 컴퓨터에서 이어서 진행하기

이 저장소는 일반 전진보행 체크포인트에 갤롭 조건을 추가해 Isaac Lab에서 학습하고 MuJoCo에서 검증하는 전체 프로젝트다. 최신 완료 학습은58k이고, 다음 작업은 실제 관절속도 제약을 적용한 추가4k다. 새 제약으로62k 학습을 실행한 기록은 아직 없다. 최신 체크포인트 SHA256과 실제 진행상태는 `handoff/resume-state.json`에 있다.

## 1. 코드·모델 복원

```bash
git clone https://github.com/PGH-snfl/go2-gallop-rl.git
cd go2-gallop-rl
python3 -m scripts.data.restore_artifacts
python3 -m scripts.data.bootstrap_repositories
```

최소 복원 묶음에는 최신58k,54k,보호30k 및 동결 일반보행 teacher의 원래 run·manifest·소스·설정·체크포인트가 있다. 모든 중간 실험·로그·영상·원본 자료까지 복원하려면 다음을 추가한다.

```bash
sudo apt install zstd
python3 -m scripts.data.restore_artifacts --full
```

Release 첨부파일의 SHA256과 각 파일의 원래 SHA256을 검증한다. 기존 내용이 다른 파일은 덮어쓰지 않는다. 전체 원본 코드 사본은 `archive/original_source_20261116/`에 보존하고, Git의 이식성 수정 코드는 유지한다. 가상환경·캐시·컴파일 산출물은 컴퓨터별로 재생성한다. 전체 파일 목록과 제외 이유는 `handoff/artifact-files.json`에 있다.

## 2. Isaac 학습 환경

검증한 호스트는 Ubuntu22.04·Python3.11·NVIDIA GPU이며 기존 버전은 `handoff/environment-versions.json`을 따른다. 주요 조합은 Isaac Sim5.1.0,고정 Isaac Lab 커밋,PyTorch2.7.0+cu128,RSL-RL3.1.2,Pinocchio2.7.0이다. 다른 GPU/드라이버에서의 설치와 학습 성능까지 검증한 것은 아니다.

이미 이 조합의 Isaac Lab Python 환경이 있으면 해당 환경을 활성화하고 바로 `setup_isaac.sh`를 실행한다. 새 환경의 기본 설치 순서는 다음과 같다. Isaac Sim 설치는 NVIDIA의 라이선스 동의와 GPU 지원 드라이버가 필요하다.

```bash
conda create -n go2-gallop -c conda-forge python=3.11 pinocchio=2.7 numpy=1.26
conda activate go2-gallop
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
cd third_party/IsaacLab
./isaaclab.sh --install rsl_rl
cd ../..
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'rsl-rl-lib==3.1.2' 'gymnasium==1.2.1' 'tensordict==0.14.1' tensorboard imageio imageio-ffmpeg matplotlib
GALLOP_ISAAC_PYTHON="$(command -v python)" ./setup_isaac.sh
```

`bootstrap_repositories`는 정확한 upstream commit을 받는다. 수정된 기존 checkout을 덮어쓰지 않는다. 기본 경로는 `third_party/{IsaacLab,unitree_rl_lab,unitree_mujoco}`이며 필요하면 `GALLOP_ISAACLAB`, `GALLOP_UNITREE_RL_LAB`, `GALLOP_UNITREE_MUJOCO`로 지정한다. Go2 USD 전체 파일과 BSD 라이선스는 `assets/vendor/unitree_go2/`에 포함했다. 예전 manifest의 절대경로를 수정하지 않고 원본 asset SHA256이 일치하는 번들 파일을 선택한다.

## 3. 비학습 확인 후 추가학습

```bash
./isaac.sh gallop check --headless --num-envs 16 --steps 1000 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/model_final.pt \
  --config configs/gallop_2mps_50cm_velocity_guard.json

./isaac.sh gallop train --headless --num-envs 256 \
  --iterations 4000 --save-interval 500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/model_final.pt \
  --config configs/gallop_2mps_50cm_velocity_guard.json \
  --video --video-length 500 --video-interval 20000
```

현재 목표는2.0m/s·실제 공중 전진거리50cm·최소 공중시간0.24초 고정이며 승급이 없다. 부모는 이미 고정 목표 분기이므로 `--reset-curriculum`을 넣지 않는다. actor/critic/std는 복원하고 Adam은 새로 시작한다. 상세 보상·종료·제동 방식은 [현재 설계](gallop_velocity_guard.md)를 따른다.

## 4. 기존 정책 플레이·MuJoCo

```bash
./isaac.sh gallop play --headless --num-envs 1 --steps 1500 \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/model_final.pt \
  --video --video-length 1500 --video-interval 2000

python -m pip install 'mujoco==3.1.6' imageio imageio-ffmpeg
python -m scripts.validation.validate_gallop_edu \
  --checkpoint runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/model_final.pt \
  --official-urdf reports/20261111_gallop_46000_sim2sim_review/sources/official_go2_description.urdf \
  --profile trained --seconds 30 --video --viewer
```

MuJoCo는검증전용이고 학습하지 않는다. 58k 전체 플레이·Sim2Sim의 성공은 아직 검증하지 않았다. 과거54k는50cm 미달·실제 관절속도 초과·MuJoCo 전환 실패였으며 실배포 보류다. 제동 제약을 강화한 동결 부모 검사도 잦은 종료를 보였으므로 코드 검사 통과를 안전한 실기 정책으로 해석하지 않는다. 실제 하드웨어 명령은 위 명령에 포함돼 있지 않다.

전체 과거 검사를 실행하려면 전체 복원과 dependency bootstrap이 필요하다. 설치된 Isaac Python에서 `OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_gallop*.py'`를 사용한다. 원래 실험 기록은 `docs/PROJECT_HISTORY.md`,작업 규칙은 `AGENTS.md`,현황은 `docs/CONTINUE_HERE.md`에 보존했다.
