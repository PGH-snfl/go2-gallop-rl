# Go2 Gallop RL

일반 전진보행 정책을 기반으로 갤롭 조건을 추가하는 Unitree Go2 프로젝트입니다. **Isaac Lab에서 학습하고, MuJoCo는 Sim2Sim 검증에만 사용합니다.**

현재 서버에서 완료된 모델은 **62,000회**입니다. 다음 수정은 안전 접촉·안정 착지의 거리 보상 분리와 착지 실패 사유 기록입니다. 목표는 **2.0m/s·실제 공중 전진거리50cm·공중시간0.24초 고정**이며 승급은 없습니다. 새 보상으로 학습한 결과는 아직 없습니다. 과거 정책의 실제 관절속도 초과·MuJoCo 전환 실패로 **실배포는 보류** 상태입니다.

## 다른 컴퓨터에서 시작

코드 편집만 진행하는 컴퓨터에서는 [코드 편집 환경](docs/CODE_EDITING.md)을 사용합니다. 아래 모델 복원과 학습 환경 설치는 학습·시뮬레이션 컴퓨터용입니다.

```bash
git clone https://github.com/PGH-snfl/go2-gallop-rl.git
cd go2-gallop-rl
python3 -m scripts.data.restore_artifacts
python3 -m scripts.data.bootstrap_repositories
```

[환경 설치·학습·플레이·검증 절차](docs/NEW_COMPUTER.md)를 이어서 진행하세요. 특정 사용자 홈 경로 대신 저장소 기준 경로와 번들 Go2 모델을 사용하며, 체크포인트·teacher·소스·asset의 원래 해시 검증을 유지합니다.

전체 중간 체크포인트·로그·영상·데이터·분석 자료까지 복원하려면:

```bash
sudo apt install zstd
python3 -m scripts.data.restore_artifacts --full
```

전체 원본은5,426파일/약8.54GB이며 Release에 분할 압축돼 있습니다. 최소 이어학습 묶음은 약360MB입니다. 첨부파일과 복원 파일은SHA256으로 검증합니다. 가상환경·캐시·컴파일 결과물은 컴퓨터별로 재생성하며, 원본 코드 사본도 전체 묶음에 보존했습니다.

## 다음 학습

서버에 수정본을 반영하고 [비학습 검사](docs/gallop_landing_credit.md)를 먼저 진행합니다. 아래62k 부모는 서버에 있으며 Release 최소 복원 묶음은58k까지입니다. 실제 학습은 사용자가 진행하며 영상은 저장하지 않습니다.

```bash
./isaac.sh gallop train \
  --headless --num-envs 256 --iterations 4000 --save-interval 500 \
  --finetune-from runs/isaac/train/unitree_go2_gallop/20261117_160234_3e2363/model_final.pt \
  --config configs/gallop_2mps_50cm_landing_credit.json
```

기존 고정 목표 부모이므로 `--reset-curriculum`을 넣지 않습니다. 새Adam으로 시작하고 새 결과는 `runs/isaac/train/unitree_go2_gallop/<새ID>/`에 저장합니다. 완료 시 누적66k입니다. 보상 수정과 코드 검사 통과가50cm 달성이나 실기 안전을 입증하지는 않습니다.

| 위치 | 내용 |
|---|---|
| `isaac_task/`, `gpd/` | 보상·환경·제동/제약·MuJoCo 공통 제어 |
| `configs/` | 현 설정과 과거 실험 설정 |
| `scripts/` | 학습·검증·수집·다른 컴퓨터 복원 |
| `reports/` | 분석 문서·그래프·검증; 대용량 파일은 전체복원 |
| `handoff/` | 이어학습 상태·환경 버전·전체 파일/Release 해시 |
| `assets/vendor/unitree_go2/` | 원래 Go2 모델과 BSD 라이선스 |
| `runs/`, `archive/`, `data/` | Release에서 복원하는 원본 산출물 |

[현재 보상 설계·명령](docs/gallop_landing_credit.md) · [관절 보호](docs/gallop_velocity_guard.md) · [새 컴퓨터 안내](docs/NEW_COMPUTER.md) · [전체 실험 경과](docs/PROJECT_HISTORY.md) · [작업 규칙](AGENTS.md) · [제3자 자료](THIRD_PARTY_NOTICES.md)
