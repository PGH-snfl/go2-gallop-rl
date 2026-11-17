# Go2 Gallop RL

일반 전진보행 정책을 기반으로 갤롭 조건을 추가하는 Unitree Go2 프로젝트입니다. **Isaac Lab에서 학습하고, MuJoCo는 Sim2Sim 검증에만 사용합니다.**

현재 서버에서 완료된 모델은 **66,000회**입니다. 안전 접촉·안정 착지 보상 분리 후 단일 플레이에서 평균 도약10.54cm·최대14.34cm·안정 착지21/40을 기록했습니다. 목표는 **2.0m/s·실제 공중 전진거리50cm·공중시간0.24초 고정**이며 아직 미달입니다. [수집 결과와 안전 한계](docs/gallop_66k_results.md)를 확인하세요. **실배포는 보류** 상태입니다.

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

## 최근 완료 학습

62k→66k 추가4,000회는 완료됐습니다. [보상 설계](docs/gallop_landing_credit.md)의62k 부모 명령은 이번 완료 학습의 재현용입니다. 최신66k 모델은 서버와 로컬 수집 보고서에 있고 Release 최소 복원 묶음은58k까지입니다. 다음66k→70k 준비는 아래를 확인하세요.

새 보상은 지급됐지만 체공 시간과 접촉 발 속도 조건이 남았습니다. 단일 플레이의 관절 한도 위반0회와 훈련 전체의 한도 종료13.56%를 구분합니다. 다음에는 여러 seed의 같은 조건 평가와 상승·접촉 동작을 검토해야 합니다.

## 다음 학습 준비

[실제 체공 시간 보상·접촉 발 속도 비용](docs/gallop_airtime.md)을 적용한66k→70k 추가4,000회 설정을 준비했습니다. 새 설정은 `configs/gallop_2mps_50cm_airtime.json`이며 관절·착지 한도와50cm·240ms 목표는 유지합니다. CPU 검사와 부모 텐서 복원은 확인했고,서버 비학습 검사 후 사용자가 영상 없이 학습합니다. 새 학습 성능은 아직 미검증입니다.

| 위치 | 내용 |
|---|---|
| `isaac_task/`, `gpd/` | 보상·환경·제동/제약·MuJoCo 공통 제어 |
| `configs/` | 현 설정과 과거 실험 설정 |
| `scripts/` | 학습·검증·수집·다른 컴퓨터 복원 |
| `reports/` | 분석 문서·그래프·검증; 대용량 파일은 전체복원 |
| `handoff/` | 이어학습 상태·환경 버전·전체 파일/Release 해시 |
| `assets/vendor/unitree_go2/` | 원래 Go2 모델과 BSD 라이선스 |
| `runs/`, `archive/`, `data/` | Release에서 복원하는 원본 산출물 |

[현재 보상 설계·명령](docs/gallop_airtime.md) · [66k 결과](docs/gallop_66k_results.md) · [관절 보호](docs/gallop_velocity_guard.md) · [새 컴퓨터 안내](docs/NEW_COMPUTER.md) · [전체 실험 경과](docs/PROJECT_HISTORY.md) · [작업 규칙](AGENTS.md) · [제3자 자료](THIRD_PARTY_NOTICES.md)
