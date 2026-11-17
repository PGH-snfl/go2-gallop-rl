# 이 컴퓨터의 코드 편집 환경

Python 3.11.16 전용 가상환경(`.venv`)과 Ruff 0.17.5만 설치했습니다. Ruff는 코드를 실행하지 않고 오류를 찾는 도구입니다. 시스템 Python은 변경하지 않았습니다.

## 시작하기

VS Code에서 상위 폴더의 `Gallop.code-workspace` 또는 이 저장소 폴더를 엽니다. Python 환경은 `.venv/bin/python`을 사용하도록 설정했습니다. 이전에 다른 환경을 선택했다면 `Python: Select Interpreter`에서 이 경로를 선택하세요. Python·Pylance·Ruff 확장 기능은 추천 목록에 등록했습니다.

저장소 폴더의 터미널에서:

```bash
source .venv/bin/activate
python -B -m scripts.validation.check_code_editing
```

활성화 없이 `.venv/bin/python -B -m scripts.validation.check_code_editing`으로도 실행할 수 있습니다. VS Code에서는 `Tasks: Run Test Task` → `Gallop: code checks only`를 선택합니다.

검사 범위는 현재 Python 소스의 문법, 설정 JSON, 기본 정적 오류, 셸 스크립트 문법과 기존 파일 복원·해시 검사 4개입니다. 과거 보고서의 소스 사본은 검사 대상에서 제외합니다. 학습·시뮬레이션·외부 다운로드는 실행하지 않습니다. `unittest discover` 전체 실행은 학습용 의존성과 산출물이 필요하므로 편집기에서는 `test_portable_handoff.py`만 검색합니다.

## 수정할 위치

| 위치 | 내용 |
| --- | --- |
| `isaac_task/gallop_core.py` | 갤롭 조건과 제어·제약 로직 |
| `isaac_task/` | Isaac 학습 환경·실행 지원 |
| `gpd/` | 공통 제어 코드 |
| `configs/gallop_2mps_50cm_velocity_guard.json` | 다음 학습에 사용할 설정 |
| `tests/` | 기존 검사 코드 |
| `handoff/resume-state.json` | 다른 학습 컴퓨터에 전달할 진행 상태 |

학습 패키지(PyTorch·Isaac Sim/Lab·RSL-RL), MuJoCo 및 외부 저장소는 설치하지 않았고, Release의 대용량 모델·로그도 복원하지 않았습니다. 따라서 외부 라이브러리 내부 자동완성과 전체 실행 검증은 제한됩니다. 미설치 라이브러리 경고는 편집기에서 숨겼습니다. 학습 보상·정책·설정 값과 기존 모델 해시는 변경하지 않았습니다.

## 편집 환경 다시 만들기

`uv`가 설치된 컴퓨터에서 저장소 루트 기준으로 실행합니다. 기존 `.venv`는 지우지 말고 필요할 때만 환경을 새로 준비하세요.

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.tooling/python" UV_CACHE_DIR=/tmp/go2-gallop-uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=/tmp/go2-gallop-uv-cache uv pip install --python .venv/bin/python -r requirements-edit.txt
```

`.venv`와 `.tooling`은 Git에서 제외되어 있습니다. 다른 컴퓨터에서 학습할 때는 [NEW_COMPUTER.md](NEW_COMPUTER.md)를 사용합니다.
