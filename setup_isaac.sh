#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
BASE_PYTHON="${GALLOP_ISAAC_PYTHON:-$(command -v python)}"
"$BASE_PYTHON" -c 'import importlib.util; assert all(importlib.util.find_spec(m) for m in ("isaaclab","isaaclab_rl","rsl_rl","pinocchio","torch")), "Base Python needs Isaac Lab, RSL-RL and Pinocchio"'
"$BASE_PYTHON" -m venv --system-site-packages .isaacvenv
.isaacvenv/bin/python -m pip install casadi==3.6.7 mujoco==3.1.6 imageio imageio-ffmpeg matplotlib
