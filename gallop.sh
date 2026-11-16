#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/go2-gallop-matplotlib"
if [[ "${1:-}" == *.py && "${1:-}" != */* ]]; then
  SCRIPT_NAME="${1%.py}"
  for SCRIPT_GROUP in training validation analysis data experiments; do
    if [[ -f "scripts/$SCRIPT_GROUP/$SCRIPT_NAME.py" ]]; then
      shift
      exec .venv/bin/python -m "scripts.$SCRIPT_GROUP.$SCRIPT_NAME" "$@"
    fi
  done
fi
exec .venv/bin/python "$@"
