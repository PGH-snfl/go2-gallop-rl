"""Canonical project destinations; all new run outputs use these paths."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

def new_run(backend, mode):
    if backend not in ("isaac", "mujoco", "experiments"):
        raise ValueError(f"Unknown run backend: {backend}")
    if mode not in ("train", "check", "play", "export", "evaluate", "dds", "sweep", "tune"):
        raise ValueError(f"Unknown run mode: {mode}")
    return RUNS / backend / mode / (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:6])
