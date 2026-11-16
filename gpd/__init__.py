"""Flat-ground, fixed-lead GPD-MPC and bounded residual learning."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT.parent / "go2-convex-mpc"
sys.path.insert(0, str(VENDOR / "src"))
