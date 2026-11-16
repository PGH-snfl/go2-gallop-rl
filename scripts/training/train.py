"""Compatibility entrypoint: training always runs in Isaac Lab, never MuJoCo."""
import os
from pathlib import Path
import sys
from gpd.paths import ROOT as root
python = root / ".isaacvenv/bin/python"
os.execv(str(python), [str(python), "-u", "-m", "scripts.training.launch_isaac", "train", *sys.argv[1:]])
