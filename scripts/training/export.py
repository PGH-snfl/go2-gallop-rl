"""Export an Isaac Lab RSL-RL checkpoint with its deployment configuration."""
import argparse
import os
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", type=Path)
p.add_argument("--output", type=Path, required=True, help="New bundle directory, not a .pt file")
a = p.parse_args()
from gpd.paths import ROOT as root
python = root / ".isaacvenv/bin/python"
os.execv(str(python), [str(python), "-u", "-m", "scripts.training.launch_isaac", "export", "--headless", "--num-envs", "1", "--checkpoint", str(a.checkpoint.resolve()), "--output", str(a.output.resolve())])
