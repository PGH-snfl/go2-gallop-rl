import json
from pathlib import Path
import numpy as np
from . import ROOT

def load_config(path=None, **overrides):
    cfg = json.loads(Path(path or ROOT / "configs/flat_gallop.json").read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("simulation_dt", "control_dt", "mpc_dt", "policy_dt", "period"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for fast, slow in (("simulation_dt", "control_dt"), ("control_dt", "mpc_dt"), ("control_dt", "policy_dt")):
        ratio = cfg[slow] / cfg[fast]
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(f"{slow} must be an integer multiple of {fast}")
    if cfg["lead"] not in ("left", "right"):
        raise ValueError("lead must be left or right")
    if not 0 <= cfg["gpd_gamma"] <= 1:
        raise ValueError("gpd_gamma must be within [0,1]")
    return cfg
