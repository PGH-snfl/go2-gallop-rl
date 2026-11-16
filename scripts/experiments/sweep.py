"""Small explicit baseline sweep; exports all trials, never relabels a failure as success."""
import argparse
from gpd.paths import new_run
import itertools
import json
from pathlib import Path
import numpy as np
from gpd.config import load_config
from scripts.validation.evaluate import evaluate, save_run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--output", default=str(new_run("experiments","sweep")))
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    for i, (gamma, kd, gain) in enumerate(itertools.product([0, 0.2, 0.5], [16, 40, 75], [0.03, 0.10])):
        cfg = load_config(gpd_gamma=gamma, swing_kd=kd, foot_velocity_gain=gain, episode_seconds=args.seconds)
        env, report = evaluate(cfg)
        survived = env.records[-1]["time"] - cfg["settle_seconds"]
        score = survived + 3 * report.get("valid_cycle_fraction", 0) - report.get("velocity_rmse", 1)
        save_run(env, report, out / f"trial_{i:02d}")
        row = {"trial": i, "gamma": gamma, "kd": kd, "gain": gain, "survived": survived, "score": score, "accepted": report["accepted"], "speed": report.get("mean_speed"), "gallop_fraction": report.get("valid_cycle_fraction", 0)}
        trials.append(row)
        print(json.dumps(row), flush=True)
    trials.sort(key=lambda r: r["score"], reverse=True)
    (out / "trials.json").write_text(json.dumps(trials, indent=2))

if __name__ == "__main__":
    main()
