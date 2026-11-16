import itertools
from gpd.paths import new_run
import json
from pathlib import Path
from gpd.config import load_config
from scripts.validation.evaluate import evaluate, save_run

def main():
    out = new_run("experiments","tune")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (speed, gamma, kd, height_weight) in enumerate(itertools.product([0.8, 1.2, 1.5], [0.25, 0.5], [16, 50], [500, 1500])):
        cfg = load_config(speed=speed, gpd_gamma=gamma, swing_kd=kd, foot_velocity_gain=0.03, episode_seconds=6, swing_feedforward=0.0)
        cfg["mpc_q"] = [1, 1, height_weight, 20, 40, 5, 8, 12, 20, 2, 2, 2]
        env, report = evaluate(cfg)
        survived = env.records[-1]["time"] - cfg["settle_seconds"]
        row = {"trial": i, "speed": speed, "gamma": gamma, "kd": kd, "height_weight": height_weight, "survived": survived, "valid_fraction": report.get("valid_cycle_fraction", 0), "accepted": report["accepted"], "velocity_rmse": report.get("velocity_rmse", 2)}
        rows.append(row)
        save_run(env, report, out / f"trial_{i:02d}")
        print(json.dumps(row), flush=True)
    (out / "trials.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
