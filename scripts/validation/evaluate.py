"""Evaluate nominal or exported actor on the flat plant, recording measured contacts."""
import argparse
from gpd.paths import new_run
import json
from pathlib import Path
import numpy as np
from gpd.config import load_config
from gpd.env import GallopEnv
from gpd.metrics import summarize

def evaluate(cfg, policy=None, seed=0):
    env = GallopEnv(cfg)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.zeros(5) if policy is None else policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
    report = summarize(env.records, cfg, terminated)
    return env, report

def save_run(env, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "config.json").write_text(json.dumps(env.cfg, indent=2))
    if env.records:
        keys = [k for k in env.records[0] if k != "fault"]
        np.savez_compressed(output / "rollout.npz", **{k: np.array([r[k] for r in env.records]) for k in keys})
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--speed", type=float)
    p.add_argument("--lead", choices=["left", "right"])
    p.add_argument("--seconds", type=float)
    p.add_argument("--actor", help="Exported TorchScript actor")
    p.add_argument("--output", default=str(new_run("mujoco","evaluate")))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config, speed=args.speed, lead=args.lead, episode_seconds=args.seconds)
    policy = None
    if args.actor:
        import torch
        torch.set_num_threads(1)
        actor = torch.jit.load(args.actor, map_location="cpu").eval()
        def policy(obs):
            with torch.no_grad():
                return actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
    env, report = evaluate(cfg, policy, args.seed)
    save_run(env, report, args.output)
    print(json.dumps(report, indent=2))
    if args.require_pass and not report["accepted"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
