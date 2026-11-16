"""Interactive live evaluation or replay of a recorded physical rollout."""
import argparse
from gpd.paths import new_run
import json
from pathlib import Path
import time
import numpy as np
import mujoco as mj
import mujoco.viewer
from gpd.config import load_config
from gpd.env import GallopEnv
from gpd.sim import FlatPlant
from gpd.metrics import summarize
from scripts.validation.evaluate import save_run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--actor")
    p.add_argument("--replay", help="A saved rollout.npz")
    p.add_argument("--output", default=str(new_run("mujoco","play")))
    args = p.parse_args()
    if args.replay and not args.config:
        args.config = str(Path(args.replay).parent / "config.json")
    cfg = load_config(args.config, episode_seconds=args.seconds)
    if args.replay:
        data = np.load(args.replay, allow_pickle=False)
        plant = FlatPlant(cfg)
        with mj.viewer.launch_passive(plant.model, plant.data) as viewer:
            viewer.cam.distance = 1.7
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -15
            started = time.perf_counter()
            for index in range(len(data["time"])):
                if not viewer.is_running():
                    break
                plant.data.qpos[:] = data["qpos"][index]
                mj.mj_forward(plant.model, plant.data)
                viewer.cam.lookat[:] = plant.data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, started + data["time"][index]-data["time"][0]-time.perf_counter()))
        return
    env = GallopEnv(cfg)
    obs, _ = env.reset()
    actor = None
    if args.actor:
        import torch
        torch.set_num_threads(1)
        actor = torch.jit.load(args.actor, map_location="cpu").eval()
    terminated = truncated = False
    with mj.viewer.launch_passive(env.plant.model, env.plant.data) as viewer:
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        started = time.perf_counter()
        while viewer.is_running() and not (terminated or truncated):
            action = np.zeros(5)
            if actor is not None:
                with torch.no_grad():
                    action = actor(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.cam.lookat[:] = env.plant.data.qpos[:3]
            viewer.sync()
            target = started + env.plant.data.time - cfg["settle_seconds"]
            time.sleep(max(0, target-time.perf_counter()))
    report = summarize(env.records, cfg, terminated or not truncated)
    save_run(env, report, args.output)
    print(json.dumps({k:v for k,v in report.items() if k != "cycles"}, indent=2))

if __name__ == "__main__":
    main()
