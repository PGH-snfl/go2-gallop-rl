"""Generate a contact/attitude plot and an optional MP4 from actual logged qpos."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gpd.config import load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="Directory with rollout.npz and config.json")
    p.add_argument("--video", action="store_true")
    p.add_argument("--seconds", type=float, default=10)
    args = p.parse_args()
    run = Path(args.run)
    data = np.load(run / "rollout.npz", allow_pickle=False)
    cfg = load_config(run / "config.json")
    t = data["time"] - cfg["settle_seconds"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for axis, key, title in zip(axes[:2], ["desired", "measured"], ["Commanded contact", "Force-detected contact"]):
        axis.imshow(data[key].T, origin="lower", aspect="auto", interpolation="nearest", extent=[t[0],t[-1],-.5,3.5], vmin=0, vmax=1, cmap="Blues")
        axis.set_yticks(range(4), ["FL","FR","RL","RR"])
        axis.set_ylabel(title)
    axes[2].plot(t, data["truth"][:,6], label="measured vx")
    axes[2].axhline(cfg["speed"], color="black", linestyle="--", label="command")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].legend(loc="upper right")
    axes[3].plot(t, np.rad2deg(data["truth"][:,3]), label="roll")
    axes[3].plot(t, np.rad2deg(data["truth"][:,4]), label="pitch")
    axes[3].set_ylabel("Angle (deg)")
    axes[3].set_xlabel("Time after Gallop start (s)")
    axes[3].legend(loc="upper right")
    for axis in axes[2:]:
        axis.grid(alpha=.2)
    fig.suptitle(f"Go2 {cfg['lead']}-lead transverse contact pattern — actual physics rollout")
    fig.savefig(run / "contact_report.png", dpi=160)
    plt.close(fig)
    if args.video:
        import mujoco as mj
        import imageio.v2 as imageio
        from gpd.sim import FlatPlant
        plant = FlatPlant(cfg)
        camera = mj.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 1.7, 90, -15
        renderer = mj.Renderer(plant.model, height=540, width=960)
        try:
            with imageio.get_writer(run / "gallop.mp4", fps=60, macro_block_size=1) as writer:
                for at in np.arange(t[0], min(t[-1],args.seconds), 1/60):
                    index = min(np.searchsorted(t, at),len(t)-1)
                    plant.data.qpos[:] = data["qpos"][index]
                    mj.mj_forward(plant.model, plant.data)
                    camera.lookat[:] = plant.data.qpos[:3]
                    renderer.update_scene(plant.data, camera=camera)
                    writer.append_data(renderer.render())
        finally:
            renderer.close()
    print(run / "contact_report.png")

if __name__ == "__main__":
    main()
