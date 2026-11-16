"""Exact Python controller through Unitree SDK2 DDS against a separate C++ MuJoCo plant."""
import argparse
from gpd.paths import new_run
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np
from gpd import ROOT
from gpd.config import load_config
from gpd.controller import GallopController
from gpd.state import LowState
from gpd.metrics import summarize

def collect(stream, mailbox):
    for line in stream:
        if not line.startswith("S "):
            continue
        data = np.fromstring(line[2:], sep=" ")
        if len(data) != 39:
            continue
        try:
            mailbox.put_nowait(data)
        except queue.Full:
            try:
                mailbox.get_nowait()
            except queue.Empty:
                pass
            mailbox.put_nowait(data)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--actor")
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--output", default=str(new_run("mujoco","dds")))
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--binary", default=str(ROOT / "deploy/build/gallop_dds"))
    args = p.parse_args()
    cfg = load_config(args.config, episode_seconds=args.seconds)
    # C++ adapter currently implements this fixed plant contract.
    if any(abs(cfg[key]-value) > 1e-9 for key, value in
           (("simulation_dt", .001), ("height", .27), ("friction", .8))):
        raise SystemExit("DDS plant requires simulation_dt=.001, height=.27, friction=.8")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise SystemExit("Build deploy/CMakeLists.txt first")
    actor = None
    if args.actor:
        import torch
        torch.set_num_threads(1)
        actor = torch.jit.load(args.actor, map_location="cpu").eval()
    controller = GallopController(cfg)
    mailbox = queue.Queue(maxsize=1)
    records = []
    gateway = plant = None
    try:
        with (out / "gateway.log").open("w") as gateway_log, (out / "plant.log").open("w") as plant_log:
            gateway = subprocess.Popen([str(binary), "gateway"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=gateway_log, text=True, bufsize=1)
            reader = threading.Thread(target=collect, args=(gateway.stdout, mailbox), daemon=True)
            reader.start()
            plant = subprocess.Popen([str(binary), "plant", str(ROOT / "assets/flat_scene.xml"), str(args.seconds + cfg["settle_seconds"]), str(out / "plant.csv")], stdout=plant_log, stderr=plant_log)
            previous = -1
            next_action = cfg["settle_seconds"]
            deadline = time.monotonic() + args.seconds + 25
            timeout = False
            while time.monotonic() < deadline:
                try:
                    data = mailbox.get(timeout=0.2)
                except queue.Empty:
                    if plant.poll() is not None:
                        break
                    if records:
                        timeout = True
                        break
                    continue
                ts = float(data[0])
                if ts <= previous:
                    continue
                if previous >= 0 and ts-previous < cfg["control_dt"]-1e-7:
                    continue
                started = time.perf_counter()
                state = LowState(ts, data[1:5], data[5:8], data[8:11], data[11:23], data[23:35], data[35:39])
                policy_ms = 0.0
                if ts + 1e-9 >= next_action:
                    t0 = time.perf_counter()
                    if actor is None:
                        action = np.zeros(5)
                    else:
                        with torch.no_grad():
                            action = actor(torch.from_numpy(controller.observation()).unsqueeze(0))[0].numpy()
                    controller.set_action(action)
                    policy_ms = (time.perf_counter()-t0)*1000
                    next_action = ts + cfg["policy_dt"]
                torque, info = controller.step(state)
                gateway.stdin.write("T " + " ".join(str(float(v)) for v in torque) + "\n")
                gateway.stdin.flush()
                previous = ts
                if ts >= cfg["settle_seconds"]:
                    records.append({"time": ts, "desired": info.get("desired", controller.schedule.mask()), "measured": controller.estimator.contact.mask.copy(), "estimated": info.get("x_est", np.zeros(12)), "torque": torque.copy(), "dq": state.dq.copy(), "foot_force": state.foot_force.copy(), "phase": info.get("phase", 0), "clip_fraction": info.get("clip_fraction", 0), "cycle_ms": (time.perf_counter()-started)*1000, "solver_ms": info.get("solver_ms", 0), "policy_ms": policy_ms, "fault": info.get("fault")})
                if info.get("fault"):
                    break
                if ts >= args.seconds + cfg["settle_seconds"] - cfg["control_dt"]:
                    break
            try:
                plant.wait(timeout=1)
            except subprocess.TimeoutExpired:
                plant.terminate()
                plant.wait(timeout=3)
    finally:
        for process in (plant, gateway):
            if process is not None and process.poll() is None:
                if process is gateway and process.stdin:
                    process.stdin.close()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if not (out / "plant.csv").exists() or (out / "plant.csv").stat().st_size == 0:
        report = {"accepted": False, "sim2sim_ready": False, "reason": "DDS plant failed to start; inspect plant.log and gateway.log", "plant_exit": None if plant is None else plant.returncode, "gateway_exit": None if gateway is None else gateway.returncode}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    truth_rows = np.genfromtxt(out / "plant.csv", delimiter=",", skip_header=1)
    truth_rows = np.atleast_2d(truth_rows)
    if truth_rows.shape[1] != 43 or not np.isfinite(truth_rows).all():
        report = {"accepted": False, "sim2sim_ready": False, "reason": "Empty or invalid plant telemetry"}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(2)
    import pinocchio as pin
    for r in records:
        idx = min(np.searchsorted(truth_rows[:, 0], r["time"]), len(truth_rows)-1)
        row = truth_rows[idx]
        q, v = row[1:20], row[20:38]
        R = pin.Quaternion(*q[3:7]).toRotationMatrix()
        r["truth"] = np.r_[q[:3], pin.rpy.matrixToRpy(R), v[:3], R @ v[3:6]]
        r["qpos"] = q.copy()
    terminated = bool(timeout or controller.fault)
    report = summarize(records, cfg, terminated, transport="dds")
    report.update({"gateway": "Unitree SDK2 / loopback / domain 1", "controller": "gpd.controller.GallopController", "physical_hardware": False, "timeout": timeout, "fault": controller.fault, "plant_command_stale_count": int(truth_rows[:, -1].sum())})
    if report["plant_command_stale_count"] or plant.returncode != 0:
        report["accepted"] = report["sim2sim_ready"] = False
    report["configured_in_process_delay_seconds"] = cfg.get("command_delay_seconds", 0.)
    report["delay_mode"] = "actual DDS latency; synthetic in-process delay is not applied"
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if records:
        np.savez_compressed(out / "rollout.npz", **{k: np.array([r[k] for r in records]) for k in records[0] if k != "fault"})
    print(json.dumps(report, indent=2))
    if args.require_pass and not report.get("sim2sim_ready", False):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
