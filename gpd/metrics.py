import numpy as np

def summarize(records, cfg, terminated=False, transport="in_process"):
    rows = [r for r in records if r["time"] >= cfg["settle_seconds"] + cfg["warmup_metric_seconds"]]
    if not rows:
        return {"accepted": False, "sim2sim_ready": False, "reason": "No samples after warmup", "terminated": bool(terminated), "transport": transport}
    t = np.array([r["time"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    desired = np.array([r["desired"] for r in rows])
    measured = np.array([r["measured"] for r in rows])
    estimated = np.array([r["estimated"] for r in rows])
    order = [2, 3, 0, 1] if cfg["lead"] == "right" else [3, 2, 1, 0]
    events = []
    for j in range(1, len(rows)):
        for leg in np.flatnonzero(measured[j] & ~measured[j-1]):
            events.append((float(t[j]), int(leg)))
    anchors = [i for i, event in enumerate(events) if event[1] == order[0]]
    cycles, valid, periods = [], [], []
    for a, b in zip(anchors[:-1], anchors[1:]):
        seq = [leg for _, leg in events[a:b]]
        times = [at for at, _ in events[a:b]]
        unambiguous = len(times) == 4 and np.all(np.diff(times) > cfg["control_dt"] * 0.5)
        duration = events[b][0] - events[a][0]
        ok = seq == order and unambiguous and 0.65*cfg["period"] <= duration <= 1.35*cfg["period"]
        cycles.append({"start": events[a][0], "end": events[b][0], "sequence": seq, "valid": bool(ok)})
        valid.append(ok)
        periods.append(duration)
    rms = np.sqrt(np.mean(truth[:, 3:5]**2, axis=0))
    velocity_rmse = float(np.sqrt(np.mean((truth[:, 6]-cfg["speed"])**2)))
    contact_accuracy = float(np.mean(desired == measured))
    clip = float(np.mean([r["clip_fraction"] for r in rows]))
    timing = np.array([r["cycle_ms"] for r in rows])
    valid_fraction = float(np.mean(valid)) if valid else 0.
    suspension = float(np.mean(~measured.any(axis=1)))
    full_duration = t[-1] >= cfg["settle_seconds"] + cfg["episode_seconds"] - 2*cfg["control_dt"] - 1e-7
    gate = {"full_duration": bool(full_duration),
            "survived": not terminated and not any(r["fault"] for r in records),
            "enough_cycles": len(cycles) >= cfg["acceptance_min_cycles"],
            "transverse_contact_order": valid_fraction >= cfg["acceptance_valid_fraction"],
            "velocity": velocity_rmse <= cfg["acceptance_velocity_rmse"],
            "contact_tracking": contact_accuracy >= cfg["acceptance_contact_accuracy"],
            "attitude": float(rms.max()) <= cfg["acceptance_attitude_rms"],
            "torque_clipping": clip <= cfg["acceptance_clip_fraction"]}
    if cfg.get("require_suspension", False):
        gate["suspension"] = suspension >= cfg.get("acceptance_suspension_fraction", .02)
    return {"accepted": bool(all(gate.values())), "gates": gate, "transport": transport,
            "acceptance_scope": "transverse contact pattern; aerial phase required" if cfg.get("require_suspension", False) else "transverse contact pattern; aerial phase NOT required",
            "duration_after_warmup": float(t[-1]-t[0]), "lead": cfg["lead"], "cycles": cycles,
            "valid_cycle_fraction": valid_fraction, "velocity_rmse": velocity_rmse,
            "mean_speed": float(np.mean(truth[:, 6])), "roll_pitch_rms": rms.tolist(),
            "contact_accuracy": contact_accuracy, "measured_duty": measured.mean(axis=0).tolist(),
            "suspension_fraction": suspension, "torque_clip_fraction": clip,
            "estimator_velocity_rmse": float(np.sqrt(np.mean((truth[:, 6:9]-estimated[:, 6:9])**2))),
            "loop_ms": {"mean": float(timing.mean()), "p95": float(np.percentile(timing,95)), "p99": float(np.percentile(timing,99)), "max": float(timing.max()), "deadline_miss_fraction": float(np.mean(timing > cfg["control_dt"]*1000))},
            "sim2sim_ready": bool(all(gate.values()) and transport == "dds" and np.max(timing) <= cfg["control_dt"]*1000)}
