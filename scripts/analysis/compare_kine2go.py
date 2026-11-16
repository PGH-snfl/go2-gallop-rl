"""Read-only comparison of every published Kine2Go reference.

Low-foot-height masks are kinematic proxies, NOT measured contact forces.
Both actual config slices and complete source arrays are retained in the report.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

from gpd.paths import ROOT
sys.path.insert(0, str(ROOT.parent / "scripts"))
from motion_reference import forward_kinematics, joint_limits

FEET = ["FL", "RL", "FR", "RR"]
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03]

def circ_distance(a, b):
    return min(abs(a-b), 1-abs(a-b))

def gait_from_phases(phases, tolerance=0.08):
    fl, rl, fr, rr = phases
    df, dh = circ_distance(fl, fr), circ_distance(rl, rr)
    diag = [circ_distance(fl, rr), circ_distance(fr, rl)]
    if max(diag) < tolerance:
        return "trot-like"
    if max(df, dh) < tolerance:
        return "bound-like"
    if max(circ_distance(fl, rl), circ_distance(fr, rr)) < tolerance:
        return "pace-like"
    if dh < tolerance and df >= tolerance:
        return "half-bound-like"
    if abs(df-0.5) < tolerance and abs(dh-0.5) < tolerance:
        return "symmetric-four-beat"
    if df >= tolerance and dh >= tolerance and abs(df-0.5) >= tolerance and abs(dh-0.5) >= tolerance:
        if min(diag) < tolerance:
            return "canter-like"
        fore_lead = "R" if (fr-fl) % 1 < 0.5 else "L"
        hind_lead = "R" if (rr-rl) % 1 < 0.5 else "L"
        return "transverse-gallop-like" if fore_lead == hind_lead else "rotary-gallop-like"
    return "other/ambiguous"

def proxy_metrics(feet, baseline, fps, threshold, cyclic=False):
    mask = feet[:, :, 2] <= baseline + threshold
    # Remove isolated one-frame toggles without bridging sustained flight.
    if len(mask) >= 3:
        original = mask.copy()
        agree = original[:-2] == original[2:]
        mask[1:-1] = np.where(agree, original[:-2], original[1:-1])
    n = len(mask)
    cycles = []
    starts = [np.flatnonzero((~mask[:-1, i]) & mask[1:, i]) + 1 for i in range(4)]
    mode = "observed consecutive RL touchdowns"
    if cyclic and n >= 12:
        # Only for short single-cycle references: hypothetical seam is explicit.
        extended = np.tile(mask, (3, 1))
        starts = [np.flatnonzero((~extended[:-1,i]) & extended[1:,i]) + 1 for i in range(4)]
        mode = "hypothetical repeated short clip; seam not physically validated"
    for start, end in zip(starts[1][:-1], starts[1][1:]):
        if cyclic and not (n <= start < 2*n):
            continue
        period = int(end-start)
        if not 0.15 <= period/fps <= 2.5:
            continue
        phases = []
        counts = []
        for events in starts:
            within = events[(events >= start) & (events < end)]
            counts.append(len(within))
            if len(within) == 1:
                phases.append(float((within[0]-start)/period))
        if counts != [1,1,1,1]:
            continue
        cycles.append({"start_local": int(start % n), "period_frames": period,
                       "period_s": period/fps, "phases_FL_RL_FR_RR": phases,
                       "pattern": gait_from_phases(phases)})
    votes = Counter(x["pattern"] for x in cycles)
    return {"threshold_above_each_foot_baseline_m": threshold,
            "near_ground_fraction_FL_RL_FR_RR": mask.mean(axis=0).tolist(),
            "all_feet_above_threshold_fraction": float((~mask.any(axis=1)).mean()),
            "touchdowns_per_foot": [int(len(x)) for x in starts],
            "cycle_mode": mode, "cycles": cycles, "pattern_counts": dict(votes)}

def segment_metrics(a, baseline, fps, original_start=0):
    if len(a) < 3:
        return {"frames": len(a)}
    feet = a[:,36:48].reshape(-1,4,3)
    q = a[:,6:18]
    qdot = np.gradient(q, 1/fps, axis=0)
    rpy = np.unwrap(Rotation.from_quat(a[:,[52,53,54,51]]).as_euler("xyz"), axis=0)
    velocity = np.gradient(a[:,48:51],1/fps,axis=0)
    local_v = Rotation.from_quat(a[:,[52,53,54,51]]).inv().apply(velocity)
    limits = joint_limits()
    violations = np.maximum(limits[:,0]-q,q-limits[:,1]).clip(min=0)
    # Compare endpoint positions only; this is not a periodic-feasibility certificate.
    qseam = np.degrees(np.abs(q[-1]-q[0])).max()
    return {"frames": len(a), "start_frame": original_start,
            "sample_duration_s": len(a)/fps, "elapsed_s": (len(a)-1)/fps,
            "mean_forward_speed_m_s": float(local_v[:,0].mean()),
            "mean_horizontal_path_speed_m_s": float(np.linalg.norm(velocity[:,:2],axis=1).mean()),
            "base_height_range_m": [float(a[:,50].min()),float(a[:,50].max())],
            "base_height_excursion_m": float(np.ptp(a[:,50])),
            "pitch_range_deg": np.degrees([rpy[:,1].min(),rpy[:,1].max()]).tolist(),
            "pitch_excursion_deg": float(np.degrees(np.ptp(rpy[:,1]))),
            "yaw_change_deg": float(np.degrees(rpy[-1,2]-rpy[0,2])),
            "peak_foot_height_above_baseline_m": (feet[:,:,2]-baseline).max(axis=0).tolist(),
            "max_joint_velocity_by_group_hip_thigh_calf_rad_s": np.abs(qdot).max(axis=0).reshape(3,4).max(axis=1).tolist(),
            "knee_speed_over_15_7_fraction": float((np.abs(qdot[:,8:12]) > 15.7).mean()),
            "joint_position_max_excess_rad": float(violations.max()),
            "endpoint_joint_max_difference_deg": float(qseam),
            "proxy": [proxy_metrics(feet,baseline,fps,t,len(a)<=50) for t in THRESHOLDS]}

def main():
    (ROOT/"reports/kine2go").mkdir(parents=True,exist_ok=True)
    ROOT.joinpath("reports/kine2go").mkdir(parents=True,exist_ok=True)
    records = []
    hashes = defaultdict(list)
    catalogue = {}
    for line in (ROOT.parent/"third_party/kine2go-pipeline/motion_imitation/motion_ranges.txt").read_text().splitlines():
        name, source, start, end = line.strip('"').split()
        catalogue[name] = {"source": source,"start":int(start),"end":int(end)}
    for directory in sorted((ROOT/"data/references").iterdir()):
        a = np.load(directory/"motion.npy",allow_pickle=False).astype(float)
        if a.ndim != 2 or a.shape[1] != 61 or not np.isfinite(a).all():
            raise ValueError(f"Invalid reference {directory}")
        meta = json.loads((directory/"clip.json").read_text())
        config = json.loads((directory/"config.json").read_text())
        fps = float(meta["fps"])
        selected_range = config.get("policy_cfg",{}).get("motion_range",[0,None])
        indices = np.arange(len(a))[slice(*selected_range)]
        selected = a[slice(*selected_range)]
        baseline = np.quantile(a[:,36:48].reshape(-1,4,3)[:,:,2],0.02,axis=0)
        fk = forward_kinematics(a[:,48:51],a[:,51:55],a[:,6:18])
        err = np.linalg.norm(fk-a[:,36:48].reshape(-1,4,3),axis=2)
        filehash = hashlib.sha256((directory/"motion.npy").read_bytes()).hexdigest()
        hashes[filehash].append(directory.name)
        record = {"name":directory.name,"fps":fps,"metadata":meta,
                  "array_frames":len(a),"config_motion_range":selected_range,
                  "catalogue":catalogue[directory.name],"sha256":filehash,
                  "foot_baseline_2pct_m":baseline.tolist(),
                  "fk_mean_error_m":float(err.mean()),"fk_max_error_m":float(err.max()),
                  "selected":segment_metrics(selected,baseline,fps,int(indices[0]) if len(indices) else 0),
                  "full_array":segment_metrics(a,baseline,fps)}
        records.append(record)
    manifest = json.loads((ROOT/"data/download_manifest.json").read_text())
    output = {"dataset_revision":manifest["revision"],"n_clips":len(records),
              "method":"Kinematic low-foot-height proxy with 2nd-percentile per-foot baseline; not GRF contact.",
              "short_clip_cyclic_limit":"Clips <=50 frames are repeated only for phase comparison; does not establish observed multi-stride stability.",
              "identical_reference_groups":[names for names in hashes.values() if len(names)>1],
              "records":records}
    (ROOT/"reports/kine2go/comparison.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    fields=["name","array_frames","selected_frames","start","end","forward_speed_m_s","pitch_excursion_deg","height_excursion_m","kinematic_air_fraction_2cm","patterns_2cm","knee_peak_rad_s","q_limit_excess_rad","fk_max_m"]
    with (ROOT/"reports/kine2go/comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            s=r["selected"];p=s["proxy"][2]
            writer.writerow(dict(name=r["name"],array_frames=r["array_frames"],selected_frames=s["frames"],start=r["config_motion_range"][0],end=r["config_motion_range"][1],forward_speed_m_s=s["mean_forward_speed_m_s"],pitch_excursion_deg=s["pitch_excursion_deg"],height_excursion_m=s["base_height_excursion_m"],kinematic_air_fraction_2cm=p["all_feet_above_threshold_fraction"],patterns_2cm=json.dumps(p["pattern_counts"]),knee_peak_rad_s=s["max_joint_velocity_by_group_hip_thigh_calf_rad_s"][2],q_limit_excess_rad=s["joint_position_max_excess_rad"],fk_max_m=r["fk_max_error_m"]))
    print(f"Compared {len(records)} references; max FK error: {max(r['fk_max_error_m'] for r in records):.7f} m")
    for r in records:
        s=r["selected"]; p=s["proxy"][2]
        print(r["name"],f"N={s['frames']} v={s['mean_forward_speed_m_s']:.2f} pitch={s['pitch_excursion_deg']:.1f} dz={s['base_height_excursion_m']:.3f} air_proxy={p['all_feet_above_threshold_fraction']:.2f}",p["pattern_counts"])

if __name__ == "__main__":
    main()
