"""Compare 48 published Genesis rollouts without launching or controlling a robot.

The export has no ground reaction forces. Height-based patterns remain proxies.
The source arrays use group-major joints; rollouts use SDK leg-major joints.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scripts.analysis.compare_kine2go import ROOT, FEET, proxy_metrics, segment_metrics, forward_kinematics

SDK_TO_GROUP = [3,0,9,6,4,1,10,7,5,2,11,8]
EXPECTED_LINKS = [13,15,14,16]
RADIUS = 0.022
THRESHOLDS = [0.0,0.005,0.01,0.015]

def load_array(path):
    frames = torch.load(path,map_location="cpu",weights_only=True)
    if isinstance(frames,dict):
        frames=frames["frames"]
    def stack(key):
        return np.stack([f[key].numpy().reshape(-1) for f in frames])
    q=stack("dof_pos")[:,SDK_TO_GROUP]
    qdot=stack("dof_vel")[:,SDK_TO_GROUP]
    position=stack("base_pos");quat=stack("base_quat")
    links=np.stack([f["links_pos"].numpy() for f in frames])
    fk=forward_kinematics(position,quat,q)
    # Independently validate joint and foot mapping for every rollout.
    cost=np.linalg.norm(fk[:,:,None]-links[:,None],axis=-1).mean(axis=0)
    rows,columns=linear_sum_assignment(cost)
    if columns.tolist()!=EXPECTED_LINKS or cost[rows,columns].max()>1e-4:
        raise ValueError(f"Joint/link mapping mismatch: {path}, {columns}, {cost[rows,columns]}")
    feet=links[:,EXPECTED_LINKS]
    a=np.zeros((len(frames),61),dtype=float)
    a[:,6:18]=q;a[:,24:36]=qdot;a[:,36:48]=feet.reshape(-1,12)
    a[:,48:51]=position;a[:,51:55]=quat
    a[:,55:58]=stack("base_lin_vel");a[:,58:61]=stack("base_ang_vel")
    frame_ids=stack("frame")[:,0]
    return a,frame_ids,cost[rows,columns]

def main():
    (ROOT/"reports/kine2go").mkdir(parents=True,exist_ok=True)
    references=json.loads((ROOT/"reports/kine2go/comparison.json").read_text())
    byname={r["name"]:r for r in references["records"]}
    results=[]
    for path in sorted((ROOT/"data/rollouts").glob("*/*.pkl")):
        a,frames,errors=load_array(path)
        name=path.parent.name;fps=byname[name]["fps"]
        if not np.all(np.diff(frames)==1):
            raise ValueError(f"Nonuniform frame IDs: {path}")
        metrics=segment_metrics(a,np.full(4,RADIUS),fps)
        metrics["proxy"]=[proxy_metrics(a[:,36:48].reshape(-1,4,3),np.full(4,RADIUS),fps,t,False) for t in THRESHOLDS]
        result={"name":name,"trajectory":path.stem,"frame_first":int(frames[0]),"frame_last":int(frames[-1]),
                "mapping_fk_error_m":errors.tolist(),"metrics":metrics}
        results.append(result)
        p=metrics["proxy"][1]
        print(name,path.stem,f"v={metrics['mean_forward_speed_m_s']:.2f} pitch={metrics['pitch_excursion_deg']:.1f} airproxy={p['all_feet_above_threshold_fraction']:.2f}",p["pattern_counts"],flush=True)
    (ROOT/"reports/kine2go/rollout_comparison.json").write_text(json.dumps({"dataset_revision":references["dataset_revision"],"rollouts":results,
        "contact_note":"No GRF fields exist in the export. Geometric proxy: foot link z <= 0.022 m radius + tolerance; ground plane assumed z=0.",
        "coverage":"One rollout per 40 clips; three rollouts each for canter, run_00, run_02, run_04. Not all 800 trajectories."},ensure_ascii=False,indent=2)+"\n")
    with (ROOT/"reports/kine2go/rollout_comparison.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["name","trajectory","frames","forward_speed","pitch_excursion_deg","air_proxy_radius_plus_5mm","pattern_counts"])
        for r in results:
            m=r["metrics"];p=m["proxy"][1];writer.writerow([r["name"],r["trajectory"],m["frames"],m["mean_forward_speed_m_s"],m["pitch_excursion_deg"],p["all_feet_above_threshold_fraction"],json.dumps(p["pattern_counts"])])
    plot_comparison(byname,results)

def plot_comparison(byname,results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=["ai4_dog_canter","ai4_dog_run_00","ai4_dog_run_02","ai4_dog_run_04","ai4_dog_trot_01","solo8_jump_forward_a","ai4_dog_synth_half_flip_jump"]
    fig,axes=plt.subplots(len(names),2,figsize=(15,15),constrained_layout=True)
    for row,name in enumerate(names):
        r=byname[name];a=np.load(ROOT/"data/references"/name/"motion.npy")[slice(*r["config_motion_range"])]
        b,_,_=load_array(ROOT/"data/rollouts"/name/"traj_0000.pkl")
        for col,(data,baseline,threshold) in enumerate([(a,np.array(r["foot_baseline_2pct_m"]),0.02),(b,np.full(4,RADIUS),0.005)]):
            ax=axes[row,col];feet=data[:,36:48].reshape(-1,4,3)
            mask=feet[:,:,2]<=baseline+threshold;n=min(len(data),120);time=np.arange(n)/r["fps"]
            for i,foot in enumerate(FEET):
                ax.fill_between(time,i+0.15,i+0.85,where=mask[:n,i],step="mid",color=["#377eb8","#4daf4a","#e41a1c","#984ea3"][i])
            flight=~mask[:n].any(axis=1)
            ax.fill_between(time,-0.2,4.1,where=flight,step="mid",color="gray",alpha=0.18)
            ax.set_yticks(np.arange(4)+0.5,FEET);ax.set_ylim(-0.2,4.1);ax.invert_yaxis();ax.set_xlim(0,2)
            ax.set_title(name+(" / selected reference" if col==0 else " / Genesis rollout 0000"),fontsize=10)
            ax.grid(axis="x",alpha=.2)
            if row==len(names)-1:ax.set_xlabel("Time (s), no artificial loop repetition")
    fig.suptitle("Kine2Go gallop candidates: kinematic near-ground proxies\nColored = foot near ground; gray = all four feet above threshold. No measured contact forces.",fontsize=14)
    fig.savefig(ROOT/"reports/kine2go/foot_patterns.png",dpi=160)
    fig.savefig(ROOT/"reports/kine2go/foot_patterns.pdf")
    plt.close(fig)

if __name__=="__main__":
    main()
