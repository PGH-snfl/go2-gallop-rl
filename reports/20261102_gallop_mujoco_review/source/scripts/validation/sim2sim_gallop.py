"""Run a saved Gallop/startup checkpoint on nominal flat-ground MuJoCo; no PPO."""
import argparse
import json
import os
from pathlib import Path
import time

from gpd.paths import ROOT,new_run


def render_trace(plant,trace,path):
    import imageio.v2 as imageio
    import mujoco as mj
    renderer=mj.Renderer(plant.model,height=480,width=640)
    camera=mj.MjvCamera();mj.mjv_defaultCamera(camera)
    camera.distance=2.7;camera.azimuth=135;camera.elevation=-18
    writer=imageio.get_writer(path,fps=round(1/plant.control_dt),codec='libx264',quality=7,macro_block_size=16)
    try:
        for q in trace['mujoco_qpos'][:,0]:
            plant.data.qpos[:]=q;mj.mj_forward(plant.model,plant.data)
            camera.lookat[:]=q[:3];renderer.update_scene(plant.data,camera=camera);writer.append_data(renderer.render())
    finally:writer.close();renderer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml')
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--isaac-play',type=Path)
    p.add_argument('--video',action='store_true');p.add_argument('--observation-noise',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not 0<a.seconds<=30:p.error('Use 0 < --seconds <= 30 for one validation episode')
    os.environ.setdefault('MUJOCO_GL','egl')
    import numpy as np
    import torch
    import mujoco as mj
    from gpd.rl_mujoco import GallopMuJoCo,sha
    from scripts.analysis.collect_gallop import analyze_rollout
    torch.set_num_threads(1)
    folder=a.output or (lambda r:r.parent/'unitree_go2_gallop'/r.name)(new_run('mujoco','evaluate'))
    folder.mkdir(parents=True,exist_ok=False)
    status=dict(status='starting',checkpoint=str(a.checkpoint.resolve()),policy_updates=0,physical_hardware=False,
        engine='MuJoCo',mujoco_version=mj.__version__,seconds_requested=a.seconds,observation_noise=a.observation_noise)
    def save():(folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def emit(record):
        line=json.dumps(record,ensure_ascii=False)
        with (folder/'progress.jsonl').open('a') as stream:stream.write(line+'\n')
        print(line,flush=True)
    save();started=time.monotonic()
    try:
        plant=GallopMuJoCo(a.checkpoint,a.model,folder,noise=a.observation_noise)
        (folder/'model_contract.json').write_text(json.dumps(plant.contract,indent=2)+'\n')
        status.update(status='running',physics_dt=plant.physics_dt,control_dt=plant.control_dt)
        save();emit(dict(event='READY',output=str(folder),physics_dt=plant.physics_dt))
        samples={};steps=round(a.seconds/plant.control_dt)
        for i in range(steps):
            sample=plant.step(last=i==steps-1)
            for key,value in sample.items():samples.setdefault(key,[]).append(value)
            if (i+1)%250==0:
                emit(dict(event='PROGRESS',time=round(float(plant.data.time),2),stage=int(plant.state.used_stage[0]),
                    vx=float(plant.tensor.root_lin_vel_b[0,0]),height=float(plant.data.qpos[2])))
            if plant.fault:break
        trace={key:np.stack(value) for key,value in samples.items()}
        n=len(samples['phase']);trace.update(time=np.arange(1,n+1)*plant.control_dt,valid=np.ones((n,1),bool),episode_reset=np.zeros((n,1),bool))
        np.savez_compressed(folder/'gallop_trace.npz',**trace)
        np.savez_compressed(folder/'physics_trace.npz',**{key:np.array([row[key] for row in plant.physics]) for key in plant.physics[0]})
        unchanged=plant.unchanged()
        if not unchanged:raise RuntimeError('Policy checkpoint or weights changed')
        episode=plant.state.last_episode
        analysis=analyze_rollout(trace,plant.manifest['gallop_settings'],[episode])
        phases=[]
        for stage in range(4):
            mask=trace['startup_stage'][:,0]==stage
            if mask.any():phases.append(dict(stage=stage,first_seconds=float(trace['time'][mask][0]),last_seconds=float(trace['time'][mask][-1]),
                samples=int(mask.sum()),mean_vx=float(trace['body_velocity'][mask,0,0].mean())))
        report=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),completed_seconds=n*plant.control_dt,
            terminated=bool(plant.fault),termination_reason=plant.fault,policy_updates=0,policy_state_unchanged=unchanged,
            startup_stages=phases,episode=episode,analysis=analysis,
            full_criteria_passed=bool(episode.get('success',False)),
            nominal_motion_transfer=bool(not plant.fault and episode.get('startup_completed') and episode.get('flight') and episode.get('motion')),
            limitations=['One nominal deterministic seed; not a robustness or hardware acceptance test.',
                'Separate official Unitree MuJoCo collision/inertia model; nominal MJ mass differs from training USD.',
                'No random mass/material/gain perturbations, pushes, estimator error or transport delay.',
                'Passive armature/damping set to zero; 0.01Nm frictionloss approximates configured joint friction.',
                'Body COM velocity, joint order and observations match the saved policy contract; no retraining or threshold tuning.'])
        if a.isaac_play:
            isaac_manifest=json.loads((a.isaac_play/'manifest.json').read_text())
            if isaac_manifest['source_checkpoint']['sha256']!=plant.checkpoint_hash:raise ValueError('Isaac comparison uses a different checkpoint')
            ep=[json.loads(x) for x in (a.isaac_play/'episodes.jsonl').read_text().splitlines() if x]
            with np.load(a.isaac_play/'gallop_trace.npz') as archive:d={k:archive[k] for k in archive.files}
            report['isaac_comparison']=dict(play=str(a.isaac_play.resolve()),episode=ep[0],analysis=analyze_rollout(d,plant.manifest['gallop_settings'],ep))
        (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        if a.video:
            try:render_trace(plant,trace,folder/'mujoco_gallop.mp4');report['video']=str(folder/'mujoco_gallop.mp4')
            except Exception as error:report['video_error']=str(error)
            (folder/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        status.update(status='completed',terminated=report['terminated'],termination_reason=plant.fault,
            simulated_seconds=n*plant.control_dt,steps=n,policy_state_unchanged=unchanged,elapsed_seconds=round(time.monotonic()-started,2))
        save();emit(dict(event='COMPLETED',output=str(folder),terminated=report['terminated'],episode=episode,
            full_criteria_passed=report['full_criteria_passed'],video_error=report.get('video_error')))
    except Exception as error:
        status.update(status='error',error=str(error));save();raise


if __name__=='__main__':main()
