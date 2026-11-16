"""Flat-ground deterministic command tests for stock Unitree policy. Never trains."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from gpd.paths import ROOT,new_run
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--profile',choices=['nominal','training-randomized'],default='nominal')
p.add_argument('--repeats',type=int,default=1)
p.add_argument('--hold-seconds',type=float,default=20.)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--start-mode',choices=['stand-first','immediate'],default='stand-first')
p.add_argument('--video',action='store_true')
p.add_argument('--output',type=Path)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.repeats<1 or a.hold_seconds<5:p.error('repeats >= 1 and hold-seconds >= 5 required')
if not a.checkpoint.is_file():p.error('Checkpoint missing')
if a.video:a.enable_cameras=True
out=a.output or new_run('isaac','check')
out.mkdir(parents=True,exist_ok=True)
if (out/'status.json').exists():p.error('Use a fresh output folder')
status=dict(status='starting',checkpoint=str(a.checkpoint.resolve()),policy_updates=0,profile=a.profile,seed=a.seed)
def save(): (out/'status.json').write_text(json.dumps(status,indent=2))
save()
app=env=writer=None

def main():
    global env,writer
    import csv
    import inspect
    import shutil
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from tensordict import TensorDict
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import euler_xyz_from_quat
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from scripts.analysis.collect_unitree_baseline import load_yaml
    lab=ROOT.parents[1]/'unitree/unitree_rl_lab'
    sys.path.insert(0,str(lab/'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    def no_updates(*args,**kwargs):raise RuntimeError('Policy updates are forbidden in evaluation')
    PPO.update=no_updates
    cases=[('stand',[0,0,0],True),('forward_02',[.2,0,0],True),('forward_04',[.4,0,0],True),
           ('turn_left',[.2,0,.5],True),('turn_right',[.2,0,-.5],True),('forward_08_stress',[.8,0,0],False)]
    trials=[dict(name=name,command=cmd,required=required,repeat=repeat)
            for name,cmd,required in cases for repeat in range(a.repeats)]
    n=len(trials);cfg=RobotEnvCfg()
    source=Path(inspect.getfile(RobotEnvCfg))
    original=a.checkpoint.parent/'params/velocity_env_cfg.py'
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if original.exists() and sha(original)!=sha(source):raise RuntimeError('Environment source changed since training; review before evaluation')
    cfg.seed=a.seed;cfg.sim.device=a.device;cfg.scene.num_envs=n;cfg.scene.env_spacing=5.
    cfg.scene.terrain.terrain_type='plane';cfg.scene.terrain.terrain_generator=None
    cfg.scene.height_scanner=None
    cfg.curriculum.terrain_levels=None;cfg.curriculum.lin_vel_cmd_levels=None
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.resampling_time_range=(1e6,1e6)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.heading_command=False
    for key in ['lin_vel_x','lin_vel_y','ang_vel_z']:setattr(cfg.commands.base_velocity.ranges,key,(0.,0.))
    cfg.observations.policy.enable_corruption=False
    if a.profile=='nominal':
        cfg.events.physics_material=None;cfg.events.add_base_mass=None
        cfg.sim.physics_material.static_friction=.8;cfg.sim.physics_material.dynamic_friction=.8
        cfg.scene.terrain.physics_material=cfg.sim.physics_material
        cfg.events.reset_base.params['pose_range']={}
        cfg.events.reset_base.params['velocity_range']={}
        cfg.events.reset_robot_joints.params['velocity_range']=(0.,0.)
    else:
        cfg.observations.policy.enable_corruption=True
    hold_start=3. if a.start_mode=='stand-first' else 0.
    hold_end=hold_start+a.hold_seconds
    total_s=hold_end+5.
    cfg.episode_length_s=total_s+1.
    focal=next(i for i,c in enumerate(trials) if c['name']=='forward_04')
    cfg.viewer.resolution=(640,480)

    class MeasuredEnv(ManagerBasedRLEnv):
        def __init__(self,*args,**kwargs):
            self.terminal_snapshots={};self.terminal_reasons={};self.ids_ready=False
            super().__init__(*args,**kwargs)
            self.foot_ids,_=self.scene['robot'].find_bodies([f'{x}_foot' for x in ['FL','FR','RL','RR']],preserve_order=True)
            self.ids_ready=True
        def snapshot(self):
            r=self.scene['robot'].data
            roll,pitch,yaw=euler_xyz_from_quat(r.root_quat_w)
            rpy=torch.remainder(torch.stack([roll,pitch,yaw],dim=1)+torch.pi,2*torch.pi)-torch.pi
            sensor=self.scene['contact_forces'];ids=[sensor.body_names.index(f'{x}_foot') for x in ['FL','FR','RL','RR']]
            return {k:v.detach().cpu().numpy().copy() for k,v in dict(
                position=r.root_pos_w-self.scene.env_origins,rpy=rpy,body_velocity=r.root_lin_vel_b,
                angular_velocity=r.root_ang_vel_b,torque=r.applied_torque,q=r.joint_pos,dq=r.joint_vel,
                foot_force=sensor.data.net_forces_w[:,ids,2].clamp_min(0),
                foot_speed=r.body_lin_vel_w[:,self.foot_ids,:2].norm(dim=-1)).items()}
        def _reset_idx(self,ids):
            if self.ids_ready and len(ids) and (self.episode_length_buf[ids]>0).any():
                sample=self.snapshot()
                for i in ids.tolist():
                    if self.episode_length_buf[i]==0:continue
                    self.terminal_snapshots[i]={k:v[i].copy() for k,v in sample.items()}
                    self.terminal_reasons[i]=[name for name in self.termination_manager.active_terms if self.termination_manager.get_term(name)[i]]
            return super()._reset_idx(ids)
        def render(self,recompute=False):
            if a.video:
                target=self.scene['robot'].data.root_pos_w[focal].detach().cpu().numpy()+np.array([0,0,.05])
                self.sim.set_camera_view(target+np.array([1.6,-2.,1.]),target)
            return super().render(recompute=recompute)
    env=MeasuredEnv(cfg,render_mode='rgb_array' if a.video else None)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    if a.profile=='nominal':
        props=env.scene['robot'].root_physx_view.get_material_properties()
        props[...,0:2]=.8;props[...,2]=0
        env.scene['robot'].root_physx_view.set_material_properties(props,torch.arange(n,dtype=torch.int32))
    agent=load_yaml(a.checkpoint.parent/'params/agent.yaml')
    agent['device']=a.device
    runner=OnPolicyRunner(wrapped,agent,log_dir=None,device=a.device)
    runner.load(str(a.checkpoint.resolve()),load_optimizer=False,map_location=a.device)
    actor=runner.get_inference_policy(device=a.device)
    term=env.command_manager.get_term('base_velocity')
    dt=env.step_dt;steps=round(total_s/dt)
    targets=np.array([c['command'] for c in trials],dtype=np.float32)
    alive=np.ones(n,bool);failures=[None]*n
    trace={};masks=[];commands=[];times=[]
    if a.video:
        (out/'videos').mkdir()
        writer=imageio.get_writer(out/'videos/forward_04.mp4',fps=round(1/dt),macro_block_size=1)
    status.update(status='running',trials=trials,total_steps=steps,physics_dt=cfg.sim.dt,policy_dt=dt,
        start_mode=a.start_mode,
        protocol=dict(initial_stand_s=2. if a.start_mode=='stand-first' else 0.,
            ramp_s=1. if a.start_mode=='stand-first' else 0.,hold_s=a.hold_seconds,stop_ramp_s=1.,final_stand_s=4.),
        threshold_note='Engineering thresholds for this flat-ground baseline test, not official certification')
    save();print('[EVAL_START] '+str(out.resolve()),flush=True)
    last_video_t=None
    with torch.inference_mode():
        for step in range(steps):
            t=step*dt
            gain=(0. if t<2 else min(1.,t-2.)) if t<hold_start else 1. if t<hold_end else max(0.,1.-(t-hold_end))
            command=targets*gain
            term.vel_command_b[:]=torch.as_tensor(command,device=a.device)
            term.is_standing_env[:]=torch.as_tensor(np.abs(command).sum(axis=1)<1e-7,device=a.device)
            term.time_left.fill_(1e6)
            # Native terms keep a one-frame history. Refresh it after overriding
            # commands; otherwise compute(False) returns the previous zero command.
            obs=TensorDict(env.observation_manager.compute(update_history=True),batch_size=[n])
            input_command=obs['policy'][:,6:9].detach().cpu().numpy().copy()
            if not np.allclose(input_command,command,atol=1e-6):
                raise RuntimeError('Requested command does not match policy observation')
            action=actor(obs)
            if not torch.isfinite(action).all():raise RuntimeError('Nonfinite action')
            action[torch.as_tensor(~alive,device=a.device)]=0.
            env.terminal_snapshots={};env.terminal_reasons={}
            _,reward,done,_=wrapped.step(action)
            sample=env.snapshot()
            valid=alive.copy()
            for i,terminal in env.terminal_snapshots.items():
                for k,v in terminal.items():sample[k][i]=v
                if alive[i]:
                    failures[i]=dict(time_seconds=(step+1)*dt,reasons=env.terminal_reasons[i])
                    print('[FAIL] '+json.dumps(dict(trial=trials[i],**failures[i])),flush=True)
                    alive[i]=False
            for i in range(n):
                if alive[i] and (not all(np.isfinite(v[i]).all() for v in sample.values())):
                    failures[i]=dict(time_seconds=(step+1)*dt,reasons=['nonfinite_state']);alive[i]=False
            sample['action']=action.detach().cpu().numpy().copy()
            sample['policy_command']=input_command
            for k,v in sample.items():trace.setdefault(k,[]).append(v)
            masks.append(valid);commands.append(command.copy());times.append((step+1)*dt)
            if writer is not None and alive[focal]:
                writer.append_data(env.render());last_video_t=(step+1)*dt
            if (step+1)%250==0 or step==steps-1:
                status.update(step=step+1,surviving=int(alive.sum()),failures=failures);save()
                print(f'[PROGRESS] {step+1}/{steps}, sim={(step+1)*dt:.1f}s, surviving={alive.sum()}/{n}',flush=True)
    if writer is not None:writer.close();writer=None
    trace={k:np.stack(v) for k,v in trace.items()};time=np.array(times);valid=np.array(masks);command=np.stack(commands)
    np.savez_compressed(out/'trace.npz',time=time,valid=valid,command=command,**trace)
    thresholds=dict(vx_rmse=.1,vy_rmse=.1,yaw_rmse=.25,stand_speed_rms=.08,stop_speed_rms=.1,
        roll_pitch_rms=.2,roll_pitch_peak=.65,min_height=.18,contact_slip_rms=.2)
    summaries=[]
    for i,c in enumerate(trials):
        hold=(time>=hold_start+1.)&(time<hold_end)&valid[:,i]
        stop=(time>=hold_end+2.)&valid[:,i]
        data=trace['body_velocity'][:,i];ang=trace['angular_velocity'][:,i]
        rpy=trace['rpy'][:,i,:2];contacts=trace['foot_force'][:,i]>5
        rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else None
        errors=dict(vx_rmse=rms(data[hold,0]-c['command'][0]),vy_rmse=rms(data[hold,1]-c['command'][1]),
            yaw_rmse=rms(ang[hold,2]-c['command'][2]),roll_pitch_rms=float(np.sqrt((rpy[hold]**2).mean(axis=0)).max()) if hold.any() else None,
            roll_pitch_peak=float(np.abs(rpy[valid[:,i]]).max()),min_height=float(trace['position'][valid[:,i],i,2].min()),
            stop_speed_rms=rms(np.linalg.norm(data[stop,:2],axis=1)),contact_slip_rms=rms(trace['foot_speed'][:,i][hold[:,None]&contacts]),
            mean_vx=float(data[hold,0].mean()) if hold.any() else None)
        gates={'survived':bool(alive[i]),'full_hold':bool(hold.sum()>=round((a.hold_seconds-1)/dt)-1),
            'stop_measured':bool(stop.any())}
        for k,limit in thresholds.items():
            if k=='stand_speed_rms':continue
            v=errors[k];gates[k]=v is not None and (v>=limit if k=='min_height' else v<=limit)
        if c['name']=='stand':
            errors['stand_speed_rms']=rms(np.linalg.norm(data[hold,:2],axis=1));gates['stand_speed_rms']=errors['stand_speed_rms'] is not None and errors['stand_speed_rms']<=thresholds['stand_speed_rms']
        summaries.append(dict(**c,passed=all(gates.values()),gates=gates,metrics=errors,failure=failures[i],hold_samples=int(hold.sum())))
    report=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),profile=a.profile,
        observation_commands_verified=True,
        seed=a.seed,start_mode=a.start_mode,policy_updates=0,trials=summaries,thresholds=thresholds,
        required_trials_passed=all(s['passed'] for s in summaries if s['required']),
        video_case='forward_04' if a.video else None,video_last_sim_time=last_video_t,
        caveats=['First terminal state is captured before auto-reset. No subsequent episode is counted as recovery.',
            'Nominal profile disables observation noise, external mass/material randomization and initial-state perturbations.',
            '0.8 m/s is a separate stress case above the final training command curriculum.',
            'Slip is contact-weighted RMS horizontal foot-link speed; not an exact measurement of contact-patch sliding.'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    with (out/'trials.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['case','repeat','required','passed','failure_s','failure_reason',*summaries[0]['metrics']])
        for s in summaries:w.writerow([s['name'],s['repeat'],s['required'],s['passed'],s['failure']['time_seconds'] if s['failure'] else '',s['failure']['reasons'] if s['failure'] else '',*s['metrics'].values()])
    shutil.copy2(a.checkpoint.parent/'params/agent.yaml',out/'agent.yaml')
    (out/'protocol.json').write_text(json.dumps(status,indent=2))
    status.update(status='completed',required_trials_passed=report['required_trials_passed']);save()
    print('[RESULT] '+json.dumps(report),flush=True)

try:
    app=AppLauncher(a).app
    main()
except BaseException as e:
    status.update(status='failed',error=str(e),traceback=traceback.format_exc());save();raise
finally:
    if writer is not None:writer.close()
    if env is not None:env.close()
    if app is not None:app.close()
