"""Isaac paired-start, frozen-policy diagnostics; never invokes learning."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from gpd.paths import ROOT,new_run
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pairs',type=int,default=8);p.add_argument('--steps',type=int,default=1500)
p.add_argument('--warmup-steps',type=int,default=600);p.add_argument('--output',type=Path)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=a.output or (lambda x:x.parent/'unitree_go2_gallop'/x.name)(new_run('isaac','check'))
out.mkdir(parents=True,exist_ok=False)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
status=dict(status='starting',policy_updates=0,mode='paired_candidate_validation');write(out/'status.json',status)
app=AppLauncher(a).app
env=None
try:
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import quat_from_euler_xyz
    from isaac_task.gallop_core import GallopSettings,FrozenWalkingPolicy
    from isaac_task.gallop_env import GallopEnv,GallopState,configure_gallop
    from isaac_task.gallop_runtime import checkpoint_source_directory
    from scripts.analysis.collect_unitree_baseline import load_yaml
    torch.set_num_threads(1)
    base=ROOT/'runs/isaac/train/unitree_go2_gallop'
    parent=base/'20261106_000029_077a6f';child=base/'20261110_195814_959c90'
    paths=[parent/'model_final.pt',child/'model_500.pt',child/'model_1000.pt',child/'model_final.pt']
    labels=['parent30k','early30500','early31000','final34k'];hashes=[sha(x) for x in paths]
    manifest=json.loads((parent/'manifest.json').read_text());other=json.loads((child/'manifest.json').read_text())
    for folder,m in [(parent,manifest),(child,other)]:
        checkpoint_source_directory(folder/'model_final.pt',m)
        assert sha(ROOT/'isaac_task/gallop_core.py')==m['gallop_core_sha256']
        assert sha(ROOT/'isaac_task/gallop_env.py')==m['gallop_env_sha256']
    repo=Path(manifest['official_repo']);sys.path.insert(0,str(repo/'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    old=GallopSettings(**manifest['gallop_settings']);new=GallopSettings(**other['gallop_settings'])
    cfg=RobotEnvCfg();cfg.scene.robot.spawn.usd_path=manifest['asset_path'];cfg.scene.num_envs=a.pairs*len(paths)
    cfg.sim.device=a.device;cfg.seed=42;configure_gallop(cfg,old)
    cfg.observations.policy.enable_corruption=False;cfg.observations.critic.enable_corruption=False
    disabled=[]
    for name,value in cfg.events.to_dict().items():
        if value is not None and name not in ('reset_base','reset_robot_joints'):
            setattr(cfg.events,name,None);disabled.append(name)
    cfg.validate()
    teacher=FrozenWalkingPolicy(parent/'source/walking_teacher.pt',a.device,manifest['walking_teacher']['action_scale'],old.action_scale)
    env=GallopEnv(cfg,gallop_settings=old,training_targets=False,curriculum_enabled=False,walking_policy=teacher)
    env.capture_terminals=True
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    agent=load_yaml(parent/'params/agent.yaml');runners=[];actors=[];initial=[]
    def forbid(*args,**kwargs):raise RuntimeError('PPO and optimizer updates forbidden in candidate validation')
    for path in paths:
        runner=OnPolicyRunner(wrapped,agent.copy()|{'policy':agent['policy'].copy(),'algorithm':agent['algorithm'].copy()},log_dir=None,device=a.device)
        runner.load(str(path),load_optimizer=False,map_location=a.device)
        runner.learn=forbid;runner.alg.update=forbid;runner.alg.optimizer.step=forbid
        runners.append(runner);actors.append(runner.get_inference_policy(device=a.device))
        initial.append({k:v.detach().clone() for k,v in runner.alg.policy.state_dict().items()})
    groups=torch.arange(env.num_envs,device=a.device)%len(paths)
    def install(settings,scenario,warm=False):
        previous=env.gallop
        state=GallopState(env,settings,None,scenario/'episodes.jsonl',False,False,teacher)
        if warm:
            state.age.copy_(previous.age);state.first.copy_(previous.first);state.phase.copy_(previous.phase)
            state.startup.stage.fill_(3);state.startup.goal_speed.fill_(settings.start_speed)
            for field in ('previous','target','requested'):getattr(state.guard,field).copy_(getattr(previous.guard,field))
        env.gallop=state
        term=env.command_manager.get_term('base_velocity')
        term.cfg.ranges.lin_vel_x=(settings.start_speed,settings.start_speed)
        term.vel_command_b.zero_();term.vel_command_b[:,0]=state.startup.command
        return state
    records=[]
    scenarios=[('native_old',old,False),('native_new',new,False),('warm_old',old,True),('warm_new',new,True)]
    write(out/'manifest.json',dict(labels=labels,checkpoints=[str(x) for x in paths],checkpoint_sha256=hashes,
        pairs=a.pairs,steps=a.steps,warmup_steps=a.warmup_steps,disabled_events=disabled,
        observation_noise=False,nominal_physics=True,policy_updates=0,
        protocol='Four policy groups with paired tiny initial joint/tilt perturbations. Warm cases use the parent actor and old task for 12s in all groups, then reconstruct only controller/metrics state, preserving physical state, reference phase and previous targets. Rewards are ignored; no PPO updates.'))
    for name,settings,warm in scenarios:
        folder=out/name;folder.mkdir();install(old if warm else settings,folder)
        env.common_step_counter=0;obs,_=wrapped.reset()
        robot=env.scene['robot'];gen=torch.Generator(device=a.device);gen.manual_seed(42)
        q=robot.data.default_joint_pos.clone()+torch.randn((a.pairs,12),generator=gen,device=a.device).repeat_interleave(len(paths),0)*.002
        robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        pose=robot.data.default_root_state[:,:7].clone();pose[:,:3]+=env.scene.env_origins
        tilt=torch.randn((a.pairs,2),generator=gen,device=a.device).repeat_interleave(len(paths),0)*.003
        pose[:,3:]=quat_from_euler_xyz(tilt[:,0],tilt[:,1],torch.zeros(env.num_envs,device=a.device))
        robot.write_root_pose_to_sim(pose);robot.write_root_velocity_to_sim(torch.zeros(env.num_envs,6,device=a.device))
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt)
        obs=wrapped.get_observations();alive=np.ones(env.num_envs,dtype=bool);samples={};valid=[]
        for step in range(a.steps):
            if warm and step==a.warmup_steps:
                install(settings,folder,True);obs=wrapped.get_observations()
            # Isaac caches tensors created during stepping; no_grad also permits
            # explicit resets between scenarios without inference-tensor errors.
            with torch.no_grad():
                if warm and step<a.warmup_steps:action=actors[0](obs)
                else:
                    action=actors[0](obs)
                    for index in range(1,len(actors)):
                        candidate=actors[index](obs);action=torch.where((groups==index)[:,None],candidate,action)
                obs,_,done,_=wrapped.step(action)
            sample=env.gallop.snapshot()
            for i,terminal in env.terminal_snapshots.items():
                for key,value in terminal.items():sample[key][i]=value
            for key,value in sample.items():samples.setdefault(key,[]).append(value)
            valid.append(alive.copy());alive&=~done.cpu().numpy().astype(bool)
            if (step+1)%250==0:print(json.dumps(dict(scenario=name,step=step+1,alive=int(alive.sum()))),flush=True)
        z={k:np.stack(v) for k,v in samples.items()};z['valid']=np.array(valid);z['time']=np.arange(1,a.steps+1)*env.step_dt
        np.savez_compressed(folder/'trace.npz',**z)
        for i in range(env.num_envs):
            mask=z['valid'][:,i].astype(bool)&z['gallop_metric_active'][:,i].astype(bool)
            if warm:mask&=z['time']>a.warmup_steps*env.step_dt+1.
            valid_i=z['valid'][:,i].astype(bool);last=np.flatnonzero(valid_i)[-1]
            rows=[json.loads(l) for l in (folder/'episodes.jsonl').read_text().splitlines() if l.strip()] if (folder/'episodes.jsonl').exists() else []
            episode=next((r for r in rows if r['env']==i),{})
            item=dict(scenario=name,label=labels[i%len(paths)],pair=i//len(paths),env=i,formal_seconds=float(mask.sum()*env.step_dt),
                final_stage=int(z['startup_stage'][last,i]),final_mix=float(z['startup_gallop_mix'][last,i]),
                mean_vx=float(z['body_velocity'][mask,i,0].mean()) if mask.any() else None,
                vx_rmse=float(np.sqrt(np.mean((z['body_velocity'][mask,i,0]-z['command'][mask,i,0])**2))) if mask.any() else None,
                all_off_fraction=float((~z['contact'][mask,i]).all(1).mean()) if mask.any() else None,
                episode=episode)
            records.append(item)
        write(out/'cases.json',records)
        print(json.dumps(dict(scenario_complete=name)),flush=True)
    unchanged=all(sha(p)==h for p,h in zip(paths,hashes)) and all(torch.equal(v,initial[i][k]) for i,r in enumerate(runners) for k,v in r.alg.policy.state_dict().items())
    assert unchanged
    status.update(status='completed',checkpoints_unchanged=True,cases=len(records));write(out/'status.json',status)
    print(json.dumps(dict(output=str(out),status='completed',cases=len(records))),flush=True)
except BaseException as error:
    import traceback
    traceback.print_exc()
    status.update(status='failed',error=str(error));write(out/'status.json',status);raise
finally:
    if env is not None:env.close()
    app.close()
