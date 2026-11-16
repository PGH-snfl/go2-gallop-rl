"""Compare pilot checkpoints under a shared, explicitly selected controller."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

from gpd.paths import ROOT,new_run
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
p.add_argument('--labels',nargs='+',required=True)
p.add_argument('--config',type=Path,default=ROOT/'configs/gallop_pilot_18cm.json')
p.add_argument('--pairs',type=int,default=8)
p.add_argument('--steps',type=int,default=1500)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if len(a.checkpoints)!=len(a.labels):p.error('One label per checkpoint is required')
temp=new_run('isaac','check');out=temp.parent/'unitree_go2_gallop'/temp.name;out.mkdir(parents=True)
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
status=dict(status='starting',policy_updates=0);write('status.json',status)
app=AppLauncher(a).app;env=None
try:
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import quat_from_euler_xyz
    from isaac_task.gallop_core import GallopSettings,FrozenWalkingPolicy
    from isaac_task.gallop_env import GallopEnv,configure_gallop
    from isaac_task.gallop_runtime import checkpoint_source_directory
    from scripts.analysis.collect_unitree_baseline import load_yaml
    paths=[x.resolve() for x in a.checkpoints];parents=[json.loads((x.parent/'manifest.json').read_text()) for x in paths]
    m=parents[0];hashes=[sha(x) for x in paths]
    for path,parent in zip(paths,parents):
        checkpoint_source_directory(path,parent)
        for key in ('asset_sha256','environment_source_sha256','interface'):
            assert parent[key]==m[key],key
        assert parent['walking_teacher']['sha256']==m['walking_teacher']['sha256']
    sys.path.insert(0,str(Path(m['official_repo'])/'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    settings=GallopSettings.load(a.config)
    cfg=RobotEnvCfg();cfg.scene.robot.spawn.usd_path=m['asset_path'];cfg.scene.num_envs=a.pairs*len(paths)
    cfg.sim.device=a.device;cfg.seed=42;configure_gallop(cfg,settings)
    cfg.observations.policy.enable_corruption=False;cfg.observations.critic.enable_corruption=False
    disabled=[]
    for name,value in cfg.events.to_dict().items():
        if value is not None and name not in ('reset_base','reset_robot_joints'):
            setattr(cfg.events,name,None);disabled.append(name)
    cfg.validate()
    teacher=FrozenWalkingPolicy(paths[0].parent/'source/walking_teacher.pt',a.device,m['walking_teacher']['action_scale'],settings.action_scale)
    env=GallopEnv(cfg,gallop_settings=settings,training_targets=False,curriculum_enabled=False,
        walking_policy=teacher,episode_path=out/'episodes.jsonl')
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None);env.capture_terminals=True
    runners=[];actors=[];initial=[];agent=load_yaml(paths[0].parent/'params/agent.yaml')
    def forbid(*args,**kwargs):raise RuntimeError('Policy updates prohibited during evaluation')
    for path in paths:
        runner=OnPolicyRunner(wrapped,copy.deepcopy(agent),log_dir=None,device=a.device)
        runner.load(str(path),load_optimizer=False,map_location=a.device)
        runner.learn=forbid;runner.alg.update=forbid;runner.alg.optimizer.step=forbid
        runners.append(runner);actors.append(runner.get_inference_policy(device=a.device))
        initial.append({k:v.detach().clone() for k,v in runner.alg.policy.state_dict().items()})
    write('manifest.json',dict(checkpoints=[str(x) for x in paths],labels=a.labels,checkpoint_sha256=hashes,
        pairs=a.pairs,steps=a.steps,settings=settings.to_dict(),config_sha256=sha(a.config),
        runtime_core_sha256=sha(ROOT/'isaac_task/gallop_core.py'),runtime_env_sha256=sha(ROOT/'isaac_task/gallop_env.py'),
        observation_noise=False,disabled_events=disabled,policy_updates=0,
        protocol='Shared pilot runtime and fixed task for every candidate; nominal plane; eight paired tiny initial perturbation requests; first episode only; no cloned intermediate states or warm switching.'))
    obs,_=wrapped.reset();n=len(paths);groups=torch.arange(env.num_envs,device=a.device)%n
    robot=env.scene['robot'];gen=torch.Generator(device=a.device);gen.manual_seed(42)
    q=robot.data.default_joint_pos.clone()+torch.randn((a.pairs,12),generator=gen,device=a.device).repeat_interleave(n,0)*.002
    robot.write_joint_state_to_sim(q,torch.zeros_like(q))
    pose=robot.data.default_root_state[:,:7].clone();pose[:,:3]+=env.scene.env_origins
    tilt=torch.randn((a.pairs,2),generator=gen,device=a.device).repeat_interleave(n,0)*.003
    pose[:,3:]=quat_from_euler_xyz(tilt[:,0],tilt[:,1],torch.zeros(env.num_envs,device=a.device))
    robot.write_root_pose_to_sim(pose);robot.write_root_velocity_to_sim(torch.zeros(env.num_envs,6,device=a.device))
    env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.step_dt);obs=wrapped.get_observations()
    samples={};valid=[];alive=np.ones(env.num_envs,dtype=bool)
    for step in range(a.steps):
        with torch.no_grad():
            action=actors[0](obs)
            for i in range(1,n):action=torch.where((groups==i)[:,None],actors[i](obs),action)
            obs,_,done,_=wrapped.step(action)
        sample=env.gallop.snapshot()
        for i,terminal in env.terminal_snapshots.items():
            for key,value in terminal.items():sample[key][i]=value
        for key,value in sample.items():samples.setdefault(key,[]).append(value)
        valid.append(alive.copy());alive&=~done.cpu().numpy().astype(bool)
        if (step+1)%250==0:print(json.dumps(dict(step=step+1,alive=int(alive.sum()))),flush=True)
    z={k:np.stack(v) for k,v in samples.items()};z['valid']=np.array(valid);z['time']=np.arange(1,a.steps+1)*env.step_dt
    np.savez_compressed(out/'trace.npz',**z)
    rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines() if line.strip()]
    cases=[]
    for i in range(env.num_envs):
        e=next(r for r in rows if r['env']==i);mask=z['valid'][:,i]&z['gallop_metric_active'][:,i].astype(bool)
        angles=z['rpy'][mask,i]
        cases.append(dict(label=a.labels[i%n],pair=i//n,env=i,episode=e,formal_seconds=float(mask.sum()*env.step_dt),
            mean_vx=float(z['body_velocity'][mask,i,0].mean()) if mask.any() else None,
            yaw_rms_deg=float(np.rad2deg(np.sqrt((angles[:,2]**2).mean()))) if mask.any() else None,
            roll_pitch_rms_deg=np.rad2deg(np.sqrt((angles[:,:2]**2).mean(0))).tolist() if mask.any() else None,
            all_off_fraction=float((~z['contact'][mask,i].astype(bool)).all(1).mean()) if mask.any() else None))
    write('cases.json',cases)
    assert all(sha(x)==h for x,h in zip(paths,hashes))
    assert all(torch.equal(v,initial[i][k]) for i,r in enumerate(runners) for k,v in r.alg.policy.state_dict().items())
    status.update(status='completed',policies_unchanged=True,cases=len(cases));write('status.json',status)
    print(json.dumps(dict(output=str(out),status='completed')),flush=True)
except BaseException as error:
    traceback.print_exc();status.update(status='failed',error=str(error));write('status.json',status);raise
finally:
    if env is not None:env.close()
    app.close()
