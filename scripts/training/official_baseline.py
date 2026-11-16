"""Pinned Go2 source with original, curriculum-fix and forward-only training variants."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from gpd.paths import ROOT,new_run
from gpd.portable import external_path, resolve_go2_asset
from isaaclab.app import AppLauncher

COMMIT='4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'
REPO=external_path('unitree_rl_lab_official')
ASSET=resolve_go2_asset()
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_source():
    if git('rev-parse','HEAD')!=COMMIT:raise RuntimeError('Official checkout revision changed')
    if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Official checkout has tracked edits; refusing to train')
    if git('remote','get-url','origin')!='https://github.com/unitreerobotics/unitree_rl_lab.git':raise RuntimeError('Unexpected official source remote')
    if not ASSET.is_file():raise RuntimeError(f'Missing local USD: {ASSET}')

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['train','check','play'])
variant=p.add_mutually_exclusive_group()
variant.add_argument('--curriculum-fixed',action='store_true',help='Separate baseline variant: periodic command curriculum; upstream rewards unchanged')
variant.add_argument('--forward-only',action='store_true',help='Positive forward commands only; default vx=0.2..0.4, vy=yaw=0')
p.add_argument('--forward-speed-range',type=float,nargs=2,metavar=('MIN','MAX'),help='Forward command range; defaults to checkpoint range or 0.2..0.4')
p.add_argument('--num-envs',type=int)
p.add_argument('--iterations',type=int,default=2000)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--seed',type=int,default=42)
weights=p.add_mutually_exclusive_group()
weights.add_argument('--checkpoint',type=Path,help='Inference checkpoint for check/play')
weights.add_argument('--finetune-from',type=Path,help='Forward-only train/check: restore policy, critic and Adam; count a new stage from zero')
p.add_argument('--command',type=float,nargs=3,metavar=('VX','VY','YAW'),help='Fixed body-frame velocity command for check/play only')
p.add_argument('--video',action='store_true')
p.add_argument('--video-length',type=int,default=500)
p.add_argument('--video-interval',type=int,default=10000)
p.add_argument('--output',type=Path)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.mode=='play' and not a.checkpoint:p.error('play requires --checkpoint')
if a.mode=='train' and a.checkpoint:p.error('Use --finetune-from for forward additional training; --checkpoint is inference-only')
if a.mode=='train' and a.command is not None:p.error('--command is evaluation-only; train uses the selected command distribution')
if a.finetune_from and (not a.forward_only or a.mode=='play'):p.error('--finetune-from is only for forward train/check')
if a.forward_speed_range is not None and not a.forward_only:p.error('--forward-speed-range requires forward mode')
source_checkpoint=a.finetune_from or a.checkpoint
parent_manifest={}
if source_checkpoint:
    if not source_checkpoint.is_file():p.error(f'Missing checkpoint: {source_checkpoint}')
    parent_manifest=json.loads((source_checkpoint.parent/'manifest.json').read_text())
if a.forward_only:
    from isaac_task.forward_only import FORWARD_SPEED_RANGE
    a.forward_speed_range=a.forward_speed_range or parent_manifest.get('forward_command_settings',{}).get('lin_vel_x',FORWARD_SPEED_RANGE)
    import math
    if not all(math.isfinite(v) for v in a.forward_speed_range) or not (0<a.forward_speed_range[0]<=a.forward_speed_range[1]):
        p.error('forward speed range requires 0 < MIN <= MAX and finite values')
    if a.command is not None and not (a.forward_speed_range[0]<=a.command[0]<=a.forward_speed_range[1] and a.command[1:]==[0.,0.]):
        p.error(f'forward mode requires vx in {a.forward_speed_range} and vy=yaw=0')
if a.num_envs is None:a.num_envs=256 if a.mode=='train' else 1
if min(a.num_envs,a.iterations,a.steps,a.video_length)<1 or a.video_interval<a.video_length:p.error('Invalid counts/video interval')
if a.video:a.enable_cameras=True
verify_source()
path=new_run('isaac',a.mode)
tag=('unitree_go2_forward_only' if a.forward_only else 'unitree_go2_curriculum_fixed' if a.curriculum_fixed else 'unitree_go2_official')
interface=('unitree_go2_forward_only_45_60_12_v1' if a.forward_only else
           'unitree_go2_curriculum_fixed_45_60_12_v1' if a.curriculum_fixed else 'unitree_go2_official_45_60_12_v1')
out=a.output or path.parent/tag/path.name
out.mkdir(parents=True,exist_ok=True)
if (out/'status.json').exists():p.error('Choose a fresh output directory')
status=dict(status='starting',mode=a.mode,source=('Unitree Go2 forward-only commands' if a.forward_only else
    'Unitree Go2 with curriculum fix' if a.curriculum_fixed else 'official Unitree Go2'),official_commit=COMMIT,
    training_variant='forward_only' if a.forward_only else 'curriculum_fixed' if a.curriculum_fixed else 'original',
    num_envs=a.num_envs,iterations=a.iterations,policy_updates=0 if a.mode!='train' else None,
    training_initialization='finetune' if a.finetune_from and a.mode=='train' else 'scratch' if a.mode=='train' else None,
    checkpoint=str(a.checkpoint) if a.checkpoint else None,finetune_from=str(a.finetune_from) if a.finetune_from else None)
def save():(out/'status.json').write_text(json.dumps(status,indent=2))
save()
app=env=video=None

def main():
    global env,video
    import inspect
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from isaac_task.logging import RunLog,logged_runner
    sys.path.insert(0,str(REPO/'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    for cls in (RobotEnvCfg,BasePPORunnerCfg):
        if not Path(inspect.getfile(cls)).resolve().is_relative_to(REPO):raise RuntimeError('Imported a different local unitree_rl_lab package')
    cfg,agent=RobotEnvCfg(),BasePPORunnerCfg()
    # Audit critical upstream defaults before applying runtime-only settings.
    audit=dict(action_scale=cfg.actions.JointPositionAction.scale,
        feet_air_time=cfg.rewards.feet_air_time.weight,joint_limits=cfg.rewards.dof_pos_limits.weight,
        action_rate=cfg.rewards.action_rate.weight,joint_position=cfg.rewards.joint_pos.weight,
        undesired_contacts=cfg.rewards.undesired_contacts.weight,
        calf_initial_angle=cfg.scene.robot.init_state.joint_pos['.*_calf_joint'],
        push_enabled=cfg.events.push_robot is not None,
        terrain_subtypes=list(cfg.scene.terrain.terrain_generator.sub_terrains))
    expected=dict(action_scale=.25,feet_air_time=.1,joint_limits=-10.,action_rate=-.1,
        joint_position=-.7,undesired_contacts=-1.,calf_initial_angle=-1.5,push_enabled=True,terrain_subtypes=['flat'])
    if audit!=expected:raise RuntimeError(f'Unexpected pinned task settings: {audit}')
    # Preserve task/rewards/physics. Resolve local USD and spell out legacy
    # empirical_normalization=False semantics for RSL-RL 3.x.
    cfg.scene.robot.spawn.usd_path=str(ASSET)
    cfg.scene.num_envs=a.num_envs
    cfg.seed=agent.seed=a.seed
    cfg.sim.device=agent.device=a.device
    agent.max_iterations=a.iterations
    agent.experiment_name=tag
    agent.obs_groups={'policy':['policy'],'critic':['critic']}
    agent.policy.actor_obs_normalization=False
    agent.policy.critic_obs_normalization=False
    if a.curriculum_fixed:
        # The fixed environment collects reset evidence itself and makes decisions
        # after every step; never execute the upstream reset-only command term.
        cfg.curriculum.lin_vel_cmd_levels=None
    if a.forward_only:
        from isaac_task import forward_only
        forward_settings=forward_only.configure_forward_only(cfg,a.forward_speed_range)
    if a.mode!='train':
        def forbid_updates(*args,**kwargs):raise RuntimeError('No policy updates allowed in check/play')
        PPO.update=forbid_updates
    if a.video:cfg.viewer.resolution=(640,480)
    cfg.validate();agent.validate()
    manifest=dict(interface=interface,official_repo=str(REPO),official_commit=COMMIT,
        training_variant=status['training_variant'],
        upstream_url='https://github.com/unitreerobotics/unitree_rl_lab',source_clean=True,
        environment_source_sha256=sha(Path(inspect.getfile(RobotEnvCfg))),
        asset_path=str(ASSET),asset_sha256=sha(ASSET),upstream_task_audit=audit,
        runtime_adaptations=['Local USD path','Explicit obs_groups for RSL-RL 3.x',
            'actor/critic_obs_normalization=False, equivalent to upstream empirical_normalization=False'],
        cli_overrides=dict(num_envs=a.num_envs,iterations=a.iterations,seed=a.seed,device=a.device,video=a.video),
        observation_dim=45,critic_observation_dim=60,action_dim=12)
    if a.curriculum_fixed:
        from isaac_task import velocity_curriculum
        from dataclasses import asdict
        settings=velocity_curriculum.VelocityCurriculumSettings(min_episodes=min(32,max(1,a.num_envs//4)))
        manifest['task_changes']=['Replace reset/global-step command curriculum with periodic completed-episode aggregation']
        manifest['curriculum_settings']=asdict(settings)
        manifest['curriculum_source_sha256']=sha(Path(velocity_curriculum.__file__))
        manifest['curriculum_active']=a.mode!='play'
    if a.forward_only:
        manifest['task_changes']=['Positive forward commands only; zero standing/lateral/yaw commands; disable symmetric command curriculum']
        manifest['forward_command_settings']=forward_settings
        manifest['forward_source_sha256']=sha(Path(forward_only.__file__))
    manifest['evaluation_command']=a.command
    if source_checkpoint:
        old=parent_manifest
        if old.get('interface')!=manifest['interface'] or old.get('official_commit')!=COMMIT:
            raise RuntimeError('Checkpoint variant mismatch; use forward, baseline or official to match the training variant')
        if old.get('asset_sha256')!=manifest['asset_sha256'] or old.get('environment_source_sha256')!=manifest['environment_source_sha256']:
            raise RuntimeError('Checkpoint asset/environment provenance differs from this pinned environment')
        manifest['source_checkpoint']=dict(path=str(source_checkpoint.resolve()),sha256=sha(source_checkpoint))
    if a.finetune_from:
        from isaac_task import finetune
        manifest['fine_tuning']=dict(parent_checkpoint=str(a.finetune_from.resolve()),
            parent_checkpoint_sha256=sha(a.finetune_from),additional_iterations=a.iterations,
            source_sha256=sha(Path(finetune.__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    dump_yaml(str(out/'params/env.yaml'),cfg);dump_yaml(str(out/'params/agent.yaml'),agent)
    source=out/'source';source.mkdir()
    for cls,name in [(RobotEnvCfg,'official_env_cfg.py'),(BasePPORunnerCfg,'official_ppo_cfg.py')]:
        shutil.copy2(inspect.getfile(cls),source/name)
    shutil.copy2(REPO/'source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py',source/'official_robot_cfg.py')
    shutil.copy2(__file__,source/'official_runtime.py')
    if a.curriculum_fixed:
        shutil.copy2(velocity_curriculum.__file__,source/'velocity_curriculum.py')
    if a.forward_only:
        shutil.copy2(forward_only.__file__,source/'forward_only.py')
    if a.finetune_from:
        shutil.copy2(finetune.__file__,source/'finetune.py')
        shutil.copy2(a.finetune_from.parent/'manifest.json',source/'parent_manifest.json')
    log=RunLog(out,status)
    base_env=velocity_curriculum.curriculum_env_class(ManagerBasedRLEnv) if a.curriculum_fixed else ManagerBasedRLEnv
    class OfficialEnv(base_env):
        def render(self,recompute=False):
            if a.video:
                target=self.scene['robot'].data.root_pos_w[0].detach().cpu().numpy()+np.array([0.,0.,.05])
                self.sim.set_camera_view(target+np.array([1.6,-2.,1.]),target)
            return super().render(recompute=recompute)
    env_kwargs=dict(curriculum_emit=log.emit,curriculum_enabled=a.mode!='play') if a.curriculum_fixed else {}
    env=OfficialEnv(cfg,render_mode='rgb_array' if a.video else None,**env_kwargs)
    env_api=env
    if a.video:
        video=gym.wrappers.RecordVideo(env,video_folder=str(out/'videos'),
            step_trigger=(lambda step:step%a.video_interval==0) if a.mode=='train' else (lambda step:step==0),
            video_length=a.video_length,name_prefix=f'official-{a.mode}',fps=round(1/env.step_dt),disable_logger=True)
        env_api=video
    wrapped=RslRlVecEnvWrapper(env_api,clip_actions=agent.clip_actions)
    curriculum=getattr(env,'velocity_curriculum',None)
    metrics=curriculum.metrics if curriculum is not None else None
    runner=logged_runner(OnPolicyRunner,log.emit,metrics=metrics)(wrapped,agent.to_dict(),log_dir=str(out) if a.mode=='train' else None,device=a.device)
    if a.finetune_from:
        restored=finetune.restore_finetune(runner,a.finetune_from.resolve())
        restored['parent_cumulative_iterations']=parent_manifest.get('fine_tuning',{}).get('parent_cumulative_iterations',0)+restored['parent_stage_iterations']
        manifest['fine_tuning'].update(restored)
        status['fine_tuning']=manifest['fine_tuning']
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        log.emit('FINETUNE_RESTORED',**manifest['fine_tuning'])
    elif a.checkpoint:
        runner.load(str(a.checkpoint.resolve()),load_optimizer=False,map_location=a.device)
    status['status']='running';save()
    log.emit('FORWARD_READY' if a.forward_only else 'BASELINE_READY' if a.curriculum_fixed else 'OFFICIAL_READY',commit=COMMIT,source_clean=True,
        training_variant=status['training_variant'],output=str(out.resolve()),audit=audit)
    if a.mode=='train':
        # Same initialization and PPO learn call as the pinned official train.py.
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
        runner.save(str(out/'model_final.pt'))
        status['policy_updates']=int(runner.current_learning_iteration)+1
        status['cumulative_iterations']=status['policy_updates']+manifest.get('fine_tuning',{}).get('parent_cumulative_iterations',0)
    else:
        initial_policy_state={k:v.detach().clone() for k,v in runner.alg.policy.state_dict().items()}
        obs=wrapped.get_observations()
        assert obs['policy'].shape==(a.num_envs,45) and obs['critic'].shape==(a.num_envs,60)
        actor=runner.get_inference_policy(device=a.device)
        done_count=0
        samples=[]
        contact=env.scene['contact_forces']
        feet=[contact.body_names.index(f'{leg}_foot') for leg in ['FL','FR','RL','RR']]
        from tensordict import TensorDict
        from isaaclab.utils.math import euler_xyz_from_quat
        command_term=env.command_manager.get_term('base_velocity')
        with torch.inference_mode():
            for step in range(a.steps):
                if a.command is not None:
                    fixed=torch.tensor(a.command,dtype=torch.float32,device=a.device).expand(a.num_envs,-1)
                    command_term.vel_command_b[:]=fixed
                    command_term.is_standing_env[:]=(fixed.abs().sum(dim=1)<1e-7)
                    command_term.time_left.fill_(1e6)
                    obs=TensorDict(env.observation_manager.compute(update_history=True),batch_size=[a.num_envs])
                    if not torch.allclose(obs['policy'][:,6:9],fixed,atol=1e-6):raise RuntimeError('Fixed command not delivered to actor')
                requested=obs['policy'][:,6:9].detach().cpu().numpy().copy()
                if a.forward_only and not (np.all(requested[:,0]>=a.forward_speed_range[0]-1e-6) and np.all(requested[:,0]<=a.forward_speed_range[1]+1e-6)
                                           and np.all(np.abs(requested[:,1:])<1e-7)):
                    raise RuntimeError('Non-forward command reached actor in forward-only check/play')
                action=actor(obs)
                obs,reward,done,_=wrapped.step(action)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                done_count+=int(done.sum())
                robot=env.scene['robot'].data
                roll,pitch,yaw=euler_xyz_from_quat(robot.root_quat_w)
                rpy=torch.remainder(torch.stack((roll,pitch,yaw),dim=1)+torch.pi,2*torch.pi)-torch.pi
                samples.append(dict(time=(step+1)*env.step_dt,command=requested,
                    body_velocity=robot.root_lin_vel_b.detach().cpu().numpy().copy(),
                    angular_velocity=robot.root_ang_vel_b.detach().cpu().numpy().copy(),
                    position=(robot.root_pos_w-env.scene.env_origins).detach().cpu().numpy().copy(),
                    rpy=rpy.detach().cpu().numpy().copy(),
                    action=action.detach().cpu().numpy().copy(),post_reset_state=done.detach().cpu().numpy().copy(),
                    foot_force=contact.data.net_forces_w[:,feet,2].detach().cpu().numpy().copy()))
        np.savez_compressed(out/'play_trace.npz',**{key:np.array([s[key] for s in samples]) for key in samples[0]})
        import csv
        with (out/'play_trace.csv').open('w') as stream:
            csv_writer=csv.writer(stream);csv_writer.writerow(['time','env','post_reset_state','vx_cmd','vy_cmd','yaw_cmd','vx','vy','yaw_rate','height','roll','pitch'])
            for s in samples:
                for i in range(a.num_envs):
                    csv_writer.writerow([s['time'],i,bool(s['post_reset_state'][i]),*s['command'][i],
                        *s['body_velocity'][i,:2],s['angular_velocity'][i,2],s['position'][i,2],*s['rpy'][i,:2]])
        log.emit('PLAY_TRACE_SAVED',path=str(out/'play_trace.npz'),fixed_command=a.command,
            note='State rows with post_reset_state=True are reset states, not pre-failure states')
        if not all(torch.equal(v,initial_policy_state[k]) for k,v in runner.alg.policy.state_dict().items()):
            raise RuntimeError('Policy changed during non-learning check/play')
        status['policy_state_unchanged']=True
        status.update(steps=a.steps,reset_count=done_count)
        log.emit('INFERENCE_COMPLETE',steps=a.steps,policy_updates=0,reset_count=done_count,
            note='Execution check only; not a walking performance acceptance test')
    if curriculum is not None:
        (out/'curriculum_summary.json').write_text(json.dumps(curriculum.summary(),indent=2))
    if video is not None:
        if video.recording:video.stop_recording()
        status['videos']=[str(f.resolve()) for f in sorted((out/'videos').glob('*.mp4'))]
        if not status['videos']:raise RuntimeError('Requested video was not saved')
    verify_source()
    if source_checkpoint and sha(source_checkpoint)!=manifest['source_checkpoint']['sha256']:
        raise RuntimeError('Source checkpoint file changed during run')
    status.update(status='completed',source_clean_after_run=True);save()
    log.emit('COMPLETED',output=str(out.resolve()))

try:
    app=AppLauncher(a).app
    main()
except BaseException as e:
    status.update(status='failed',error=str(e),traceback=traceback.format_exc());save();raise
finally:
    if video is not None:video.close()
    elif env is not None:env.close()
    if app is not None:app.close()
