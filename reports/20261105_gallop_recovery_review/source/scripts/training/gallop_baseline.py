"""High-amplitude Gallop-style training with a success-gated speed curriculum."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from gpd.paths import ROOT,new_run
from isaaclab.app import AppLauncher

COMMIT='4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'
REPO=ROOT.parents[1]/'unitree/unitree_rl_lab_official'
BASE_INTERFACE='unitree_go2_gallop_47_62_12_v1'
HEADING_INTERFACE='unitree_go2_gallop_heading_49_64_12_v2'
BOUNDED_INTERFACE='unitree_go2_gallop_bounded_49_64_12_v3'
GUARDED_INTERFACE='unitree_go2_gallop_guarded_49_64_12_v4'
RELEARN_INTERFACE='unitree_go2_gallop_relearn_49_64_12_v5'
FORWARD_INTERFACE='unitree_go2_forward_only_45_60_12_v1'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['train','check','play'])
weights=p.add_mutually_exclusive_group()
weights.add_argument('--finetune-from',type=Path,help='Forward, Gallop or heading-Gallop parent for a new training stage')
weights.add_argument('--checkpoint',type=Path,help='Gallop or heading-Gallop policy for check/play')
p.add_argument('--config',type=Path)
p.add_argument('--walk-checkpoint',type=Path,help='Frozen 45/12 forward policy for walking startup')
p.add_argument('--num-envs',type=int)
p.add_argument('--iterations',type=int,default=6000)
p.add_argument('--steps',type=int,default=1200)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--command',type=float,nargs=3)
p.add_argument('--video',action='store_true')
p.add_argument('--video-length',type=int,default=500)
p.add_argument('--video-interval',type=int,default=10000)
p.add_argument('--output',type=Path)
p.add_argument('--distill-steps',type=int,default=1500,help='Frozen parent rollout before bounded PPO')
p.add_argument('--distill-epochs',type=int,default=30)
p.add_argument('--distill-rounds',type=int,default=2,help='Parent fit, then frozen-parent labels on student states')
p.add_argument('--distill-batch-size',type=int,default=4096)
p.add_argument('--adapt-check-steps',type=int,default=1500,help='New student preservation rollout before PPO')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.mode=='train' and not a.finetune_from:p.error('Initial Gallop train requires --finetune-from a forward checkpoint')
if a.mode=='play' and not a.checkpoint:p.error('Gallop play requires --checkpoint')
if a.mode=='train' and a.command is not None:p.error('Train uses the success-gated command curriculum, not --command')
source_checkpoint=a.finetune_from or a.checkpoint
parent=json.loads((source_checkpoint.parent/'manifest.json').read_text()) if source_checkpoint else {}
from isaac_task.gallop_runtime import activate_checkpoint_source,parent_cumulative_iterations
replay_source=activate_checkpoint_source(a.checkpoint,parent) if a.checkpoint else None
from isaac_task.gallop_core import GallopSettings
settings=GallopSettings.load(a.config) if a.config else GallopSettings(**parent['gallop_settings']) if a.checkpoint else GallopSettings.load(ROOT/'configs/gallop_rl.json')
heading_motion=getattr(settings,'heading_motion_enabled',False)
stabilization=getattr(settings,'stabilization_enabled',False)
bounded=getattr(settings,'bounded_actions_enabled',False)
guarded=getattr(settings,'guarded_enabled',False)
relearn=getattr(settings,'relearn_enabled',False)
recovery=getattr(settings,'relearn_recovery_enabled',False)
INTERFACE=RELEARN_INTERFACE if relearn else GUARDED_INTERFACE if guarded else BOUNDED_INTERFACE if bounded else HEADING_INTERFACE if heading_motion else BASE_INTERFACE
if relearn and a.finetune_from and parent.get('interface') not in (FORWARD_INTERFACE,RELEARN_INTERFACE):
    p.error('Relearning starts from the forward walking policy or a saved relearning branch')
if parent.get('interface')==RELEARN_INTERFACE and not relearn:p.error('Relearning checkpoints require their staged controller')
relearn_parent_progress=None
if parent.get('interface')==RELEARN_INTERFACE:
    from isaac_task.gallop_runtime import relearn_checkpoint_progress
    relearn_parent_progress=relearn_checkpoint_progress(source_checkpoint,parent)
if guarded and not relearn and a.finetune_from and parent.get('interface') not in (HEADING_INTERFACE,GUARDED_INTERFACE):
    p.error('Guarded fine-tuning requires the pre-imitation heading policy or an existing guarded policy')
if parent.get('interface')==GUARDED_INTERFACE and not guarded:p.error('Guarded checkpoints require their guarded controller')
needs_adaptation=bounded and a.mode=='train' and parent.get('interface')!=BOUNDED_INTERFACE
if needs_adaptation and (parent.get('interface')!=HEADING_INTERFACE or not parent['gallop_settings'].get('stabilization_enabled')):
    p.error('Bounded migration needs a completed 49/64 stabilization parent')
if parent.get('interface')==BOUNDED_INTERFACE and not bounded:p.error('Bounded checkpoints require the bounded target transform')
observation_dim,critic_dim=(49,64) if heading_motion else (47,62)
startup_enabled=getattr(settings,'startup_enabled',False)
if startup_enabled and a.command is not None:p.error('Startup controls the full command sequence; omit --command')
if a.walk_checkpoint and (not startup_enabled or a.checkpoint):p.error('--walk-checkpoint applies only to a new startup training/check stage')
walk_source=None;walk_info=None
if startup_enabled:
    if a.checkpoint:
        walk_info=parent.get('walking_teacher')
        if not walk_info:raise ValueError('Startup checkpoint has no bundled walking teacher')
        walk_source=a.checkpoint.parent/'source/walking_teacher.pt'
    else:
        walk_source=a.walk_checkpoint or ROOT/'runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt'
        wm=json.loads((walk_source.parent/'manifest.json').read_text())
        from scripts.analysis.collect_unitree_baseline import load_yaml
        wp=load_yaml(walk_source.parent/'params/agent.yaml')['policy']
        if wm.get('interface')!='unitree_go2_forward_only_45_60_12_v1' or wp['activation']!='elu' or wp['actor_obs_normalization']:
            raise ValueError('Walking startup needs the verified unnormalized ELU 45/12 forward actor')
        if wm.get('asset_sha256')!=parent.get('asset_sha256') or wm.get('environment_source_sha256')!=parent.get('environment_source_sha256'):
            raise ValueError('Walking and Gallop robot/source differ')
        walk_info=dict(original_checkpoint=str(walk_source.resolve()),sha256=hashlib.sha256(walk_source.read_bytes()).hexdigest(),
            action_scale=wm['upstream_task_audit']['action_scale'],bundled_file='source/walking_teacher.pt',activation='elu')
if a.command is not None and not (0<a.command[0]<=settings.max_speed and a.command[1:]==[0.,0.]):p.error('Only positive forward commands up to configured max_speed are allowed')
if a.num_envs is None:a.num_envs=256 if a.mode=='train' else 1 if a.mode=='play' else 16
if needs_adaptation and (a.num_envs<2 or min(a.distill_steps,a.distill_epochs,a.distill_rounds,a.distill_batch_size,a.adapt_check_steps)<1):
    p.error('Bounded adaptation needs >=2 environments and positive adaptation counts')
if min(a.num_envs,a.iterations,a.steps,a.video_length)<1 or a.video_interval<a.video_length:p.error('Invalid counts')
if a.video:a.enable_cameras=True
out=a.output or (lambda path:path.parent/'unitree_go2_gallop'/path.name)(new_run('isaac',a.mode))
out.mkdir(parents=True,exist_ok=True)
if (out/'status.json').exists():p.error('Use a fresh output directory')
status=dict(status='starting',mode=a.mode,training_variant='gallop_success_curriculum',num_envs=a.num_envs,
            planned_iterations=a.iterations,policy_updates=0 if a.mode!='train' else None,
            training_initialization='migrate_forward_new_optimizer' if a.finetune_from else 'load_gallop' if a.checkpoint else 'random_check')
if relearn:status['training_variant']='gallop_from_walk_constraints'
def save():(out/'status.json').write_text(json.dumps(status,indent=2))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_source():
    def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
    if git('rev-parse','HEAD')!=COMMIT or git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Pinned official source is not clean')
    if git('remote','get-url','origin')!='https://github.com/unitreerobotics/unitree_rl_lab.git':raise RuntimeError('Unexpected source remote')
save();app=env=video=None


def main():
    global env,video
    import inspect
    import numpy as np
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.algorithms import PPO
    from isaac_task import gallop_core,gallop_env
    from isaac_task.gallop_core import transplant_forward_policy
    from isaac_task.logging import RunLog,logged_runner
    verify_source()
    sys.path.insert(0,str(REPO/'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    if not Path(inspect.getfile(RobotEnvCfg)).resolve().is_relative_to(REPO):raise RuntimeError('Wrong task import')
    cfg,agent=RobotEnvCfg(),BasePPORunnerCfg()
    asset=Path(parent.get('asset_path',str(ROOT.parents[1]/'unitree/unitree_rl_lab/unitree_model/Go2/usd/go2.usd')))
    native_sha=sha(inspect.getfile(RobotEnvCfg));asset_sha=sha(asset)
    if source_checkpoint:
        expected=((FORWARD_INTERFACE,RELEARN_INTERFACE) if relearn else (HEADING_INTERFACE,GUARDED_INTERFACE) if guarded else (HEADING_INTERFACE,BOUNDED_INTERFACE) if bounded else (BASE_INTERFACE,HEADING_INTERFACE) if heading_motion else (FORWARD_INTERFACE,BASE_INTERFACE)) if a.finetune_from else (INTERFACE,)
        if parent.get('interface') not in expected or parent.get('official_commit')!=COMMIT:raise RuntimeError('Checkpoint interface/source mismatch')
        if parent['asset_sha256']!=asset_sha or parent['environment_source_sha256']!=native_sha:raise RuntimeError('Checkpoint robot or source changed')
        if parent['interface'] in (BASE_INTERFACE,HEADING_INTERFACE,BOUNDED_INTERFACE,GUARDED_INTERFACE,RELEARN_INTERFACE) and parent['action_scale']!=settings.action_scale:raise RuntimeError('Gallop fine-tuning requires the same action scale')
        if guarded and parent['interface']!=FORWARD_INTERFACE:
            from isaac_task.gallop_runtime import checkpoint_source_directory
            checkpoint_source_directory(source_checkpoint,parent)
        if needs_adaptation:
            from isaac_task.gallop_runtime import checkpoint_source_directory
            checkpoint_source_directory(source_checkpoint,parent)
            # The rollout teacher uses this exact parent controller contract.
            allowed={'bounded_actions_enabled','bounded_limit_margin','bounded_limit_weight','bounded_target_rate_weight',
                     'bounded_target_rate_scale','bounded_max_limit_fraction','bounded_latent_std'}
            for key,value in parent['gallop_settings'].items():
                if key not in allowed and settings.to_dict()[key]!=value:raise ValueError('Keep parent gait settings during bounded migration: '+key)
    if a.checkpoint:
        if parent['gallop_core_sha256']!=sha(gallop_core.__file__) or parent['gallop_env_sha256']!=sha(gallop_env.__file__):raise RuntimeError('Gallop task changed since training')
        if settings.to_dict()!=GallopSettings(**parent['gallop_settings']).to_dict():raise RuntimeError('Play settings differ from the trained checkpoint')
    if cfg.actions.JointPositionAction.scale!=.25:raise RuntimeError('Unexpected original action scaling')
    cfg.scene.robot.spawn.usd_path=str(asset);cfg.scene.num_envs=a.num_envs
    cfg.seed=agent.seed=a.seed;cfg.sim.device=agent.device=a.device
    gallop_env.configure_gallop(cfg,settings)
    agent.max_iterations=a.iterations;agent.experiment_name='unitree_go2_gallop'
    agent.obs_groups={'policy':['policy'],'critic':['critic']}
    agent.policy.actor_obs_normalization=False;agent.policy.critic_obs_normalization=False
    agent.algorithm.learning_rate=1e-4 if bounded or guarded else 3e-4
    if recovery:agent.algorithm.entropy_coef=0.
    if a.video:cfg.viewer.resolution=(640,480)
    cfg.validate();agent.validate()
    if a.mode!='train':
        def forbid(*args,**kwargs):raise RuntimeError('PPO updates forbidden in check/play')
        PPO.update=forbid
    manifest=dict(interface=INTERFACE,training_variant=status['training_variant'],official_repo=str(REPO),official_commit=COMMIT,
        environment_source_sha256=native_sha,asset_path=str(asset),asset_sha256=asset_sha,
        gallop_settings=settings.to_dict(),gallop_core_sha256=sha(gallop_core.__file__),gallop_env_sha256=sha(gallop_env.__file__),
        observation_dim=observation_dim,critic_observation_dim=critic_dim,action_dim=12,action_scale=settings.action_scale,
        gait_goal='Gallop-style: relaxed foot order, required forelimb participation and suspension',
        speed_increase_rule='80% episode success in two consecutive eligible windows; no time-only speed increase',
        replay_source=replay_source,
        source_checkpoint=dict(path=str(source_checkpoint.resolve()),sha256=sha(source_checkpoint)) if source_checkpoint else None)
    if startup_enabled:
        manifest.update(walking_teacher=walk_info,startup_controller=dict(stages=['hold','walk','blend','gallop'],
            teacher_control_until_blend=True,blend_pauses_on_instability=True,phase_observation_radius=[0.,.5,1.],
            random_initial_episode_length=False,metrics='full Gallop only, after per-episode metric warmup'))
    if heading_motion:
        manifest['posture_design']=dict(version=2,target_yaw_rad=settings.target_yaw,
            heading_observation='append sin(target-yaw), cos(target-yaw)-1 after gait phase',
            yaw_penalty='wrapped absolute world heading error squared',
            roll_pitch_reward='exp of negative mean squared wrapped Euler angle rate; first sample excluded',
            front_shoulder_width_m=settings.front_shoulder_width,hind_shoulder_width_m=settings.hind_shoulder_width,
            width_ratio=settings.stance_shoulder_ratio,front_width_m=settings.front_stance_width,hind_width_m=settings.hind_stance_width,
            width_reward='independent positive Gaussian scores; signed lateral width in body yaw frame',
            foot_center_penalty=False,static_roll_pitch_penalty=False,
            direction_reward='actual world velocity projected onto target heading, with speed and travel-angle scores; no stationary credit',
            reset_yaw_rad=settings.target_yaw,stages='all phases, no gait strength or full-Gallop mask')
    elif getattr(settings,'posture_enabled',False):
        manifest['posture_design']=dict(version=1,body_axis='body +Z aligned with world +Z on flat ground',
            pitch_reference_rad=0.,upright_tolerance_rad=settings.upright_tolerance,
            upright_weight=settings.upright_weight,front_width_m=settings.front_stance_width,
            hind_width_m=settings.hind_stance_width,width_frame='signed lateral distance in yaw-aligned horizontal frame',
            width_weight=settings.stance_width_weight,center_weight=settings.stance_center_weight,
            width_tolerance_m=settings.stance_width_tolerance,center_tolerance_m=settings.stance_center_tolerance,
            swing_tolerance_multiplier=settings.stance_swing_tolerance_multiplier,
            stages='all startup and gait phases; no velocity, strength or warmup gate',
            statistics='all-stage denominator, separate from full-Gallop success metrics')
    if getattr(settings,'leap_reward_weight',0)>0:
        manifest['leap_measurement']=dict(min_distance_m=settings.leap_min_distance,min_air_seconds=settings.flight_min_seconds,
            physics_sample_seconds=cfg.sim.dt,position='first all-off body pose to first touchdown; projected onto takeoff forward heading',
            reward_weight=settings.leap_reward_weight,max_score_per_phase_cycle=1.,forelimbs_required=settings.leap_require_forelimbs,
            initial_free_fall_excluded=True,completed_landings_only=True)
    if getattr(settings,'flight_shaping_enabled',False):
        manifest['flight_design']=dict(version=1,prepare_seconds=settings.flight_prepare_seconds,
            shared_air_phase_start=max(t+d for t,d in zip(settings.touchdown,settings.duty)),
            recovery_seconds=settings.flight_recovery_seconds,distance_reward_confirmation='continuous stable supported recovery after actual flight',
            partial_distance_credit=settings.leap_partial_credit,takeoff_vertical_speed_mps=settings.flight_takeoff_vz,
            force_normalization='sum of positive vertical foot force / measured robot weight',
            touchdown_order_required=settings.require_touchdown_order,
            gait_reward_floor_during_flight_violation=settings.flight_air_reward_floor)
    if settings.start_speed==settings.max_speed and settings.speed_band==0:
        manifest['speed_increase_rule']=f'Fixed positive forward command {settings.start_speed} m/s; no speed promotion'
    if startup_enabled:manifest['speed_increase_rule']=f'Stationary hold -> stable walking {settings.startup_walk_speed} m/s -> smooth stable handover -> {settings.start_speed} m/s Gallop'
    if stabilization:
        manifest['posture_design'].update(version=3,
            roll_pitch_reward='positive temporal quiet score multiplied by soft pose quality',
            width_reward='independent positive 1/(1+normalized_error^2), support weight 1 and configured air weight')
        manifest['stabilization_design']=dict(version=1,initial_speed=settings.start_speed,final_speed=settings.max_speed,
            initial_leap_m=settings.stabilization_min_leap_distance,final_leap_m=settings.leap_min_distance,
            target_update='latch both speed and leap on environment reset; never change an ongoing episode',
            promotion='full gait gates plus pose/support width/landing/distance; >=128 episodes, >=80% for two windows',
            regression='below 50% at raised cap, return one speed step',width_reward='1/(1+normalized_error^2), contact priority',
            pose_reward='positive quiet score multiplied by soft pose quality',
            calf_control='student only, smoothly 1.0 to 0.85 with startup blend; frozen teacher unchanged',
            calf_multiplier=settings.stabilization_calf_multiplier,
            target_excess='pre-clip target excess Huber, calf weight x2')
        manifest['speed_increase_rule']='Success-conditioned stabilization speed/leap curriculum; no time-only advancement'
        manifest['leap_measurement']['target_rule']='Per-episode speed mapped linearly from initial to final leap distance'
        manifest['leap_measurement'].update(initial_min_distance_m=settings.stabilization_min_leap_distance,
            final_min_distance_m=settings.leap_min_distance,min_distance_m_is_final=True)
    if bounded:
        manifest['stabilization_design']['calf_control']='Old 0.85 scaling included in imitation targets; no additional scaling of bounded latent'
        manifest['bounded_action_design']=dict(version=1,transform='joint midpoint + half soft range * tanh(Gaussian latent)',
            startup='convex blend of separately bounded walking/student targets; hold uses default pose',
            previous_action='(executed bounded target - default pose) / action_scale',
            ppo_storage='unmodified Gaussian latent and its latent-space log probability',
            limit_margin_fraction=settings.bounded_limit_margin,max_near_limit_fraction=settings.bounded_max_limit_fraction,
            joint_reference='keep references out of the outer configured soft-range margin',
            target_rate_scale_rad_s=settings.bounded_target_rate_scale,
            note='Zero final clipping is structural, not proof of successful gait; near-limit gate is also required')
    if guarded:
        manifest['guarded_design']=dict(version=1,imitation=False,action_semantics='legacy default + 0.5 * mixed delta; student calf multiplier retained',
            physics_limit_factor_of_original=settings.guarded_sim_limit_factor,target_margin_fraction=settings.guarded_target_margin,
            target_rate_rad_s=dict(hip=settings.guarded_hip_rate,thigh_calf=settings.guarded_leg_rate),
            foot_height_caps_m=dict(front=settings.guarded_front_height,hind=settings.guarded_hind_height),
            physical_limit_write='Isaac articulation write_joint_position_limit_to_sim; soft factor 1 prevents double shrink',
            actual_limit_monitor='Measured excess penalty plus sustained violation termination; never overwrite actual positions')
        if settings.start_speed==settings.max_speed:
            manifest['speed_increase_rule']=f'Fixed Gallop goal {settings.start_speed}m/s after stationary/walking/blend; no promotion'
        else:
            manifest['speed_increase_rule']='Guarded success-conditioned speed/leap curriculum after stationary/walking/blend; all stability and guard gates required; no time-only advancement'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'gallop_config.json').write_text(json.dumps(settings.to_dict(),indent=2))
    dump_yaml(str(out/'params/env.yaml'),cfg);dump_yaml(str(out/'params/agent.yaml'),agent)
    src=out/'source';src.mkdir()
    for path in [Path(__file__),Path(gallop_core.__file__),Path(gallop_env.__file__),Path(inspect.getfile(RobotEnvCfg))]:shutil.copy2(path,src/path.name)
    shutil.copy2(ROOT/'isaac_task/gallop_runtime.py',src/'gallop_runtime.py')
    shutil.copy2(ROOT/'isaac_task/logging.py',src/'logging.py')
    if bounded:
        shutil.copy2(ROOT/'isaac_task/gallop_adaptation.py',src/'gallop_adaptation.py')
        manifest['adaptation_source_sha256']=sha(src/'gallop_adaptation.py')
    if source_checkpoint:shutil.copy2(source_checkpoint.parent/'manifest.json',src/'parent_manifest.json')
    walking_policy=None
    if startup_enabled:
        if sha(walk_source)!=walk_info['sha256']:raise RuntimeError('Walking teacher changed')
        shutil.copy2(walk_source,src/'walking_teacher.pt')
        from isaac_task.gallop_core import FrozenWalkingPolicy
        walking_policy=FrozenWalkingPolicy(src/'walking_teacher.pt',a.device,walk_info['action_scale'],settings.action_scale)
    log=RunLog(out,status)
    class FollowingEnv(gallop_env.GallopEnv):
        def render(self,recompute=False):
            if a.video:
                target=self.scene['robot'].data.root_pos_w[0].detach().cpu().numpy()+np.array([0.,0.,.05])
                self.sim.set_camera_view(target+np.array([2.,-2.5,1.2]),target)
            return super().render(recompute=recompute)
    extra=dict(walking_policy=walking_policy) if startup_enabled else {}
    env=FollowingEnv(cfg,gallop_settings=settings,gallop_emit=log.emit,episode_path=out/'episodes.jsonl',
        training_targets=a.mode!='play',curriculum_enabled=a.mode!='play',render_mode='rgb_array' if a.video else None,**extra)
    if guarded:
        manifest['guarded_runtime_joint_contract']=dict(joint_names=env.scene['robot'].joint_names,
            original_limits=env.guard_original_limits[0].detach().cpu().tolist(),
            physics_limits=env.gallop.guard.limits[0].detach().cpu().tolist(),
            target_limits=torch.stack((env.gallop.guard.lower[0],env.gallop.guard.upper[0]),dim=-1).detach().cpu().tolist())
    if relearn:
        if parent.get('interface')==RELEARN_INTERFACE:
            from isaac_task.gallop_runtime import map_relearn_progress
            restored_progress=map_relearn_progress(relearn_parent_progress,parent,env.gallop.curriculum.plan)
            env.gallop.restore_relearn_progress(int(restored_progress['level']),restored_progress.get('highest_confirmed_speed'),freeze=a.mode!='train')
            manifest['relearn_stage_migration']=restored_progress
        manifest['relearn_design']=dict(plan=env.gallop.curriculum.plan,
            phase_observation='Zero while learning walking; radius grows with reset-latched gait strength',
            request_failure='Unfiltered requested operating angle exceeds the reset-latched tolerance for the configured duration; failure, not timeout',
            startup_exemption='Frozen startup walking/handover is protected by filters; request termination starts with full student control',
            stage_success='Walking/partial-gait gates advance early stages; full gait and higher speeds require all original final gates plus zero requested angle overflow',
            final_success='success always means full Gallop gates, never walking-stage progress')
        if recovery:
            manifest['relearn_design'].update(
                evaluation='One completed episode per environment per cohort; wait for every environment before stage decisions; accumulate complete cohorts if num_envs < min_episodes',
                exploration='Frozen state-independent Gaussian std, scheduled by constraint stage only before complete PPO rollouts; entropy coefficient zero; actor/critic continue PPO learning')
        manifest['speed_increase_rule']='Two successful eligible windows advance the walking/constraint/gait/speed plan; targets latch on reset; no time-only advancement'
    elif stabilization and a.checkpoint:env.gallop.set_evaluation_speed(float(parent.get('trained_speed_cap',settings.start_speed)),parent.get('highest_confirmed_speed'))
    env.capture_terminals=a.mode!='train'
    api=env
    if a.video and not needs_adaptation:
        video=gym.wrappers.RecordVideo(env,video_folder=str(out/'videos'),
            step_trigger=(lambda step:step%a.video_interval==0) if a.mode=='train' else (lambda step:step==0),
            video_length=a.video_length,name_prefix=f'gallop-{a.mode}',fps=round(1/env.step_dt),disable_logger=True)
        api=video
    wrapped=RslRlVecEnvWrapper(api,clip_actions=agent.clip_actions)
    def runner_emit(event,**fields):
        if relearn and event=='CHECKPOINT_SAVED':
            path=Path(fields['path']);iterations=int(runner.current_learning_iteration)+1
            inherited=manifest.get('fine_tuning',manifest.get('migration',{})).get('parent_cumulative_iterations',0)
            progress=dict(level=env.gallop.curriculum.index,speed_cap=env.gallop.curriculum.cap,
                highest_confirmed_speed=env.gallop.curriculum.highest_confirmed_speed,sha256=sha(path),
                stage_iterations=iterations,cumulative_iterations=inherited+iterations)
            manifest.setdefault('relearn_checkpoints',{})[path.name]=progress
            manifest.update(trained_relearn_level=progress['level'],trained_speed_cap=progress['speed_cap'],
                highest_confirmed_speed=progress['highest_confirmed_speed'])
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        log.emit(event,**fields)
    exploration=None
    def runner_metrics():
        values=env.gallop.metrics()
        if exploration is not None:values.update({'Relearn/noise_std':exploration.std,'Relearn/entropy_coef':runner.alg.entropy_coef})
        return values
    runner=logged_runner(OnPolicyRunner,runner_emit,metrics=runner_metrics,compact_episode_log=True)(wrapped,agent.to_dict(),log_dir=str(out) if a.mode=='train' else None,device=a.device)
    obs=wrapped.get_observations()
    if obs['policy'].shape!=(a.num_envs,observation_dim) or obs['critic'].shape!=(a.num_envs,critic_dim):raise RuntimeError('Unexpected augmented observation shape/order')
    if a.finetune_from:
        if parent['interface'] in (BASE_INTERFACE,HEADING_INTERFACE,BOUNDED_INTERFACE,GUARDED_INTERFACE,RELEARN_INTERFACE):
            from isaac_task.gallop_core import restore_gallop_policy
            info=restore_gallop_policy(runner,a.finetune_from.resolve())
            key='fine_tuning';status['training_initialization']='restore_gallop_new_optimizer'
        else:
            info=transplant_forward_policy(runner,a.finetune_from.resolve(),parent['upstream_task_audit']['action_scale'],settings.action_scale)
            key='migration'
        info['parent_cumulative_iterations']=parent_cumulative_iterations(parent,info['parent_stage_iterations'])
        if relearn_parent_progress is not None:info['parent_cumulative_iterations']=int(relearn_parent_progress['cumulative_iterations'])
        manifest[key]=info;status[key]=info;log.emit('GALLOP_INITIALIZED',**info)
    elif a.checkpoint:runner.load(str(a.checkpoint.resolve()),load_optimizer=False,map_location=a.device)
    if recovery and a.mode=='train':
        from isaac_task.gallop_core import RelearnExploration
        policy=runner.alg.policy
        previous_std=(policy.log_std.exp() if hasattr(policy,'log_std') else policy.std).detach().cpu().tolist()
        exploration=RelearnExploration(runner,env.gallop.curriculum,emit=log.emit)
        manifest['relearn_exploration']=dict(parent_std=previous_std,initial_std=exploration.std,
            final_std=settings.relearn_noise_std_end,std_trainable=False,entropy_coef=0.,
            actor_critic_preserved=True,action_samples_and_log_probabilities_unmodified=True)
    if needs_adaptation:
        from isaac_task.gallop_adaptation import adapt_bounded_policy
        status['status']='adapting';status['policy_updates']=0;save()
        # Store the frozen teacher with the new bundle for provenance.
        shutil.copy2(a.finetune_from,src/'gallop_teacher.pt')
        manifest['gallop_teacher']=dict(sha256=sha(src/'gallop_teacher.pt'),bundled_file='source/gallop_teacher.pt')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        info=adapt_bounded_policy(runner,wrapped,env,a.finetune_from,out/'adaptation',steps=a.distill_steps,
            epochs=a.distill_epochs,rounds=a.distill_rounds,batch_size=a.distill_batch_size,check_steps=a.adapt_check_steps,emit=log.emit)
        manifest['bounded_adaptation']=info;status['supervised_updates']=info['supervised_updates']
        log.emit('BOUNDED_ADAPTATION_COMPLETED',supervised_updates=info['supervised_updates'],ppo_updates=0)
        if a.video:
            # Training clips begin after migration, rather than showing the
            # frozen teacher rollout as if it were the new PPO policy.
            video=gym.wrappers.RecordVideo(env,video_folder=str(out/'videos'),
                step_trigger=lambda step:step%a.video_interval==0,video_length=a.video_length,
                name_prefix='gallop-train',fps=round(1/env.step_dt),disable_logger=True)
            wrapped.env=video
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    status['status']='running';save()
    ready=dict(output=str(out.resolve()),start_speed=settings.start_speed,max_speed=settings.max_speed,strict_foot_order=settings.require_touchdown_order)
    if getattr(settings,'leap_reward_weight',0)>0:ready.update(leap_min_distance_m=settings.leap_min_distance,leap_sample_seconds=cfg.sim.dt)
    if getattr(settings,'flight_shaping_enabled',False):ready.update(flight_shaping=True,stable_landing_seconds=settings.flight_recovery_seconds)
    if startup_enabled:ready.update(startup='hold -> walk -> stable blend -> Gallop',walk_speed=settings.startup_walk_speed)
    if heading_motion:ready.update(posture='yaw zero + quiet roll/pitch + shoulder tracks + forward direction',
        front_width_m=settings.front_stance_width,hind_width_m=settings.hind_stance_width)
    elif getattr(settings,'posture_enabled',False):ready.update(posture='upright + centered foot tracks',
        front_width_m=settings.front_stance_width,hind_width_m=settings.hind_stance_width)
    if stabilization:ready.update(stabilization=True,current_speed_cap=env.gallop.curriculum.cap,
        leap_min_distance_m=float(env.gallop.leap.distance_target[0]),
        initial_leap_m=settings.stabilization_min_leap_distance,final_leap_m=settings.leap_min_distance)
    if guarded:ready.update(guarded=True,imitation=False,front_height_cap=settings.guarded_front_height,hind_height_cap=settings.guarded_hind_height)
    if relearn:ready.update(relearn_level=env.gallop.curriculum.index,relearn_goal=env.gallop.curriculum.goal)
    log.emit('GALLOP_READY',**ready)
    if a.mode=='train':
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=not startup_enabled)
        runner.save(str(out/'model_final.pt'));status['policy_updates']=int(runner.current_learning_iteration)+1
        lineage=manifest.get('fine_tuning',manifest.get('migration',{}))
        status['cumulative_iterations']=status['policy_updates']+lineage.get('parent_cumulative_iterations',0)
        manifest['stage_iterations']=status['policy_updates'];manifest['cumulative_iterations']=status['cumulative_iterations']
    else:
        initial={k:v.detach().clone() for k,v in runner.alg.policy.state_dict().items()}
        actor=runner.get_inference_policy(device=a.device);term=env.command_manager.get_term('base_velocity')
        command=None if startup_enabled else a.command or ([float(parent.get('trained_speed_cap',settings.start_speed)),0.,0.] if a.mode=='play' else None)
        manifest['evaluation_command']=command
        samples={};alive=np.ones(a.num_envs,bool);valid=[];resets=[];times=[];done_count=0
        with torch.inference_mode():
            for step in range(a.steps):
                if command is not None:
                    term.vel_command_b[:]=torch.tensor(command,device=a.device);term.is_standing_env[:]=False;term.time_left.fill_(1e6)
                    obs=TensorDict(env.observation_manager.compute(update_history=True),batch_size=[a.num_envs])
                requested=obs['policy'][:,6:9].detach().cpu().numpy().copy()
                if not (np.all(requested[:,0]>=0 if startup_enabled else requested[:,0]>0) and np.all(requested[:,1:]==0)):raise RuntimeError('Non-forward Gallop command')
                if startup_enabled:
                    np.testing.assert_allclose(requested[:,0],env.gallop.startup.command.detach().cpu().numpy(),atol=1e-6,rtol=0)
                if command is not None:np.testing.assert_allclose(requested,np.broadcast_to(command,requested.shape),atol=1e-6,rtol=0)
                action=actor(obs);obs,reward,done,_=wrapped.step(action)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs['policy']).all():raise RuntimeError('Nonfinite rollout')
                sample=env.gallop.snapshot()
                for i,terminal in env.terminal_snapshots.items():
                    for key,value in terminal.items():sample[key][i]=value
                sample['command']=requested;sample['action']=action.detach().cpu().numpy().copy()
                for key,value in sample.items():samples.setdefault(key,[]).append(value)
                valid.append(alive.copy());reset=done.detach().cpu().numpy().astype(bool)
                alive&=~reset;resets.append(reset);done_count+=int(reset.sum());times.append((step+1)*env.step_dt)
                if (step+1)%250==0:log.emit('GALLOP_CHECK',step=step+1,total_steps=a.steps,speed_cap=env.gallop.curriculum.cap,reset_count=done_count)
        if not all(torch.equal(v,initial[k]) for k,v in runner.alg.policy.state_dict().items()):raise RuntimeError('Policy changed in check/play')
        np.savez_compressed(out/'gallop_trace.npz',time=np.array(times),valid=np.array(valid),episode_reset=np.array(resets),**{k:np.stack(v) for k,v in samples.items()})
        status.update(steps=a.steps,reset_count=done_count,policy_state_unchanged=True,
                      rollout_note='First segment validity only; reset rows use pre-reset terminal snapshots')
    manifest['trained_speed_cap']=env.gallop.curriculum.cap if a.mode=='train' else parent.get('trained_speed_cap',settings.start_speed)
    if relearn:manifest['trained_relearn_level']=env.gallop.curriculum.index
    if stabilization:manifest['highest_confirmed_speed']=env.gallop.curriculum.highest_confirmed_speed if a.mode=='train' else parent.get('highest_confirmed_speed')
    if relearn:
        manifest['trained_speed_cap']=env.gallop.curriculum.cap
        manifest['highest_confirmed_speed']=env.gallop.curriculum.highest_confirmed_speed
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'curriculum_summary.json').write_text(json.dumps(dict(speed_cap=env.gallop.curriculum.cap,events=env.gallop.curriculum.events,last_episode=env.gallop.last_episode),indent=2))
    if video is not None:
        if video.recording:video.stop_recording()
        status['videos']=[str(p.resolve()) for p in (out/'videos').glob('*.mp4')]
        if not status['videos']:raise RuntimeError('Requested video missing')
    verify_source()
    if source_checkpoint and sha(source_checkpoint)!=manifest['source_checkpoint']['sha256']:raise RuntimeError('Parent checkpoint changed')
    if startup_enabled and sha(walk_source)!=walk_info['sha256']:raise RuntimeError('Walking checkpoint changed')
    status.update(status='completed',source_clean_after_run=True);save();log.emit('COMPLETED',output=str(out.resolve()))


try:
    app=AppLauncher(a).app;main()
except BaseException as error:
    status.update(status='failed',error=str(error),traceback=traceback.format_exc());save();raise
finally:
    if video is not None:video.close()
    elif env is not None:env.close()
    if app is not None:app.close()
