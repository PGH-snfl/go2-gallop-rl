"""Import only after Isaac AppLauncher. Gallop task atop the pinned Unitree robot."""
import json
import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils.math import euler_xyz_from_quat

from isaac_task.gallop_core import SuccessSpeedCurriculum, phase_targets, forelimb_success, gait_pattern_success, pair_joint_penalty, PairBalanceTracker, LeapDistanceTracker, FlightStabilityTracker, flight_phase_targets, posture_terms
from isaac_task.gallop_core import WalkingToGallop,blend_startup_actions,heading_features,HeadingMotionTracker
from isaac_task.gallop_core import stabilization_leap_target,stabilized_student_actions,target_excess_penalty
from isaac_task.gallop_core import bounded_startup_targets,bounded_limit_cost
from isaac_task.gallop_core import GuardedJointController,guarded_physics_limits,guarded_foot_height_cost
from isaac_task.gallop_core import RelearnCurriculum,RequestLimitMonitor
from isaac_task.gallop_core import StagedGallopCurriculum,StagedEpisodeProfile,CycleSpeedTracker

LEGS=('FL','FR','RL','RR')


class BoundedJointPositionAction(JointPositionAction):
    def process_actions(self,actions):
        state=getattr(self._env,'gallop',None)
        if state is not None and getattr(state.cfg,'guarded_enabled',False):
            calf_mix=state.startup.mix*(state.relearn_gait if state.cfg.relearn_enabled else 1.)
            student=stabilized_student_actions(actions,state.calf_ids,calf_mix,state.cfg)
            proposal=blend_startup_actions(student,state.walking_policy(self._env.obs_buf['policy']),state.startup.stage,state.startup.mix)
            super().process_actions(proposal)
            requested=self._processed_actions.clone();limits=state.guard.limits
            angle_clipped=torch.clamp(requested,min=limits[...,0],max=limits[...,1])
            state.action_clip.copy_(((angle_clipped-requested).abs()>1e-6).float().mean(-1))
            state.joint_target_excess.copy_((requested-angle_clipped).abs())
            state.action_excess.copy_(target_excess_penalty(requested,angle_clipped,state.calf_ids,state.cfg))
            target=state.guard.process(requested,self._env.step_dt)
            if state.cfg.relearn_enabled:state.observe_requested(requested)
            default=self._asset.data.default_joint_pos[:,self._joint_ids]
            executed=(target-default)/state.cfg.action_scale
            state.executed_actions.copy_(executed);self._env.action_manager.action.copy_(executed)
            self._processed_actions=target
            return
        if state is not None and getattr(state.cfg,'bounded_actions_enabled',False) and not state.bounded_teacher_rollout:
            # Keep Gaussian latent samples intact for PPO. Only executed target
            # equivalents are exposed as previous actions to the robot/teacher.
            super().process_actions(actions)
            limits=self._asset.data.soft_joint_pos_limits[:,self._joint_ids]
            default=self._asset.data.default_joint_pos[:,self._joint_ids]
            target=bounded_startup_targets(actions,state.walking_policy(self._env.obs_buf['policy']),
                default,limits,state.startup.stage,state.startup.mix,state.cfg)
            clipped=torch.clamp(target,min=limits[...,0],max=limits[...,1])
            state.action_clip.copy_(((clipped-target).abs()>1e-6).float().mean(-1))
            state.joint_target_excess.copy_((target-clipped).abs())
            state.action_excess.copy_(target_excess_penalty(target,clipped,state.calf_ids,state.cfg))
            state.bounded_near_cost[:],state.bounded_near_fraction[:]=bounded_limit_cost(clipped,limits,state.calf_ids,state.cfg.bounded_limit_margin)
            rate=(clipped-state.bounded_previous_target)/self._env.step_dt
            state.bounded_target_rate.copy_(torch.where(state.bounded_previous_valid[:,None],rate,0.))
            state.bounded_rate_cost.copy_(torch.nn.functional.huber_loss(state.bounded_target_rate/state.cfg.bounded_target_rate_scale,
                torch.zeros_like(rate),reduction='none').mean(-1))
            state.bounded_previous_target.copy_(clipped);state.bounded_previous_valid.fill_(True)
            state.bounded_target.copy_(clipped)
            executed=(clipped-default)/state.cfg.action_scale
            state.executed_actions.copy_(executed);self._env.action_manager.action.copy_(executed)
            self._processed_actions=clipped
            return
        if state is not None and state.startup is not None:
            walking=state.walking_policy(self._env.obs_buf['policy'])
            if state.cfg.stabilization_enabled:
                actions=stabilized_student_actions(actions,state.calf_ids,state.startup.mix,state.cfg)
            actions=blend_startup_actions(actions,walking,state.startup.stage,state.startup.mix)
            state.executed_actions.copy_(actions)
            # PPO keeps its sampled latent actions. Physics, previous-action
            # observations and action-rate rewards use the executed mixture.
            self._env.action_manager.action.copy_(actions)
        super().process_actions(actions)
        limits=self._asset.data.soft_joint_pos_limits[:,self._joint_ids]
        clipped=torch.clamp(self._processed_actions,min=limits[...,0],max=limits[...,1])
        if state is not None:state.action_clip=((clipped-self._processed_actions).abs()>1e-6).float().mean(dim=1)
        if state is not None and state.cfg.stabilization_enabled:
            state.joint_target_excess.copy_((self._processed_actions-clipped).abs())
            state.action_excess.copy_(target_excess_penalty(self._processed_actions,clipped,state.calf_ids,state.cfg))
        self._processed_actions=clipped
        if state is not None and getattr(state.cfg,'bounded_actions_enabled',False):
            # Legacy teacher collection only; keep its original pre-clip
            # previous-action semantics, but log the real physical target.
            state.bounded_target.copy_(clipped)


class StartupVelocityCommand(UniformVelocityCommand):
    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids]=0.
        self.is_standing_env[env_ids]=False

    def _update_command(self):
        state=getattr(self._env,'gallop',None)
        if state is None:return
        d=state.robot.data
        rpy=torch.stack(euler_xyz_from_quat(d.root_quat_w),dim=1)
        rpy=(rpy+math.pi)%(2*math.pi)-math.pi
        state.startup.advance(self._env.step_dt,d.root_lin_vel_b,d.root_ang_vel_b,rpy,
            d.root_pos_w[:,2]-self._env.scene.env_origins[:,2],state.age>0)
        self.vel_command_b.zero_();self.vel_command_b[:,0]=state.startup.command
        self.is_standing_env[:]=False


def phase_observation(env):
    if getattr(env,'gallop',None) is None:return torch.zeros(env.num_envs,2,device=env.device)
    p=2*math.pi*env.gallop.phase
    values=torch.stack((torch.sin(p),torch.cos(p)),dim=1)
    if env.gallop.startup is not None:
        radius=torch.where(env.gallop.startup.stage==0,0.,.5+.5*env.gallop.startup.mix)
        values*=radius[:,None]
    if env.gallop.cfg.relearn_enabled:values*=env.gallop.relearn_gait[:,None]
    return values


def heading_observation(env,target_yaw):
    return heading_features(env.scene['robot'].data.root_quat_w,target_yaw)


def gallop_reward(env,key):
    env.gallop.update_measurements()
    return env.gallop.components[key]


def height_termination(env,minimum,maximum):
    z=env.scene['robot'].data.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return (z<minimum)|(z>maximum)


def guarded_limit_termination(env):
    if getattr(env,'gallop',None) is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    state=env.gallop;state.guard.observe_actual(state.robot.data.joint_pos,env.step_dt,env.common_step_counter)
    return state.guard.violation_run>=state.cfg.guarded_violation_seconds-1e-6


def requested_limit_termination(env):
    state=getattr(env,'gallop',None)
    if state is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    return state.request_monitor.failed


def startup_failure(env):
    state=getattr(env,'gallop',None)
    if state is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    return state.startup.failed


def apply_guarded_sim_limits(robot,settings):
    """Constrain the per-run physics instance; the original USD stays intact."""
    if robot.cfg.soft_joint_pos_limit_factor!=1.:raise ValueError('Guarded physics must not shrink the limits twice')
    original=robot.data.joint_pos_limits.clone()
    limits=guarded_physics_limits(original,settings.guarded_sim_limit_factor)
    robot.write_joint_position_limit_to_sim(limits)
    actual=robot.root_physx_view.get_dof_limits().to(limits.device)
    if not torch.allclose(actual,limits,atol=1e-6) or not torch.allclose(robot.data.soft_joint_pos_limits,limits,atol=1e-6):
        raise RuntimeError('Guarded physics limits were not applied')
    return original


def configure_gallop(cfg,settings):
    # High-speed 20-second episodes must not run off the original finite tiles.
    cfg.scene.terrain.terrain_type='plane';cfg.scene.terrain.terrain_generator=None
    cfg.scene.height_scanner=None;cfg.curriculum.terrain_levels=None
    cmd=cfg.commands.base_velocity
    cmd.rel_standing_envs=0.;cmd.heading_command=False;cmd.debug_vis=False
    for r in (cmd.ranges,cmd.limit_ranges):
        r.lin_vel_x=(settings.start_speed-settings.speed_band,settings.start_speed)
        r.lin_vel_y=(0.,0.);r.ang_vel_z=(0.,0.)
    cmd.limit_ranges.lin_vel_x=(settings.start_speed-settings.speed_band,settings.max_speed)
    cfg.curriculum.lin_vel_cmd_levels=None
    if settings.startup_enabled:
        cmd.class_type=StartupVelocityCommand
        cfg.episode_length_s=settings.startup_episode_seconds
        cfg.events.reset_base.params['velocity_range']={key:(0.,0.) for key in ('x','y','z','roll','pitch','yaw')}
        cfg.events.reset_robot_joints.params['velocity_range']=(0.,0.)
        if settings.startup_pose_handover:
            cfg.terminations.gallop_startup_deadline=TerminationTermCfg(func=startup_failure,time_out=False)
            # RewardManager multiplies by step_dt: -1000 * .02 = -20 on failure.
            cfg.rewards.gallop_startup_failure=RewardTermCfg(func=startup_failure,weight=-1000.)
    if settings.heading_motion_enabled:
        # Absolute yaw zero is world +X; do not spawn the frozen forward-only
        # walking teacher facing a uniformly random world direction.
        cfg.events.reset_base.params['pose_range']['yaw']=(settings.target_yaw,settings.target_yaw)
    cfg.actions.JointPositionAction.class_type=BoundedJointPositionAction
    cfg.actions.JointPositionAction.scale=settings.action_scale
    if settings.guarded_enabled:
        cfg.scene.robot.soft_joint_pos_limit_factor=1.
        cfg.events.reset_robot_joints.params['position_range']=(1.,1.)
    if settings.leap_reward_weight>0:cfg.scene.contact_forces.update_period=cfg.sim.dt
    cfg.observations.policy.gallop_phase=ObservationTermCfg(func=phase_observation)
    cfg.observations.critic.gallop_phase=ObservationTermCfg(func=phase_observation)
    if settings.heading_motion_enabled:
        for group in (cfg.observations.policy,cfg.observations.critic):
            group.gallop_heading=ObservationTermCfg(func=heading_observation,params={'target_yaw':settings.target_yaw})
    for key in ('feet_air_time','air_time_variance','joint_pos','flat_orientation_l2'):setattr(cfg.rewards,key,None)
    cfg.rewards.track_lin_vel_xy.weight=3.
    cfg.rewards.base_linear_velocity.weight=-.1
    cfg.rewards.base_angular_velocity.weight=-.02
    cfg.rewards.action_rate.weight=-.02
    weights=dict(contact=2.,order=settings.order_reward_weight,swing=1.,joint=.6,motion=1.,flight=.75,fore_use=1.,attitude=1.,
                 pair_joint=settings.pair_joint_weight,pair_rom=settings.pair_rom_weight,
                 pair_load=settings.pair_load_weight,pair_stance=settings.pair_stance_weight,
                 leap_distance=settings.leap_reward_weight)
    if settings.pair_hip_weight>0:weights['pair_hip']=settings.pair_hip_weight
    if settings.posture_enabled:
        if settings.heading_motion_enabled:
            weights['attitude']=0.
            weights.update(yaw_heading=settings.yaw_penalty_weight,roll_pitch_quiet=settings.roll_pitch_quiet_weight,
                front_width=settings.stance_width_weight/2,hind_width=settings.stance_width_weight/2,
                forward_progress=settings.forward_progress_weight)
            if settings.stabilization_enabled:weights.update(yaw_damping=settings.stabilization_yaw_rate_weight,target_excess=settings.stabilization_excess_weight)
        else:
            for key in ('upright','stance_width','stance_center'):weights[key]=getattr(settings,key+'_weight')
    if settings.flight_shaping_enabled:
        for key in ('drive','clearance','support','balance','takeoff','landing','impact','clip'):
            weights['flight_'+key]=getattr(settings,'flight_'+key+'_weight')
    if settings.bounded_actions_enabled:
        weights.update(bounded_near_limit=settings.bounded_limit_weight,bounded_target_rate=settings.bounded_target_rate_weight)
    if settings.guarded_enabled:
        weights.update(guarded_projection=settings.guarded_projection_weight,guarded_actual=settings.guarded_actual_weight,
            guarded_front_height=settings.guarded_height_weight,guarded_hind_height=settings.guarded_height_weight)
        cfg.terminations.guarded_joint_limit=TerminationTermCfg(func=guarded_limit_termination)
    if settings.relearn_enabled:
        cfg.terminations.requested_joint_limit=TerminationTermCfg(func=requested_limit_termination,time_out=False)
    for key,weight in weights.items():
        setattr(cfg.rewards,'gallop_'+key,RewardTermCfg(func=gallop_reward,weight=weight,params={'key':key}))
    cfg.terminations.gallop_height=TerminationTermCfg(func=height_termination,params={'minimum':settings.min_height,'maximum':settings.max_height})


class GallopState:
    def __init__(self,env,settings,emit,episode_path,training_targets=True,curriculum_enabled=True,walking_policy=None):
        self.env=env;self.cfg=settings;self.episode_path=episode_path
        self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces']
        self.foot_ids,_=self.robot.find_bodies([f'{x}_foot' for x in LEGS],preserve_order=True)
        self.sensor_ids=[self.sensor.body_names.index(f'{x}_foot') for x in LEGS]
        ids,_=self.robot.find_joints([f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip','thigh','calf')],preserve_order=True)
        if len(ids)!=12 or len(self.foot_ids)!=4:raise RuntimeError('Gallop joint/foot mapping mismatch')
        self.joint_ids=torch.tensor(ids,device=env.device).reshape(4,3)
        self.calf_ids=self.joint_ids[:,2]
        self.permutations=torch.tensor([[0,1,2,3],[1,0,2,3],[0,1,3,2],[1,0,3,2]] if settings.flexible_left_right else [[0,1,2,3]],device=env.device)
        self.rows=torch.arange(env.num_envs,device=env.device)
        self.curriculum=(StagedGallopCurriculum if settings.staged_enabled else RelearnCurriculum if settings.relearn_enabled else SuccessSpeedCurriculum)(settings,emit);self.curriculum_enabled=curriculum_enabled
        self.training_targets=training_targets;self.cache_step=-1;self.last_episode={}
        n=env.num_envs;self.phase=torch.zeros(n,device=env.device)
        self.profile=StagedEpisodeProfile(self.phase,settings,self.curriculum.plan) if settings.staged_enabled else None
        self.speed_diagnostics=CycleSpeedTracker(self.phase) if settings.staged_speed_adaptation_enabled else None
        if settings.staged_enabled:self.curriculum.configure_envs(n,env.step_dt)
        if settings.relearn_enabled:self.curriculum.configure_envs(n)
        def zeros(*shape):return torch.zeros(n,*shape,device=env.device)
        self.age=zeros().long();self.first=torch.ones(n,dtype=torch.bool,device=env.device);self.level=zeros().long()
        self.episode_speed=zeros()+settings.start_speed
        if settings.relearn_enabled:
            self.relearn_index=zeros().long();self.relearn_gait=zeros()
            self.relearn_tolerance=zeros();self.relearn_request_seconds=zeros();self.relearn_allowance=zeros()
            self.relearn_request_sum=zeros();self.request_monitor=RequestLimitMonitor(zeros())
            self.latch_relearn(torch.arange(n,device=env.device))
        self.startup=WalkingToGallop(zeros(),settings) if settings.startup_enabled else None
        if self.startup is not None and walking_policy is None:raise ValueError('Startup requires the frozen walking policy')
        self.walking_policy=walking_policy;self.executed_actions=zeros(12)
        if settings.guarded_enabled:
            self.guard=GuardedJointController(self.robot.data.default_joint_pos,self.robot.data.soft_joint_pos_limits,
                self.joint_ids[:,0],self.calf_ids,settings)
            self.guard_sums={k:zeros() for k in ('intervention','actual','front_high','hind_high','front_cost','hind_cost','tracking2')}
            self.guard_height=zeros(4)
        if settings.bounded_actions_enabled:
            self.bounded_teacher_rollout=False
            self.bounded_target=zeros(12);self.bounded_previous_target=zeros(12);self.bounded_previous_valid=zeros().bool()
            self.bounded_target_rate=zeros(12);self.bounded_near_cost=zeros();self.bounded_near_fraction=zeros();self.bounded_rate_cost=zeros()
            self.bounded_tracking_error=zeros();self.bounded_torque_fraction=zeros()-1
            self.bounded_sums={k:zeros() for k in ('near','near_cost','rate_cost','tracking2','torque','torque_points')}
        self.used_stage=zeros().long();self.used_mix=zeros()+1.;self.cycle_metric_ok=zeros().bool();self.metric_used=zeros().bool()
        self.contacts=zeros(4).bool();self.seeded=zeros().bool();self.last_order=zeros().long()-1
        self.streak=zeros().long();self.events=zeros();self.good_events=zeros();self.sequences=zeros()
        self.cycles=zeros();self.flight_cycles=zeros();self.air_run=zeros();self.air_max=zeros();self.rom_sum=zeros();self.last_rom=zeros()
        self.cycle_impulse=zeros(4);self.total_impulse=zeros(4);self.fore_stance=zeros(2);self.fore_swing=zeros(2)
        self.fore_participation=zeros(2);self.fore_rom_sum=zeros(2)
        self.last_fore_use=zeros()
        self.last_fore_ok=zeros().bool()
        self.leap_enabled=settings.leap_reward_weight>0
        self.leap=LeapDistanceTracker(self.robot.data.root_pos_w-env.scene.env_origins,settings)
        if settings.stabilization_enabled:self.leap.distance_target.copy_(stabilization_leap_target(self.episode_speed,settings))
        if self.profile is not None:
            self.leap.cfg=self.profile;self.leap.distance_target.fill_(self.curriculum.goal['leap'])
        self.flight=None
        if settings.flight_shaping_enabled:
            mass=self.robot.root_physx_view.get_masses().sum(dim=1).to(env.device)
            gravity=math.sqrt(sum(float(x)**2 for x in env.cfg.sim.gravity))
            self.flight=FlightStabilityTracker(self.robot.data.root_pos_w,settings,mass*gravity)
            if self.profile is not None:self.flight.cfg=self.profile
        self.leap_score=zeros();self.leap_reward_sum=zeros();self.leap_reward_steps=zeros()
        q=self.robot.data.joint_pos[:,self.joint_ids][:,:,1:]
        self.qmin=q.clone();self.qmax=q.clone();self.action_clip=zeros()
        self.pair=PairBalanceTracker(self.robot.data.joint_pos[:,self.joint_ids],settings)
        if self.profile is not None:self.pair.cfg=self.profile
        self.pair_costs={key:zeros() for key in ('joint','rom','load','stance')}
        if settings.pair_hip_weight>0:self.pair_costs['hip']=zeros()
        self.sums={k:zeros() for k in ('points','vx2','vy2','yaw2','roll2','pitch2','match','clip','command','pair_joint','pair_rom','pair_load','pair_stance')}
        if settings.staged_enabled:self.sums['heading2']=zeros()
        if settings.stabilization_enabled:
            self.action_excess=zeros();self.joint_target_excess=zeros(12)
            self.stability_sums={key:zeros() for key in ('front_support','hind_support','front_width_good','hind_width_good','excess')}
        self.components={k:zeros() for k in ('contact','order','swing','joint','motion','flight','fore_use','attitude','pair_joint','pair_rom','pair_load','pair_stance','leap_distance')}
        if settings.posture_enabled:
            if settings.heading_motion_enabled:
                self.motion=HeadingMotionTracker(self.robot.data.root_pos_w,settings)
                self.posture_values=self.motion.values
                self.components.update({key:zeros() for key in ('yaw_heading','roll_pitch_quiet','front_width','hind_width','forward_progress')})
                if settings.stabilization_enabled:self.components.update(yaw_damping=zeros(),target_excess=zeros())
            else:
                self.components.update({key:zeros() for key in ('upright','stance_width','stance_center')})
                self.posture_values=dict(tilt=zeros(),width=zeros(2),center=zeros(2),upright_cost=zeros(),width_cost=zeros(),center_cost=zeros())
                # Independent all-stage denominator: startup may never enter Gallop.
                self.posture_sums={key:zeros() for key in ('points','tilt2','upright_cost','width_cost','center_cost',
                    'front_width','hind_width','front_width_error2','hind_width_error2','front_center2','hind_center2')}
        if settings.pair_hip_weight>0:self.sums['pair_hip']=zeros();self.components['pair_hip']=zeros()
        if self.flight is not None:
            self.components.update({'flight_'+key:zeros() for key in ('drive','clearance','support','balance','takeoff','landing','impact','clip')})
        if settings.bounded_actions_enabled:self.components.update(bounded_near_limit=zeros(),bounded_target_rate=zeros())
        if settings.guarded_enabled:self.components.update(guarded_projection=zeros(),guarded_actual=zeros(),guarded_front_height=zeros(),guarded_hind_height=zeros())
        self.expected=zeros(4).bool();self.qref=self.robot.data.joint_pos[:,self.joint_ids].clone();self.height_ref=zeros(4)
        self.command=env.command_manager.get_command('base_velocity').clone()
        self.period=phase_targets(self.phase,self.command[:,0],settings,1.,profile=self.profile)['period']

    def restore_staged_progress(self,index,confirmed=None,freeze=True):
        if not self.cfg.staged_enabled or (self.age>0).any():raise ValueError('Restore staged progress before stepping a staged policy')
        self.curriculum.restore(index,confirmed);self.curriculum_enabled=not freeze
        self.level.fill_(self.curriculum.level)
        self.profile.latch(torch.arange(self.env.num_envs,device=self.env.device),index)
        self.episode_speed.fill_(self.curriculum.goal['speed']);self.startup.goal_speed.copy_(self.episode_speed)
        self.leap.distance_target.fill_(self.curriculum.goal['leap'])

    def latch_relearn(self,ids):
        goal=self.curriculum.goal
        self.relearn_index[ids]=self.curriculum.index;self.relearn_gait[ids]=goal['gait']
        self.relearn_tolerance[ids]=goal['request_tolerance'];self.relearn_request_seconds[ids]=goal['request_seconds']
        self.relearn_allowance[ids]=goal['request_fraction'];self.episode_speed[ids]=goal['speed']

    def restore_relearn_progress(self,index,confirmed_speed=None,freeze=True):
        if not self.cfg.relearn_enabled:raise ValueError('Not a relearning policy')
        if (self.age>0).any():raise ValueError('Restore relearning progress before stepping')
        self.curriculum.restore(index,confirmed_speed)
        self.curriculum_enabled=not freeze
        self.level.fill_(self.curriculum.level)
        self.latch_relearn(torch.arange(self.env.num_envs,device=self.env.device))
        self.startup.goal_speed.copy_(self.episode_speed)
        self.leap.distance_target.copy_(stabilization_leap_target(self.episode_speed,self.cfg))

    def observe_requested(self,requested):
        self.request_monitor.observe(requested,self.guard.lower,self.guard.upper,self.relearn_tolerance,
            self.relearn_request_seconds,self.used_stage==3,self.env.step_dt)

    def set_evaluation_speed(self,speed,confirmed_speed=None):
        if self.cfg.staged_enabled:raise ValueError('Staged replay requires checkpoint-specific stage, not just speed')
        if not self.cfg.stabilization_enabled:return
        if not self.cfg.start_speed<=speed<=self.cfg.max_speed:raise ValueError('Saved evaluation speed outside stabilization curriculum')
        if confirmed_speed is not None and not self.cfg.start_speed<=confirmed_speed<=self.cfg.max_speed:raise ValueError('Saved confirmed speed outside curriculum')
        if (self.age>0).any():raise ValueError('Set replay speed before the first control step')
        self.curriculum_enabled=False
        self.curriculum.cap=float(speed);self.episode_speed.fill_(speed);self.startup.goal_speed.fill_(speed)
        self.curriculum.highest_confirmed_speed=None if confirmed_speed is None else float(confirmed_speed)
        self.leap.distance_target.copy_(stabilization_leap_target(self.episode_speed,self.cfg))

    @property
    def strength(self):
        if self.cfg.relearn_enabled:return 1.
        if not self.training_targets:return 1.
        fraction=min(1.,self.env.common_step_counter/(24*self.cfg.reward_ramp_iterations))
        return self.cfg.initial_gait_weight+(1-self.cfg.initial_gait_weight)*fraction

    def metric_active(self):
        if self.startup is None:return self.age*self.env.step_dt>=self.cfg.metric_warmup_seconds
        return (self.startup.stage==3)&(self.startup.times[:,3]>=self.cfg.metric_warmup_seconds)

    def begin_step(self):
        self.command=self.env.command_manager.get_command('base_velocity').clone()
        if self.startup is not None:
            self.used_stage=self.startup.stage.clone();self.used_mix=self.startup.mix.clone()
        target=phase_targets(self.phase,self.command[:,0],self.cfg,self.strength,profile=self.profile)
        if self.startup is not None:target['period']=.5*(1-self.used_mix)+target['period']*self.used_mix
        next_phase=(self.phase+self.env.step_dt/target['period'])%1.
        if self.startup is not None:next_phase=torch.where(self.used_stage==0,self.phase,next_phase)
        wrapped=next_phase<self.phase
        count_wrap=wrapped
        if self.startup is not None:
            count_wrap=wrapped&self.cycle_metric_ok
            self.cycle_metric_ok=torch.where(wrapped,self.metric_active(),self.cycle_metric_ok&self.metric_active())
        self.leap.begin_step(wrapped)
        if self.speed_diagnostics is not None:self.speed_diagnostics.finish_cycles(wrapped)
        if self.flight is not None:self.flight.begin_step(wrapped)
        self.period=target['period']
        self.pair.finish_cycles(wrapped,self.robot.data.joint_pos[:,self.joint_ids])
        target_rom=torch.stack((target['rom_thigh'],target['rom_calf']),dim=1)[:,None,:]
        rom=((self.qmax-self.qmin)/target_rom.clamp_min(.01)).clamp(0,1)
        fraction=rom.mean(dim=(1,2));fore_rom=rom[:,:2].amin(dim=2)
        participation=(self.fore_stance>=self.cfg.min_fore_stance_seconds)&(self.fore_swing>=self.cfg.min_fore_swing_seconds)
        load=self.cycle_impulse/self.cycle_impulse.sum(dim=1,keepdim=True).clamp_min(.01)
        load_score=torch.stack((load[:,:2].sum(dim=1)/self.cfg.min_fore_load_share,
                                load[:,2:].sum(dim=1)/self.cfg.min_hind_load_share,
                                load[:,:2].amin(dim=1)/self.cfg.min_each_fore_load_share),dim=1).amin(dim=1).clamp(0,1)
        fore_use=.4*load_score+.3*fore_rom.mean(dim=1)+.3*participation.float().mean(dim=1)
        self.last_fore_use=torch.where(wrapped,fore_use,self.last_fore_use)
        fore_flags=forelimb_success(load,fore_rom,participation.float(),self.cfg)
        self.last_fore_ok=torch.where(wrapped,fore_flags[0]&fore_flags[1]&fore_flags[2],self.last_fore_ok)
        self.fore_rom_sum+=count_wrap[:,None]*fore_rom
        self.fore_participation+=count_wrap[:,None]*participation.float()
        self.cycle_impulse=torch.where(wrapped[:,None],0.,self.cycle_impulse)
        self.fore_stance=torch.where(wrapped[:,None],0.,self.fore_stance)
        self.fore_swing=torch.where(wrapped[:,None],0.,self.fore_swing)
        self.last_rom=torch.where(wrapped,fraction,self.last_rom)
        self.rom_sum+=torch.where(count_wrap,fraction,0.)
        self.cycles+=count_wrap.float()
        self.flight_cycles+=(count_wrap&(self.air_max*self.env.step_dt>=(self.profile or self.cfg).flight_min_seconds-1e-6)).float()
        self.air_max=torch.where(wrapped,0.,self.air_max);self.air_run=torch.where(wrapped,0.,self.air_run)
        q=self.robot.data.joint_pos[:,self.joint_ids][:,:,1:]
        self.qmin=torch.where(wrapped[:,None,None],q,self.qmin);self.qmax=torch.where(wrapped[:,None,None],q,self.qmax)
        self.phase=next_phase;self.age+=1

    def observe_leap_physics(self,dt):
        data=self.robot.data;position=data.root_pos_w-self.env.scene.env_origins
        eligible=self.last_fore_ok if self.cfg.leap_require_forelimbs else torch.ones_like(self.last_fore_ok)
        if self.flight is not None:
            eligible=torch.where(self.leap.active,self.flight.fore_ok_at_takeoff,eligible)
        eligible=eligible&(position[:,2]>=self.cfg.min_height)&(position[:,2]<=self.cfg.max_height)
        force=self.sensor.data.net_forces_w[:,self.sensor_ids,2]
        start,finish,gain=self.leap.observe(position,data.root_quat_w,force,dt,eligible,
            measure_enabled=self.metric_active() if self.startup is not None else None)
        if self.flight is not None:
            # self.phase is the end-of-policy-step phase. Evaluate each substep
            # at its actual timestamp rather than shifting the flight window.
            phase=(self.phase-(self.env.cfg.decimation-self.leap.samples)*dt/self.period)%1.
            rpy=torch.stack(euler_xyz_from_quat(data.root_quat_w),dim=1)
            rpy=(rpy+math.pi)%(2*math.pi)-math.pi
            self.flight.observe(self.leap,start,finish,gain,phase,self.period,position,data.root_lin_vel_w[:,2],
                data.root_lin_vel_b,data.root_ang_vel_b,rpy,
                data.body_pos_w[:,self.foot_ids,2]-self.env.scene.env_origins[:,None,2]-self.cfg.foot_radius,
                data.body_lin_vel_w[:,self.foot_ids],force,self.command,self.last_fore_ok,dt,
                self.used_mix*(self.relearn_gait if self.cfg.relearn_enabled else 1.))

    def update_measurements(self):
        if self.cache_step==self.env.common_step_counter:return
        self.cache_step=self.env.common_step_counter
        if self.leap_enabled and self.leap.samples!=self.env.cfg.decimation:
            raise RuntimeError(f'Expected {self.env.cfg.decimation} physics samples for leap reward; got {self.leap.samples}')
        force=self.sensor.data.net_forces_w[:,self.sensor_ids,2]
        contact=torch.where(self.contacts,force>self.cfg.contact_off,force>=self.cfg.contact_on)
        touchdown=contact&~self.contacts&self.seeded[:,None]
        self.contacts=contact;self.seeded[:]=True
        count=touchdown.sum(dim=1);foot=touchdown.long().argmax(dim=1)
        order_idx=foot.new_tensor([2,3,0,1])[foot]
        good=(count==1)&((self.last_order<0)|(order_idx==(self.last_order+1)%4))
        self.last_order=torch.where(count>0,order_idx,self.last_order)
        self.streak=torch.where(count==0,self.streak,torch.where(good,self.streak+1,0))
        self.sequences+=(self.streak>=4).float();self.streak%=4
        self.events+=count.float();self.good_events+=good.float()
        self.air_run=torch.where(~contact.any(dim=1),self.air_run+1,0.)
        self.air_max=torch.maximum(self.air_max,self.air_run)
        t=phase_targets(self.phase,self.command[:,0],self.cfg,self.strength,profile=self.profile);self.expected=t['stance']
        data=self.robot.data;v=data.root_lin_vel_b
        gate=(v[:,0]/(.5*self.command[:,0]).clamp_min(.05)).clamp(0,1)
        scale=self.strength*gate
        height=data.body_pos_w[:,self.foot_ids,2]-self.env.scene.env_origins[:,None,2]-self.cfg.foot_radius
        q=data.joint_pos[:,self.joint_ids];default=data.default_joint_pos[:,self.joint_ids]
        limits=data.soft_joint_pos_limits[:,self.joint_ids]
        permutations=self.permutations
        contacts_variants=t['stance'][:,permutations]
        contact_scores=(contact[:,None,:]==contacts_variants).float().mean(dim=2)
        qref=default[:,None,:,:].expand(-1,len(permutations),-1,-1).clone()
        qref[:,:,:,1]+=t['thigh'][:,permutations];qref[:,:,:,2]+=t['calf'][:,permutations]
        lower,upper=limits[...,0],limits[...,1]
        if self.cfg.bounded_actions_enabled or self.cfg.guarded_enabled:
            margin=(upper-lower)*(self.cfg.guarded_target_margin if self.cfg.guarded_enabled else self.cfg.bounded_limit_margin)
            lower,upper=lower+margin,upper-margin
        qref=torch.clamp(qref,min=lower[:,None],max=upper[:,None])
        heights=t['foot_height'][:,permutations]
        joint_scores=torch.exp(-((q[:,None]-qref)/.4).square().mean(dim=(2,3)))
        swing_scores=torch.exp(-((height[:,None]-heights)/.04).square().mean(dim=2))
        choice=(2*contact_scores+.6*joint_scores+swing_scores).argmax(dim=1)
        rows=self.rows
        self.expected=contacts_variants[rows,choice];self.qref=qref[rows,choice];self.height_ref=heights[rows,choice]
        match=contact_scores[rows,choice]
        active=self.metric_active()
        if self.speed_diagnostics is not None:self.speed_diagnostics.observe(v[:,0],self.command[:,0],active)
        self.metric_used=active.clone()
        balance_active=(self.age*self.env.step_dt>=self.cfg.metric_warmup_seconds)
        self.pair.observe(q,force,contact,self.env.step_dt,balance_active)
        self.pair_costs=dict(joint=pair_joint_penalty(q,self.qref,self.cfg),**self.pair.costs)
        if self.cfg.pair_hip_weight>0:self.pair_costs['hip']=self.pair.hip_cost
        impulse=force.clamp_min(0)*self.env.step_dt
        self.cycle_impulse+=impulse;self.total_impulse+=impulse*active[:,None] if self.startup is not None else impulse
        self.fore_stance+=contact[:,:2].float()*self.env.step_dt
        self.fore_swing+=(~contact[:,:2]&(height[:,:2]>=self.cfg.min_fore_swing_height)).float()*self.env.step_dt
        self.qmin=torch.minimum(self.qmin,q[:,:,1:]);self.qmax=torch.maximum(self.qmax,q[:,:,1:])
        rpy=torch.stack(euler_xyz_from_quat(data.root_quat_w),dim=1)
        rpy=(rpy+math.pi)%(2*math.pi)-math.pi
        distance=(self.phase-self.phase.new_tensor(self.cfg.touchdown)[foot]).abs()
        distance=torch.minimum(distance,1-distance)
        order_score=torch.where(good,torch.exp(-(distance/.12)**2),torch.zeros_like(distance))
        quality=self.good_events/self.events.clamp_min(1)
        air_target=~self.expected.any(dim=1)
        self.components=dict(contact=scale*match,order=scale*order_score,
            swing=scale*swing_scores[rows,choice],joint=scale*joint_scores[rows,choice],
            motion=scale*self.last_rom,
            flight=scale*air_target.float()*(~contact.any(dim=1)).float(),
            fore_use=scale*self.last_fore_use,
            attitude=-(3*rpy[:,0].square()+(rpy[:,1]-self.strength*t['pitch']).square()))
        gait_mix=self.used_mix if self.startup is not None else torch.ones_like(self.phase)
        if self.cfg.relearn_enabled:gait_mix=gait_mix*self.relearn_gait
        if self.startup is not None:
            self.components['contact']=scale*((1-gait_mix)*contact.any(dim=1).float()+gait_mix*match)
            for key in ('order','swing','joint','motion','flight','fore_use'):self.components[key]*=gait_mix
            self.components['attitude']=-(3*rpy[:,0].square()+(rpy[:,1]-self.strength*gait_mix*t['pitch']).square())
        distance_points=self.leap.pending_reward
        if self.flight is not None:
            self.flight.reject_terminated(self.env.reset_terminated)
            distance_points=self.flight.points['distance']
            # Keep the forward-tracking objective available. Reduce gait-style
            # positives when support remains in the required shared flight phase.
            floor=self.cfg.flight_air_reward_floor
            adherence=1-gait_mix*air_target.float()*(1-floor)*contact.float().mean(dim=1)
            for key in ('contact','swing','joint','motion','fore_use'):self.components[key]*=adherence
            for key,value in self.flight.rates.items():
                sign=-1 if key in ('support','balance','impact') else 1
                self.components['flight_'+key]=sign*self.strength*value/self.env.step_dt
            for key in ('takeoff','landing'):
                self.components['flight_'+key]=self.strength*self.flight.points[key]/self.env.step_dt
                if self.cfg.relearn_enabled:self.components['flight_'+key]*=self.relearn_gait
            self.components['flight_clip']=-self.strength*balance_active.float()*self.action_clip
        self.leap_score=self.strength*distance_points*(~self.env.reset_terminated).float() if self.leap_enabled else torch.zeros_like(self.leap_score)
        if self.cfg.relearn_enabled:self.leap_score*=self.relearn_gait
        # RewardManager multiplies by step_dt: an event earns fixed points,
        # independent of the policy decimation, with one full score per cycle.
        self.components['leap_distance']=self.leap_score/self.env.step_dt
        if self.cfg.guarded_enabled:
            self.guard.observe_actual(data.joint_pos,self.env.step_dt,self.env.common_step_counter)
            front,hind,high=guarded_foot_height_cost(height,self.cfg);self.guard_height.copy_(height)
            self.components.update(guarded_projection=-self.guard.cost,guarded_actual=-self.guard.actual_cost,
                guarded_front_height=-front,guarded_hind_height=-hind)
            measured=dict(intervention=self.guard.intervention,actual=self.guard.actual_fraction,
                front_high=high[:,:2].mean(-1),hind_high=high[:,2:].mean(-1),front_cost=front,hind_cost=hind,
                tracking2=(data.joint_pos-self.guard.target).square().mean(-1))
            for key,value in measured.items():self.guard_sums[key]+=active.float()*value
            if self.cfg.relearn_enabled:self.relearn_request_sum+=active.float()*self.request_monitor.fraction
        self.leap_reward_sum+=self.leap_score;self.leap_reward_steps+=(self.leap_score>0).float()
        # Pair penalties do not disappear by slowing down; startup is excluded.
        for key,value in self.pair_costs.items():
            self.components['pair_'+key]=-self.strength*balance_active.float()*value*(gait_mix if key=='joint' else 1)
        if self.cfg.heading_motion_enabled:
            terms,self.posture_values=self.motion.observe(data.root_quat_w,data.root_pos_w,
                data.body_pos_w[:,self.foot_ids],contact,data.root_lin_vel_w,self.command[:,0],self.env.step_dt,
                angular_velocity=data.root_ang_vel_b)
            self.components.update(terms)
            self.components['attitude']=torch.zeros_like(self.phase)
            if self.cfg.stabilization_enabled:
                self.components['target_excess']=-self.used_mix*self.action_excess
                self.stability_sums['excess']+=active.float()*self.action_excess
                for pair,name in enumerate(('front','hind')):
                    both=contact[:,pair*2:pair*2+2].all(1)&active
                    width=self.posture_values['width'][:,pair]
                    good=(width>=self.cfg.stabilization_width_min)&(width<=self.cfg.stabilization_width_max)
                    self.stability_sums[name+'_support']+=both.float()
                    self.stability_sums[name+'_width_good']+=(both&good).float()
        if self.cfg.bounded_actions_enabled:
            self.components['bounded_near_limit']=-self.used_mix*self.bounded_near_cost
            self.components['bounded_target_rate']=-self.used_mix*self.bounded_rate_cost
            self.bounded_tracking_error.copy_((data.joint_pos-self.bounded_target).square().mean(-1).sqrt())
            computed=getattr(data,'computed_torque',None);applied=getattr(data,'applied_torque',None)
            if computed is not None and applied is not None:
                self.bounded_torque_fraction.copy_(((computed-applied).abs()>1e-4).float().mean(-1))
                self.bounded_sums['torque']+=active.float()*self.bounded_torque_fraction
                self.bounded_sums['torque_points']+=active.float()
            for key,value in dict(near=self.bounded_near_fraction,near_cost=self.bounded_near_cost,
                    rate_cost=self.bounded_rate_cost,tracking2=self.bounded_tracking_error.square()).items():
                self.bounded_sums[key]+=active.float()*value
        if self.cfg.posture_enabled and not self.cfg.heading_motion_enabled:
            terms,self.posture_values=posture_terms(data.root_quat_w,data.root_pos_w,
                data.body_pos_w[:,self.foot_ids],contact,self.cfg)
            # No speed, phase, warmup or gait-strength multiplier: retain body
            # alignment and foot placement in support, takeoff, air and landing.
            self.components.update(terms)
            p=self.posture_values
            measured=dict(points=torch.ones_like(self.phase),tilt2=p['tilt'].square(),
                upright_cost=p['upright_cost'],width_cost=p['width_cost'],center_cost=p['center_cost'],
                front_width=p['width'][:,0],hind_width=p['width'][:,1],
                front_width_error2=(p['width'][:,0]-self.cfg.front_stance_width).square(),
                hind_width_error2=(p['width'][:,1]-self.cfg.hind_stance_width).square(),
                front_center2=p['center'][:,0].square(),hind_center2=p['center'][:,1].square())
            for key,value in measured.items():self.posture_sums[key]+=value
        active=active.float()
        values=dict(points=torch.ones_like(active),vx2=(v[:,0]-self.command[:,0]).square(),vy2=v[:,1].square(),
                    yaw2=data.root_ang_vel_b[:,2].square(),roll2=rpy[:,0].square(),pitch2=rpy[:,1].square(),
                    match=match,clip=self.action_clip,command=self.command[:,0])
        if self.cfg.staged_enabled:
            heading=(rpy[:,2]-self.cfg.target_yaw+math.pi)%(2*math.pi)-math.pi
            values['heading2']=heading.square()
        values.update({'pair_'+key:value for key,value in self.pair_costs.items()})
        for key,value in values.items():self.sums[key]+=active*value

    def complete_episodes(self,ids):
        ids=torch.as_tensor(ids,device=self.env.device,dtype=torch.long)
        ids=ids[self.age[ids]>0]
        if not len(ids):return
        den=self.sums['points'].clamp_min(1)
        vx=torch.sqrt(self.sums['vx2']/den);mean_cmd=self.sums['command']/den
        contact=self.sums['match']/den;order=self.good_events/self.events.clamp_min(1)
        flight=self.flight_cycles/self.cycles.clamp_min(1);rom=self.rom_sum/self.cycles.clamp_min(1)
        attitude=torch.sqrt(torch.maximum(self.sums['roll2'],self.sums['pitch2'])/den)
        failed=self.env.reset_terminated
        tracking=(vx<=torch.maximum(mean_cmd*self.cfg.tracking_relative_rmse,mean_cmd.new_full(mean_cmd.shape,self.cfg.tracking_absolute_rmse)))
        tracking&=(torch.sqrt(self.sums['vy2']/den)<=.15)&(torch.sqrt(self.sums['yaw2']/den)<=.3)
        pattern=gait_pattern_success(contact,order,self.cycles,self.sequences,self.cfg)
        flight_ok=flight>=self.cfg.flight_cycle_fraction;motion=rom>=self.cfg.min_rom_fraction
        load=self.total_impulse/self.total_impulse.sum(dim=1,keepdim=True).clamp_min(.01)
        fore_rom=self.fore_rom_sum/self.cycles[:,None].clamp_min(1)
        fore_participation=self.fore_participation/self.cycles[:,None].clamp_min(1)
        front_load,front_motion,front_cycle=forelimb_success(load,fore_rom,fore_participation,self.cfg)
        stable=(~failed)&(self.age>=self.env.max_episode_length-1)&(attitude<=self.cfg.max_attitude_rms)&(self.sums['clip']/den<=self.cfg.max_action_clipping)
        guarded_columns={}
        if self.cfg.guarded_enabled:
            gs=self.guard_sums;enough=self.sums['points']*self.env.step_dt>=self.cfg.stabilization_min_metric_seconds
            limit_ok=(gs['actual']==0)&(gs['intervention']/den<=self.cfg.guarded_max_intervention)&enough
            feet_ok=(gs['front_high']/den<=self.cfg.guarded_max_high_fraction)&(gs['hind_high']/den<=self.cfg.guarded_max_high_fraction)&enough
            stable&=limit_ok&feet_ok
            guarded_columns=dict(guarded_limits=limit_ok,guarded_feet=feet_ok,
                **{'guarded_'+k:gs[k]/den for k in ('intervention','actual','front_high','hind_high','front_cost','hind_cost')},
                guarded_tracking_rmse=torch.sqrt(gs['tracking2']/den))
        bounded_columns={}
        if self.cfg.bounded_actions_enabled:
            fraction=self.bounded_sums['near']/den
            bounded_ok=(fraction<=self.cfg.bounded_max_limit_fraction)&(self.sums['points']*self.env.step_dt>=self.cfg.stabilization_min_metric_seconds)
            stable&=bounded_ok
            bounded_columns=dict(bounded_limits=bounded_ok,bounded_near_limit_fraction=fraction,
                bounded_near_limit_cost=self.bounded_sums['near_cost']/den,bounded_target_rate_cost=self.bounded_sums['rate_cost']/den,
                bounded_tracking_rmse=torch.sqrt(self.bounded_sums['tracking2']/den),
                bounded_torque_saturation=torch.where(self.bounded_sums['torque_points']>0,
                    self.bounded_sums['torque']/self.bounded_sums['torque_points'].clamp_min(1),-1.))
        mature=torch.full_like(stable,self.strength>=.95)
        if self.cfg.relearn_enabled:mature&=self.relearn_gait>=1-1e-6
        if self.startup is not None:mature&=self.startup.stage==3
        success=tracking&pattern&flight_ok&motion&front_load&front_motion&front_cycle&stable&mature
        stabilization_columns={}
        if self.cfg.stabilization_enabled:
            widths=[]
            for name in ('front','hind'):
                count=self.stability_sums[name+'_support'];fraction=self.stability_sums[name+'_width_good']/count.clamp_min(1)
                widths.append((count>=self.cfg.stabilization_min_support_samples)&(fraction>=self.cfg.stabilization_width_fraction))
                stabilization_columns['stabilization_'+name+'_support_samples']=count
                stabilization_columns['stabilization_'+name+'_width_fraction']=fraction
            pose=(torch.sqrt(self.sums['roll2']/den)<=self.cfg.stabilization_roll_limit)&(torch.sqrt(self.sums['pitch2']/den)<=self.cfg.stabilization_pitch_limit)
            finished=self.flight.totals['stable_landings']+self.flight.totals['aborted_landings']
            landing_fraction=self.flight.totals['stable_landings']/finished.clamp_min(1)
            landing=(finished>=self.cfg.stabilization_min_recoveries)&(landing_fraction>=self.cfg.stabilization_landing_fraction)
            distance=(self.leap.events>=self.cfg.stabilization_min_recoveries)&(self.leap.qualified_events/self.leap.events.clamp_min(1)>=self.cfg.stabilization_distance_fraction)
            enough=self.sums['points']*self.env.step_dt>=self.cfg.stabilization_min_metric_seconds
            tracking&=enough;stable&=enough
            success&=pose&widths[0]&widths[1]&landing&distance&enough
            stabilization_columns.update(stabilization_pose=pose,stabilization_width=widths[0]&widths[1],
                stabilization_landing=landing,stabilization_distance=distance,stabilization_measured=enough,
                stabilization_landing_fraction=landing_fraction,stabilization_target_speed=self.episode_speed,
                stabilization_leap_target=self.leap.distance_target,stabilization_excess_cost=self.stability_sums['excess']/den,
                stabilization_highest_confirmed_speed=self.episode_speed.new_full(self.episode_speed.shape,self.curriculum.highest_confirmed_speed or 0.))
        relearn_columns={}
        if self.cfg.relearn_enabled:
            request_fraction=self.relearn_request_sum/den
            success&=(request_fraction==0)&~self.request_monitor.failed
            base=tracking&(~failed)&(self.age>=self.env.max_episode_length-1)&enough&pose&feet_ok&front_load
            base&=(self.guard_sums['actual']==0)&~self.request_monitor.failed&(request_fraction<=self.relearn_allowance)
            shaping=(contact>=.5)&(self.cycles>=self.cfg.min_cycles)&motion&front_load&front_motion&front_cycle
            stage_ok=base&torch.where(self.relearn_gait==0,torch.ones_like(base),shaping)
            stage_ok&=torch.where(self.relearn_gait>=.5,pattern&(flight>=.3),torch.ones_like(base))
            recoveries=self.flight.totals['stable_landings']+self.flight.totals['aborted_landings']
            intermediate_landing=(recoveries>=self.cfg.stabilization_min_recoveries)&(landing_fraction>=.6)
            stage_ok&=torch.where(self.relearn_gait>=.75,(flight>=self.cfg.flight_cycle_fraction)&distance&intermediate_landing,torch.ones_like(base))
            stage_ok=torch.where(self.relearn_gait>=1-1e-6,success,stage_ok)
            relearn_columns=dict(relearn_level=self.relearn_index,relearn_gait=self.relearn_gait,
                relearn_stage_success=stage_ok,relearn_request_fraction=request_fraction,
                relearn_request_tolerance=self.relearn_tolerance,relearn_request_seconds=self.relearn_request_seconds,
                relearn_request_terminated=self.request_monitor.failed)
        eligible=self.age>0 if self.cfg.relearn_enabled else (self.age>0)&~(self.first&(self.age<self.env.max_episode_length))
        columns=dict(age=self.age,level=self.level,eligible=eligible,success=success,tracking=tracking,
                     pattern=pattern,flight=flight_ok,motion=motion,front_load=front_load,front_motion=front_motion,
                     front_cycle=front_cycle,stable=stable,mature=mature,
                     vx_rmse=vx,contact_accuracy=contact,order_accuracy=order,flight_fraction=flight,
                     rom_fraction=rom,fore_load_share=load[:,:2].sum(dim=1),fore_min_load_share=load[:,:2].amin(dim=1),
                     fore_rom_fraction=fore_rom.mean(dim=1),fore_min_rom_fraction=fore_rom.amin(dim=1),
                     fore_participation=fore_participation.amin(dim=1),cycles=self.cycles,sequences=self.sequences,attitude_rms=attitude)
        columns.update(action_clip_fraction=self.sums['clip']/den,terminated=failed,timed_out=self.env.reset_time_outs,
                       pair_valid_cycles=self.pair.valid_cycles,
                       **{'pair_'+key+'_cost':self.sums['pair_'+key]/den for key in self.pair_costs})
        columns.update(stabilization_columns)
        if self.speed_diagnostics is not None:columns.update(self.speed_diagnostics.diagnostics())
        columns.update(leap_events=self.leap.events,leap_qualified_events=self.leap.qualified_events,
                       leap_qualified_fraction=self.leap.qualified_events/self.leap.events.clamp_min(1),
                       leap_distance_mean=self.leap.distance_sum/self.leap.events.clamp_min(1),leap_distance_max=self.leap.maximum_distance,
                       leap_air_seconds_mean=self.leap.seconds_sum/self.leap.events.clamp_min(1),
                       leap_reward_points=self.cfg.leap_reward_weight*self.leap_reward_sum,
                       leap_reward_steps=self.leap_reward_steps,leap_physics_samples=self.leap.physics_samples)
        columns.update(bounded_columns);columns.update(guarded_columns)
        columns.update(relearn_columns)
        if self.flight is not None:columns.update(self.flight.diagnostics())
        if self.cfg.heading_motion_enabled:columns.update(self.motion.diagnostics())
        elif self.cfg.posture_enabled:
            p=self.posture_sums;count=p['points'].clamp_min(1)
            columns.update(posture_seconds=p['points']*self.env.step_dt,
                posture_tilt_rms=torch.sqrt(p['tilt2']/count),
                **{'posture_'+key:p[key]/count for key in ('upright_cost','width_cost','center_cost','front_width','hind_width')},
                **{'posture_'+key.replace('2','_rms'):torch.sqrt(p[key]/count)
                   for key in ('front_width_error2','hind_width_error2','front_center2','hind_center2')})
        if self.startup is not None:
            columns.update(startup_stage=self.startup.stage,startup_completed=self.startup.stage==3,
                startup_failed=self.startup.failed,
                startup_hold_seconds=self.startup.times[:,0],startup_walk_seconds=self.startup.times[:,1],
                startup_blend_seconds=self.startup.times[:,2],startup_gallop_seconds=self.startup.times[:,3],
                gallop_metric_seconds=self.sums['points']*self.env.step_dt)
        if self.cfg.staged_enabled:
            measured=self.sums['points']*self.env.step_dt>=self.cfg.staged_min_metric_seconds
            yaw_rms=torch.sqrt(self.sums['heading2']/den)
            precise=(vx<=self.cfg.staged_max_vx_rmse)&(yaw_rms<=self.cfg.staged_max_yaw_rms)
            success&=measured&precise
            strict_success=success.clone()
            if self.cfg.staged_core_first_enabled:
                requirements=self.profile.promotion_requirements()
                core=(tracking&pattern&flight_ok&motion&front_load&front_motion&front_cycle&mature
                    &(~failed)&(self.age>=self.env.max_episode_length-1)&measured&distance&limit_ok
                    &(attitude<=self.cfg.max_attitude_rms)&(self.sums['clip']/den<=self.cfg.max_action_clipping)
                    &(vx<=self.cfg.staged_max_vx_rmse))
                progressive_pose=(torch.sqrt(self.sums['roll2']/den)<=requirements['roll_limit'])&(torch.sqrt(self.sums['pitch2']/den)<=requirements['pitch_limit'])
                progressive_heading=yaw_rms<=requirements['yaw_limit']
                progressive_landing=(finished>=self.cfg.stabilization_min_recoveries)&(landing_fraction>=requirements['landing_fraction'])
                progressive_details=~requirements['details_required']|(widths[0]&widths[1]&feet_ok)
                progress_success=core&progressive_pose&progressive_heading&progressive_landing&progressive_details
                # Stay at the final task to refine style while core locomotion
                # remains viable; regress when core safety/landing deteriorates.
                regression_success=(core&(torch.sqrt(self.sums['roll2']/den)<=math.radians(10))
                    &(torch.sqrt(self.sums['pitch2']/den)<=math.radians(12))
                    &(yaw_rms<=math.radians(8))&(finished>=self.cfg.stabilization_min_recoveries)
                    &(landing_fraction>=.60))
                # Final certification always uses all original strict criteria.
                success=torch.where(requirements['details_required'],strict_success,progress_success)
                columns.update(staged_core=core,staged_pose=progressive_pose,staged_heading=progressive_heading,
                    staged_landing=progressive_landing,staged_details=progressive_details,staged_regression=regression_success,
                    staged_required_landing_fraction=requirements['landing_fraction'],
                    staged_required_yaw_rms=requirements['yaw_limit'],staged_details_required=requirements['details_required'])
            columns.update(success=success,staged_index=self.profile.index,staged_measured=measured,
                staged_tracking=precise,staged_yaw_rms=yaw_rms,staged_generation=self.level,
                staged_leap_target=self.leap.distance_target,
                staged_strict_success=strict_success,
                staged_final_success=strict_success&(self.profile.index==len(self.curriculum.plan)-1),
                staged_flight_min_seconds=self.profile.flight_min_seconds,eligible=self.age>0)
        matrix=torch.stack([v.float() for v in columns.values()],dim=1)[ids].detach().cpu().tolist()
        records=[]
        for idx,row in zip(ids.tolist(),matrix):
            record=dict(zip(columns,row));record.update(env=idx,step=int(self.env.common_step_counter))
            if record['age']<=0:continue
            if record['eligible'] and self.curriculum_enabled:
                criteria={k:bool(record[k]) for k in ('tracking','pattern','flight','motion','front_load','front_motion','front_cycle','stable','mature')}
                criteria.update({k:bool(record[k]) for k in ('stabilization_pose','stabilization_width','stabilization_landing','stabilization_distance','stabilization_measured') if k in record})
                if 'bounded_limits' in record:criteria['bounded_limits']=bool(record['bounded_limits'])
                for key in ('guarded_limits','guarded_feet'):
                    if key in record:criteria[key]=bool(record[key])
                progress=bool(record['success'])
                if self.cfg.relearn_enabled:
                    progress=bool(record['relearn_stage_success']);criteria['relearn_stage_success']=progress
                if self.cfg.relearn_enabled and self.cfg.relearn_recovery_enabled:
                    self.curriculum.record(progress,int(record['level']),criteria,env_id=idx)
                elif self.cfg.staged_enabled:
                    criteria.update({k:bool(record[k]) for k in ('staged_measured','staged_tracking')})
                    if self.cfg.staged_core_first_enabled:
                        criteria.update({k:bool(record[k]) for k in ('staged_core','staged_pose','staged_heading','staged_landing','staged_details','staged_regression')})
                    record['staged_cohort_accepted']=self.curriculum.record(progress,int(record['level']),criteria,
                        env_id=idx,step=int(self.env.common_step_counter))
                else:self.curriculum.record(progress,int(record['level']),criteria)
            records.append(record);self.last_episode=record
        if records and self.episode_path is not None:
            with self.episode_path.open('a') as f:
                for record in records:f.write(json.dumps(record)+'\n')

    def reset(self,ids):
        self.first[ids]&=self.age[ids]==0
        self.level[ids]=self.curriculum.level
        if self.profile is not None:self.profile.latch(ids,self.curriculum.index)
        if self.cfg.stabilization_enabled:self.episode_speed[ids]=self.curriculum.cap
        if self.cfg.relearn_enabled:self.latch_relearn(ids)
        for value in [self.phase,self.age,self.streak,self.events,self.good_events,self.sequences,self.cycles,
                      self.flight_cycles,self.air_run,self.air_max,self.rom_sum,self.last_rom,self.action_clip,*self.sums.values()]:value[ids]=0
        for value in (self.cycle_impulse,self.total_impulse,self.fore_stance,self.fore_swing,self.fore_participation,self.fore_rom_sum,self.last_fore_use):value[ids]=0
        self.last_order[ids]=-1;self.seeded[ids]=False;self.contacts[ids]=False
        q=self.robot.data.joint_pos[:,self.joint_ids][:,:,1:]
        self.qmin[ids]=q[ids];self.qmax[ids]=q[ids]
        self.pair.reset(ids,self.robot.data.joint_pos[:,self.joint_ids])
        if self.speed_diagnostics is not None:self.speed_diagnostics.reset(ids)
        for value in self.pair_costs.values():value[ids]=0
        self.leap.reset(ids)
        if self.startup is not None:self.startup.reset(ids,self.episode_speed[ids] if self.cfg.stabilization_enabled else None)
        if self.cfg.stabilization_enabled:
            self.leap.distance_target[ids]=stabilization_leap_target(self.episode_speed[ids],self.cfg)
            if self.profile is not None:self.leap.distance_target[ids]=self.curriculum.goal['leap']
            self.action_excess[ids]=0;self.joint_target_excess[ids]=0
            for value in self.stability_sums.values():value[ids]=0
        self.executed_actions[ids]=0.;self.cycle_metric_ok[ids]=False;self.metric_used[ids]=False
        if self.cfg.guarded_enabled:
            self.guard.reset(ids);self.guard_height[ids]=0
            for value in self.guard_sums.values():value[ids]=0
            if self.cfg.relearn_enabled:
                self.request_monitor.reset(ids);self.relearn_request_sum[ids]=0
        if self.cfg.bounded_actions_enabled:
            self.bounded_previous_valid[ids]=False
            for value in (self.bounded_target,self.bounded_previous_target,self.bounded_target_rate,self.bounded_near_cost,
                    self.bounded_near_fraction,self.bounded_rate_cost,self.bounded_tracking_error,*self.bounded_sums.values()):value[ids]=0
            self.bounded_target[ids]=self.robot.data.default_joint_pos[ids]
            self.bounded_torque_fraction[ids]=-1
        if self.flight is not None:self.flight.reset(ids)
        for value in (self.last_fore_ok,self.leap_score,self.leap_reward_sum,self.leap_reward_steps):value[ids]=0
        if self.cfg.heading_motion_enabled:self.motion.reset(ids)
        elif self.cfg.posture_enabled:
            for value in (*self.posture_sums.values(),*self.posture_values.values()):value[ids]=0

    def after_step(self):
        if not self.curriculum_enabled:return
        event=self.curriculum.decide(self.env.common_step_counter)
        if event is not None:
            term=self.env.command_manager.get_term('base_velocity')
            term.cfg.ranges.lin_vel_x=self.curriculum.bounds()

    def metrics(self):
        result=dict(**{'Gallop/speed_cap':self.curriculum.cap,'Gallop/gait_weight':self.strength,
                       'Gallop/success_rate':self.curriculum.last_rate or 0.},
                     **{'Gallop/last_'+k:v for k,v in self.last_episode.items() if k in ('vx_rmse','contact_accuracy','order_accuracy','flight_fraction','rom_fraction','action_clip_fraction','pair_joint_cost','pair_rom_cost','pair_load_cost','pair_stance_cost','pair_hip_cost','pair_valid_cycles') or k.startswith(('leap_','flight_','startup_','posture_','stabilization_','bounded_','guarded_','relearn_'))})
        if self.cfg.staged_enabled:
            result.update({'Staged/index':self.curriculum.index,'Staged/leap_target':self.curriculum.goal['leap'],
                'Staged/air_seconds':self.curriculum.goal['air_seconds'],'Staged/cohort_completed':len(self.curriculum.cohort_seen),
                'Staged/highest_confirmed_index':self.curriculum.highest_confirmed_stage if self.curriculum.highest_confirmed_stage is not None else -1})
        if self.speed_diagnostics is not None:
            result.update({'StagedAdapt/'+k:v for k,v in self.last_episode.items() if k.startswith('gallop_')})
        if self.cfg.relearn_enabled:
            result.update({'Relearn/level':self.curriculum.index,'Relearn/gait':self.curriculum.goal['gait'],
                'Relearn/request_tolerance':self.curriculum.goal['request_tolerance']})
        return result

    def snapshot(self):
        d=self.robot.data;rpy=torch.stack(euler_xyz_from_quat(d.root_quat_w),dim=1)
        rpy=(rpy+math.pi)%(2*math.pi)-math.pi
        tensors=dict(phase=self.phase,command=self.command,body_velocity=d.root_lin_vel_b,
                     angular_velocity=d.root_ang_vel_b,position=d.root_pos_w-self.env.scene.env_origins,rpy=rpy,
                     foot_force=self.sensor.data.net_forces_w[:,self.sensor_ids,2],contact=self.contacts,
                     target_contact=self.expected,joint_pos=d.joint_pos[:,self.joint_ids],joint_reference=self.qref,
                     foot_height=d.body_pos_w[:,self.foot_ids,2]-self.env.scene.env_origins[:,None,2]-self.cfg.foot_radius,
                     foot_height_reference=self.height_ref,rom_fraction=self.last_rom,action_clip=self.action_clip,
                     cycles=self.cycles,good_events=self.good_events,events=self.events)
        if self.startup is not None:
            tensors.update(startup_stage=self.used_stage,startup_gallop_mix=self.used_mix,
                startup_failed=self.startup.failed,
                startup_gallop_seconds=self.startup.times[:,3],gallop_metric_active=self.metric_used,
                executed_action=self.executed_actions)
        if self.profile is not None:
            tensors.update(staged_index=self.profile.index,staged_generation=self.level,
                staged_flight_min_seconds=self.profile.flight_min_seconds,
                staged_touchdown=self.profile.touchdown,staged_duty=self.profile.duty)
        if self.speed_diagnostics is not None:tensors.update(self.speed_diagnostics.diagnostics())
        tensors.update({'pair_'+key+'_cost':value for key,value in self.pair_costs.items()})
        tensors.update({'pair_'+key+'_asymmetry':value for key,value in self.pair.diagnostics.items()})
        if self.cfg.pair_hip_weight>0:tensors['pair_hip_asymmetry']=self.pair.hip_asymmetry
        if self.cfg.posture_enabled:
            tensors.update({'posture_'+key:value for key,value in self.posture_values.items()})
        if self.cfg.stabilization_enabled:
            tensors.update(stabilization_target_speed=self.episode_speed,stabilization_leap_target=self.leap.distance_target,
                stabilization_target_excess=self.joint_target_excess,stabilization_excess_cost=self.action_excess)
        if self.cfg.bounded_actions_enabled:
            tensors.update(bounded_target=self.bounded_target,bounded_target_rate=self.bounded_target_rate,
                bounded_near_limit_fraction=self.bounded_near_fraction,bounded_near_limit_cost=self.bounded_near_cost,
                bounded_target_rate_cost=self.bounded_rate_cost,bounded_tracking_error=self.bounded_tracking_error,
                bounded_torque_saturation=self.bounded_torque_fraction)
        if self.cfg.guarded_enabled:
            tensors.update(guarded_requested=self.guard.requested,guarded_target=self.guard.target,
                guarded_target_rate=self.guard.target_rate,guarded_projection_cost=self.guard.cost,
                guarded_intervention=self.guard.intervention,guarded_rate_intervention=self.guard.rate_intervention,
                guarded_actual_excess=self.guard.actual_excess,guarded_actual_fraction=self.guard.actual_fraction,
                guarded_violation_seconds=self.guard.violation_run,guarded_foot_height=self.guard_height)
        if self.cfg.relearn_enabled:
            tensors.update(relearn_level=self.relearn_index,relearn_gait=self.relearn_gait,
                relearn_request_tolerance=self.relearn_tolerance,relearn_request_seconds=self.relearn_request_seconds,
                relearn_request_peak=self.request_monitor.peak,relearn_request_fraction=self.request_monitor.fraction,
                relearn_request_run=self.request_monitor.seconds,relearn_request_failed=self.request_monitor.failed)
        if self.flight is not None:
            _,_,_,stage=flight_phase_targets(self.phase,self.period,self.profile or self.cfg)
            tensors.update(self.flight.diagnostics())
            tensors.update(flight_planned_stage=stage,flight_recovering=self.flight.recovering,
                           flight_recovery_elapsed=self.flight.recovery_elapsed,flight_pending_distance_score=self.flight.recovery_bank,
                           flight_step_takeoff_score=self.flight.points['takeoff'],flight_step_landing_score=self.flight.points['landing'],
                           flight_step_confirmed_distance_score=self.flight.points['distance'],world_vertical_velocity=d.root_lin_vel_w[:,2])
        tensors.update(leap_active=self.leap.active,leap_current_distance=self.leap.current_distance,
                       leap_air_seconds=self.leap.air_seconds,leap_landed=self.leap.landed,
                       leap_qualified_landed=self.leap.qualified_landed,leap_last_distance=self.leap.last_distance,
                       leap_last_seconds=self.leap.last_seconds,leap_takeoff_position=self.leap.takeoff,
                       leap_landing_position=self.leap.landing,leap_reward_score=self.leap_score,
                       leap_events=self.leap.events,leap_qualified_events=self.leap.qualified_events,
                       leap_distance_sum=self.leap.distance_sum,leap_air_seconds_sum=self.leap.seconds_sum,
                       leap_distance_max=self.leap.maximum_distance,leap_physics_samples=self.leap.physics_samples,
                       leap_reward_points_total=self.cfg.leap_reward_weight*self.leap_reward_sum)
        return {k:v.detach().cpu().numpy().copy() for k,v in tensors.items()}


class GallopEnv(ManagerBasedRLEnv):
    def __init__(self,*args,gallop_settings,gallop_emit=None,episode_path=None,training_targets=True,curriculum_enabled=True,walking_policy=None,**kwargs):
        self.gallop=None;self.terminal_snapshots={};self.capture_terminals=not curriculum_enabled
        self._leap_substeps_remaining=0;self._scene_update_without_leap=None
        super().__init__(*args,**kwargs)
        if gallop_settings.guarded_enabled:self.guard_original_limits=apply_guarded_sim_limits(self.scene['robot'],gallop_settings)
        self.gallop=GallopState(self,gallop_settings,gallop_emit,episode_path,training_targets,curriculum_enabled,walking_policy)
        if self.gallop.leap_enabled:
            self._scene_update_without_leap=self.scene.update
            self.scene.update=self._update_scene_for_leap

    def _update_scene_for_leap(self,dt):
        self._scene_update_without_leap(dt)
        if self._leap_substeps_remaining>0 and dt>0:
            self.gallop.observe_leap_physics(dt)
            self._leap_substeps_remaining-=1

    def _reset_idx(self,ids):
        if self.gallop is not None:
            self.gallop.complete_episodes(ids)
            if self.capture_terminals and (self.gallop.age[ids]>0).any():
                sample=self.gallop.snapshot()
                for i in ids.tolist():
                    if self.gallop.age[i]>0:self.terminal_snapshots[i]={k:v[i].copy() for k,v in sample.items()}
        super()._reset_idx(ids)
        if self.gallop is not None:self.gallop.reset(ids)

    def step(self,action):
        self.terminal_snapshots={};self.gallop.begin_step()
        self._leap_substeps_remaining=self.cfg.decimation if self.gallop.leap_enabled else 0
        try:result=super().step(action)
        finally:self._leap_substeps_remaining=0
        self.gallop.after_step()
        return result

    def close(self):
        if getattr(self,'_scene_update_without_leap',None) is not None:
            self.scene.update=self._scene_update_without_leap
            self._scene_update_without_leap=None
        super().close()
