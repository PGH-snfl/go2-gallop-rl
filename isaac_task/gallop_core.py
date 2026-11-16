"""Simulator-independent Gallop timing, success curriculum and policy migration."""
from dataclasses import asdict, dataclass
import json
import math

import torch


@dataclass(frozen=True)
class GallopSettings:
    start_speed: float = 1.0
    max_speed: float = 3.0
    reference_start_speed: float = 1.0
    reference_max_speed: float | None = None
    speed_band: float = .2
    speed_increment: float = .25
    promote_success: float = .8
    regress_success: float = .5
    pass_windows: int = 2
    min_episodes: int = 128
    decision_steps: int = 1000
    slow_period: float = .42
    fast_period: float = .32
    touchdown: tuple = (.25,.5,0.,.125)  # FL FR RL RR, right-lead transverse
    duty: tuple = (.45,.35,.35,.45)
    action_scale: float = .5
    thigh_amplitude_slow: float = .45
    thigh_amplitude_fast: float = .75
    calf_amplitude_slow: float = .45
    calf_amplitude_fast: float = .8
    swing_height_slow: float = .065
    swing_height_fast: float = .12
    pitch_amplitude: float = .08
    reward_ramp_iterations: int = 500
    initial_gait_weight: float = .1
    contact_on: float = 8.
    contact_off: float = 3.
    flight_min_seconds: float = .04
    flight_cycle_fraction: float = .6
    metric_warmup_seconds: float = 1.
    contact_accuracy: float = .6
    order_accuracy: float = .8
    require_touchdown_order: bool = False
    order_reward_weight: float = .1
    flexible_left_right: bool = True
    min_cycles: int = 8
    min_rom_fraction: float = .7
    min_fore_rom_fraction: float = .7
    min_each_fore_rom_fraction: float = .5
    min_fore_load_share: float = .25
    min_each_fore_load_share: float = .08
    min_hind_load_share: float = .25
    min_fore_participation: float = .7
    min_fore_swing_height: float = .035
    min_fore_stance_seconds: float = .06
    min_fore_swing_seconds: float = .04
    max_action_clipping: float = .1
    max_attitude_rms: float = .25
    tracking_relative_rmse: float = .25
    tracking_absolute_rmse: float = .15
    velocity_tracking_weight: float = 3.
    velocity_tracking_std: float = .5
    progress_speed_scale_fraction: float = .25
    min_height: float = .18
    max_height: float = .6
    foot_radius: float = .022
    pair_joint_weight: float = 0.
    pair_rom_weight: float = 0.
    pair_load_weight: float = 0.
    pair_stance_weight: float = 0.
    pair_joint_tolerance: float = .20
    pair_joint_scale: float = .40
    pair_rom_tolerance: float = .25
    pair_load_tolerance: float = .30
    pair_stance_tolerance: float = .15
    leap_reward_weight: float = 0.
    leap_min_distance: float = .12
    leap_require_forelimbs: bool = True
    leap_partial_credit: float = .25
    leap_progress_exponent: float = 1.
    flight_landing_distance_scaled: bool = False
    pair_hip_weight: float = 0.
    pair_hip_tolerance: float = .2
    flight_shaping_enabled: bool = False
    flight_prepare_seconds: float = .08
    flight_recovery_seconds: float = .08
    flight_takeoff_vz: float = .4
    flight_clearance: float = .025
    flight_landing_min_air_seconds: float = .02
    flight_max_tilt: float = .3
    flight_max_angular_speed: float = 2.
    flight_max_lateral_speed: float = .35
    flight_max_landing_vz: float = 1.2
    flight_max_force_bw: float = 3.
    flight_max_foot_slide: float = .5
    flight_air_reward_floor: float = .2
    flight_drive_weight: float = 2.
    flight_clearance_weight: float = 2.
    flight_support_weight: float = 8.
    flight_balance_weight: float = 1.5
    flight_takeoff_weight: float = .5
    flight_landing_weight: float = 1.
    flight_impact_weight: float = 1.
    flight_clip_weight: float = 1.
    startup_enabled: bool = False
    startup_settle_seconds: float = .5
    startup_walk_speed: float = 1.
    startup_walk_ramp_seconds: float = 1.5
    startup_walk_min_seconds: float = 3.
    startup_ready_seconds: float = .5
    startup_blend_seconds: float = 3.
    startup_max_tilt: float = .18
    startup_max_angular_speed: float = 1.5
    startup_tracking_error: float = .25
    startup_episode_seconds: float = 30.
    startup_pose_handover: bool = False
    startup_deadline_seconds: float = 12.
    posture_enabled: bool = False
    upright_weight: float = 6.
    upright_tolerance: float = .05
    upright_scale: float = .2
    stance_width_weight: float = 2.
    stance_center_weight: float = 1.
    front_stance_width: float = .30
    hind_stance_width: float = .30
    stance_width_tolerance: float = .025
    stance_width_scale: float = .06
    stance_center_tolerance: float = .015
    stance_center_scale: float = .04
    stance_swing_tolerance_multiplier: float = 1.5
    heading_motion_enabled: bool = False
    target_yaw: float = 0.
    yaw_penalty_weight: float = 2.
    yaw_error_scale: float = .35
    roll_pitch_quiet_weight: float = 2.
    roll_pitch_rate_scale: float = 1.
    forward_progress_weight: float = 2.
    progress_direction_scale: float = .15
    front_shoulder_width: float = .284
    hind_shoulder_width: float = .284
    stance_shoulder_ratio: float = .95
    stabilization_enabled: bool = False
    stabilization_min_leap_distance: float = .14
    stabilization_roll_deadband: float = math.radians(3)
    stabilization_pitch_deadband: float = math.radians(5)
    stabilization_roll_scale: float = math.radians(6)
    stabilization_pitch_scale: float = math.radians(8)
    stabilization_roll_limit: float = math.radians(5)
    stabilization_pitch_limit: float = math.radians(8)
    stabilization_air_width_weight: float = .3
    stabilization_yaw_rate_weight: float = 1.
    stabilization_yaw_rate_scale: float = .3
    stabilization_excess_weight: float = 2.
    stabilization_excess_scale: float = .35
    stabilization_calf_multiplier: float = .85
    stabilization_width_min: float = .24
    stabilization_width_max: float = .30
    stabilization_width_fraction: float = .8
    stabilization_min_support_samples: int = 10
    stabilization_min_recoveries: int = 10
    stabilization_landing_fraction: float = .8
    stabilization_distance_fraction: float = .6
    stabilization_min_metric_seconds: float = 5.
    bounded_actions_enabled: bool = False
    bounded_limit_margin: float = .05
    bounded_limit_weight: float = 1.
    bounded_target_rate_weight: float = .05
    bounded_target_rate_scale: float = 10.
    bounded_max_limit_fraction: float = .1
    bounded_latent_std: float = .1
    guarded_enabled: bool = False
    guarded_sim_limit_factor: float = .9
    guarded_target_margin: float = .05
    guarded_hip_rate: float = 6.
    guarded_leg_rate: float = 16.
    guarded_projection_scale: float = .25
    guarded_projection_weight: float = 4.
    guarded_actual_scale: float = .03
    guarded_actual_weight: float = 8.
    guarded_actual_tolerance: float = .01
    guarded_violation_seconds: float = .06
    guarded_max_intervention: float = .35
    guarded_front_height: float = .16
    guarded_hind_height: float = .10
    guarded_height_scale: float = .04
    guarded_height_weight: float = 3.
    guarded_max_high_fraction: float = .10
    relearn_enabled: bool = False
    relearn_gallop_start_speed: float = 1.5
    relearn_initial_request_tolerance: float = .12
    relearn_final_request_tolerance: float = 1e-6
    relearn_initial_request_seconds: float = .06
    relearn_final_request_seconds: float = .02
    relearn_recovery_enabled: bool = False
    relearn_noise_std_start: float = .12
    relearn_noise_std_end: float = .06
    staged_enabled: bool = False
    staged_max_vx_rmse: float = .25
    staged_max_yaw_rms: float = math.radians(3)
    staged_min_metric_seconds: float = 10.
    staged_min_dwell_seconds: float = 60.
    staged_regress_windows: int = 2
    staged_core_first_enabled: bool = False
    staged_speed_adaptation_enabled: bool = False
    staged_noise_target_std: float = .20
    staged_noise_ramp_iterations: int = 500
    managed_exploration_enabled: bool = False
    fixed_goal_enabled: bool = False
    edu_limits_enabled: bool = False
    edu_velocity_margin: float = .9
    edu_effort_margin: float = .9
    edu_velocity_weight: float = 6.
    edu_effort_weight: float = 1.
    edu_saturation_weight: float = 2.
    edu_violation_seconds: float = .02
    edu_calf_target_rate: float = 12.
    velocity_brake_enabled: bool = False
    velocity_brake_margin: float = .8
    velocity_brake_gain: float = 2.
    velocity_brake_stiffness: float = 25.
    velocity_brake_damping: float = .5
    edu_velocity_peak_weight: float = 0.
    edu_terminate_on_exceedance: bool = False
    flight_height_compensation: bool = False

    def __post_init__(self):
        if type(self.velocity_brake_enabled) is not bool or type(self.edu_terminate_on_exceedance) is not bool:
            raise ValueError('Velocity safety switches must be boolean')
        if self.velocity_brake_enabled:
            if not self.edu_limits_enabled or not self.guarded_enabled:raise ValueError('Velocity braking requires guarded EDU monitoring')
            if not 0<self.velocity_brake_margin<1:raise ValueError('Invalid velocity braking margin')
            for key in ('velocity_brake_gain','velocity_brake_stiffness','velocity_brake_damping'):
                if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('Invalid braking option: '+key)
        if not math.isfinite(self.edu_velocity_peak_weight) or self.edu_velocity_peak_weight<0:raise ValueError('Invalid peak velocity cost weight')
        if self.edu_terminate_on_exceedance and not self.edu_limits_enabled:raise ValueError('Immediate velocity termination requires EDU monitoring')
        for key in ('tracking_relative_rmse','tracking_absolute_rmse','velocity_tracking_std','progress_speed_scale_fraction'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('Invalid tracking option: '+key)
        if not math.isfinite(self.velocity_tracking_weight) or self.velocity_tracking_weight<0:raise ValueError('Invalid velocity tracking weight')
        if not math.isfinite(self.leap_progress_exponent) or self.leap_progress_exponent<1:raise ValueError('Invalid leap progress exponent')
        if type(self.flight_landing_distance_scaled) is not bool:raise ValueError('Landing distance scaling must be boolean')
        if type(self.fixed_goal_enabled) is not bool or type(self.managed_exploration_enabled) is not bool or type(self.edu_limits_enabled) is not bool or type(self.flight_height_compensation) is not bool:
            raise ValueError('New fixed-goal options must be booleans')
        if self.fixed_goal_enabled and (self.staged_enabled or self.relearn_enabled or self.start_speed!=self.max_speed or self.speed_band!=0 or self.stabilization_min_leap_distance!=self.leap_min_distance):
            raise ValueError('Fixed goal excludes promotion/regression and requires one speed/distance')
        if self.edu_limits_enabled:
            if not self.guarded_enabled or self.leap_reward_weight<=0:raise ValueError('EDU monitoring requires guarded physics sampling')
            if not 0<self.edu_velocity_margin<1 or not 0<self.edu_effort_margin<1:raise ValueError('Invalid EDU margin')
            for key in ('edu_velocity_weight','edu_effort_weight','edu_saturation_weight','edu_violation_seconds','edu_calf_target_rate'):
                if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('Invalid EDU option: '+key)
            if self.edu_calf_target_rate>15.7*self.edu_velocity_margin:raise ValueError('Calf target rate must retain velocity margin')
        if self.flight_height_compensation and not self.flight_shaping_enabled:raise ValueError('Height compensation requires flight tracking')
        if type(self.staged_speed_adaptation_enabled) is not bool or (self.staged_speed_adaptation_enabled and not self.staged_core_first_enabled):
            raise ValueError('Speed adaptation requires core-first staged promotion')
        if not math.isfinite(self.staged_noise_target_std) or self.staged_noise_target_std<=0 or type(self.staged_noise_ramp_iterations) is not int or self.staged_noise_ramp_iterations<1:
            raise ValueError('Invalid managed exploration schedule')
        if type(self.staged_core_first_enabled) is not bool or (self.staged_core_first_enabled and not self.staged_enabled):
            raise ValueError('Core-first promotion requires the staged controller and a boolean flag')
        if self.staged_enabled:
            if not (self.guarded_enabled and self.startup_pose_handover and self.heading_motion_enabled) or self.relearn_enabled or self.bounded_actions_enabled:
                raise ValueError('Staged Gallop requires guarded pose handover without relearning/bounded migration')
            if (self.start_speed,self.max_speed,self.stabilization_min_leap_distance,self.leap_min_distance)!=(2.5,3.,.18,.8):
                raise ValueError('Staged Gallop plan requires 2.5m/s/18cm through 3m/s/80cm')
            if not all(math.isfinite(v) for v in (self.staged_max_vx_rmse,self.staged_max_yaw_rms,self.staged_min_metric_seconds,self.staged_min_dwell_seconds)) or min(self.staged_max_vx_rmse,self.staged_max_yaw_rms,self.staged_min_metric_seconds)<=0 or self.staged_min_dwell_seconds<0 or type(self.staged_regress_windows) is not int or self.staged_regress_windows<1:
                raise ValueError('Invalid staged evaluation or dwell settings')
            if (self.slow_period,self.fast_period,self.reference_start_speed,self.reference_max_speed)!=(.42,.38,1.,3.) or tuple(self.touchdown)!=(.25,.5,0.,.125) or tuple(self.duty)!=(.45,.30,.35,.45):
                raise ValueError('Staged plan v1 requires the original reference anchors/template')
        if self.relearn_enabled:
            if not self.guarded_enabled or not self.startup_enabled or not self.heading_motion_enabled:
                raise ValueError('Relearning requires guarded heading startup')
            if not self.start_speed<self.relearn_gallop_start_speed<self.max_speed:
                raise ValueError('Relearning needs walking, initial Gallop and final speed stages')
            for key,value in asdict(self).items():
                if key.startswith('relearn_') and key not in ('relearn_enabled','relearn_recovery_enabled') and (isinstance(value,bool) or not math.isfinite(value) or value<=0):
                    raise ValueError('Invalid relearning setting: '+key)
            if self.relearn_noise_std_end>self.relearn_noise_std_start:
                raise ValueError('Relearning exploration must not increase with stricter constraints')
            if self.relearn_final_request_tolerance>self.relearn_initial_request_tolerance or self.relearn_final_request_seconds>self.relearn_initial_request_seconds:
                raise ValueError('Request termination must tighten during relearning')
        if self.relearn_recovery_enabled and not self.relearn_enabled:
            raise ValueError('Recovery requires the relearning controller')
        if self.relearn_recovery_enabled and (self.relearn_initial_request_tolerance<.06 or self.relearn_initial_request_seconds<.04 or self.relearn_final_request_tolerance>.002 or self.relearn_final_request_seconds>.04):
            raise ValueError('Recovery refinement requires initial >= .06rad/.04s and final <= .002rad/.04s')
        if self.guarded_enabled:
            if self.bounded_actions_enabled or not self.stabilization_enabled:raise ValueError('Guarded control uses legacy delta actions and stabilization')
            for key,value in asdict(self).items():
                if key.startswith('guarded_') and key!='guarded_enabled' and (not math.isfinite(value) or value<=0):raise ValueError('Invalid guarded setting: '+key)
            if self.start_speed>self.max_speed or self.speed_band!=0:raise ValueError('Guarded profile requires a nondecreasing speed curriculum and zero speed band')
            if not 0<self.guarded_sim_limit_factor<=1 or self.guarded_target_margin>=.5:raise ValueError('Invalid guarded joint range')
            if self.guarded_max_intervention>1 or self.guarded_max_high_fraction>1:raise ValueError('Invalid guarded fraction')
            if max(self.guarded_hip_rate,self.guarded_leg_rate)>30:raise ValueError('Target rate exceeds saved motor no-load speed')
            if max(self.swing_height_slow,self.swing_height_fast)>min(self.guarded_front_height,self.guarded_hind_height):raise ValueError('Swing reference exceeds foot height cap')
        if self.bounded_actions_enabled:
            if not self.stabilization_enabled:raise ValueError('Bounded actions require the stabilization profile')
            for key in ('bounded_limit_margin','bounded_limit_weight','bounded_target_rate_weight',
                        'bounded_target_rate_scale','bounded_max_limit_fraction','bounded_latent_std'):
                if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError('Invalid bounded action setting: '+key)
            if self.bounded_limit_margin>=.5 or self.bounded_max_limit_fraction>1:raise ValueError('Invalid bounded limit fractions')
        if not (0 <= self.speed_band < self.start_speed <= self.max_speed):
            raise ValueError('Need 0 <= speed_band < start_speed <= max_speed')
        for key,value in asdict(self).items():
            if key.startswith('stabilization_') and key!='stabilization_enabled':
                if not (math.isfinite(value) and value>0):raise ValueError('Stabilization scales/counts must be finite and positive')
        for key in ('stabilization_air_width_weight','stabilization_calf_multiplier','stabilization_width_fraction',
                    'stabilization_landing_fraction','stabilization_distance_fraction'):
            if getattr(self,key)>1:raise ValueError('Stabilization fractions must not exceed one')
        if self.stabilization_width_min>=self.stabilization_width_max:raise ValueError('Invalid stabilization support width range')
        if self.stabilization_enabled:
            if not self.heading_motion_enabled or not self.startup_enabled or not self.flight_shaping_enabled or self.speed_band!=0 or self.start_speed>self.max_speed or (self.start_speed==self.max_speed and not self.guarded_enabled):
                raise ValueError('Stabilization needs heading startup and a fixed command per curriculum stage')
            if self.stabilization_min_leap_distance>self.leap_min_distance:raise ValueError('Initial leap target exceeds final target')
            if not all(self.stabilization_width_min<getattr(self,k+'_stance_width')<self.stabilization_width_max for k in ('front','hind')):
                raise ValueError('Foot width targets must lie inside stabilization support bounds')
        for key in ('upright_weight','stance_width_weight','stance_center_weight','upright_tolerance',
                    'stance_width_tolerance','stance_center_tolerance'):
            if not (math.isfinite(getattr(self,key)) and getattr(self,key)>=0):
                raise ValueError('Posture weights and tolerances must be finite and nonnegative')
        for key in ('upright_scale','stance_width_scale','stance_center_scale','front_stance_width','hind_stance_width'):
            if not (math.isfinite(getattr(self,key)) and getattr(self,key)>0):
                raise ValueError('Posture scales and widths must be finite and positive')
        if not (math.isfinite(self.stance_swing_tolerance_multiplier) and self.stance_swing_tolerance_multiplier>=1):
            raise ValueError('Swing tolerance multiplier must be at least one')
        if self.upright_tolerance>=math.pi/2 or self.stance_width_tolerance*self.stance_swing_tolerance_multiplier>=min(self.front_stance_width,self.hind_stance_width):
            raise ValueError('Posture tolerances must exclude sideways bodies and crossed feet')
        if self.posture_enabled and self.pitch_amplitude!=0:
            raise ValueError('Upright posture requires zero pitch reference amplitude')
        for key in ('yaw_penalty_weight','roll_pitch_quiet_weight','forward_progress_weight'):
            if not (math.isfinite(getattr(self,key)) and getattr(self,key)>=0):raise ValueError('Invalid heading/motion reward weight')
        for key in ('yaw_error_scale','roll_pitch_rate_scale','progress_direction_scale','front_shoulder_width','hind_shoulder_width'):
            if not (math.isfinite(getattr(self,key)) and getattr(self,key)>0):raise ValueError('Invalid heading/motion scale')
        if not math.isfinite(self.target_yaw) or not 0<self.stance_shoulder_ratio<1:raise ValueError('Invalid target heading or shoulder ratio')
        if self.heading_motion_enabled:
            if not self.posture_enabled or self.upright_weight!=0 or self.stance_center_weight!=0:
                raise ValueError('Heading/motion replaces upright and foot-center rewards')
            for which in ('front','hind'):
                if not math.isclose(getattr(self,which+'_stance_width'),getattr(self,which+'_shoulder_width')*self.stance_shoulder_ratio,abs_tol=1e-8):
                    raise ValueError('Foot targets must match their independent shoulder widths')
        for key,value in asdict(self).items():
            if key.startswith('startup_') and key not in ('startup_enabled','startup_pose_handover') and not (math.isfinite(value) and value>0):
                raise ValueError('Startup durations and limits must be finite and positive')
        if self.startup_enabled:
            if self.startup_pose_handover and not (self.startup_settle_seconds+self.startup_walk_min_seconds+self.startup_blend_seconds < self.startup_deadline_seconds < self.startup_episode_seconds):
                raise ValueError('Pose handover deadline must leave time for startup and full student measurement')
            if not (self.speed_band==0 and (self.start_speed==self.max_speed or self.stabilization_enabled) and self.startup_walk_speed<self.start_speed):
                raise ValueError('Startup requires a fixed final speed above walking speed')
            if self.startup_walk_min_seconds<self.startup_walk_ramp_seconds+self.startup_ready_seconds:
                raise ValueError('Walking must include ramp and stable tracking time')
            if self.startup_episode_seconds<self.startup_settle_seconds+self.startup_walk_min_seconds+self.startup_blend_seconds+5:
                raise ValueError('Episode leaves insufficient full-gallop time')
        if not (math.isfinite(self.leap_reward_weight) and self.leap_reward_weight>=0 and math.isfinite(self.leap_min_distance) and self.leap_min_distance>0):
            raise ValueError('Leap reward must be nonnegative and its distance target positive')
        if not (0 < self.reference_start_speed < self.max_speed):
            raise ValueError('Reference speed anchors must be positive and increasing')
        if self.reference_max_speed is not None and not (math.isfinite(self.reference_max_speed) and self.reference_max_speed>self.reference_start_speed):
            raise ValueError('Invalid upper reference speed anchor')
        if not (0<=self.leap_partial_credit<=1 and 0<=self.flight_air_reward_floor<=1 and 0<=self.pair_hip_tolerance<1):
            raise ValueError('Invalid flight credit, reward floor or hip tolerance')
        for key,value in asdict(self).items():
            if (key.startswith('flight_') and key.endswith('_weight')) or key=='pair_hip_weight':
                if not (math.isfinite(value) and value>=0):raise ValueError('Flight weights must be finite and nonnegative')
        for key in ('flight_prepare_seconds','flight_recovery_seconds','flight_takeoff_vz','flight_clearance',
                    'flight_landing_min_air_seconds','flight_max_tilt','flight_max_angular_speed',
                    'flight_max_lateral_speed','flight_max_landing_vz','flight_max_force_bw','flight_max_foot_slide'):
            if not (math.isfinite(getattr(self,key)) and getattr(self,key)>0):raise ValueError('Flight scales must be finite and positive')
        if self.flight_shaping_enabled and (self.leap_reward_weight<=0 or self.flight_landing_min_air_seconds>self.flight_min_seconds):
            raise ValueError('Flight shaping requires leap rewards and a valid landing minimum air time')
        if any(not math.isfinite(x) or x<0 for x in (self.pair_joint_weight,self.pair_rom_weight,self.pair_load_weight,self.pair_stance_weight)):
            raise ValueError('Pair penalty weights must be finite and nonnegative')
        if not (0<=self.pair_joint_tolerance and self.pair_joint_scale>0):
            raise ValueError('Invalid pair joint deadband/scale')
        if not all(0<=x<1 for x in (self.pair_rom_tolerance,self.pair_load_tolerance,self.pair_stance_tolerance)):
            raise ValueError('Invalid pair imbalance tolerances')
        if not (0 <= self.regress_success < self.promote_success <= 1):
            raise ValueError('Invalid success thresholds')
        if min(self.pass_windows,self.min_episodes,self.decision_steps,self.reward_ramp_iterations)<1:
            raise ValueError('Positive curriculum counts required')
        if not (self.speed_increment>0 and self.slow_period>0 and self.fast_period>0 and self.action_scale>0):
            raise ValueError('Positive increments, periods and action scale required')
        if len(self.duty)!=4 or len(self.touchdown)!=4 or not all(0<d<1 for d in self.duty):
            raise ValueError('Four valid foot duty fractions required')
        if any(abs(self.duty[i]-self.duty[i+1])+self.pair_stance_tolerance>=1 for i in (0,2)):
            raise ValueError('Pair stance tolerance leaves no imbalance constraint')
        if not (self.touchdown[2]==0 < self.touchdown[3] < self.touchdown[0] < self.touchdown[1] < 1):
            raise ValueError('Touchdowns must follow RL, RR, FL, FR')
        if not (0 <= self.contact_off < self.contact_on):raise ValueError('Invalid contact hysteresis')
        if not (0 < self.flight_cycle_fraction <= 1 and 0 <= self.initial_gait_weight <= 1):
            raise ValueError('Invalid flight/gait fractions')
        gap=1-max(t+d for t,d in zip(self.touchdown,self.duty))
        if gap*min(self.fast_period,self.slow_period)<self.flight_min_seconds:
            raise ValueError('Template has insufficient shared airborne time at its fastest period')
        if self.flight_shaping_enabled and self.flight_prepare_seconds+self.flight_recovery_seconds >= (1-gap)*min(self.fast_period,self.slow_period):
            raise ValueError('Preparation and recovery windows overlap')

    @classmethod
    def load(cls,path):return cls(**json.loads(path.read_text()))

    def to_dict(self):return asdict(self)


def smoothstep(value):
    x=value.clamp(0,1)
    return x*x*(3-2*x)


class WalkingToGallop:
    """Per-environment stationary hold, walking, smooth handover and Gallop.

    Walking is performed by a frozen forward policy. Handover progress pauses
    on poor pose/velocity tracking; resets always restart stationary.
    """
    def __init__(self,like,settings):
        self.cfg=settings
        self.stage=torch.zeros_like(like,dtype=torch.long)
        self.stage_seconds=torch.zeros_like(like);self.ready_seconds=torch.zeros_like(like)
        self.blend_seconds=torch.zeros_like(like)
        self.times=torch.zeros(len(like),4,device=like.device,dtype=like.dtype)
        self.goal_speed=torch.full_like(like,settings.start_speed)
        self.failed=torch.zeros_like(like,dtype=torch.bool)

    @property
    def mix(self):
        blend=smoothstep(self.blend_seconds/self.cfg.startup_blend_seconds)
        return torch.where(self.stage==3,1.,torch.where(self.stage==2,blend,0.))

    @property
    def command(self):
        s=self.cfg
        walk=s.startup_walk_speed*smoothstep(self.stage_seconds/s.startup_walk_ramp_seconds)
        progress=smoothstep(self.stage_seconds/s.startup_blend_seconds) if s.startup_pose_handover else self.mix
        transition=s.startup_walk_speed+(self.goal_speed-s.startup_walk_speed)*torch.where(self.stage==3,1.,progress)
        return torch.where(self.stage==0,0.,torch.where(self.stage==1,walk,transition))

    def advance(self,dt,velocity,angular_velocity,rpy,height,active):
        s=self.cfg;old=self.stage.clone()
        self.times.scatter_add_(1,old[:,None],(active.float()*dt)[:,None])
        self.stage_seconds+=active.float()*dt
        pose=(rpy[:,:2].abs()<=s.startup_max_tilt).all(1)&(angular_velocity.norm(dim=1)<=s.startup_max_angular_speed)
        pose&=(height>=s.min_height)&(height<=s.max_height)&(velocity[:,1].abs()<=.15)
        stationary=pose&(velocity[:,:2].norm(dim=1)<=.15)
        tracking=pose&((velocity[:,0]-self.command).abs()<=s.startup_tracking_error)
        ready=torch.where(old==0,stationary,tracking)&active
        self.ready_seconds=torch.where(ready,self.ready_seconds+dt,0.)
        blend_ready=pose if s.startup_pose_handover else tracking
        self.blend_seconds+=((old==2)&blend_ready&active).float()*dt
        leave_hold=(old==0)&(self.stage_seconds>=s.startup_settle_seconds)&(self.ready_seconds>=s.startup_ready_seconds)
        leave_walk=(old==1)&(self.stage_seconds>=s.startup_walk_min_seconds)&(self.ready_seconds>=s.startup_ready_seconds)
        leave_blend=(old==2)&(self.blend_seconds>=s.startup_blend_seconds-1e-6)&blend_ready
        changed=(leave_hold|leave_walk|leave_blend)&active
        self.stage+=changed.long()
        self.stage_seconds=torch.where(changed,0.,self.stage_seconds)
        self.ready_seconds=torch.where(changed,0.,self.ready_seconds)
        if s.startup_pose_handover:
            self.failed|=(self.times.sum(1)>=s.startup_deadline_seconds-1e-6)&(self.stage<3)&active
        return changed

    def reset(self,ids,target_speed=None):
        for tensor in (self.stage,self.stage_seconds,self.ready_seconds,self.blend_seconds,self.times):tensor[ids]=0
        self.goal_speed[ids]=self.cfg.start_speed if target_speed is None else target_speed
        self.failed[ids]=False


class FrozenWalkingPolicy(torch.nn.Module):
    """Load the verified 45-observation ELU forward actor; never optimized."""
    def __init__(self,checkpoint,device,old_action_scale=.25,new_action_scale=.5):
        super().__init__()
        state=torch.load(checkpoint,map_location=device,weights_only=True)['model_state_dict']
        weights=sorted((k for k in state if k.startswith('actor.') and k.endswith('.weight')),key=lambda k:int(k.split('.')[1]))
        if [int(k.split('.')[1]) for k in weights]!=[0,2,4,6] or state[weights[0]].shape[1]!=45 or state[weights[-1]].shape[0]!=12:
            raise ValueError('Expected the verified 45/12 forward actor with three ELU hidden layers')
        layers=[]
        for i,key in enumerate(weights):
            value=state[key];layer=torch.nn.Linear(value.shape[1],value.shape[0],device=device)
            layer.weight.data.copy_(value);layer.bias.data.copy_(state[key.replace('.weight','.bias')])
            layers.append(layer)
            if i<len(weights)-1:layers.append(torch.nn.ELU())
        self.actor=torch.nn.Sequential(*layers).eval().requires_grad_(False)
        self.old_action_scale=old_action_scale;self.new_action_scale=new_action_scale

    @torch.no_grad()
    def forward(self,observations):
        obs=observations[:,:45].clone()
        obs[:,33:45]*=self.new_action_scale/self.old_action_scale
        return self.actor(obs)*(self.old_action_scale/self.new_action_scale)


def blend_startup_actions(student,walking,stage,mix):
    blended=walking*(1-mix[:,None])+student*mix[:,None]
    return torch.where((stage==0)[:,None],torch.zeros_like(blended),blended)


def stabilization_leap_target(speed,settings):
    if not settings.stabilization_enabled:return torch.full_like(speed,settings.leap_min_distance)
    if settings.start_speed==settings.max_speed:return torch.full_like(speed,settings.leap_min_distance)
    ratio=((speed-settings.start_speed)/(settings.max_speed-settings.start_speed)).clamp(0,1)
    result=settings.stabilization_min_leap_distance+ratio*(settings.leap_min_distance-settings.stabilization_min_leap_distance)
    result=torch.where(speed<=settings.start_speed+1e-6,torch.full_like(speed,settings.stabilization_min_leap_distance),result)
    return torch.where(speed>=settings.max_speed-1e-6,torch.full_like(speed,settings.leap_min_distance),result)


def stabilized_student_actions(actions,calf_ids,mix,settings):
    result=actions.clone()
    multiplier=1-(1-settings.stabilization_calf_multiplier)*mix
    result[:,calf_ids]*=multiplier[:,None]
    return result


def target_excess_penalty(proposed,clipped,calf_ids,settings):
    normalized=(proposed-clipped).abs()/settings.stabilization_excess_scale
    huber=torch.where(normalized<1,.5*normalized.square(),normalized-.5)
    weights=torch.ones(proposed.shape[1],device=proposed.device,dtype=proposed.dtype);weights[calf_ids]=2
    return (huber*weights).mean(dim=1)


def bounded_joint_targets(latent,limits):
    """A fixed tanh transform of PPO latent actions into each joint's soft range.

    PPO stores/log-probs the original Gaussian latent, never this target.
    No second calf scaling: imitation targets already include the old scaling.
    """
    lower,upper=limits.unbind(-1)
    return (lower+upper)*.5+(upper-lower)*.5*torch.tanh(latent)


def bounded_startup_targets(latent,walking,default,limits,stage,mix,settings):
    lower,upper=limits.unbind(-1)
    student=bounded_joint_targets(latent,limits)
    teacher=torch.clamp(default+settings.action_scale*walking,min=lower,max=upper)
    target=teacher*(1-mix[:,None])+student*mix[:,None]
    return torch.where((stage==0)[:,None],default,target)


def bounded_limit_cost(target,limits,calf_ids,margin):
    lower,upper=limits.unbind(-1)
    distance=torch.minimum(target-lower,upper-target)/(upper-lower)
    penetration=((margin-distance).clamp_min(0)/margin).square()
    weights=torch.ones(target.shape[-1],device=target.device,dtype=target.dtype);weights[calf_ids]=2
    return (penetration*weights).mean(-1),(distance<margin).float().mean(-1)


def guarded_physics_limits(original,factor):
    center=original.mean(-1);radius=(original[...,1]-original[...,0])*.5*factor
    return torch.stack((center-radius,center+radius),dim=-1)


class GuardedJointController:
    """Delta-action safety filter with explicit pre/post intervention costs.

    This constrains targets. Actual joints use simulator position constraints
    too, and residual solver violations are measured rather than teleported.
    """
    def __init__(self,default,limits,hip_ids,calf_ids,settings):
        self.cfg=settings;self.default=default.clone();self.limits=limits.clone();self.calf_ids=calf_ids
        span=limits[...,1]-limits[...,0]
        self.lower=limits[...,0]+settings.guarded_target_margin*span
        self.upper=limits[...,1]-settings.guarded_target_margin*span
        if ((default<self.lower)|(default>self.upper)).any():raise ValueError('Default pose outside guarded operating range')
        self.rate=default.new_full((default.shape[-1],),settings.guarded_leg_rate);self.rate[hip_ids]=settings.guarded_hip_rate
        if settings.edu_limits_enabled:self.rate[calf_ids]=settings.edu_calf_target_rate
        self.previous=default.clone();self.requested=default.clone();self.target=default.clone();self.target_rate=torch.zeros_like(default)
        self.cost=default.new_zeros(len(default));self.intervention=self.cost.clone();self.rate_intervention=self.cost.clone()
        self.actual_excess=torch.zeros_like(default);self.actual_cost=self.cost.clone();self.violation_run=self.cost.clone()
        self.actual_fraction=self.cost.clone();self.actual_step=None
        self.velocity_limit=default.new_full((default.shape[-1],),30.1);self.velocity_limit[calf_ids]=15.7
        self.brake_active=torch.zeros_like(default,dtype=torch.bool);self.brake_infeasible=torch.zeros_like(default,dtype=torch.bool);self.brake_intervention=self.cost.clone()
        self.brake_position=default.clone();self.brake_velocity=torch.zeros_like(default);self.brake_bound=default.clone()

    def process(self,requested,dt,actual_position=None,actual_velocity=None):
        lower=torch.maximum(self.lower,self.previous-self.rate*dt)
        upper=torch.minimum(self.upper,self.previous+self.rate*dt)
        target=torch.clamp(requested,min=lower,max=upper)
        if self.cfg.velocity_brake_enabled:
            q,dq=actual_position,actual_velocity
            if q is None or dq is None or q.shape!=target.shape or dq.shape!=target.shape:
                raise ValueError('Velocity braking requires measured joint position/velocity')
            if not torch.isfinite(q).all() or not torch.isfinite(dq).all():raise ValueError('Nonfinite braking state')
            soft=self.velocity_limit*self.cfg.velocity_brake_margin
            positive=dq>soft;negative=dq<-soft;active=positive|negative
            # For tau=kp*(target-q)-kd*dq, request no accelerating torque
            # above the margin, then progressively more opposing torque.
            desired=-torch.sign(dq)*self.cfg.velocity_brake_gain*(dq.abs()-soft).clamp_min(0)
            bound=q+(self.cfg.velocity_brake_damping*dq+desired)/self.cfg.velocity_brake_stiffness
            adjusted=torch.where(positive,torch.minimum(target,bound),torch.where(negative,torch.maximum(target,bound),target))
            # Preserve angle/rate feasibility even if the braking torque bound
            # cannot be attained this control step. Log that conflict explicitly.
            adjusted=torch.clamp(adjusted,min=lower,max=upper)
            self.brake_active.copy_(active)
            self.brake_infeasible.copy_((positive&(lower>bound+1e-6))|(negative&(upper<bound-1e-6)))
            self.brake_intervention.copy_(((adjusted-target).abs()>1e-6).float().mean(-1))
            self.brake_position.copy_(q);self.brake_velocity.copy_(dq);self.brake_bound.copy_(bound)
            target=adjusted
        self.requested.copy_(requested);self.target.copy_(target)
        self.target_rate.copy_((target-self.previous)/dt)
        delta=(requested-target).abs();normalized=delta/self.cfg.guarded_projection_scale
        weights=torch.ones_like(self.rate);weights[self.calf_ids]=2
        self.cost.copy_((torch.where(normalized<1,.5*normalized.square(),normalized-.5)*weights).mean(-1))
        self.intervention.copy_((delta>1e-6).float().mean(-1))
        self.rate_intervention.copy_(((requested-self.previous).abs()>self.rate*dt+1e-6).float().mean(-1))
        self.previous.copy_(target)
        return target

    def observe_actual(self,q,dt,step):
        if self.actual_step==step:return
        self.actual_step=step
        excess=(self.limits[...,0]-q).clamp_min(0)+(q-self.limits[...,1]).clamp_min(0)
        self.actual_excess.copy_(excess)
        self.actual_cost.copy_((excess/self.cfg.guarded_actual_scale).square().mean(-1))
        outside=excess>self.cfg.guarded_actual_tolerance
        self.actual_fraction.copy_(outside.float().mean(-1))
        self.violation_run.copy_(torch.where(outside.any(-1),self.violation_run+dt,0.))

    def reset(self,ids):
        for value in (self.previous,self.target,self.requested):value[ids]=self.default[ids]
        for value in (self.target_rate,self.cost,self.intervention,self.rate_intervention,self.actual_excess,
                      self.actual_cost,self.violation_run,self.actual_fraction,self.brake_active,self.brake_infeasible,
                      self.brake_intervention,self.brake_velocity):value[ids]=0
        self.brake_position[ids]=self.default[ids];self.brake_bound[ids]=self.default[ids]


class OfficialJointMonitor:
    """URDF velocity/effort checks on actual physics samples, including startup.

    No state clipping. A single hard-bound exceedance disqualifies success;
    sustained exceedance ends the episode. Peak/continuous motor certification
    remains separate from these official generic Go2 software bounds.
    """
    def __init__(self,like,calf_ids,settings):
        self.cfg=settings
        self.velocity=like.new_full((like.shape[-1],),30.1);self.velocity[calf_ids]=15.7
        self.effort=like.new_full((like.shape[-1],),23.7);self.effort[calf_ids]=45.43
        self.step_cost=like.new_zeros((len(like),3));self.step_seconds=like.new_zeros(len(like))
        self.step_velocity_peak=like.new_zeros(len(like))
        self.seconds=self.step_seconds.clone();self.run=self.seconds.clone()
        self.bad_samples=self.seconds.clone();self.peak_velocity_ratio=self.seconds.clone();self.peak_effort_ratio=self.seconds.clone()
        self.failed=self.seconds.bool();self.last_velocity=like.clone();self.last_computed=like.clone();self.last_applied=like.clone()

    def begin_step(self):
        self.step_cost.zero_();self.step_seconds.zero_()
        self.step_velocity_peak.zero_()

    def observe(self,dq,computed,applied,dt):
        if dt<=0 or dq.shape!=self.last_velocity.shape or computed.shape!=dq.shape or applied.shape!=dq.shape:
            raise ValueError('Invalid official joint physics sample')
        self.last_velocity.copy_(dq);self.last_computed.copy_(computed);self.last_applied.copy_(applied)
        vr=torch.nan_to_num(dq.abs()/self.velocity,nan=10.,posinf=10.,neginf=10.)
        tr=torch.nan_to_num(applied.abs()/self.effort,nan=10.,posinf=10.,neginf=10.)
        def shape(ratio,margin):
            x=((ratio-margin)/(1-margin)).clamp_min(0)
            value=torch.where(x<1,.5*x.square(),x-.5)
            return .5*(value.mean(-1)+value.amax(-1))
        velocity_cost=shape(vr,self.cfg.edu_velocity_margin)
        self.step_velocity_peak=torch.maximum(self.step_velocity_peak,velocity_cost)
        requested_ratio=torch.nan_to_num(computed.abs()/self.effort,nan=10.,posinf=10.,neginf=10.)
        effort_cost=shape(requested_ratio,self.cfg.edu_effort_margin)
        saturation=torch.nan_to_num((computed-applied).abs()/self.effort,nan=10.,posinf=10.,neginf=10.).square().mean(-1)
        self.step_cost+=torch.stack((velocity_cost,effort_cost,saturation),-1)*dt
        self.step_seconds+=dt;self.seconds+=dt
        finite=torch.isfinite(dq).all(-1)&torch.isfinite(computed).all(-1)&torch.isfinite(applied).all(-1)
        bad=(dq.abs()>self.velocity+1e-3).any(-1)|(applied.abs()>self.effort+1e-3).any(-1)|~finite
        self.bad_samples+=bad.float();self.run=torch.where(bad,self.run+dt,0.)
        self.failed|=(self.run>=self.cfg.edu_violation_seconds-1e-6)|~finite
        if self.cfg.edu_terminate_on_exceedance:self.failed|=bad
        self.peak_velocity_ratio=torch.maximum(self.peak_velocity_ratio,vr.amax(-1))
        self.peak_effort_ratio=torch.maximum(self.peak_effort_ratio,tr.amax(-1))

    def costs(self):
        value=self.step_cost/self.step_seconds.clamp_min(1e-9)[:,None]
        value[:,0]+=self.cfg.edu_velocity_peak_weight*self.step_velocity_peak
        return value

    def diagnostics(self):
        return dict(edu_samples_seconds=self.seconds,edu_bad_physics_samples=self.bad_samples,
            edu_peak_velocity_ratio=self.peak_velocity_ratio,edu_peak_effort_ratio=self.peak_effort_ratio,
            edu_terminated=self.failed,edu_limits=(self.bad_samples==0)&(self.seconds>0))

    def reset(self,ids):
        for value in (self.step_cost,self.step_seconds,self.step_velocity_peak,self.seconds,self.run,self.bad_samples,
                      self.peak_velocity_ratio,self.peak_effort_ratio,self.failed,self.last_velocity,self.last_computed,self.last_applied):value[ids]=0


class RequestLimitMonitor:
    """Monitor the unfiltered angle request once per action, never actual q."""
    def __init__(self,like):
        self.seconds=torch.zeros_like(like);self.peak=torch.zeros_like(like);self.fraction=torch.zeros_like(like)
        self.failed=torch.zeros_like(like,dtype=torch.bool)

    def observe(self,requested,lower,upper,tolerance,required_seconds,active,dt):
        excess=(lower-requested).clamp_min(0)+(requested-upper).clamp_min(0)
        self.peak.copy_(excess.amax(-1));self.fraction.copy_((excess>1e-6).float().mean(-1))
        self.seconds.copy_(torch.where(active&(self.peak>tolerance),self.seconds+dt,0.))
        self.failed|=active&(self.seconds>=required_seconds-1e-6)
        return self.failed

    def reset(self,ids):
        for value in (self.seconds,self.peak,self.fraction,self.failed):value[ids]=0


def guarded_foot_height_cost(height,settings):
    caps=height.new_tensor([settings.guarded_front_height]*2+[settings.guarded_hind_height]*2)
    normalized=(height-caps).clamp_min(0)/settings.guarded_height_scale
    cost=torch.where(normalized<1,.5*normalized.square(),normalized-.5)
    return cost[:,:2].mean(-1),cost[:,2:].mean(-1),(height>caps+.001).float()


def phase_targets(phase,speed,settings,strength=1.,profile=None):
    s=settings
    upper=s.reference_max_speed if s.reference_max_speed is not None else s.max_speed
    ratio=((speed-s.reference_start_speed)/(upper-s.reference_start_speed)).clamp(0,1)
    period=s.slow_period+(s.fast_period-s.slow_period)*ratio
    if profile is not None:period=period*(1-profile.alpha)+.60*profile.alpha
    local=(phase[:,None]-(phase.new_tensor(s.touchdown) if profile is None else profile.touchdown))%1.
    duty=phase.new_tensor(s.duty) if profile is None else profile.duty
    stance=local<duty
    swing=((local-duty)/(1-duty)).clamp(0,1)
    stance_progress=(local/duty).clamp(0,1)
    a=s.thigh_amplitude_slow+(s.thigh_amplitude_fast-s.thigh_amplitude_slow)*ratio
    c=s.calf_amplitude_slow+(s.calf_amplitude_fast-s.calf_amplitude_slow)*ratio
    h=s.swing_height_slow+(s.swing_height_fast-s.swing_height_slow)*ratio
    amplitude_factor=.35+.65*strength
    thigh=torch.where(stance,-torch.cos(math.pi*stance_progress),torch.cos(math.pi*swing))*a[:,None]*amplitude_factor
    lift=torch.where(stance,torch.zeros_like(swing),torch.sin(math.pi*swing))
    return dict(period=period,local=local,stance=stance,swing=swing,
                thigh=thigh,calf=-c[:,None]*amplitude_factor*lift,
                foot_height=h[:,None]*(.5+.5*strength)*lift,
                rom_thigh=1.8*a,rom_calf=.8*c,
                pitch=s.pitch_amplitude*torch.sin(2*math.pi*phase))


class CycleSpeedTracker:
    """Diagnostic only: full measured-cycle velocity, excluding partial/reset cycles."""
    def __init__(self,like):
        self.values={k:torch.zeros_like(like) for k in ('points','vx','vx2','command','cycle_points','cycle_error',
            'cycles','cycle_mean_error2')}
        self.started=torch.zeros_like(like,dtype=torch.bool);self.all_active=self.started.clone()

    def observe(self,vx,command,active):
        v=self.values;weight=active.float()
        v['points']+=weight;v['vx']+=weight*vx;v['vx2']+=weight*vx.square();v['command']+=weight*command
        v['cycle_points']+=weight;v['cycle_error']+=weight*(vx-command)
        self.all_active&=active

    def finish_cycles(self,wrapped):
        v=self.values;good=wrapped&self.started&self.all_active&(v['cycle_points']>0)
        v['cycles']+=good.float()
        v['cycle_mean_error2']+=good.float()*(v['cycle_error']/v['cycle_points'].clamp_min(1)).square()
        v['cycle_points'][wrapped]=0;v['cycle_error'][wrapped]=0
        self.started|=wrapped;self.all_active=torch.where(wrapped,True,self.all_active)

    def diagnostics(self):
        v=self.values;n=v['points'].clamp_min(1);mean=v['vx']/n;command=v['command']/n
        return dict(gallop_mean_vx=mean,gallop_mean_command_vx=command,gallop_mean_vx_error=mean-command,
            gallop_vx_std=torch.sqrt((v['vx2']/n-mean.square()).clamp_min(0)),
            gallop_cycle_mean_vx_rmse=torch.sqrt(v['cycle_mean_error2']/v['cycles'].clamp_min(1)),
            gallop_speed_cycles=v['cycles'],gallop_speed_samples=v['points'])

    def reset(self,ids):
        for value in self.values.values():value[ids]=0
        self.started[ids]=False;self.all_active[ids]=False


class LeapDistanceTracker:
    """First all-off pose to first touchdown, projected onto takeoff heading.

    Call once per physics update after scene buffers have advanced. No reward is
    given for initial free fall, a reset, or an unfinished flight. Phase wraps
    reset only the reward budget, never the ongoing flight measurement.
    """
    def __init__(self,positions,settings):
        self.cfg=settings;n=positions.shape[0]
        self.contact=torch.zeros(n,4,dtype=torch.bool,device=positions.device)
        self.seeded=torch.zeros(n,dtype=torch.bool,device=positions.device)
        self.active=self.seeded.clone()
        self.elapsed=positions.new_zeros(n);self.air_seconds=self.elapsed.clone()
        self.takeoff=positions.clone();self.landing=positions.clone()
        self.direction=positions.new_zeros(n,2);self.direction[:,0]=1.
        self.current_distance=self.elapsed.clone();self.last_distance=self.elapsed.clone();self.last_seconds=self.elapsed.clone()
        self.distance_target=torch.full_like(self.elapsed,settings.leap_min_distance)
        self.events=self.elapsed.clone();self.qualified_events=self.elapsed.clone()
        self.distance_sum=self.elapsed.clone();self.seconds_sum=self.elapsed.clone();self.maximum_distance=self.elapsed.clone()
        self.cycle_best=self.elapsed.clone();self.pending_reward=self.elapsed.clone()
        self.landed=self.elapsed.clone();self.qualified_landed=self.elapsed.clone()
        self.samples=0;self.physics_samples=self.elapsed.clone()

    def begin_step(self,wrapped):
        self.pending_reward.zero_();self.landed.zero_();self.qualified_landed.zero_();self.samples=0
        self.cycle_best=torch.where(wrapped,0.,self.cycle_best)

    def observe(self,positions,quaternions,force,dt,reward_eligible,measure_enabled=None):
        self.samples+=1;self.physics_samples+=1;self.elapsed+=dt
        contact=torch.where(self.contact,force>self.cfg.contact_off,force>=self.cfg.contact_on)
        was_grounded=self.contact.any(dim=1);grounded=contact.any(dim=1)
        start=self.seeded&was_grounded&~grounded&~self.active&(self.elapsed>=self.cfg.metric_warmup_seconds)
        if measure_enabled is not None:start&=measure_enabled
        # Existing flights include the elapsed interval; new flights start at
        # the timestamp of the first observed all-off pose (duration zero).
        self.air_seconds=torch.where(self.active,self.air_seconds+dt,self.air_seconds)
        self.air_seconds=torch.where(start,0.,self.air_seconds)
        self.takeoff=torch.where(start[:,None],positions,self.takeoff)
        w,x,y,z=quaternions.unbind(dim=1)
        direction=torch.stack((1-2*(y.square()+z.square()),2*(x*y+w*z)),dim=1)
        direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-6)
        self.direction=torch.where(start[:,None],direction,self.direction)
        self.active|=start
        forward=((positions[:,:2]-self.takeoff[:,:2])*self.direction).sum(dim=1)
        finish=self.active&grounded
        qualified=finish&(self.air_seconds>=self.cfg.flight_min_seconds-1e-6)&(forward>=self.distance_target-1e-5)
        self.events+=finish.float();self.qualified_events+=qualified.float()
        self.distance_sum+=torch.where(finish,forward,0.)
        self.seconds_sum+=torch.where(finish,self.air_seconds,0.)
        self.maximum_distance=torch.where(finish,torch.maximum(self.maximum_distance,forward),self.maximum_distance)
        self.last_distance=torch.where(finish,forward,self.last_distance)
        self.last_seconds=torch.where(finish,self.air_seconds,self.last_seconds)
        self.landing=torch.where(finish[:,None],positions,self.landing)
        self.landed+=finish.float();self.qualified_landed+=qualified.float()
        # Partial credit approaches the fixed distance/time threshold; one
        # complete score per reference cycle, independent of physics sampling.
        progress=(forward/self.distance_target).clamp(0,1)*(self.air_seconds/self.cfg.flight_min_seconds).clamp(0,1)
        score=torch.where(qualified,torch.ones_like(progress),self.cfg.leap_partial_credit*progress.pow(self.cfg.leap_progress_exponent))
        score=torch.where(finish&reward_eligible,score,0.)
        gain=(score-self.cycle_best).clamp_min(0)
        self.pending_reward+=gain;self.cycle_best=torch.maximum(self.cycle_best,score)
        self.active&=~finish
        self.current_distance=torch.where(self.active,forward,0.)
        self.air_seconds=torch.where(self.active,self.air_seconds,0.)
        self.contact=contact;self.seeded[:]=True
        return start,finish,gain

    def reset(self,ids):
        for value in (self.contact,self.seeded,self.active):value[ids]=False
        for value in (self.elapsed,self.air_seconds,self.takeoff,self.landing,self.direction,self.current_distance,
                      self.last_distance,self.last_seconds,self.events,self.qualified_events,self.distance_sum,self.seconds_sum,
                      self.maximum_distance,self.cycle_best,self.pending_reward,self.landed,self.qualified_landed,self.physics_samples):value[ids]=0
        self.direction[ids,0]=1.


def flight_phase_targets(phase,period,settings):
    """Shared flight window, preceded by push-off and followed by recovery."""
    air_start=getattr(settings,'shared_air_start',None)
    if air_start is None:air_start=max(t+d for t,d in zip(settings.touchdown,settings.duty))
    prepare=(phase>=air_start-settings.flight_prepare_seconds/period)&(phase<air_start)
    air=phase>=air_start
    recovery=phase<settings.flight_recovery_seconds/period
    stage=torch.where(prepare,1,torch.where(air,2,torch.where(recovery,3,0)))
    return prepare,air,recovery,stage


class FlightStabilityTracker:
    """5ms shaping and delayed landing confirmation. All rewards are bounded.

    Rates integrate in seconds; event points are paid once. Distance is held
    through recovery, so a crash or another takeoff cannot earn a landing bonus.
    This tracker does not prescribe an individual foot's touchdown order.
    """
    RATE_KEYS=('drive','clearance','support','balance','impact')
    def __init__(self,positions,settings,body_weight):
        self.cfg=settings;self.body_weight=body_weight
        n=len(positions)
        def zero():return positions.new_zeros(n)
        self.rates={key:zero() for key in self.RATE_KEYS}
        self.points={key:zero() for key in ('takeoff','landing','distance')}
        self.fore_ok_at_takeoff=zero().bool();self.flight_valid=zero().bool();self.recovering=zero().bool()
        self.recovery_elapsed=zero();self.recovery_bank=zero();self.recovery_quality=zero();self.recovery_progress=zero()
        self.previous_vz=zero();self.takeoff_best=zero();self.landing_best=zero();self.step_stable=zero()
        self.totals={key:zero() for key in ('takeoffs','landings','stable_landings','aborted_landings',
                    'takeoff_vz_sum','landing_peak_force_bw','landing_peak_down_vz','balance_integral','support_integral','seconds')}

    def begin_step(self,wrapped):
        for value in (*self.rates.values(),*self.points.values(),self.step_stable):value.zero_()
        self.takeoff_best=torch.where(wrapped,0.,self.takeoff_best)
        self.landing_best=torch.where(wrapped,0.,self.landing_best)
        # Actual flights and recovery banks survive reference phase wraps.

    def observe(self,leap,start,finish,distance_gain,phase,period,position,vz,body_velocity,angular_velocity,
                rpy,foot_height,foot_velocity,force,command,fore_ok,dt,gait_mix=None):
        s=self.cfg
        prepare,air,_,_=flight_phase_targets(phase,period,s)
        contact=leap.contact;grounded=contact.any(dim=1)
        warm=leap.elapsed>=s.metric_warmup_seconds
        height_ok=(position[:,2]>=s.min_height)&(position[:,2]<=s.max_height)
        tilt_ok=(rpy[:,:2].abs()<=s.flight_max_tilt).all(dim=1)
        angular_ok=angular_velocity.norm(dim=1)<=s.flight_max_angular_speed
        lateral_ok=body_velocity[:,1].abs()<=s.flight_max_lateral_speed
        speed=(body_velocity[:,0]/(.5*command[:,0]).clamp_min(.05)).clamp(0,1)
        pose_cost=(rpy[:,:2]/s.flight_max_tilt).square().mean(dim=1)
        if s.heading_motion_enabled:pose_cost=torch.zeros_like(pose_cost)
        rotation_cost=(angular_velocity/s.flight_max_angular_speed).square().mean(dim=1)
        lateral_cost=(body_velocity[:,1]/s.flight_max_lateral_speed).square()
        quality=torch.exp(-pose_cost-rotation_cost-lateral_cost)
        safe=height_ok&tilt_ok&angular_ok&lateral_ok
        positive=warm.float()*safe.float()*speed
        mix=torch.ones_like(speed) if gait_mix is None else gait_mix
        upward=(vz/s.flight_takeoff_vz).clamp(0,1)*torch.exp(-((vz-s.flight_takeoff_vz).clamp_min(0)/s.flight_takeoff_vz).square())
        unload=1-contact.float().mean(dim=1)
        clearance=(foot_height.amin(dim=1)/s.flight_clearance).clamp(0,1)
        self.rates['drive']+=dt*mix*positive*prepare.float()*grounded.float()*upward*quality
        self.rates['clearance']+=dt*mix*positive*air.float()*(.5*unload+.5*clearance)*quality
        support=mix*air.float()*contact.float().mean(dim=1)
        balance=(pose_cost+lateral_cost+(angular_velocity[:,0]/s.flight_max_angular_speed).square()).clamp_max(4)/3
        self.rates['support']+=dt*warm.float()*support
        self.rates['balance']+=dt*warm.float()*balance
        self.totals['seconds']+=dt*warm.float()
        self.totals['support_integral']+=dt*warm.float()*support
        self.totals['balance_integral']+=dt*warm.float()*balance
        valid_start=start&safe&(speed>=1)&fore_ok&(prepare|air)
        self.fore_ok_at_takeoff=torch.where(start,fore_ok,self.fore_ok_at_takeoff)
        self.flight_valid=torch.where(start,valid_start,self.flight_valid)
        self.totals['takeoffs']+=valid_start.float()
        self.totals['takeoff_vz_sum']+=valid_start.float()*vz
        takeoff_score=valid_start.float()*upward*quality
        self.points['takeoff']+=(takeoff_score-self.takeoff_best).clamp_min(0)
        self.takeoff_best=torch.maximum(self.takeoff_best,takeoff_score)

        # Include the pre-touchdown velocity, before the collision decelerates the root.
        downward=torch.maximum(-self.previous_vz,-vz).clamp_min(0)
        force_bw=force.clamp_min(0).sum(dim=1)/self.body_weight.clamp_min(1.)
        slide=(foot_velocity[:,:,:2].norm(dim=2)*contact.float()).amax(dim=1)
        impact_cost=(((force_bw/s.flight_max_force_bw-1).clamp_min(0)).square()
                    +((downward/s.flight_max_landing_vz-1).clamp_min(0)).square()
                    +(slide/s.flight_max_foot_slide).square()).clamp_max(4)/3
        begin_recovery=finish&self.flight_valid&(leap.last_seconds>=s.flight_landing_min_air_seconds-1e-6)&(leap.last_distance>0)
        monitoring=self.recovering|begin_recovery
        self.rates['impact']+=dt*monitoring.float()*impact_cost
        self.totals['landings']+=begin_recovery.float()
        self.totals['landing_peak_force_bw']=torch.maximum(self.totals['landing_peak_force_bw'],torch.where(monitoring,force_bw,0.))
        self.totals['landing_peak_down_vz']=torch.maximum(self.totals['landing_peak_down_vz'],torch.where(monitoring,downward,0.))
        landing_safe=safe&grounded&(force_bw<=s.flight_max_force_bw)&(downward<=s.flight_max_landing_vz)&(slide<=s.flight_max_foot_slide)
        aborted=self.recovering&(start|~landing_safe)
        self.totals['aborted_landings']+=aborted.float()
        self.recovering&=~aborted
        self.recovery_bank=torch.where(self.recovering,self.recovery_bank,0.)
        self.recovery_elapsed=torch.where(self.recovering,self.recovery_elapsed+dt,0.)
        self.recovery_bank=torch.where(begin_recovery,distance_gain,self.recovery_bank)
        self.recovery_elapsed=torch.where(begin_recovery,0.,self.recovery_elapsed)
        landing_quality=quality*torch.exp(-(downward/s.flight_max_landing_vz).square()-.5*(slide/s.flight_max_foot_slide).square())
        self.recovery_quality=torch.where(begin_recovery,landing_quality,self.recovery_quality)
        progress=(leap.last_distance/leap.distance_target).clamp(0,1)*(leap.last_seconds/s.flight_min_seconds).clamp(0,1)
        self.recovery_progress=torch.where(begin_recovery,progress,self.recovery_progress)
        self.recovering|=begin_recovery
        failed_touch=begin_recovery&~landing_safe
        self.totals['aborted_landings']+=failed_touch.float()
        self.recovering&=~failed_touch
        self.recovery_quality=torch.where(self.recovering,torch.minimum(self.recovery_quality,landing_quality),self.recovery_quality)
        settled=self.recovering&(self.recovery_elapsed>=s.flight_recovery_seconds-1e-6)
        landing_score=torch.where(settled,self.recovery_quality,0.)
        if s.flight_landing_distance_scaled:landing_score*=self.recovery_progress
        self.points['landing']+=(landing_score-self.landing_best).clamp_min(0)
        self.landing_best=torch.maximum(self.landing_best,landing_score)
        self.points['distance']+=torch.where(settled,self.recovery_bank*self.recovery_quality,0.)
        self.totals['stable_landings']+=settled.float();self.step_stable+=settled.float()
        self.recovering&=~settled
        self.recovery_bank=torch.where(self.recovering,self.recovery_bank,0.)
        self.previous_vz=vz.clone()

    def reject_terminated(self,mask):
        self.totals['aborted_landings']+=mask.float()*(self.recovering.float()+self.step_stable)
        self.totals['stable_landings']-=mask.float()*self.step_stable
        self.recovering&=~mask
        self.recovery_bank[mask]=0.
        for value in (*self.points.values(),self.rates['drive'],self.rates['clearance']):value[mask]=0.

    def diagnostics(self):
        values={'flight_'+key:value for key,value in self.totals.items() if not key.endswith(('_sum','_integral')) and key!='seconds'}
        values.update(flight_takeoff_vz_mean=self.totals['takeoff_vz_sum']/self.totals['takeoffs'].clamp_min(1),
                      flight_balance_cost=self.totals['balance_integral']/self.totals['seconds'].clamp_min(1e-6),
                      flight_support_cost=self.totals['support_integral']/self.totals['seconds'].clamp_min(1e-6))
        return values

    def reset(self,ids):
        for value in (*self.rates.values(),*self.points.values(),*self.totals.values(),self.fore_ok_at_takeoff,
                      self.flight_valid,self.recovering,self.recovery_elapsed,self.recovery_bank,self.recovery_quality,self.recovery_progress,
                      self.previous_vz,self.takeoff_best,self.landing_best,self.step_stable):value[ids]=0


def quaternion_rpy(quaternion):
    w,x,y,z=quaternion.unbind(dim=1)
    return torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
        torch.asin((2*(w*y-z*x)).clamp(-1,1)),torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),dim=1)


def wrap_angle(value):return torch.atan2(torch.sin(value),torch.cos(value))


def heading_features(quaternion,target_yaw):
    error=wrap_angle(target_yaw-quaternion_rpy(quaternion)[:,2])
    # Both new columns are zero at the target heading. Existing observation
    # columns retain their meaning, including the frozen walking actor's 45.
    return torch.stack((torch.sin(error),torch.cos(error)-1),dim=1)


class HeadingMotionTracker:
    """Yaw penalty, temporal roll/pitch quietness and positive motion/width scores.

    No foot-center or static roll/pitch angle penalty. History is reset per env;
    the first observation earns no temporal score. Statistics cover all stages.
    """
    def __init__(self,positions,settings):
        self.cfg=settings;n=len(positions)
        def zero(*shape):return positions.new_zeros(n,*shape)
        self.previous=zero(2);self.seeded=torch.zeros(n,dtype=torch.bool,device=positions.device)
        self.values={key:zero() for key in ('yaw_error','along_speed','cross_speed','direction_error','front_width_score','hind_width_score','forward_score','quiet_score')}
        self.values.update(width=zero(2),roll_pitch_rate=zero(2),rate_valid=self.seeded.clone())
        self.sums={key:zero() for key in ('points','seconds','rate_points','yaw2','roll_rate2','pitch_rate2',
            'front_width','hind_width','front_width_score','hind_width_score','forward_score','quiet_score','along_speed','cross2')}
        if settings.stabilization_enabled:
            self.values.update(pose_quality=zero(),yaw_rate=zero(),yaw_damping_cost=zero())
            self.sums.update(pose_quality=zero(),yaw_rate2=zero())

    def observe(self,quaternion,position,feet,contact,world_velocity,command_speed,dt,angular_velocity=None):
        if not math.isfinite(dt) or dt<=0:raise ValueError('Positive motion sample timestep required')
        s=self.cfg;rpy=quaternion_rpy(quaternion)
        valid=self.seeded.clone()
        rate=wrap_angle(rpy[:,:2]-self.previous)/dt
        rate=torch.where(valid[:,None],rate,0.)
        self.previous.copy_(rpy[:,:2]);self.seeded[:]=True
        yaw_error=wrap_angle(rpy[:,2]-s.target_yaw)
        yaw_cost=(yaw_error/s.yaw_error_scale).square().clamp_max(4)
        quiet=torch.exp(-(rate/s.roll_pitch_rate_scale).square().mean(dim=1))*valid.float()
        if s.stabilization_enabled:
            if angular_velocity is None:raise ValueError('Stabilization requires measured body angular velocity')
            deadband=rpy.new_tensor([s.stabilization_roll_deadband,s.stabilization_pitch_deadband])
            pose_scale=rpy.new_tensor([s.stabilization_roll_scale,s.stabilization_pitch_scale])
            pose_quality=torch.exp(-((rpy[:,:2].abs()-deadband).clamp_min(0)/pose_scale).square().sum(1))
            quiet*=pose_quality
            yaw_rate=angular_velocity[:,2]
            normalized=(yaw_rate/s.stabilization_yaw_rate_scale).abs()
            yaw_damping=torch.where(normalized<1,.5*normalized.square(),normalized-.5)
        relative=feet-position[:,None,:];yaw=rpy[:,2]
        lateral=-torch.sin(yaw)[:,None]*relative[:,:,0]+torch.cos(yaw)[:,None]*relative[:,:,1]
        pairs=lateral.reshape(-1,2,2);widths=pairs[:,:,0]-pairs[:,:,1]
        multiplier=torch.where(contact.reshape(-1,2,2).all(dim=2),1.,s.stance_swing_tolerance_multiplier)
        target=widths.new_tensor([s.front_stance_width,s.hind_stance_width])
        error=((widths-target).abs()-s.stance_width_tolerance*multiplier).clamp_min(0)/s.stance_width_scale
        width_scores=torch.exp(-error.square())
        if s.stabilization_enabled:
            width_scores=1/(1+error.square())
            support=contact.reshape(-1,2,2).any(dim=2)
            width_scores*=torch.where(support,1.,s.stabilization_air_width_weight)
        along=math.cos(s.target_yaw)*world_velocity[:,0]+math.sin(s.target_yaw)*world_velocity[:,1]
        cross=-math.sin(s.target_yaw)*world_velocity[:,0]+math.cos(s.target_yaw)*world_velocity[:,1]
        direction=torch.atan2(cross,along)
        moving=(command_speed>.05)&(along>0)
        speed_scale=(s.progress_speed_scale_fraction*command_speed).clamp_min(.25)
        progress=(along/command_speed.clamp_min(.05)).clamp(0,1)
        forward=progress*torch.exp(-((along-command_speed)/speed_scale).square()-(direction/s.progress_direction_scale).square())*moving.float()
        terms=dict(yaw_heading=-yaw_cost,roll_pitch_quiet=quiet,front_width=width_scores[:,0],
                   hind_width=width_scores[:,1],forward_progress=forward)
        measured=dict(yaw_error=yaw_error,along_speed=along,cross_speed=cross,direction_error=direction,
            front_width_score=width_scores[:,0],hind_width_score=width_scores[:,1],forward_score=forward,
            quiet_score=quiet,width=widths,roll_pitch_rate=rate,rate_valid=valid)
        if s.stabilization_enabled:
            terms['yaw_damping']=-yaw_damping
            measured.update(pose_quality=pose_quality,yaw_rate=yaw_rate,yaw_damping_cost=yaw_damping)
        for key,value in measured.items():self.values[key].copy_(value)
        summed=dict(points=torch.ones_like(along),seconds=torch.full_like(along,dt),rate_points=valid.float(),yaw2=yaw_error.square(),
            roll_rate2=rate[:,0].square(),pitch_rate2=rate[:,1].square(),front_width=widths[:,0],hind_width=widths[:,1],
            front_width_score=width_scores[:,0],hind_width_score=width_scores[:,1],forward_score=forward,
            quiet_score=quiet,along_speed=along,cross2=cross.square())
        if s.stabilization_enabled:summed.update(pose_quality=pose_quality,yaw_rate2=yaw_rate.square())
        for key,value in summed.items():self.sums[key]+=value
        return terms,self.values

    def diagnostics(self):
        p=self.sums;count=p['points'].clamp_min(1);rates=p['rate_points'].clamp_min(1)
        result=dict(posture_seconds=p['seconds'],posture_rate_samples=p['rate_points'],posture_yaw_rmse=torch.sqrt(p['yaw2']/count),
            posture_roll_rate_rms=torch.sqrt(p['roll_rate2']/rates),posture_pitch_rate_rms=torch.sqrt(p['pitch_rate2']/rates),
            posture_cross_speed_rms=torch.sqrt(p['cross2']/count),posture_quiet_score=p['quiet_score']/rates,
            **{'posture_'+key:p[key]/count for key in ('front_width','hind_width','front_width_score','hind_width_score','forward_score','along_speed')})
        if self.cfg.stabilization_enabled:result.update(posture_pose_quality=p['pose_quality']/count,posture_yaw_rate_rms=torch.sqrt(p['yaw_rate2']/count))
        return result

    def reset(self,ids):
        self.previous[ids]=0;self.seeded[ids]=False
        for value in (*self.values.values(),*self.sums.values()):value[ids]=0


def posture_terms(quaternion,position,feet,contact,settings):
    """Body-up alignment and signed foot tracks on a flat, yaw-aligned plane.

    Feet are FL, FR, RL, RR. Forward excursion and foot height are free. Signed
    widths reject crossed legs; pair centers reject common lateral translation.
    Swing/flight relaxes lateral deadbands, never disables the terms. These are
    rates: RewardManager integrates each term with the policy timestep.
    """
    s=settings
    w,x,y,z=quaternion.unbind(dim=1)
    tilt=2*torch.atan2(torch.linalg.vector_norm(torch.stack((x,y),dim=1),dim=1),
                       torch.linalg.vector_norm(torch.stack((w,z),dim=1),dim=1))
    upright=((tilt-s.upright_tolerance).clamp_min(0)/s.upright_scale).square().clamp_max(4)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    relative=feet-position[:,None,:]
    lateral=-torch.sin(yaw)[:,None]*relative[:,:,0]+torch.cos(yaw)[:,None]*relative[:,:,1]
    pairs=lateral.reshape(-1,2,2)
    widths=pairs[:,:,0]-pairs[:,:,1]
    centers=pairs.mean(dim=2)
    supported=contact.reshape(-1,2,2).all(dim=2)
    multiplier=torch.where(supported,1.,s.stance_swing_tolerance_multiplier)
    targets=widths.new_tensor([s.front_stance_width,s.hind_stance_width])
    width_cost=((widths-targets).abs()-s.stance_width_tolerance*multiplier).clamp_min(0)
    width_cost=(width_cost/s.stance_width_scale).square().clamp_max(4).mean(dim=1)
    center_cost=(centers.abs()-s.stance_center_tolerance*multiplier).clamp_min(0)
    center_cost=(center_cost/s.stance_center_scale).square().clamp_max(4).mean(dim=1)
    terms=dict(upright=-upright,stance_width=1-width_cost,stance_center=-center_cost)
    diagnostics=dict(tilt=tilt,width=widths,center=centers,upright_cost=upright,
                     width_cost=width_cost,center_cost=center_cost)
    return terms,diagnostics


def pair_joint_penalty(joints,reference,settings):
    """Compare mirrored residuals after removing each leg's own phase target.

    FL/FR and RL/RR need not have the same instantaneous angle or contact state.
    Hip abduction has opposite signs on the two sides of the robot.
    """
    residual=joints-reference
    pairs=residual.reshape(-1,2,2,3)
    mirror=residual.new_tensor([-1.,1.,1.])
    difference=(pairs[:,:,0]-pairs[:,:,1]*mirror).abs()
    excess=(difference-settings.pair_joint_tolerance).clamp_min(0)/settings.pair_joint_scale
    return excess.clamp_max(1).square().mean(dim=(1,2))


def pair_cycle_penalties(excursion,impulse,stance_seconds,duration,settings):
    """Bounded costs and per-pair diagnostics over a complete observed cycle."""
    rom=excursion.reshape(-1,2,2,2).clamp_min(0)
    rom_difference=(rom[:,:,0]-rom[:,:,1]).abs()/(rom[:,:,0]+rom[:,:,1]).clamp_min(.05)
    load=impulse.reshape(-1,2,2).clamp_min(0)
    load_difference=(load[:,:,0]-load[:,:,1]).abs()/load.sum(dim=2).clamp_min(.01)
    duty=(stance_seconds/duration[:,None].clamp_min(1e-6)).reshape(-1,2,2)
    stance_difference=(duty[:,:,0]-duty[:,:,1]).abs()
    target_duty=torch.as_tensor(settings.duty,device=duty.device,dtype=duty.dtype)
    allowed=(target_duty[...,0::2]-target_duty[...,1::2]).abs()+settings.pair_stance_tolerance
    def cost(value,tolerance):
        return ((value-tolerance).clamp_min(0)/(1-tolerance)).clamp_max(1).square()
    costs=dict(rom=cost(rom_difference,settings.pair_rom_tolerance).mean(dim=(1,2)),
               load=cost(load_difference,settings.pair_load_tolerance).mean(dim=1),
               stance=cost(stance_difference,allowed).mean(dim=1))
    diagnostics=dict(rom=rom_difference.mean(dim=2),load=load_difference,stance=stance_difference)
    return costs,diagnostics


class PairBalanceTracker:
    """Hold completed-cycle costs; discard startup, partial and reset cycles."""
    def __init__(self,joints,settings):
        self.cfg=settings
        n=joints.shape[0]
        self.qmin=joints[:,:,1:].clone();self.qmax=self.qmin.clone()
        self.hip_min=joints[:,:,0].clone();self.hip_max=self.hip_min.clone()
        self.impulse=joints.new_zeros(n,4);self.stance=joints.new_zeros(n,4)
        self.duration=joints.new_zeros(n)
        self.started=torch.zeros(n,dtype=torch.bool,device=joints.device)
        self.warmup_ok=self.started.clone()
        self.costs={key:joints.new_zeros(n) for key in ('rom','load','stance')}
        self.diagnostics={key:joints.new_zeros(n,2) for key in self.costs}
        self.valid_cycles=joints.new_zeros(n)
        self.hip_cost=joints.new_zeros(n);self.hip_asymmetry=joints.new_zeros(n,2)

    def finish_cycles(self,wrapped,joints):
        costs,diagnostics=pair_cycle_penalties(self.qmax-self.qmin,self.impulse,self.stance,self.duration,self.cfg)
        eligible=self.started&self.warmup_ok&(self.duration>0)
        hip=(self.hip_max-self.hip_min).reshape(-1,2,2)
        asymmetry=(hip[:,:,0]-hip[:,:,1]).abs()/hip.sum(dim=2).clamp_min(.15)
        cost=((asymmetry-self.cfg.pair_hip_tolerance).clamp_min(0)/(1-self.cfg.pair_hip_tolerance)).clamp_max(1).square().mean(dim=1)
        self.hip_cost=torch.where(wrapped,torch.where(eligible,cost,0.),self.hip_cost)
        self.hip_asymmetry=torch.where(wrapped[:,None],torch.where(eligible[:,None],asymmetry,0.),self.hip_asymmetry)
        for key in self.costs:
            self.costs[key]=torch.where(wrapped,torch.where(eligible,costs[key],0.),self.costs[key])
            self.diagnostics[key]=torch.where(wrapped[:,None],torch.where(eligible[:,None],diagnostics[key],0.),self.diagnostics[key])
        self.valid_cycles+=(wrapped&eligible).float()
        self.impulse[wrapped]=0;self.stance[wrapped]=0;self.duration[wrapped]=0
        self.qmin[wrapped]=joints[wrapped,:,1:];self.qmax[wrapped]=joints[wrapped,:,1:]
        self.hip_min[wrapped]=joints[wrapped,:,0];self.hip_max[wrapped]=joints[wrapped,:,0]
        self.started[wrapped]=True;self.warmup_ok[wrapped]=True

    def observe(self,joints,force,contact,dt,active):
        self.qmin=torch.minimum(self.qmin,joints[:,:,1:]);self.qmax=torch.maximum(self.qmax,joints[:,:,1:])
        self.hip_min=torch.minimum(self.hip_min,joints[:,:,0]);self.hip_max=torch.maximum(self.hip_max,joints[:,:,0])
        self.impulse+=force.clamp_min(0)*dt;self.stance+=contact.float()*dt;self.duration+=dt
        self.warmup_ok&=active

    def reset(self,ids,joints):
        self.impulse[ids]=0;self.stance[ids]=0;self.duration[ids]=0
        self.started[ids]=False;self.warmup_ok[ids]=False;self.valid_cycles[ids]=0
        self.qmin[ids]=joints[ids,:,1:];self.qmax[ids]=joints[ids,:,1:]
        self.hip_min[ids]=joints[ids,:,0];self.hip_max[ids]=joints[ids,:,0]
        self.hip_cost[ids]=0.;self.hip_asymmetry[ids]=0.
        for value in (*self.costs.values(),*self.diagnostics.values()):value[ids]=0


def forelimb_success(load,rom,participation,settings):
    load_ok=(load[:,:2].sum(dim=1)>=settings.min_fore_load_share)&(load[:,:2].amin(dim=1)>=settings.min_each_fore_load_share)&(load[:,2:].sum(dim=1)>=settings.min_hind_load_share)
    motion_ok=(rom.mean(dim=1)>=settings.min_fore_rom_fraction)&(rom.amin(dim=1)>=settings.min_each_fore_rom_fraction)
    cycle_ok=participation.amin(dim=1)>=settings.min_fore_participation
    return load_ok,motion_ok,cycle_ok


def gait_pattern_success(contact_accuracy,order_accuracy,cycles,sequences,settings):
    good=(contact_accuracy>=settings.contact_accuracy)&(cycles>=settings.min_cycles)
    if settings.require_touchdown_order:good&=(order_accuracy>=settings.order_accuracy)&(sequences>=settings.min_cycles)
    return good


class FixedGallopGoal:
    """Constant targets with no episode collection, decision windows or promotion."""
    def __init__(self,settings,emit=None):
        self.cfg=settings;self.cap=settings.start_speed;self.level=0
        self.last_rate=None;self.highest_confirmed_speed=None;self.events=[]
    def record(self,*args,**kwargs):return False
    def decide(self,*args,**kwargs):return None
    def bounds(self):return (self.cap,self.cap)


class SuccessSpeedCurriculum:
    def __init__(self,settings,emit=None):
        self.cfg=settings;self.emit=emit;self.cap=settings.start_speed
        self.level=0;self.streak=0;self.count=0;self.successes=0;self.criteria={}
        self.next_step=settings.decision_steps;self.last_rate=None;self.events=[];self.discarded=0
        self.highest_confirmed_speed=None

    def record(self,success,level,criteria):
        if level!=self.level:self.discarded+=1;return
        self.count+=1;self.successes+=bool(success)
        for key,value in criteria.items():self.criteria[key]=self.criteria.get(key,0)+bool(value)

    def decide(self,step):
        if step<self.next_step:return None
        self.next_step=(step//self.cfg.decision_steps+1)*self.cfg.decision_steps
        before=self.cap;reason='waiting_for_episodes'
        self.last_rate=self.successes/self.count if self.count else None
        if self.count>=self.cfg.min_episodes:
            if self.last_rate>=self.cfg.promote_success:
                self.streak+=1;reason='confirming_success'
                if self.streak>=self.cfg.pass_windows:
                    self.highest_confirmed_speed=max(self.highest_confirmed_speed or before,before)
                    self.cap=min(self.cfg.max_speed,self.cap+self.cfg.speed_increment)
                    reason='promote' if self.cap>before else 'at_limit';self.streak=0
            else:
                self.streak=0;reason='hold'
                if self.last_rate<self.cfg.regress_success and self.cap>self.cfg.start_speed:
                    self.cap=max(self.cfg.start_speed,self.cap-self.cfg.speed_increment);reason='regress'
        changed=not math.isclose(before,self.cap,abs_tol=1e-8)
        if changed:self.level+=1
        event=dict(step=int(step),speed_before=before,speed_cap=self.cap,level=self.level,
                   episodes=self.count,success_rate=self.last_rate,reason=reason,streak=self.streak,
                   criterion_rates={k:v/self.count for k,v in self.criteria.items()} if self.count else {},
                   discarded_cross_level=self.discarded)
        if self.cfg.stabilization_enabled:event.update(leap_target=float(stabilization_leap_target(torch.tensor(self.cap),self.cfg)),highest_confirmed_speed=self.highest_confirmed_speed)
        self.events.append(event)
        if self.emit:self.emit('GALLOP_CURRICULUM',**event)
        if self.count>=self.cfg.min_episodes:
            self.count=self.successes=0;self.criteria={}
        return event

    def bounds(self):return (self.cap-self.cfg.speed_band,self.cap)


def staged_gallop_plan():
    """Version 1: retain the gait, raise speed, then extend shared flight."""
    rows=[('stability',2.5,.18,.078,.06),('speed_275',2.75,.18,.077,.06),
          ('speed_300',3.,.18,.076,.06),('leap_020',3.,.20,.076,.06),
          ('leap_030',3.,.30,.120,.100),('leap_040',3.,.40,.155,.130),
          ('leap_050',3.,.50,.190,.165),('leap_065',3.,.65,.240,.215),
          ('leap_080',3.,.80,.300,.260)]
    plan=[]
    for i,(name,speed,distance,air,minimum) in enumerate(rows):
        alpha=0. if i<4 else (-.158+math.sqrt(.158**2+4*.066*(air-.076)))/(.132)
        touchdown=[(1-alpha)*a+alpha*b for a,b in zip((.25,.5,0.,.125),(.14,.28,0.,.07))]
        duty=[(1-alpha)*a+alpha*b for a,b in zip((.45,.30,.35,.45),(.28,.22,.28,.28))]
        period=(.42-.04*((speed-1.)/2.))*(1-alpha)+.60*alpha
        plan.append(dict(name=name,speed=speed,leap=distance,alpha=alpha,period=period,
            touchdown=touchdown,duty=duty,air_seconds=air,flight_min_seconds=minimum,
            flight_prepare_seconds=.08+.02*alpha,flight_recovery_seconds=.08+.04*alpha,
            flight_takeoff_vz=.4+1.1*alpha,flight_max_landing_vz=1.2+.6*alpha))
    return plan


class StagedEpisodeProfile:
    """Per-environment settings latched on reset, including physical sampling.

    Unchanged limits/reward weights delegate to immutable base settings.
    Only the reference template and flight timing/vertical scales vary.
    """
    FIELDS=('flight_min_seconds','flight_prepare_seconds','flight_recovery_seconds',
            'flight_takeoff_vz','flight_max_landing_vz')

    def __init__(self,like,settings,plan):
        self.base=settings;self.plan=plan
        self.index=torch.zeros_like(like,dtype=torch.long);self.alpha=torch.zeros_like(like)
        self.touchdown=like.new_zeros(len(like),4);self.duty=self.touchdown.clone()
        self.shared_air_start=torch.zeros_like(like)
        for key in self.FIELDS:setattr(self,key,torch.zeros_like(like))
        self.latch(torch.arange(len(like),device=like.device),0)

    def __getattr__(self,name):return getattr(self.base,name)

    def promotion_requirements(self):
        """Version1: retain core safety; tighten quality before final certification.

        Use each episode's reset-latched index, including mixed-stage cohorts.
        Width and lift-height remain rewarded at every stage. They become hard
        success gates only at the final80cm stage; final strict gates are intact.
        """
        def table(values):return self.alpha.new_tensor(values)[self.index]
        s=self.base
        return dict(
            landing_fraction=table([.60,.60,.60,.60,.65,.65,.70,.75,s.stabilization_landing_fraction]),
            roll_limit=table([math.radians(10)]*4+[math.radians(8)]*3+[math.radians(6),s.stabilization_roll_limit]),
            pitch_limit=table([math.radians(12)]*4+[math.radians(10)]*3+[math.radians(9),s.stabilization_pitch_limit]),
            yaw_limit=table([math.radians(8)]*4+[math.radians(5)]*3+[math.radians(4),s.staged_max_yaw_rms]),
            details_required=self.index==len(self.plan)-1,
        )

    def latch(self,ids,index):
        goal=self.plan[index];self.index[ids]=index;self.alpha[ids]=goal['alpha']
        self.touchdown[ids]=self.touchdown.new_tensor(goal['touchdown'])
        self.duty[ids]=self.duty.new_tensor(goal['duty'])
        self.shared_air_start[ids]=max(t+d for t,d in zip(goal['touchdown'],goal['duty']))
        for key in self.FIELDS:getattr(self,key)[ids]=goal[key]


class StagedGallopCurriculum(SuccessSpeedCurriculum):
    """Complete per-environment cohorts; no early-finisher or time-only promotion."""
    def __init__(self,settings,emit=None):
        super().__init__(settings,emit)
        self.plan=staged_gallop_plan();self.index=0;self.cap=self.goal['speed']
        self.highest_confirmed_stage=None;self.failure_streak=0
        self.cohort_size=None;self.cohort_seen=set();self.cohort_rounds=0
        self.repeated_env_episodes=0;self.adaptation_discarded=0;self.stage_started_step=0
        self.control_dt=.02

    @property
    def goal(self):return self.plan[self.index]

    def configure_envs(self,count,control_dt=.02):
        if not isinstance(count,int) or count<1 or control_dt<=0:raise ValueError('Invalid staged cohort configuration')
        self.cohort_size=count;self.control_dt=control_dt

    def record(self,success,level,criteria,env_id=None,step=None):
        if level!=self.level:self.discarded+=1;return False
        if self.cohort_size is None or env_id is None or not 0<=env_id<self.cohort_size or step is None:
            raise ValueError('Staged recording requires configured env id and control step')
        if (step-self.stage_started_step)*self.control_dt<self.cfg.staged_min_dwell_seconds:
            self.adaptation_discarded+=1;return False
        if env_id in self.cohort_seen:self.repeated_env_episodes+=1;return False
        self.cohort_seen.add(env_id)
        super().record(success,level,criteria)
        return True

    def restore(self,index,confirmed=None):
        if type(index) is not int or not 0<=index<len(self.plan):raise ValueError('Invalid saved Gallop stage')
        if confirmed is not None and (type(confirmed) is not int or not 0<=confirmed<len(self.plan)):
            raise ValueError('Invalid confirmed Gallop stage')
        self.index=index;self.level=0;self.cap=self.goal['speed'];self.highest_confirmed_stage=confirmed
        self.highest_confirmed_speed=None if confirmed is None else self.plan[confirmed]['speed']
        self.streak=self.failure_streak=self.count=self.successes=0;self.criteria={}
        self.cohort_seen.clear();self.stage_started_step=0
        self.cohort_rounds=self.repeated_env_episodes=self.adaptation_discarded=0
        self.next_step=self.cfg.decision_steps

    def decide(self,step):
        if step<self.next_step:return None
        self.next_step=(step//self.cfg.decision_steps+1)*self.cfg.decision_steps
        before=self.index;old_goal=self.goal.copy();reason='waiting_for_cohort'
        self.last_rate=self.successes/self.count if self.count else None
        closed=self.cohort_size is not None and len(self.cohort_seen)==self.cohort_size
        evaluate=closed and self.count>=self.cfg.min_episodes
        if (step-self.stage_started_step)*self.control_dt<self.cfg.staged_min_dwell_seconds:reason='adapting_stage'
        if evaluate:
            if self.last_rate>=self.cfg.promote_success:
                self.streak+=1;self.failure_streak=0;reason='confirming_success'
                if self.streak>=self.cfg.pass_windows:
                    self.highest_confirmed_stage=max(self.highest_confirmed_stage if self.highest_confirmed_stage is not None else -1,before)
                    self.highest_confirmed_speed=self.plan[self.highest_confirmed_stage]['speed']
                    self.index=min(self.index+1,len(self.plan)-1);self.streak=0
                    reason='at_limit' if self.index==before else 'promote_stage'
            else:
                regression_rate=(self.criteria.get('staged_regression',self.successes)/self.count
                    if self.cfg.staged_core_first_enabled else self.last_rate)
                self.streak=0;self.failure_streak=self.failure_streak+1 if regression_rate<self.cfg.regress_success else 0
                reason='hold'
                if self.cfg.staged_core_first_enabled and self.index==len(self.plan)-1 and regression_rate>=self.cfg.regress_success:
                    reason='refine_quality'
                if self.failure_streak>=self.cfg.staged_regress_windows and self.index>0:
                    self.index-=1;self.failure_streak=0;reason='regress_stage'
        rates={k:v/self.count for k,v in self.criteria.items()} if self.count else {}
        episodes=self.count
        if self.index!=before:self.level+=1;self.stage_started_step=step
        self.cap=self.goal['speed']
        event=dict(step=int(step),speed_before=old_goal['speed'],speed_cap=self.cap,level=self.level,
            staged_index_before=before,staged_index=self.index,staged_goal_before=old_goal,staged_goal=self.goal.copy(),
            episodes=episodes,success_rate=self.last_rate,reason=reason,streak=self.streak,failure_streak=self.failure_streak,
            criterion_rates=rates,cohort_size=self.cohort_size,cohort_completed=len(self.cohort_seen),
            cohort_rounds=self.cohort_rounds+int(closed),repeated_env_episodes=self.repeated_env_episodes,
            discarded_cross_level=self.discarded,adaptation_discarded=self.adaptation_discarded,
            leap_target=self.goal['leap'],highest_confirmed_speed=self.highest_confirmed_speed,
            highest_confirmed_stage=self.highest_confirmed_stage)
        if self.cfg.staged_core_first_enabled:
            event['core_safety_success_rate']=self.criteria.get('staged_regression',self.successes)/self.count if self.count else None
        if closed:self.cohort_seen.clear();self.cohort_rounds+=1
        if evaluate:self.count=self.successes=0;self.criteria={}
        self.events.append(event)
        if self.emit:self.emit('GALLOP_CURRICULUM',**event)
        return event


class RelearnCurriculum(SuccessSpeedCurriculum):
    """Walking constraints, gradual gait shaping, then confirmed-speed stages.

    Goals are global proposals. GallopState latches each goal on local reset;
    episodes from an older level cannot advance the current level.
    """
    def __init__(self,settings,emit=None):
        super().__init__(settings,emit)
        s=settings;self.index=0
        def goal(name,speed,gait,tolerance,seconds,fraction):
            return dict(name=name,speed=float(speed),gait=float(gait),request_tolerance=float(tolerance),
                request_seconds=float(seconds),request_fraction=float(fraction))
        self.plan=[goal('walk_adapt',s.start_speed,0,s.relearn_initial_request_tolerance,s.relearn_initial_request_seconds,.05),
            goal('walk_precision',s.start_speed,0,min(.06,s.relearn_initial_request_tolerance),.04,.03),
            goal('walk_limits',s.start_speed,0,.02,s.relearn_final_request_seconds,.01)]
        speed=s.start_speed+.25
        while speed<s.relearn_gallop_start_speed-1e-6:
            self.plan.append(goal('walk_speed',speed,0,.02,s.relearn_final_request_seconds,.01));speed+=.25
        self.plan.append(goal('walk_speed',s.relearn_gallop_start_speed,0,.02,s.relearn_final_request_seconds,.01))
        for gait,tol,fraction in ((.25,.02,.01),(.5,.01,.005),(.75,.005,.002)):
            self.plan.append(goal('gallop_shape',s.relearn_gallop_start_speed,gait,tol,s.relearn_final_request_seconds,fraction))
        speed=s.relearn_gallop_start_speed
        while speed<s.max_speed-1e-6:
            self.plan.append(goal('gallop_speed',speed,1,s.relearn_final_request_tolerance,s.relearn_final_request_seconds,0))
            speed=min(s.max_speed,speed+s.speed_increment)
        self.plan.append(goal('gallop_speed',s.max_speed,1,s.relearn_final_request_tolerance,s.relearn_final_request_seconds,0))
        if s.relearn_recovery_enabled:
            # Preserve the original goals and insert changes to one constraint
            # at a time. Existing checkpoints are mapped by goal, not index.
            refined=[]
            for index,item in enumerate(self.plan):
                if index==1:
                    refined.append(goal('walk_precision_duration',s.start_speed,0,item['request_tolerance'],s.relearn_initial_request_seconds,.03))
                if item['name']=='walk_limits':
                    refined.extend([goal('walk_precision_004',s.start_speed,0,.04,.04,.02),
                                    goal('walk_precision_002',s.start_speed,0,.02,.04,.01)])
                if item['name']=='gallop_shape' and item['gait'] in (.5,.75):
                    previous=refined[-1]
                    refined.append(goal('gallop_shape_adapt',item['speed'],item['gait'],previous['request_tolerance'],item['request_seconds'],previous['request_fraction']))
                if item['name']=='gallop_speed' and item['speed']==s.relearn_gallop_start_speed:
                    for tolerance in (.005,.002):
                        refined.append(goal('gallop_limits',item['speed'],1,tolerance,item['request_seconds'],0))
                refined.append(item)
            self.plan=refined
        self.cohort_size=None;self.cohort_seen=set();self.cohort_rounds=0;self.repeated_env_episodes=0

    def configure_envs(self,count):
        if not isinstance(count,int) or count<1:raise ValueError('Invalid curriculum environment count')
        self.cohort_size=count

    def record(self,success,level,criteria,env_id=None):
        if self.cfg.relearn_recovery_enabled and level==self.level:
            if self.cohort_size is None:raise RuntimeError('Configure curriculum environments before recording')
            if env_id is None or not 0<=env_id<self.cohort_size:raise ValueError('Missing curriculum environment id')
            if env_id in self.cohort_seen:
                self.repeated_env_episodes+=1;return
            self.cohort_seen.add(env_id)
        super().record(success,level,criteria)

    @property
    def goal(self):return self.plan[self.index]

    def restore(self,level,confirmed=None):
        if not isinstance(level,int) or not 0<=level<len(self.plan):raise ValueError('Invalid saved relearning level')
        if confirmed is not None and not self.cfg.relearn_gallop_start_speed<=confirmed<=self.cfg.max_speed:
            raise ValueError('Invalid confirmed Gallop speed')
        self.index=level;self.level=0;self.cap=self.goal['speed'];self.highest_confirmed_speed=confirmed
        self.streak=self.count=self.successes=0;self.criteria={}
        self.cohort_seen.clear();self.cohort_rounds=0

    def decide(self,step):
        if step<self.next_step:return None
        self.next_step=(step//self.cfg.decision_steps+1)*self.cfg.decision_steps
        before=self.cap;old_index=self.index;old_goal=self.goal.copy();reason='waiting_for_episodes'
        self.last_rate=self.successes/self.count if self.count else None
        closed=not self.cfg.relearn_recovery_enabled or (self.cohort_size is not None and len(self.cohort_seen)==self.cohort_size)
        evaluate=closed and self.count>=self.cfg.min_episodes
        if self.cfg.relearn_recovery_enabled and not closed:reason='waiting_for_cohort'
        if evaluate:
            if self.last_rate>=self.cfg.promote_success:
                self.streak+=1;reason='confirming_success'
                if self.streak>=self.cfg.pass_windows:
                    if old_goal['gait']==1:self.highest_confirmed_speed=max(self.highest_confirmed_speed or before,before)
                    self.index=min(self.index+1,len(self.plan)-1);self.streak=0
                    reason='at_limit' if self.index==old_index else 'promote' if self.goal['speed']>before else 'promote_stage'
            else:
                self.streak=0;reason='hold'
                if self.last_rate<self.cfg.regress_success and self.index>0:
                    self.index-=1;reason='regress'
        if self.index!=old_index:self.level+=1
        self.cap=self.goal['speed']
        event=dict(step=int(step),speed_before=before,speed_cap=self.cap,level=self.level,episodes=self.count,
            success_rate=self.last_rate,reason=reason,streak=self.streak,
            criterion_rates={k:v/self.count for k,v in self.criteria.items()} if self.count else {},
            discarded_cross_level=self.discarded,highest_confirmed_speed=self.highest_confirmed_speed,
            leap_target=float(stabilization_leap_target(torch.tensor(self.cap),self.cfg)),
            relearn_level_before=old_index,relearn_level=self.index,relearn_goal_before=old_goal,relearn_goal=self.goal.copy())
        if self.cfg.relearn_recovery_enabled:
            event.update(cohort_size=self.cohort_size,cohort_completed=len(self.cohort_seen),
                         cohort_rounds=self.cohort_rounds+int(closed),repeated_env_episodes=self.repeated_env_episodes)
        self.events.append(event)
        if self.emit:self.emit('GALLOP_CURRICULUM',**event)
        if evaluate:self.count=self.successes=0;self.criteria={}
        if self.cfg.relearn_recovery_enabled and closed:
            self.cohort_seen.clear();self.cohort_rounds+=1
        return event


class RelearnExploration:
    """Stage-scheduled Gaussian exploration, fixed within each PPO rollout.

    The sampled action and its stored Gaussian probability are untouched.
    Only the state-independent std parameter is frozen; actor/critic still learn.
    Changes happen before collection, never between collection and PPO update.
    """
    def __init__(self,runner,curriculum,emit=None):
        self.runner=runner;self.curriculum=curriculum;self.emit=emit;self.calls=0;self.last_level=None
        policy=runner.alg.policy
        if getattr(policy,'state_dependent_std',False):raise ValueError('Managed exploration needs state-independent std')
        self.parameter=getattr(policy,'log_std',None)
        self.logarithmic=self.parameter is not None
        if self.parameter is None:self.parameter=policy.std
        self.parameter.requires_grad_(False)
        runner.alg.entropy_coef=0.
        self.apply()
        original_act=runner.alg.act
        def act(obs):
            if self.calls%runner.num_steps_per_env==0:self.apply()
            self.calls+=1
            return original_act(obs)
        runner.alg.act=act

    def apply(self):
        goal=self.curriculum.goal;s=self.curriculum.cfg;tolerance=goal['request_tolerance']
        fraction=0 if tolerance>=.06 else 1/3 if tolerance>=.04 else 2/3 if tolerance>=.02 else 5/6 if tolerance>=.005 else 1
        self.std=s.relearn_noise_std_start+(s.relearn_noise_std_end-s.relearn_noise_std_start)*fraction
        with torch.no_grad():self.parameter.fill_(math.log(self.std) if self.logarithmic else self.std)
        if self.last_level!=self.curriculum.level:
            self.last_level=self.curriculum.level
            if self.emit:self.emit('RELEARN_EXPLORATION',relearn_level=self.curriculum.index,
                generation=self.last_level,noise_std=self.std,entropy_coef=0.,std_trainable=False,
                update_boundary='before complete PPO rollout')


def transplant_forward_policy(runner,checkpoint,parent_action_scale,new_action_scale):
    """Expand observations and change action units while preserving physical output.

    Previous-action inputs are rescaled inversely; new phase columns start at zero.
    Adam is deliberately fresh because matrix dimensions and the objective changed.
    """
    data=torch.load(checkpoint,map_location=runner.device,weights_only=True)
    old=data['model_state_dict'];new=runner.alg.policy.state_dict()
    if old['actor.0.weight'].shape[1]!=45 or old['critic.0.weight'].shape[1]!=60:
        raise ValueError('Expected a 45/60 forward policy as the Gallop parent')
    actor_dim=new['actor.0.weight'].shape[1];critic_dim=new['critic.0.weight'].shape[1]
    if (actor_dim,critic_dim) not in ((47,62),(49,64)):
        raise ValueError('Expected a 47/62 or 49/64 Gallop policy')
    if set(old)!=set(new):raise ValueError('Unexpected policy state keys or normalization state')
    ratio=parent_action_scale/new_action_scale
    last=max((k for k in old if k.startswith('actor.') and k.endswith('.weight')),key=lambda k:int(k.split('.')[1]))
    last_bias=last.replace('.weight','.bias')
    for key,value in old.items():
        if key in ('actor.0.weight','critic.0.weight'):
            dim=45 if key.startswith('actor') else 60
            migrated=torch.zeros_like(new[key]);migrated[:,:dim]=value
            migrated[:,dim-12:dim]/=ratio
        else:
            if value.shape!=new[key].shape:raise ValueError(f'Unexpected state shape: {key}')
            migrated=value.clone()
            if key in (last,last_bias,'std'):migrated*=ratio
        new[key]=migrated
    if runner.alg.optimizer.state:raise ValueError('Migration requires a fresh optimizer')
    runner.alg.policy.load_state_dict(new)
    runner.current_learning_iteration=0
    return dict(parent_iteration_zero_based=int(data['iter']),parent_stage_iterations=int(data['iter'])+1,
                actor_observations=[45,actor_dim],critic_observations=[60,critic_dim],actions=12,
                parent_action_scale=parent_action_scale,new_action_scale=new_action_scale,
                actor_output_and_std_multiplier=ratio,previous_action_input_multiplier=1/ratio,
                phase_columns_zero=True,optimizer_reset=True,stage_iteration_start=0)


def restore_gallop_policy(runner,checkpoint):
    """Keep trained tensors, optionally append two zero heading input columns."""
    data=torch.load(checkpoint,map_location=runner.device,weights_only=True)
    state=data['model_state_dict']
    old_dims=(state['actor.0.weight'].shape[1],state['critic.0.weight'].shape[1])
    new=runner.alg.policy.state_dict()
    new_dims=(new['actor.0.weight'].shape[1],new['critic.0.weight'].shape[1])
    if old_dims not in ((47,62),(49,64)) or (new_dims!=old_dims and (old_dims,new_dims)!=((47,62),(49,64))):
        raise ValueError('Expected compatible Gallop 47/62 or heading 49/64 observations')
    if set(new)!=set(state):raise ValueError('Unexpected policy state keys')
    for key,value in state.items():
        if new[key].shape==value.shape:new[key]=value
        elif key in ('actor.0.weight','critic.0.weight') and new[key].shape[1]==value.shape[1]+2:
            expanded=torch.zeros_like(new[key]);expanded[:,:value.shape[1]]=value;new[key]=expanded
        else:raise ValueError('Unexpected changed policy tensor '+key)
    if runner.alg.optimizer.state:raise ValueError('Fine-tuning requires a fresh optimizer')
    runner.alg.policy.load_state_dict(new,strict=True)
    runner.current_learning_iteration=0
    return dict(parent_iteration_zero_based=int(data['iter']),parent_stage_iterations=int(data['iter'])+1,
                actor_critic_restored=True,noise_restored=True,optimizer_reset=True,stage_iteration_start=0,
                actor_observations=list((old_dims[0],new_dims[0])),critic_observations=list((old_dims[1],new_dims[1])),
                added_heading_columns_zero=new_dims!=old_dims)
