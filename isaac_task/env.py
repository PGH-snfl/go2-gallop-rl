"""PhysX plant with the same sensor-only GPD controller used in MuJoCo deployment.

The CPU QP is intentionally retained: start with 4 environments, not thousands.
No MuJoCo or SB3 imports belong in this module.
"""
import numpy as np
import time
from collections import Counter
import torch
from pathlib import Path
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from gpd.controller import GallopController
from gpd.state import LowState
from gpd.schedule import LEGS
from gpd import ROOT

JOINTS = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
USD = ROOT.parents[1] / "unitree/unitree_rl_lab/unitree_model/Go2/usd/go2.usd"

@configclass
class GallopIsaacCfg(DirectRLEnvCfg):
    decimation = 25
    episode_length_s = 10.5
    action_space = 5
    observation_space = 57
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=.001, render_interval=25,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8,
            friction_combine_mode="average", restitution=0.))
    scene = InteractiveSceneCfg(num_envs=4, env_spacing=5., replicate_physics=True)
    robot = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.288),
            joint_pos={".*_hip_joint":0., ".*_thigh_joint":.9, ".*_calf_joint":-1.8}),
        actuators={"legs":IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0.,
            effort_limit={".*_hip_joint":23.7,".*_thigh_joint":23.7,".*_calf_joint":45.43},
            velocity_limit={".*_hip_joint":30.1,".*_thigh_joint":30.1,".*_calf_joint":15.7})})

class GallopIsaacEnv(DirectRLEnv):
    def __init__(self, cfg, controller_cfg, record=False, log_callback=None, log_every=50, total_steps=None, follow_camera=False, **kwargs):
        self.follow_camera=follow_camera
        self.log_callback = log_callback
        self.log_every = log_every
        self.total_steps = total_steps
        self.run_steps = 0
        self.reset_reasons = Counter()
        self.run_started = self.last_progress = time.monotonic()
        self.control_cfg = dict(controller_cfg)
        self.record = record
        self.episodes = []
        super().__init__(cfg, **kwargs)
        self.joint_ids, names = self.robot.find_joints(JOINTS, preserve_order=True)
        if names != JOINTS:
            raise ValueError(f"Unexpected Go2 joint order: {names}")
        self.foot_ids, names = self.contact.find_bodies([f"{leg}_foot" for leg in LEGS], preserve_order=True)
        if len(self.foot_ids) != 4:
            raise ValueError(f"Missing foot contact bodies: {names}")
        self.controllers = [GallopController(self.control_cfg) for _ in range(self.num_envs)]
        self.clock_ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.records = [[] for _ in range(self.num_envs)]
        self.torques = torch.zeros((self.num_envs,12), device=self.device)
        self.previous_torque = self.torques.clone()
        self.reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.faults = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.control_stride = round(self.control_cfg["control_dt"]/self.physics_dt)
        self.control_updates = round(self.control_cfg["policy_dt"]/self.control_cfg["control_dt"])
        materials = self.robot.root_physx_view.get_material_properties()
        materials[...,0:2] = self.control_cfg["friction"]
        materials[...,2] = 0.
        self.robot.root_physx_view.set_material_properties(materials,torch.arange(self.num_envs,dtype=torch.int32))

    def step(self, actions):
        result = super().step(actions)
        self.run_steps += 1
        now = time.monotonic()
        if self.log_callback and (self.run_steps==1 or self.run_steps%self.log_every==0
                or now-self.last_progress>=5 or self.run_steps==self.total_steps):
            x = self._truth().detach().cpu().numpy()
            elapsed = now-self.run_started
            completed = sum(self.reset_reasons.values())
            c = self.controllers[0]
            self.log_callback("PROGRESS",step=self.run_steps,total_steps=self.total_steps,
                progress_pct=round(100*self.run_steps/self.total_steps,1) if self.total_steps else None,
                transitions_per_s=round(self.run_steps*self.num_envs/max(elapsed,.001),1),
                mean_vx=round(float(x[:,6].mean()),3),
                height_range=[round(float(x[:,2].min()),3),round(float(x[:,2].max()),3)],
                max_roll_pitch_deg=round(float(np.rad2deg(np.abs(x[:,3:5])).max()),1),
                env0_phase=round(c.schedule.phase,3),env0_contacts=c.estimator.contact.mask.astype(int).tolist(),
                env0_desired=c.schedule.mask().astype(int).tolist(),
                resets=completed,reset_reasons=dict(self.reset_reasons))
            self.last_progress = now
        return result

    def render(self, recompute=False):
        if self.follow_camera:
            # Keep environment 0 in frame across locomotion and automatic resets.
            target=self.robot.data.root_pos_w[0].detach().cpu().numpy()+np.array([0.,0.,.05])
            self.sim.set_camera_view(target+np.array([1.4,-1.8,.9]),target)
        return super().render(recompute=recompute)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg(
            physics_material=self.cfg.sim.physics_material))
        # Find the actual base prim in the local official USD, rather than assuming its name.
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        base = [p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/envs/env_0/Robot/")
                and p.GetName() in ("base", "base_link", "trunk") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(base) != 1:
            raise ValueError("Cannot uniquely identify Go2 IMU base prim")
        base_path = str(base[0].GetPath()).replace("env_0/", "env_.*/")
        self.contact = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*_foot",
            update_period=0., history_length=1))
        # Same IMU offset as the MuJoCo model; body orientation is unchanged.
        self.imu = Imu(ImuCfg(prim_path=base_path, update_period=0.,
            offset=ImuCfg.OffsetCfg(pos=(-.02557,0.,.04232)), gravity_bias=(0.,0.,9.81)))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["feet"] = self.contact
        self.scene.sensors["imu"] = self.imu
        sim_utils.DomeLightCfg(intensity=1500.).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.))

    def _pre_physics_step(self, actions):
        self.actions = actions.clamp(-1,1).clone()
        self.reward_sum.zero_()
        for i, action in enumerate(self.actions.detach().cpu().numpy()):
            # No residual during standing initialization, matching deployment.
            if self.clock_ticks[i]*self.physics_dt >= self.control_cfg["settle_seconds"]:
                self.controllers[i].set_action(action)

    def _apply_action(self):
        # Per-environment clocks reset independently; all resets occur at policy boundaries.
        if self.clock_ticks[0] % self.control_stride == 0:
            imu = self.imu.data
            q = self.robot.data.joint_pos[:,self.joint_ids].detach().cpu().numpy()
            dq = self.robot.data.joint_vel[:,self.joint_ids].detach().cpu().numpy()
            quat = imu.quat_w.detach().cpu().numpy()
            gyro = imu.ang_vel_b.detach().cpu().numpy()
            accel = imu.lin_acc_b.detach().cpu().numpy()
            forces = self.contact.data.net_forces_w[:,self.foot_ids,2].clamp_min(0).detach().cpu().numpy()
            truth = self._truth().detach().cpu().numpy()
            old = self.torques.detach().cpu().numpy()
            torque = np.zeros_like(old)
            reward = np.zeros(self.num_envs)
            for i,c in enumerate(self.controllers):
                at = self.clock_ticks[i]*self.physics_dt
                state = LowState(at,quat[i],gyro[i],accel[i],q[i],dq[i],forces[i])
                torque[i],info = c.step(state)
                self.faults[i] = bool(info.get("fault"))
                mask = c.estimator.contact.mask.copy()
                desired = c.schedule.mask()
                x = truth[i]
                reward[i] = (2*np.exp(-((x[6]-self.control_cfg["speed"])/.4)**2)
                    + .3*np.exp(-(x[7]/.3)**2) + .5*np.mean(mask == desired)
                    - 2*(np.sum(x[3:5]**2)+.25*x[11]**2)
                    - .001*np.abs(torque[i]*dq[i]).sum()
                    - .1*np.mean(((torque[i]-old[i])/np.tile(self.control_cfg["torque_limits"],4))**2)
                    - .05*np.mean((c.schedule.residual/np.asarray(self.control_cfg["residual_bounds"]))**2))
                if at < self.control_cfg["settle_seconds"]:
                    reward[i] = 0.
                if self.record and at >= self.control_cfg["settle_seconds"]:
                    self.records[i].append(dict(time=at, truth=x.copy(), estimated=info.get("x_est",np.zeros(12)),
                        desired=desired, measured=mask, foot_force=forces[i].copy(), torque=torque[i].copy(),
                        dq=dq[i].copy(), phase=c.schedule.phase, clip_fraction=info.get("clip_fraction",0.),
                        cycle_ms=info.get("cycle_ms",0.), fault=info.get("fault")))
            self.torques[:] = torch.as_tensor(torque,device=self.device)
            self.reward_sum += torch.as_tensor(reward,device=self.device,dtype=torch.float32)/self.control_updates
        self.robot.set_joint_effort_target(self.torques, joint_ids=self.joint_ids)
        self.clock_ticks += 1

    def _truth(self):
        data = self.robot.data
        roll,pitch,yaw = euler_xyz_from_quat(data.root_quat_w)
        rpy = torch.stack((roll,pitch,yaw),dim=1)
        rpy = torch.remainder(rpy+torch.pi,2*torch.pi)-torch.pi
        return torch.cat((data.root_pos_w-self.scene.env_origins,rpy,data.root_lin_vel_w,data.root_ang_vel_w),dim=1)

    def _get_observations(self):
        obs = np.stack([c.observation() for c in self.controllers])
        return {"policy":torch.as_tensor(obs,device=self.device)}

    def _get_dones(self):
        x = self._truth()
        fail = (self.faults | ~torch.isfinite(x).all(dim=1)
            | (x[:,2]<self.control_cfg["min_height"]) | (x[:,2]>self.control_cfg["max_height"])
            | (x[:,3:5].abs().amax(dim=1)>self.control_cfg["roll_pitch_limit"]))
        return fail, self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        return self.reward_sum-10*self.reset_terminated.float()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self.log_callback:
            x = self._truth().detach().cpu().numpy()
            for i in env_ids.tolist():
                if self.clock_ticks[i] == 0:
                    continue  # Initial scene reset is not a failed episode.
                reasons=[]
                if self.controllers[i].fault:
                    reasons.append(self.controllers[i].fault)
                if not np.isfinite(x[i]).all(): reasons.append("nonfinite_state")
                if x[i,2]<self.control_cfg["min_height"]: reasons.append("low_height")
                if x[i,2]>self.control_cfg["max_height"]: reasons.append("high_height")
                if np.abs(x[i,3:5]).max()>self.control_cfg["roll_pitch_limit"]: reasons.append("attitude_limit")
                if self.reset_time_outs[i]: reasons.append("time_limit")
                reason=" + ".join(reasons) or "reset"
                self.reset_reasons[reason] += 1
                self.log_callback("EPISODE_END",env_id=i,
                    episode_seconds=round(float(self.clock_ticks[i]*self.physics_dt),3),reason=reason,
                    vx=float(x[i,6]),height=float(x[i,2]),terminated=bool(self.reset_terminated[i]))
        if self.record:
            from gpd.metrics import summarize
            for i in env_ids.tolist():
                if self.records[i]:
                    self.episodes.append({"env_id":i,"report":summarize(self.records[i],self.control_cfg,
                        bool(self.reset_terminated[i]),transport="isaaclab_physx"),"records":self.records[i]})
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:,:3] += self.scene.env_origins[env_ids]
        pos = self.robot.data.default_joint_pos[env_ids].clone()
        vel = torch.zeros_like(pos)
        for i in env_ids.tolist():
            self.controllers[i] = GallopController(self.control_cfg)
            self.clock_ticks[i] = 0
            self.records[i] = []
        pos[:,self.joint_ids] = torch.as_tensor(self.controllers[0].nominal_q,device=self.device,dtype=pos.dtype)
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids)
        self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
        self.robot.write_joint_state_to_sim(pos,vel,env_ids=env_ids)
        self.torques[env_ids] = 0
        self.faults[env_ids] = False
