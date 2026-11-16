"""MuJoCo plant adapter. Only this module may access evaluator ground truth."""
from pathlib import Path
from collections import deque
import mujoco as mj
import numpy as np
from . import ROOT, VENDOR
from .state import LowState
from .schedule import LEGS

class FlatPlant:
    def __init__(self, cfg):
        self.cfg = cfg
        scene = ROOT / "assets/flat_scene.xml"
        self.model = mj.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = cfg["simulation_dt"]
        self.data = mj.MjData(self.model)
        self.actuator_ids = np.array([mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{joint}") for leg in LEGS for joint in ("hip", "thigh", "calf")])
        joints = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint") for leg in LEGS for joint in ("hip", "thigh", "calf")]
        self.q_indices = self.model.jnt_qposadr[joints]
        self.dq_indices = self.model.jnt_dofadr[joints]
        self.foot_ids = {mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, leg): i for i, leg in enumerate(LEGS)}
        for geom in self.foot_ids:
            self.model.geom_friction[geom, 0] = cfg["friction"]
        self.floor = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_friction[self.floor, 0] = cfg["friction"]
        self.reset()

    def reset(self):
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:3] = [0, 0, self.cfg["height"] + 0.018]
        mj.mj_forward(self.model, self.data)
        ticks = round(self.cfg.get("command_delay_seconds", 0.0) / self.cfg["simulation_dt"])
        self.command_queue = deque([np.zeros(12) for _ in range(ticks)])

    def sensor(self, name):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"Missing sensor {name}")
        start, size = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
        return self.data.sensordata[start:start+size].copy()

    def foot_forces(self):
        force = np.zeros(4)
        contact_force = np.zeros(6)
        for j in range(self.data.ncon):
            contact = self.data.contact[j]
            if contact.geom1 == self.floor:
                foot = contact.geom2
            elif contact.geom2 == self.floor:
                foot = contact.geom1
            else:
                continue
            if foot in self.foot_ids:
                mj.mj_contactForce(self.model, self.data, j, contact_force)
                force[self.foot_ids[foot]] += max(0, contact_force[0])
        return force

    def low_state(self):
        return LowState(float(self.data.time), self.sensor("imu_quat"), self.sensor("imu_gyro"), self.sensor("imu_acc"),
                        self.data.qpos[self.q_indices].copy(), self.data.qvel[self.dq_indices].copy(), self.foot_forces())

    def step(self, torque):
        for _ in range(round(self.cfg["control_dt"] / self.cfg["simulation_dt"])):
            self.command_queue.append(np.asarray(torque).copy())
            self.data.ctrl[self.actuator_ids] = self.command_queue.popleft()
            mj.mj_step(self.model, self.data)

    def truth(self):
        import pinocchio as pin
        quat = self.data.qpos[3:7]
        R = pin.Quaternion(*quat).toRotationMatrix()
        rpy = np.asarray(pin.rpy.matrixToRpy(R))
        return np.r_[self.data.qpos[:3], rpy, self.data.qvel[:3], R @ self.data.qvel[3:6]]

    def unhealthy(self):
        x = self.truth()
        return (not np.isfinite(x).all() or x[2] < self.cfg["min_height"] or x[2] > self.cfg["max_height"] or np.max(np.abs(x[3:5])) > self.cfg["roll_pitch_limit"])
