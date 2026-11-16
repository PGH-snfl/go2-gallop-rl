import time
import numpy as np
from convex_mpc.gait import Gait
from .state import StateEstimator
from .schedule import GallopSchedule, LEGS
from .qp import GallopTrajectory, BoundedMPC, GpdAttitudeQP, project_forces

class GallopController:
    """Transport-independent runtime, also used by the PPO environment."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.estimator = StateEstimator(cfg)
        self.schedule = GallopSchedule(cfg)
        self.trajectory = GallopTrajectory(self.estimator.robot, cfg)
        self.trajectory.update(self.estimator.robot, self.schedule)
        self.mpc = BoundedMPC(self.estimator.robot, self.trajectory, cfg)
        self.gpd = GpdAttitudeQP(cfg)
        self.force = np.zeros(12)
        self.last_mask = np.ones(4, dtype=bool)
        self.swing = {}
        self.ticks = 0
        self.failures = 0
        self.fault = None
        self.last_action = np.zeros(5)
        self.last_torque = np.zeros(12)
        thigh = np.arccos(np.clip(cfg["height"] / 0.426, 0.1, 0.95))
        self.nominal_q = np.tile([0, thigh, -2*thigh], 4)
        self.last_solver_ms = 0.0
        self.last_gallop_time = None

    def damping(self, dq):
        limits = np.tile(self.cfg["torque_limits"], 4)
        return np.clip(-2*np.nan_to_num(dq, nan=0.0, posinf=0.0, neginf=0.0), -limits, limits)

    def observation(self):
        robot = self.estimator.robot
        R = robot.R_body_to_world
        x = robot.compute_com_x_vec()[:, 0]
        phase = self.schedule.phase * 2 * np.pi
        return np.r_[R.T @ [0, 0, -1], robot.current_config.base_ang_vel,
                     R.T @ self.estimator.velocity,
                     robot.current_config.get_q()[7:] - self.nominal_q,
                     robot.current_config.get_dq()[6:], self.last_action,
                     self.cfg["speed"], np.sin(phase), np.cos(phase),
                     self.schedule.mask(), self.estimator.contact.mask,
                     self.schedule.residual, x[2] - self.schedule.height_reference,
                     x[4] - self.schedule.pitch_reference, min(self.failures, 3)].astype(np.float32)

    def set_action(self, action):
        self.schedule.apply(action, self.cfg["policy_dt"])
        self.last_action = np.clip(np.asarray(action), -1, 1)

    def step(self, state):
        started = time.perf_counter()
        try:
            robot, measured = self.estimator.update(state)
            x = robot.compute_com_x_vec()[:, 0]
            if state.timestamp >= self.cfg["settle_seconds"]:
                if np.max(np.abs(x[3:5])) > self.cfg["roll_pitch_limit"]:
                    raise ValueError("Attitude limit")
                if not self.cfg["min_height"] < self.estimator.position[2] < self.cfg["max_height"]:
                    raise ValueError("Estimated height limit")
            if self.fault:
                raise ValueError(self.fault)
        except (ValueError, FloatingPointError) as e:
            self.fault = str(e)
            torque = self.damping(state.dq)
            return torque, {"fault": self.fault, "cycle_ms": (time.perf_counter()-started)*1000}
        if state.timestamp < self.cfg["settle_seconds"]:
            torque = 45 * (self.nominal_q - state.q) - 3 * state.dq
            for i, leg in enumerate(LEGS):
                J = robot.compute_3x3_foot_Jacobian_world(leg)
                torque[3*i:3*i+3] -= J.T @ np.array([0, 0, float(robot.data.Ig.mass)*9.81/4])
            return np.clip(torque, -np.tile(self.cfg["torque_limits"], 4), np.tile(self.cfg["torque_limits"], 4)), {"fault": None, "settling": True, "cycle_ms": (time.perf_counter()-started)*1000}
        if self.last_gallop_time is not None:
            self.schedule.advance(state.timestamp-self.last_gallop_time)
        self.last_gallop_time = state.timestamp
        mask = self.schedule.mask()
        try:
            mpc_stride = round(self.cfg["mpc_dt"] / self.cfg["control_dt"])
            if self.ticks % mpc_stride == 0 or not np.array_equal(mask, self.last_mask):
                t = time.perf_counter()
                self.trajectory.update(robot, self.schedule)
                sol = self.mpc.solve_QP(robot, self.trajectory)
                if not self.mpc.solver.stats().get("success", False):
                    raise RuntimeError("MPC QP did not converge")
                self.force = sol["x"].full().ravel()[12*self.trajectory.N:12*self.trajectory.N+12]
                self.last_solver_ms = (time.perf_counter()-t)*1000
            gpd_force = self.gpd.solve(robot, self.schedule)
            gamma = self.cfg["gpd_gamma"]
            force = gamma * gpd_force + (1-gamma) * self.force
            self.failures = 0
        except RuntimeError:
            self.failures += 1
            force = self.force.copy()
            self.set_action(np.zeros(5))
            if self.failures >= self.cfg["max_solver_failures"]:
                self.fault = "Repeated QP failure"
                return self.damping(state.dq), {"fault": self.fault, "cycle_ms": (time.perf_counter()-started)*1000}
        # Optional stance impedance is a project whole-body command-conversion
        # choice, not a change to the paper's GPD equations. Project afterwards.
        stance_kp = self.cfg.get("stance_kp", 0.0)
        stance_kd = self.cfg.get("stance_kd", 0.0)
        if stance_kp or stance_kd:
            import pinocchio as pin
            R = robot.R_body_to_world
            R_des = pin.rpy.rpyToMatrix(0, self.schedule.pitch_reference, x[5])
            p_des = self.estimator.position.copy()
            p_des[2] += self.schedule.height_reference - x[2]
            v_des = robot.R_z @ np.array([self.cfg["speed"], 0, 0])
            for i, leg in enumerate(LEGS):
                if mask[i]:
                    p, _ = robot.get_single_foot_state_in_world(leg)
                    rel = R.T @ (p-self.estimator.position)
                    rel_des = R_des.T @ (p-p_des)
                    Jbody = robot.compute_3x3_foot_Jacobian_body(leg)
                    rel_v = Jbody @ state.dq[3*i:3*i+3]
                    adjustment = stance_kp * (rel_des-rel) + stance_kd * (-R_des.T @ v_des-rel_v)
                    force[3*i:3*i+3] -= R @ adjustment
        force = project_forces(force, mask, self.cfg["friction"], self.cfg["mpc_force_max"])
        torque = np.zeros(12)
        R = robot.R_body_to_world
        velocity = self.estimator.velocity
        for i, leg in enumerate(LEGS):
            J = robot.compute_3x3_foot_Jacobian_world(leg)
            if mask[i]:
                torque[3*i:3*i+3] = -J.T @ force[3*i:3*i+3]
            else:
                p, v = robot.get_single_foot_state_in_world(leg)
                if self.last_mask[i] or i not in self.swing:
                    remain = self.schedule.swing_remaining(i)
                    hip = robot.get_hip_offset(leg)
                    target = self.estimator.position + R @ hip
                    target[:2] += velocity[:2] * (remain + self.schedule.stance_seconds(i)/2)
                    target[:2] += self.cfg["foot_velocity_gain"] * (velocity[:2] - robot.R_z[:2, 0] * self.cfg["speed"])
                    target[:2] += robot.R_z[:2, 0] * self.cfg.get("fore_foothold_bias" if i < 2 else "hind_foothold_bias", 0.0)
                    target[2] = 0.022
                    traj = Gait.make_swing_trajectory(None, p, target, remain, self.cfg["swing_height"])
                    self.swing[i] = (self.schedule.elapsed, traj)
                takeoff, traj = self.swing[i]
                pd, vd, ad = traj(self.schedule.elapsed - takeoff)
                cartesian = self.cfg["swing_kp"] * (pd - p) + self.cfg["swing_kd"] * (vd - v)
                feedforward = self.cfg.get("swing_feedforward", 1.0)
                if feedforward:
                    full_j = robot.compute_full_foot_Jacobian_world(leg)
                    mass = np.asarray(robot.data.M)
                    effective_inv = full_j @ np.linalg.solve(mass, full_j.T)
                    cartesian += feedforward * np.linalg.solve(effective_inv + 1e-6*np.eye(3), ad - robot.compute_Jdot_dq_world(leg))
                torque[3*i:3*i+3] = J.T @ cartesian + robot.data.g[6+3*i:9+3*i]
        limits = np.tile(self.cfg["torque_limits"], 4)
        velocity_limits = np.tile(self.cfg["velocity_limits"], 4)
        raw = torque.copy()
        # Prevent outward acceleration near position and velocity limits.
        lower, upper = robot.model.lowerPositionLimit[7:], robot.model.upperPositionLimit[7:]
        outward = ((state.q < lower + 0.02) & (torque < 0)) | ((state.q > upper - 0.02) & (torque > 0)) | ((np.abs(state.dq) > velocity_limits) & (torque * state.dq > 0))
        torque[outward] = -2 * state.dq[outward]
        torque = np.clip(torque, -limits, limits)
        if not np.isfinite(torque).all():
            self.fault = "Nonfinite motor command"
            torque[:] = 0
        info = {"fault": self.fault, "desired": mask.copy(), "measured": measured,
                "phase": self.schedule.phase, "force": force, "x_est": x,
                "clip_fraction": np.mean(np.abs(raw) > limits),
                "solver_ms": self.last_solver_ms, "solver_failures": self.failures,
                "cycle_ms": (time.perf_counter()-started)*1000}
        self.last_mask = mask.copy()
        self.last_torque = torque.copy()
        self.ticks += 1
        return torque, info
