"""Paper Eq.4-8,12-13, with world-frame GRF constraints and explicit projection."""
import contextlib
import io
import time
import casadi as ca
import numpy as np
from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj

def skew(x):
    a, b, c = x
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])

def rpy_rate_world(rpy):
    # ZYX Euler rates from angular velocity expressed in world coordinates.
    roll, pitch, yaw = rpy
    cp = np.cos(pitch)
    if abs(cp) < 0.1:
        raise ValueError("Attitude too close to Euler singularity")
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy / cp, sy / cp, 0], [-sy, cy, 0], [cy * np.tan(pitch), sy * np.tan(pitch), 1]])

def friction_matrix(mu):
    return np.kron(np.eye(4), np.array([[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu]]))

def project_forces(force, mask, mu, maximum):
    f = np.asarray(force, dtype=float).reshape(4, 3).copy()
    f[~mask] = 0
    f[:, 2] = np.clip(f[:, 2], 0, maximum)
    f[:, :2] = np.clip(f[:, :2], -mu * f[:, 2, None], mu * f[:, 2, None])
    tangent = np.linalg.norm(f[:, :2], axis=1)
    scale = np.minimum(1.0, mu*f[:, 2]/np.maximum(tangent, 1e-12))
    f[:, :2] *= scale[:, None]
    return f.reshape(12)

class GallopTrajectory(ComTraj):
    def __init__(self, robot, cfg):
        self.cfg = cfg
        self.N = cfg["horizon_steps"]

    def update(self, robot, schedule):
        dt, N = self.cfg["mpc_dt"], self.N
        self.initial_x_vec = robot.compute_com_x_vec().copy()
        x = self.initial_x_vec[:, 0]
        self.m = float(robot.data.Ig.mass)
        self.I_com_world = np.asarray(robot.data.Ig.inertia)
        t = (np.arange(N) + 1) * dt
        v = robot.R_z @ np.array([self.cfg["speed"], 0, 0])
        self.pos_traj_world = x[:3, None] + v[:, None] * t
        self.pos_traj_world[2] = schedule.height_reference
        self.vel_traj_world = np.repeat(v[:, None], N, axis=1)
        self.rpy_traj_world = np.zeros((3, N))
        self.rpy_traj_world[1] = schedule.pitch_reference
        self.rpy_traj_world[2] = x[5]
        self.omega_traj_world = np.zeros((3, N))
        self.contact_table = schedule.table(dt, N).astype(int)
        feet = np.asarray(robot.get_foot_placement_in_world())
        # Anchor present stance feet; future touchdown lever uses nominal geometry.
        r = np.empty((4, 3, N))
        lever = feet - x[:3]
        for i in range(4):
            for k in range(N):
                delta = k * dt
                if schedule.mask()[i] and delta < (schedule.duty[i] - schedule.local_phase()[i]) * schedule.gait_period:
                    r[i, :, k] = feet[i] - (x[:3] + v * delta)
                else:
                    local = schedule.local_phase(delta)[i]
                    hip = robot.R_z @ robot.get_hip_offset(("FL", "FR", "RL", "RR")[i])
                    nominal = robot.current_config.base_pos - x[:3] + hip
                    nominal[:2] += v[:2] * (schedule.stance_seconds(i)/2 - local * schedule.gait_period)
                    nominal[:2] += robot.R_z[:2, 0] * self.cfg.get("fore_foothold_bias" if i < 2 else "hind_foothold_bias", 0.0)
                    nominal[2] = 0.022 - schedule.height_reference
                    r[i, :, k] = nominal
        self.r_fl_foot_world, self.r_fr_foot_world, self.r_rl_foot_world, self.r_rr_foot_world = r
        self._discreteDynamics(dt)
        self.Ad[3:6, 9:12] = dt * rpy_rate_world(x[3:6])

class BoundedMPC(CentroidalMPC):
    def __init__(self, robot, traj, cfg):
        self.cfg = cfg
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(robot, traj)
        self.Q = np.diag(cfg["mpc_q"])
        with contextlib.redirect_stdout(io.StringIO()):
            self._build_sparse_matrix(traj)

    def _precompute_friction_matrix(self, traj):
        import scipy.sparse as sp
        F = sp.block_diag([friction_matrix(self.cfg["friction"])] * traj.N)
        return self._scipy_to_casadi(sp.hstack([sp.csc_matrix((16 * traj.N, 12 * traj.N)), F]))

    def _compute_bounds(self, traj):
        low = np.full(self.nvars, -np.inf)
        high = np.full(self.nvars, np.inf)
        for k in range(self.N):
            for i in range(4):
                j = self.N * 12 + k * 12 + i * 3
                if not traj.contact_table[i, k]:
                    low[j:j+3] = high[j:j+3] = 0
                else:
                    low[j+2] = 0
                    high[j+2] = self.cfg["mpc_force_max"]
        return ca.DM(low), ca.DM(high)

class GpdAttitudeQP:
    def __init__(self, cfg):
        self.cfg = cfg
        self.F = friction_matrix(cfg["friction"])
        qp = {"h": ca.Sparsity.dense(12, 12), "a": ca.DM(self.F).sparsity()}
        opts = {"error_on_fail": False, "osqp": {"verbose": False, "eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 3000}}
        self.solver = ca.conic("gpd", "osqp", qp, opts)
        self.last = np.zeros(12)

    def solve(self, robot, schedule):
        mask = schedule.mask()
        if not mask.any():
            return np.zeros(12)
        x = robot.compute_com_x_vec()[:, 0]
        xi = np.r_[x[3:6], x[9:12]]
        target = np.array([0, schedule.pitch_reference, x[5], 0, 0, 0])
        h = self.cfg["gpd_predict_dt"]
        Rr = rpy_rate_world(x[3:6])
        Ad = np.eye(6)
        Ad[:3, 3:] = h * Rr
        invI = np.linalg.inv(np.asarray(robot.data.Ig.inertia))
        torque_map = np.hstack([invI @ skew(r) for r in robot.get_foot_lever_world()])
        Bd = np.vstack([0.5 * h*h * Rr @ torque_map, h * torque_map])
        S = np.diag(self.cfg["gpd_s"])
        H = Bd.T @ S @ Bd + self.cfg["gpd_w"] * np.eye(12)
        g = Bd.T @ S @ (Ad @ xi - target)
        nc = mask.sum()
        mg = float(robot.data.Ig.mass) * 9.81
        alpha = min(1 + max(0, self.cfg["speed"] - self.cfg["gpd_v_initial"]) / 2, 2)
        lb, ub = np.full(12, -np.inf), np.full(12, np.inf)
        for i in range(4):
            if mask[i]:
                lb[3*i+2] = self.cfg["gpd_force_min_factor"] * mg / nc
                ub[3*i+2] = max(lb[3*i+2], alpha * mg / nc)
            else:
                lb[3*i:3*i+3] = ub[3*i:3*i+3] = 0
        sol = self.solver(h=ca.DM(H), g=ca.DM(g), a=ca.DM(self.F), lba=-np.inf, uba=0, lbx=lb, ubx=ub, x0=self.last)
        if not self.solver.stats().get("success", False):
            raise RuntimeError("GPD QP did not converge")
        self.last = sol["x"].full().ravel()
        return project_forces(self.last, mask, self.cfg["friction"], self.cfg["mpc_force_max"])
