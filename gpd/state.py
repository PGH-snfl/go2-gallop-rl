from dataclasses import dataclass
import numpy as np
from convex_mpc.go2_robot_data import PinGo2Model

@dataclass
class LowState:
    timestamp: float
    quaternion: np.ndarray  # wxyz, body -> world
    gyro: np.ndarray        # body frame
    acceleration: np.ndarray  # specific force, body frame
    q: np.ndarray          # FL, FR, RL, RR / hip, thigh, calf
    dq: np.ndarray
    foot_force: np.ndarray # same leg order, normal-force magnitude

class ContactEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mask = np.zeros(4, dtype=bool)
        self.age = np.zeros(4)
        self.filtered = np.zeros(4)

    def update(self, force, dt):
        alpha = dt / (self.cfg.get("contact_filter_seconds", 0.01) + dt)
        self.filtered += alpha * (force - self.filtered)
        candidate = np.where(self.mask, self.filtered > self.cfg["contact_off"], self.filtered > self.cfg["contact_on"])
        self.age = np.where(candidate != self.mask, self.age + dt, 0)
        change = self.age + 1e-10 >= self.cfg["contact_dwell"]
        self.mask[change] = candidate[change]
        self.age[change] = 0
        return self.mask.copy()

class StateEstimator:
    """IMU propagation + contact-aided leg odometry. No simulator root-state input."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = PinGo2Model()
        self.position = np.array([0., 0., cfg["height"]])
        self.velocity = np.zeros(3)
        self.last_time = None
        self.last_feet_body = None
        self.contact = ContactEstimator(cfg)

    def update(self, state):
        fields = (state.quaternion, state.gyro, state.acceleration, state.q, state.dq, state.foot_force)
        if not np.isfinite(state.timestamp) or not all(np.isfinite(a).all() for a in fields):
            raise ValueError("Nonfinite LowState")
        norm = np.linalg.norm(state.quaternion)
        if norm < 0.5 or norm > 1.5:
            raise ValueError("Invalid IMU quaternion")
        quat = state.quaternion / norm
        import pinocchio as pin
        R = pin.Quaternion(*map(float, quat)).toRotationMatrix()
        dt = self.cfg["control_dt"] if self.last_time is None else state.timestamp - self.last_time
        if dt <= 0 or dt > self.cfg["state_timeout"]:
            raise ValueError("Stale or nonmonotonic LowState")
        self.last_time = state.timestamp
        mask = self.contact.update(state.foot_force, dt)
        acceleration = R @ state.acceleration + np.array([0, 0, -9.81])
        self.velocity += acceleration * dt
        q = np.r_[self.position, quat[[1, 2, 3, 0]], state.q]
        # Zero translation velocity isolates kinematic relative foot velocity.
        self.robot.update_model(q, np.r_[np.zeros(3), state.gyro, state.dq])
        if mask.any():
            feet = np.asarray(self.robot.get_foot_placement_in_world())
            feet_velocity = np.array([self.robot.get_single_foot_state_in_world(l)[1] for l in ("FL", "FR", "RL", "RR")])
            leg_velocity = -feet_velocity[mask].mean(axis=0)
            gain = self.cfg["estimator_contact_gain"]
            self.velocity = (1-gain) * self.velocity + gain * leg_velocity
            height = self.cfg.get("foot_radius", 0.022) - (feet[mask, 2] - self.position[2]).mean()
            self.position[2] = (1-gain) * self.position[2] + gain * height
        self.position += self.velocity * dt
        q[:3] = self.position
        self.robot.update_model(q, np.r_[R.T @ self.velocity, state.gyro, state.dq])
        return self.robot, mask
