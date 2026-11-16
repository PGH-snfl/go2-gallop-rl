import gymnasium as gym
import numpy as np
from .controller import GallopController
from .sim import FlatPlant

class GallopEnv(gym.Env):
    """MuJoCo evaluation adapter only. Training uses isaac_task.env.GallopIsaacEnv."""
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.plant = FlatPlant(self.cfg)
        self.controller = GallopController(self.cfg)
        self.action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
        obs = self.controller.observation()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.records = []
        self.previous_torque = np.zeros(12)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.plant.reset()
        self.controller = GallopController(self.cfg)
        self.records = []
        self.previous_torque[:] = 0
        # Standing initialisation only; no trot reference and no gait transition policy.
        for _ in range(round(self.cfg["settle_seconds"] / self.cfg["control_dt"])):
            torque, _ = self.controller.step(self.plant.low_state())
            self.plant.step(torque)
        return self.controller.observation(), {}

    def step(self, action):
        try:
            self.controller.set_action(action)
        except ValueError as e:
            self.controller.fault = str(e)
        reward = 0.0
        terminated = False
        info = {}
        for _ in range(round(self.cfg["policy_dt"] / self.cfg["control_dt"])):
            state = self.plant.low_state()
            torque, info = self.controller.step(state)
            truth = self.plant.truth()
            measured = self.controller.estimator.contact.mask.copy()
            desired = info.get("desired", np.ones(4, dtype=bool))
            mismatch = np.mean(measured != desired)
            velocity_reward = np.exp(-((truth[6] - self.cfg["speed"])/0.4)**2)
            lateral_reward = np.exp(-(truth[7]/0.3)**2)
            attitude_penalty = np.sum(truth[3:5]**2) + 0.25 * truth[11]**2
            power = np.sum(np.abs(torque * state.dq))
            delta = np.mean(((torque - self.previous_torque)/np.tile(self.cfg["torque_limits"], 4))**2)
            value = 2 * velocity_reward + 0.3 * lateral_reward + 0.5 * (1-mismatch) - 2 * attitude_penalty - 0.001 * power - 0.1 * delta - 0.05 * np.mean(self.controller.schedule.residual**2 / np.asarray(self.cfg["residual_bounds"])**2)
            reward += value / round(self.cfg["policy_dt"] / self.cfg["control_dt"])
            self.records.append({"time": state.timestamp, "truth": truth.copy(), "estimated": info.get("x_est", np.zeros(12)).copy(), "desired": desired.copy(), "measured": measured, "foot_force": state.foot_force.copy(), "torque": torque.copy(), "dq": state.dq.copy(), "phase": info.get("phase", 0), "solver_ms": info.get("solver_ms", 0), "cycle_ms": info.get("cycle_ms", 0), "clip_fraction": info.get("clip_fraction", 0), "fault": info.get("fault"), "qpos": self.plant.data.qpos.copy()})
            self.previous_torque = torque.copy()
            self.plant.step(torque)
            terminated = bool(info.get("fault") or self.plant.unhealthy())
            if terminated:
                reward -= 10
                break
        truncated = self.plant.data.time >= self.cfg["settle_seconds"] + self.cfg["episode_seconds"]
        return self.controller.observation(), float(reward), terminated, bool(truncated), info

    def close(self):
        pass
