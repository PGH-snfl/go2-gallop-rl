import numpy as np

LEGS = ("FL", "FR", "RL", "RR")

class GallopSchedule:
    """TD phase convention. Phase integrates continuously even when period changes."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 0.25  # paper Eq.3 origin, rotated so hind touchdown is zero
        self.elapsed = 0.0
        self.residual = np.zeros(5)
        self.apply(np.zeros(5), cfg["policy_dt"])

    def apply(self, action, dt):
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Residual must contain five finite values")
        target = np.clip(action, -1, 1) * np.asarray(self.cfg["residual_bounds"])
        rate = np.asarray(self.cfg["residual_rate"]) * dt
        self.residual += np.clip(target - self.residual, -rate, rate)
        self.gait_period = self.cfg["period"] + self.residual[0]
        hind = self.cfg["hind_separation"] + self.residual[1]
        fore = self.cfg["fore_separation"] + self.residual[2]
        start = self.cfg["fore_start"]
        if not 0 < hind < start < start + fore < 1:
            raise ValueError("Residual changes transverse touchdown order")
        if self.cfg["lead"] == "right":
            self.touchdown = np.array([start, start + fore, 0, hind])
            self.duty = np.asarray(self.cfg["duty"], dtype=float)
        else:
            self.touchdown = np.array([start + fore, start, hind, 0])
            self.duty = np.asarray(self.cfg["duty"], dtype=float)[[1, 0, 3, 2]]
        self.pitch_reference = self.residual[3]
        self.height_reference = self.cfg["height"] + self.residual[4]

    def advance(self, dt):
        self.phase = (self.phase + dt / self.gait_period) % 1.0
        self.elapsed += dt

    def local_phase(self, delta=0.0):
        return (self.phase + delta / self.gait_period - self.touchdown) % 1.0

    def mask(self, delta=0.0):
        return self.local_phase(delta) < self.duty

    def table(self, dt, count):
        return ((self.phase + np.arange(count)[None, :] * dt / self.gait_period - self.touchdown[:, None]) % 1) < self.duty[:, None]

    def stance_seconds(self, i):
        return self.duty[i] * self.gait_period

    def swing_remaining(self, i):
        return max(0.005, (1 - self.local_phase()[i]) * self.gait_period)
