"""Position-servo model for Feetech STS3215 joints.

Pipeline per step: command latency -> deadband -> first-order lag -> rate limit.
Works on a scalar joint or an array of joints (all joints share one parameter set).
"""
from collections import deque

import numpy as np


def latency_steps(latency_s: float, dt: float) -> int:
    """Number of whole steps of delay. Latency is quantized to the step size."""
    if latency_s < 0:
        raise ValueError(f"latency must be >= 0, got {latency_s}")
    return int(round(latency_s / dt))


class ServoModel:
    def __init__(self, config: dict, dt: float, init_pos=0.0):
        self.dt = dt
        self.max_rate = np.deg2rad(config["max_rate_deg_s"])
        self.tau = config["lag_tau_s"]
        self.deadband = np.deg2rad(config["deadband_deg"])
        self.latency_steps = latency_steps(config["command_latency_s"], dt)
        self.reset(init_pos)

    @property
    def effective_latency_s(self) -> float:
        return self.latency_steps * self.dt

    def reset(self, pos=0.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros_like(self.pos)
        # queue[0] is the command from `latency_steps` steps ago, after this step's append.
        n = self.latency_steps + 1
        self._queue = deque([self.pos.copy() for _ in range(n)], maxlen=n)

    def step(self, target):
        self._queue.append(np.broadcast_to(np.asarray(target, dtype=float), self.pos.shape).copy())
        err = self._queue[0] - self.pos

        # Deadband: the servo does not react to errors smaller than the deadband.
        err = np.where(np.abs(err) < self.deadband, 0.0, err)

        # First-order lag, exact discretization of tau * dx/dt = target - x.
        alpha = 1.0 - np.exp(-self.dt / self.tau) if self.tau > 0 else 1.0
        delta = alpha * err

        max_step = self.max_rate * self.dt
        delta = np.clip(delta, -max_step, max_step)

        self.vel = delta / self.dt
        self.pos = self.pos + delta
        return self.pos.copy()
