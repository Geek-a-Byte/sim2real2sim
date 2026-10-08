"""Overhead camera model: frame rate, latency, Gaussian noise, dropouts.

Call ``observe(t, true_xy)`` once per physics step with the sim time after the
step. The camera captures a frame when a frame period has passed, and the frame
becomes available ``latency_s`` later. ``observe`` returns only the frames that
became available at this step, so a consumer never sees a stale frame twice.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class Frame:
    t_capture: float    # Sim time when the image was taken
    t_available: float  # Sim time when the tracker receives it
    xy: np.ndarray      # Measured position(s), same shape as the true position


class CameraModel:
    def __init__(self, config: dict, rng: np.random.Generator | None = None):
        self.period = 1.0 / config["fps"]
        self.latency = config["latency_s"]
        self.noise_std = config["noise_std_m"]
        self.dropout_prob = config["dropout_prob"]
        self.reset(rng=rng)

    def reset(self, t0: float = 0.0, rng: np.random.Generator | None = None):
        if rng is not None:
            self.rng = rng
        elif not hasattr(self, "rng"):
            self.rng = np.random.default_rng()
        self.t_next_capture = t0
        self._pending: deque[Frame] = deque()
        self.n_captured = 0
        self.n_dropped = 0

    def observe(self, t: float, true_xy) -> list[Frame]:
        if t + _EPS >= self.t_next_capture:
            self._capture(t, np.asarray(true_xy, dtype=float))
            self.t_next_capture += self.period
            if self.t_next_capture <= t + _EPS:
                # Step is longer than one frame period; skip the missed frames.
                missed = np.floor((t + _EPS - self.t_next_capture) / self.period) + 1
                self.t_next_capture += missed * self.period

        ready = []
        while self._pending and self._pending[0].t_available <= t + _EPS:
            ready.append(self._pending.popleft())
        return ready

    def _capture(self, t: float, true_xy: np.ndarray):
        self.n_captured += 1
        if self.rng.random() < self.dropout_prob:
            self.n_dropped += 1
            return
        noisy = true_xy + self.rng.normal(0.0, self.noise_std, size=true_xy.shape)
        self._pending.append(Frame(t, t + self.latency, noisy))
