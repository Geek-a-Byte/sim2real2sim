"""Real-time puck tracks from the detections of each camera frame.

Each track is a constant-velocity Kalman filter (state [x, y, vx, vy], board frame), the same filter as
sensing/kalman_tracker.py on the main branch (Phase 1). New here: many pucks at once. Each frame, every detection
goes to the nearest predicted track inside a gate (greedy, nearest pairs first); a detection without a track starts
a new track; a track without a detection for max_missed_s ends.

estimate(t_now) predicts every track to t_now, so the camera latency can be removed from what the robot sees
(give t_now = capture time + latency).

Known limit: the constant-velocity model does not know about wall bounces, so a track lags for a few frames after
a bounce. Two pucks that touch and then move apart can swap track ids.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def _transition(dt: float, accel_std: float):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[0.5 * dt ** 2, 0.0], [0.0, 0.5 * dt ** 2], [dt, 0.0], [0.0, dt]])
    return F, G @ G.T * accel_std ** 2       # Discrete white-noise acceleration model


class KalmanTracker:
    """Constant-velocity Kalman filter for one puck."""

    def __init__(self, meas_noise_std_m: float, accel_noise_std_m_s2: float, init_vel_std_m_s: float = 2.0):
        self.R = np.eye(2) * max(meas_noise_std_m, 1e-6) ** 2
        self.accel_std = accel_noise_std_m_s2
        self.init_vel_std = init_vel_std_m_s
        self.x = self.P = self.t = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def update(self, t: float, xy):
        z = np.asarray(xy, float).reshape(2)
        if not self.initialized:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.R[0, 0], self.R[1, 1], self.init_vel_std ** 2, self.init_vel_std ** 2])
            self.t = t
            return
        dt = t - self.t
        if dt < 0:
            return                          # Out-of-order frame
        F, Q = _transition(dt, self.accel_std)
        x, P = F @ self.x, F @ self.P @ F.T + Q
        S = _H @ P @ _H.T + self.R
        K = P @ _H.T @ np.linalg.inv(S)
        self.x = x + K @ (z - _H @ x)
        I_KH = np.eye(4) - K @ _H
        self.P = I_KH @ P @ I_KH.T + K @ self.R @ K.T       # Joseph form
        self.t = t

    def predict(self, t_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pos, vel, cov) at t_now."""
        F, Q = _transition(max(t_now - self.t, 0.0), self.accel_std)
        return (F @ self.x)[:2], (F @ self.x)[2:], F @ self.P @ F.T + Q


@dataclass
class Track:
    id: int
    kf: KalmanTracker
    t_first: float
    t_last: float
    hits: int = 1
    colours: Counter = field(default_factory=Counter)

    @property
    def colour(self) -> str:
        return self.colours.most_common(1)[0][0] if self.colours else "unknown"


@dataclass
class PuckState:
    id: int
    xy: np.ndarray        # Board (x, y), m, at the asked time
    vel: np.ndarray       # m/s
    colour: str
    hits: int
    age_s: float          # Time since the last detection

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.vel))

    @property
    def side(self) -> str:
        return "robot" if self.xy[1] < 0.0 else "opponent"


class PuckTracker:
    def __init__(self, meas_noise_std_m: float = 0.002, accel_noise_std_m_s2: float = 20.0, gate_m: float = 0.03,
                 max_speed_m_s: float = 3.5, max_missed_s: float = 0.25, min_hits: int = 2):
        self.meas_noise, self.accel = meas_noise_std_m, accel_noise_std_m_s2
        self.gate_m, self.max_speed = gate_m, max_speed_m_s
        self.max_missed_s, self.min_hits = max_missed_s, min_hits
        self.tracks: list[Track] = []
        self._next_id = 0
        self.t = None

    @classmethod
    def from_config(cls, cfg: dict) -> "PuckTracker":
        m, o = cfg.get("match", {}), cfg.get("opponent", {})
        top = max(o.get("shot_speed_range_m_s", [0.0, 3.0]))
        return cls(meas_noise_std_m=m.get("puck_track_noise_m", 0.002), max_speed_m_s=1.2 * top)

    def update(self, t: float, detections) -> None:
        """detections: objects with .xy and .colour (puck_detector.Detection), all from the frame taken at time t."""
        self.t = t
        dets = list(detections)
        pred = [tr.kf.predict(t)[0] for tr in self.tracks]
        pairs = sorted(((float(np.hypot(*(d.xy - p))), i, j) for j, p in enumerate(pred) for i, d in enumerate(dets)),
                       key=lambda e: e[0])
        used_d, used_t = set(), set()
        for dist, i, j in pairs:
            tr = self.tracks[j]
            # A track with one detection has no speed yet: its gate allows the fastest puck
            gate = self.gate_m + (self.max_speed * max(t - tr.t_last, 0.0) if tr.hits < 2 else 0.0)
            if dist > gate or i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            tr.kf.update(t, dets[i].xy)
            tr.t_last, tr.hits = t, tr.hits + 1
            tr.colours[dets[i].colour] += 1
        for i, d in enumerate(dets):
            if i not in used_d:
                kf = KalmanTracker(self.meas_noise, self.accel)
                kf.update(t, d.xy)
                self.tracks.append(Track(self._next_id, kf, t, t, colours=Counter([d.colour])))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.max_missed_s]

    def estimate(self, t_now: float | None = None, confirmed: bool = True) -> list[PuckState]:
        """Every track predicted to t_now (default: the last frame time)."""
        t_now = self.t if t_now is None else t_now
        out = []
        for tr in self.tracks:
            if confirmed and tr.hits < self.min_hits:
                continue
            pos, vel, _ = tr.kf.predict(t_now)
            out.append(PuckState(tr.id, pos, vel, tr.colour, tr.hits, t_now - tr.t_last))
        return out
