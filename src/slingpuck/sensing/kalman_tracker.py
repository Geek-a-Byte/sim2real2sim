"""Constant-velocity Kalman tracker for one puck, with latency compensation.

State is [x, y, vx, vy]. Each camera Frame updates the filter at its capture
time. ``estimate(t_now)`` extrapolates the filter to the current time, which
removes the camera latency from the estimate the policy sees.

Known limit: the constant-velocity model does not know about wall bounces, so
the estimate lags for a few frames after a bounce.
"""
import numpy as np

from src.slingpuck.sensing.camera_model import Frame

_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def _transition(dt: float, accel_std: float):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    # Discrete white-noise acceleration model.
    G = np.array([[0.5 * dt**2, 0.0],
                  [0.0, 0.5 * dt**2],
                  [dt, 0.0],
                  [0.0, dt]])
    Q = G @ G.T * accel_std**2
    return F, Q


class KalmanTracker:
    def __init__(self, meas_noise_std_m: float, accel_noise_std_m_s2: float, init_vel_std_m_s: float = 2.0):
        self.R = np.eye(2) * max(meas_noise_std_m, 1e-6) ** 2
        self.accel_std = accel_noise_std_m_s2
        self.init_vel_std = init_vel_std_m_s
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict) -> "KalmanTracker":
        return cls(cfg["camera"]["noise_std_m"],
                   cfg["tracker"]["accel_noise_std_m_s2"],
                   cfg["tracker"]["init_vel_std_m_s"])

    def reset(self):
        self.x = None  # Filter state at time self.t
        self.P = None
        self.t = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def update(self, frame: Frame):
        z = np.asarray(frame.xy, dtype=float).reshape(2)
        if not self.initialized:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.R[0, 0], self.R[1, 1], self.init_vel_std**2, self.init_vel_std**2])
            self.t = frame.t_capture
            return
        dt = frame.t_capture - self.t
        if dt < 0:
            return  # Out-of-order frame; the camera model never produces one.
        F, Q = _transition(dt, self.accel_std)
        x = F @ self.x
        P = F @ self.P @ F.T + Q

        S = _H @ P @ _H.T + self.R
        K = P @ _H.T @ np.linalg.inv(S)
        x = x + K @ (z - _H @ x)
        I_KH = np.eye(4) - K @ _H
        self.P = I_KH @ P @ I_KH.T + K @ self.R @ K.T  # Joseph form, stays symmetric PSD
        self.x = x
        self.t = frame.t_capture

    def estimate(self, t_now: float):
        """Return (pos, vel, cov) predicted to t_now, or None before the first frame."""
        if not self.initialized:
            return None
        F, Q = _transition(max(t_now - self.t, 0.0), self.accel_std)
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        return x[:2].copy(), x[2:].copy(), P
