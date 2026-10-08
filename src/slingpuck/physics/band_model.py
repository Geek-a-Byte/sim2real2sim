"""Elastic band of the sling (Phase 2).

Geometry (board frame): the band runs between two anchors at
(+-anchor_span/2, band_y). The puck rests against the band line; the arm pulls
it back (toward the agent end wall, -y) to point P. The band then forms a V from
each anchor to P, and pushes P forward with force T * (u_left + u_right), where
u are the unit vectors from P to the anchors. The band vertex is taken at the
puck center (the puck radius is ignored here).

Tension law. Elongation s = (band length) - natural_length. The band is
pre-stretched: at rest s0 = anchor_span - natural_length > 0.
  Loading:   T_l(s) = k * s * (s / S_REF)^(exponent - 1)
  Unloading from s_max back to s0:
             T_u(s) = T_l(s) - h * 4 x (1 - x),  x = (s - s0) / (s_max - s0)
  The bump is zero at both ends, so the hysteresis loop closes, and h is set so
  that the loop loses exactly hysteresis_loss_factor of the loading work
  W_l = integral of T_l from s0 to s_max.

Release: the puck starts at rest at P and is pushed by the unloading band (with
sliding friction) until it crosses the band line. If the band force cannot beat
friction (a very small pull), or friction stops the puck before the band line,
the puck stays behind the band (launched = False). energy_transfer is the
fraction of that kinetic energy the puck keeps (band mass, slip-release losses),
so the exit speed is scaled by sqrt(energy_transfer).
"""
import math
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
S_REF = 0.01  # Reference elongation for the exponent, m


@dataclass(frozen=True)
class BandParams:
    anchor_span_m: float
    natural_length_m: float
    stiffness_k_n_m: float
    stiffness_exponent: float
    hysteresis_loss_factor: float
    energy_transfer: float

    @classmethod
    def from_config(cls, cfg: dict) -> "BandParams":
        b = cfg["band"]
        return cls(b["anchor_span_m"], b["natural_length_m"], b["stiffness_k_n_m"], b["stiffness_exponent"],
                   b["hysteresis_loss_factor"], b["energy_transfer"])


@dataclass(frozen=True)
class ReleaseResult:
    exit_pos: np.ndarray    # Where the puck crosses the band line
    exit_vel: np.ndarray    # After energy_transfer
    release_time_s: float
    band_work_j: float      # Work the band did on the puck (before energy_transfer)
    launched: bool = True   # False if the puck stopped behind the band line


class BandModel:
    def __init__(self, params: BandParams, band_y: float):
        if not 0.0 <= params.hysteresis_loss_factor < 1.0:
            raise ValueError("hysteresis_loss_factor must be in [0, 1)")
        if not 0.0 < params.energy_transfer <= 1.0:
            raise ValueError("energy_transfer must be in (0, 1]")
        self.p = params
        self.band_y = band_y
        half = 0.5 * params.anchor_span_m
        self.anchors = np.array([[-half, band_y], [half, band_y]])
        self.s_rest = max(params.anchor_span_m - params.natural_length_m, 0.0)

    @classmethod
    def from_config(cls, cfg: dict) -> "BandModel":
        band_y = -0.5 * cfg["board"]["length_m"] + cfg["band"]["band_offset_from_end_wall_m"]
        return cls(BandParams.from_config(cfg), band_y)

    # ------------------------------------------------------------ tension
    def tension_load(self, s):
        s = np.maximum(np.asarray(s, dtype=float), 0.0)
        a = self.p.stiffness_exponent
        return self.p.stiffness_k_n_m * s * (s / S_REF) ** (a - 1.0)

    def work_load(self, s0: float, s1: float) -> float:
        """Integral of T_l from s0 to s1."""
        a = self.p.stiffness_exponent
        c = self.p.stiffness_k_n_m / S_REF ** (a - 1.0)
        s0, s1 = max(s0, 0.0), max(s1, 0.0)
        return c * (s1 ** (a + 1.0) - s0 ** (a + 1.0)) / (a + 1.0)

    def tension_unload(self, s, s_max: float):
        """Unloading tension on the way back from s_max to the rest elongation."""
        s = np.asarray(s, dtype=float)
        s0 = self.s_rest
        span = s_max - s0
        if span <= 0.0:
            return self.tension_load(s)
        h = self.p.hysteresis_loss_factor * self.work_load(s0, s_max) / (2.0 / 3.0 * span)
        x = np.clip((s - s0) / span, 0.0, 1.0)
        return np.maximum(self.tension_load(s) - h * 4.0 * x * (1.0 - x), 0.0)

    # ----------------------------------------------------------- geometry
    def elongation(self, pos) -> float:
        pos = np.asarray(pos, dtype=float)
        length = np.linalg.norm(self.anchors[0] - pos) + np.linalg.norm(self.anchors[1] - pos)
        return float(length - self.p.natural_length_m)

    def direction_sum(self, pos) -> np.ndarray:
        """u_left + u_right: the force direction times sin-like factor (zero on the band line)."""
        d = self.anchors - np.asarray(pos, dtype=float)
        return (d / np.linalg.norm(d, axis=1, keepdims=True)).sum(axis=0)

    def force_load(self, pos) -> np.ndarray:
        """Band force on the puck at pos while the band is being stretched."""
        return float(self.tension_load(self.elongation(pos))) * self.direction_sum(pos)

    # ------------------------------------------------------------ release
    def release(self, pull_pos, mass: float, mu: float, dt: float = 1e-4, max_t: float = 0.3) -> ReleaseResult:
        pos = np.array(pull_pos, dtype=float)
        if pos[1] >= self.band_y:
            raise ValueError("pull position must be behind the band line (y < band_y)")
        s_max = self.elongation(pos)
        vel = np.zeros(2)
        work, t = 0.0, 0.0
        decel = mu * GRAVITY
        while pos[1] < self.band_y:
            if t > max_t:
                raise RuntimeError("band release did not finish; check band parameters")
            force = float(self.tension_unload(self.elongation(pos), s_max)) * self.direction_sum(pos)
            acc = force / mass
            speed = math.hypot(vel[0], vel[1])
            if speed > 0.0:
                acc = acc - decel * vel / speed
            elif np.linalg.norm(acc) <= decel:
                return ReleaseResult(pos, np.zeros(2), t, work, launched=False)  # Static friction holds it
            new_vel = vel + acc * dt
            if speed > 0.0 and float(new_vel @ vel) <= 0.0:
                return ReleaseResult(pos, np.zeros(2), t, work, launched=False)  # Friction stopped it
            step = new_vel * dt
            work += float(force @ step)
            prev = pos.copy()
            pos = pos + step
            vel = new_vel
            t += dt
        # Interpolate back to the band line.
        frac = (self.band_y - prev[1]) / (pos[1] - prev[1])
        exit_pos = prev + frac * (pos - prev)
        return ReleaseResult(exit_pos, vel * math.sqrt(self.p.energy_transfer), t, work)
