"""Arm deflection under band load (Phase 2).

The arm holds the puck against the band. The 3D-printed links and the servos
give way a little, so the actual pull-back is shorter than the command:

    d_act = d_cmd - compliance * F_par(d_act)

F_par is the band force component that resists the pull (along -pull direction).
d + compliance * F_par(d) increases with d, so bisection always finds d_act.
Only deflection along the pull direction is modeled.
"""
from dataclasses import dataclass

import numpy as np

from src.slingpuck.physics.band_model import BandModel


@dataclass(frozen=True)
class PullResult:
    pos: np.ndarray         # Actual puck position while held
    pull_m: float           # Actual pull-back distance
    force: np.ndarray       # Band force on the puck at that position (loading curve)


def pull_direction(angle_rad: float) -> np.ndarray:
    """Unit pull direction. angle 0 pulls straight back (-y), away from the gate."""
    return -np.array([np.sin(angle_rad), np.cos(angle_rad)])


def solve_pull(band: BandModel, start, angle_rad: float, pull_cmd_m: float, compliance_m_per_n: float,
               tol: float = 1e-9) -> PullResult:
    start = np.asarray(start, dtype=float)
    u = pull_direction(angle_rad)

    def resisting(d):
        return float(-band.force_load(start + d * u) @ u)

    if pull_cmd_m <= 0.0:
        return PullResult(start.copy(), 0.0, band.force_load(start))
    if compliance_m_per_n <= 0.0:
        d = pull_cmd_m  # Rigid arm
    else:
        lo, hi = 0.0, pull_cmd_m
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if mid + compliance_m_per_n * resisting(mid) > pull_cmd_m:
                hi = mid
            else:
                lo = mid
        d = 0.5 * (lo + hi)
    pos = start + d * u
    return PullResult(pos, d, band.force_load(pos))
