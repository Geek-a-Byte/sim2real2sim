"""Fast planar puck dynamics in NumPy (used for Phase 3 and for env physics).

Model:
- Coulomb sliding friction: constant deceleration mu * g against velocity, and
  the puck stops (no reversal) when its speed reaches zero.
- Side and end walls, and the center divider with a gate gap. Contacts scale the
  normal velocity by -restitution and keep the tangential velocity.
- Puck-puck contacts: equal-mass impulse with puck restitution.

No contact or friction step adds kinetic energy, so total energy never increases.
"""
from dataclasses import dataclass

import numpy as np

from slingpuck.physics.backend import GateCrossing, PuckPhysicsBackend

GRAVITY = 9.81


@dataclass(frozen=True)
class BoardGeometry:
    length_m: float          # Inner play length (y)
    width_m: float           # Inner play width (x)
    gate_width_m: float
    divider_thickness_m: float
    wall_restitution: float

    @classmethod
    def from_config(cls, cfg: dict) -> "BoardGeometry":
        b = cfg["board"]
        return cls(b["length_m"], b["width_m"], b["gate_width_m"], b["divider_thickness_m"], b["restitution"])

    @property
    def half_length(self) -> float:
        return 0.5 * self.length_m

    @property
    def half_width(self) -> float:
        return 0.5 * self.width_m

    def divider_boxes(self) -> np.ndarray:
        """Two divider blocks beside the gate, rows of [xmin, xmax, ymin, ymax]."""
        hw, hg, ht = self.half_width, 0.5 * self.gate_width_m, 0.5 * self.divider_thickness_m
        return np.array([[-hw, -hg, -ht, ht],
                         [hg, hw, -ht, ht]])


@dataclass(frozen=True)
class PuckParams:
    radius_m: float
    mass_kg: float
    friction_kinetic: float
    restitution: float

    @classmethod
    def from_config(cls, cfg: dict) -> "PuckParams":
        p = cfg["puck"]
        return cls(p["radius_m"], p["mass_kg"], p["friction_kinetic"], p["restitution"])


class Fast2DPuckSim(PuckPhysicsBackend):
    def __init__(self, board: BoardGeometry, puck: PuckParams, n_pucks: int = 1):
        for name, e in (("wall", board.wall_restitution), ("puck", puck.restitution)):
            if not 0.0 <= e <= 1.0:
                raise ValueError(f"{name} restitution must be in [0, 1], got {e}")
        if board.gate_width_m <= 2 * puck.radius_m:
            raise ValueError("gate is not wider than the puck")
        self.board = board
        self.puck = puck
        self.n_pucks = n_pucks
        self._boxes = board.divider_boxes()
        self.pos = np.zeros((n_pucks, 2))
        self.vel = np.zeros((n_pucks, 2))

    @classmethod
    def from_config(cls, cfg: dict, n_pucks: int = 1) -> "Fast2DPuckSim":
        return cls(BoardGeometry.from_config(cfg), PuckParams.from_config(cfg), n_pucks)

    def reset(self, pos, vel):
        pos = np.array(pos, dtype=float).reshape(self.n_pucks, 2)
        vel = np.array(vel, dtype=float).reshape(self.n_pucks, 2)
        r = self.puck.radius_m
        if np.any(np.abs(pos[:, 0]) > self.board.half_width - r + 1e-9) or \
           np.any(np.abs(pos[:, 1]) > self.board.half_length - r + 1e-9):
            raise ValueError(f"puck outside the board: {pos}")
        self.pos, self.vel = pos, vel

    def get_state(self):
        return self.pos.copy(), self.vel.copy()

    def kinetic_energy(self) -> float:
        return 0.5 * self.puck.mass_kg * float(np.sum(self.vel**2))

    def step(self, dt: float) -> list[GateCrossing]:
        side_before = self.pos[:, 1] > 0.0
        self._apply_friction(dt)
        self.pos += self.vel * dt
        self._collide_walls()
        self._collide_divider(side_before)
        if self.n_pucks > 1:
            self._collide_pucks()
        side_after = self.pos[:, 1] > 0.0
        return [GateCrossing(int(i), 1 if side_after[i] else -1, float(self.pos[i, 0]))
                for i in np.flatnonzero(side_before != side_after)]

    def _apply_friction(self, dt: float):
        speed = np.linalg.norm(self.vel, axis=1)
        moving = speed > 0.0
        new_speed = np.maximum(speed - self.puck.friction_kinetic * GRAVITY * dt, 0.0)
        scale = np.ones_like(speed)
        scale[moving] = new_speed[moving] / speed[moving]
        self.vel *= scale[:, None]

    def _collide_walls(self):
        r, e = self.puck.radius_m, self.board.wall_restitution
        for axis, half in ((0, self.board.half_width), (1, self.board.half_length)):
            lim = half - r
            low = self.pos[:, axis] < -lim
            high = self.pos[:, axis] > lim
            self.pos[low, axis] = -lim
            self.pos[high, axis] = lim
            into_low = low & (self.vel[:, axis] < 0)
            into_high = high & (self.vel[:, axis] > 0)
            self.vel[into_low | into_high, axis] *= -e

    def _collide_divider(self, side_before: np.ndarray):
        r, e = self.puck.radius_m, self.board.wall_restitution
        for xmin, xmax, ymin, ymax in self._boxes:
            closest = np.column_stack([np.clip(self.pos[:, 0], xmin, xmax),
                                       np.clip(self.pos[:, 1], ymin, ymax)])
            d = self.pos - closest
            dist = np.linalg.norm(d, axis=1)
            for i in np.flatnonzero(dist < r):
                if dist[i] > 1e-12:
                    n = d[i] / dist[i]
                    self.pos[i] = closest[i] + n * r
                else:
                    # Center is inside the block: push out toward the side it came from.
                    n = np.array([0.0, 1.0 if side_before[i] else -1.0])
                    self.pos[i, 1] = (ymax + r) if side_before[i] else (ymin - r)
                vn = self.vel[i] @ n
                if vn < 0.0:
                    self.vel[i] -= (1.0 + e) * vn * n

    def _collide_pucks(self):
        r2, e = 2.0 * self.puck.radius_m, self.puck.restitution
        for i in range(self.n_pucks - 1):
            for j in range(i + 1, self.n_pucks):
                d = self.pos[j] - self.pos[i]
                dist = float(np.hypot(d[0], d[1]))
                if dist >= r2:
                    continue
                n = d / dist if dist > 1e-12 else np.array([1.0, 0.0])
                overlap = r2 - dist
                self.pos[i] -= 0.5 * overlap * n
                self.pos[j] += 0.5 * overlap * n
                vn = (self.vel[j] - self.vel[i]) @ n
                if vn < 0.0:
                    impulse = 0.5 * (1.0 + e) * vn * n  # Equal masses
                    self.vel[i] += impulse
                    self.vel[j] -= impulse
