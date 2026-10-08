"""Fast planar puck dynamics (used for Phase 3 and for env physics).

Model:
- Coulomb sliding friction: constant deceleration mu * g against velocity, and
  the puck stops (no reversal) when its speed reaches zero.
- Side and end walls, and the center divider with a gate gap. Contacts scale the
  normal velocity by -restitution and keep the tangential velocity.
- Puck-puck contacts: equal-mass impulse with puck restitution.
- Optional kinematic paddle (oriented box moved by the arm). Contacts reflect the
  normal velocity relative to the paddle surface, scaled by paddle restitution.

Without a moving paddle, no step adds kinetic energy, so total energy never
increases. A moving paddle can add energy, as a real strike does.

Per-puck contact code uses Python floats, not NumPy: for 1-2 values, NumPy call
overhead is ~10x the arithmetic.
"""
import math
from dataclasses import dataclass

import numpy as np

from src.slingpuck.physics.backend import GateCrossing, PaddleState, PuckPhysicsBackend, StepEvents

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


def _reflect(vx, vy, nx, ny, e, sx=0.0, sy=0.0):
    """Reflect the velocity component along normal n (relative to surface velocity s) if approaching."""
    vn = (vx - sx) * nx + (vy - sy) * ny
    if vn < 0.0:
        k = (1.0 + e) * vn
        return vx - k * nx, vy - k * ny
    return vx, vy


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
        self._boxes = [tuple(map(float, b)) for b in board.divider_boxes()]
        self.pos = np.zeros((n_pucks, 2))
        self.vel = np.zeros((n_pucks, 2))
        self.paddle: PaddleState | None = None
        self._pad = None

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

    def set_paddle(self, paddle: PaddleState | None):
        if paddle is not None and not 0.0 <= paddle.restitution <= 1.0:
            raise ValueError(f"paddle restitution must be in [0, 1], got {paddle.restitution}")
        self.paddle = paddle
        if paddle is None:
            self._pad = None
        else:
            self._pad = (float(paddle.center[0]), float(paddle.center[1]),
                         math.cos(paddle.angle), math.sin(paddle.angle),
                         float(paddle.vel[0]), float(paddle.vel[1]), float(paddle.omega),
                         float(paddle.half_width), float(paddle.half_thickness), float(paddle.restitution))

    def kinetic_energy(self) -> float:
        return 0.5 * self.puck.mass_kg * float(np.sum(self.vel**2))

    def step(self, dt: float) -> StepEvents:
        decel = self.puck.friction_kinetic * GRAVITY * dt
        pos, vel = self.pos.tolist(), self.vel.tolist()
        side_before = [p[1] > 0.0 for p in pos]
        contacts = []
        for i in range(self.n_pucks):
            if self._step_puck(pos[i], vel[i], dt, decel, side_before[i]):
                contacts.append(i)
        self.pos[:] = pos
        self.vel[:] = vel
        if self.n_pucks > 1:
            self._collide_pucks()
        crossings = []
        for i in range(self.n_pucks):
            side_after = bool(self.pos[i, 1] > 0.0)
            if side_after != side_before[i]:
                crossings.append(GateCrossing(i, 1 if side_after else -1, float(self.pos[i, 0])))
        return StepEvents(crossings, contacts)

    def _step_puck(self, p, v, dt, decel, side_before) -> bool:
        """Advance one puck in place (p, v are [x, y] lists). Returns True on paddle contact."""
        x, y = p
        vx, vy = v
        r = self.puck.radius_m

        speed = math.hypot(vx, vy)
        if speed > 0.0:
            s = max(speed - decel, 0.0) / speed
            vx *= s
            vy *= s
        x += vx * dt
        y += vy * dt

        hit = False
        if self._pad is not None:
            x, y, vx, vy, hit = self._collide_paddle(x, y, vx, vy, r)

        # Walls
        e = self.board.wall_restitution
        lim = self.board.half_width - r
        if x < -lim:
            x = -lim
            if vx < 0.0:
                vx = -e * vx
        elif x > lim:
            x = lim
            if vx > 0.0:
                vx = -e * vx
        lim = self.board.half_length - r
        if y < -lim:
            y = -lim
            if vy < 0.0:
                vy = -e * vy
        elif y > lim:
            y = lim
            if vy > 0.0:
                vy = -e * vy

        # Divider blocks beside the gate
        for xmin, xmax, ymin, ymax in self._boxes:
            cx = min(max(x, xmin), xmax)
            cy = min(max(y, ymin), ymax)
            dx, dy = x - cx, y - cy
            d2 = dx * dx + dy * dy
            if d2 >= r * r:
                continue
            dist = math.sqrt(d2)
            if dist > 1e-12:
                nx, ny = dx / dist, dy / dist
                x, y = cx + nx * r, cy + ny * r
            else:  # Center is inside the block: push out toward the side it came from.
                nx, ny = 0.0, (1.0 if side_before else -1.0)
                y = (ymax + r) if side_before else (ymin - r)
            vx, vy = _reflect(vx, vy, nx, ny, e)

        p[0], p[1], v[0], v[1] = x, y, vx, vy
        return hit

    def _collide_paddle(self, x, y, vx, vy, r):
        pcx, pcy, c, s, pvx, pvy, omega, hw, ht, e = self._pad
        rx, ry = x - pcx, y - pcy
        lu, ln = rx * c + ry * s, -rx * s + ry * c  # Puck center in the paddle frame
        cu, cn = min(max(lu, -hw), hw), min(max(ln, -ht), ht)
        du, dn = lu - cu, ln - cn
        d2 = du * du + dn * dn
        if d2 >= r * r:
            return x, y, vx, vy, False
        dist = math.sqrt(d2)
        if dist > 1e-12:
            nu, nn = du / dist, dn / dist
        else:  # Center inside the box: push out along the thin axis.
            nu, nn = 0.0, (1.0 if ln >= 0.0 else -1.0)
            cn = ht * nn
        nx, ny = nu * c - nn * s, nu * s + nn * c
        ax, ay = cu * c - cn * s, cu * s + cn * c   # Contact point relative to paddle center
        x, y = pcx + ax + nx * r, pcy + ay + ny * r
        sx, sy = pvx - omega * ay, pvy + omega * ax  # Surface velocity at the contact point
        vx, vy = _reflect(vx, vy, nx, ny, e, sx, sy)
        return x, y, vx, vy, True

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
