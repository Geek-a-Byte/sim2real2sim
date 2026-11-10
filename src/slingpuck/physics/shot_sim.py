"""One robot shot in sim: band release (BandModel), release noise, then the puck flight in MuJoCo.

A shot is (launch_x, aim, pull) in the board frame: the puck sits on the agent band at x = launch_x, and
the robot draws it back by `pull` in the direction `aim` (rad from straight back, + toward +x). The band
releases it (physics/band_model.py), the jaws' measured effect and the release noise are added
(shot.release_* in env.yaml, to be calibrated in S4), and the puck slides in a board-only MuJoCo scene (walls, divider, gate corners, Coulomb
friction, the configured restitution) until it goes through the gate, stops, or bounces back.

Outcome:
  through   the puck crossed the divider into the opponent half
  weak      it stopped (or never launched) before the divider
  edge      it reached the divider within one puck radius of the gate opening, and bounced back
  miss      it hit the divider wall away from the gate
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.slingpuck.physics.band_model import GRAVITY, BandModel
from src.slingpuck.so101_slingpuck import WALL_PAIRS, add_board, contact_params, coulomb_along_velocity

OUTCOMES = ("through", "weak", "edge", "miss")


@dataclass
class ShotResult:
    outcome: str
    exit_speed: float        # m/s at the band line (0 if not launched)
    divider_x: float | None  # x where the puck reached the divider face
    flight_s: float          # Release to the end of the flight

    @property
    def through(self) -> bool:
        return self.outcome == "through"


class ShotSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.band = BandModel.from_config(cfg)
        b, p = cfg["board"], cfg["puck"]
        self.r, self.mass, self.mu = p["radius_m"], p["mass_kg"], p["friction_kinetic"]
        self.half_t = 0.5 * b["divider_thickness_m"]
        self.half_gate = 0.5 * b["gate_width_m"]
        self.noise = cfg["shot"]

        spec = mujoco.MjSpec()                    # Board only: the puck is planar, friction is on its joints
        spec.option.timestep = cfg["sim"]["mujoco_timestep_s"]
        board = spec.worldbody.add_body(name="board")
        add_board(board, cfg)
        z_mid = 0.5 * p["thickness_m"]
        puck = spec.worldbody.add_body(name="puck", pos=[0, 0, z_mid])
        friction = self.mu * self.mass * GRAVITY
        for jn, jt, axis in (("x", "SLIDE", (1, 0, 0)), ("y", "SLIDE", (0, 1, 0)), ("yaw", "HINGE", (0, 0, 1))):
            puck.add_joint(name=f"puck_{jn}", type=getattr(mujoco.mjtJoint, f"mjJNT_{jt}"), axis=list(axis),
                           frictionloss=friction if jt == "SLIDE" else 1e-5, damping=0.0)
        puck.add_geom(name="puck", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[self.r, z_mid, 0], mass=self.mass,
                      contype=0, conaffinity=0)
        puck.add_geom(name="puck_col", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.r, 0, 0], density=0.0,
                      contype=0, conaffinity=0)
        solref, solimp = contact_params(cfg)
        for wall in WALL_PAIRS:
            spec.add_pair(geomname1="puck_col", geomname2=wall, condim=1, solref=solref, solimp=solimp)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.friction_n = friction
        self._dofs = [self.m.joint(f"puck_{a}").dofadr[0] for a in "xy"]

    def pull_point(self, launch_x: float, aim: float, pull: float) -> np.ndarray:
        return np.array([launch_x + pull * np.sin(aim), self.band.band_y - pull * np.cos(aim)])

    def shoot(self, launch_x: float, aim: float, pull: float, rng: np.random.Generator | None = None,
              max_t: float = 1.5) -> ShotResult:
        rel = self.band.release(self.pull_point(launch_x, aim, pull), self.mass, self.mu)
        if not rel.launched:
            return ShotResult("weak", 0.0, None, 0.0)
        pos, vel = rel.exit_pos.copy(), rel.exit_vel.copy()
        n = self.noise
        dx, a, k = n["release_x_bias_m"], n["release_angle_bias_rad"], n["release_speed_factor"]
        if rng is not None:                       # Release noise
            dx += rng.normal(0, n["release_x_std_m"])
            a += rng.normal(0, n["release_angle_std_rad"])
            k *= max(0.0, 1.0 + rng.normal(0, n["release_speed_std_frac"]))
        pos[0] += dx
        c, s = np.cos(a), np.sin(a)               # + a turns the exit direction toward +x
        vel = k * np.array([c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]])
        exit_speed = float(np.linalg.norm(vel))

        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:2] = pos
        self.d.qvel[:2] = vel
        face = -(self.half_t + self.r)            # Puck center when it touches the divider (agent side)
        divider_x, t = None, 0.0
        while t < max_t:
            for _ in range(10):
                coulomb_along_velocity(self.m, self.d, self._dofs, self.friction_n)
                mujoco.mj_step(self.m, self.d)
            t = self.d.time
            x, y = self.d.qpos[0], self.d.qpos[1]
            vy = self.d.qvel[1]
            if divider_x is None and y >= face - 1e-3:
                divider_x = float(x)
            if y > self.half_t + self.r:
                return ShotResult("through", exit_speed, divider_x, t)
            if np.hypot(*self.d.qvel[:2]) < 1e-3 or (divider_x is not None and vy < 0):
                break
        if divider_x is None:
            return ShotResult("weak", exit_speed, None, t)
        near_gate = abs(divider_x) < self.half_gate + self.r
        return ShotResult("edge" if near_gate else "miss", exit_speed, divider_x, t)
