"""Shared parts of the generalized method: config, task objects and small geometry helpers.

Robot frame: origin at the SO-101 base on the table, x forward, y left, z up. Meters.
A goal is an object pose: position (center of the object) and yaw about z. The SO-101 has a 5-DoF arm
and a parallel gripper, so goals with a roll or pitch of the object are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.config import REPO_ROOT, git_state

GENERAL_CONFIG = REPO_ROOT / "configs" / "general.yaml"
SO101_XML = REPO_ROOT / "assets" / "mujoco" / "robotstudio_so101" / "so101.xml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def load_general_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    from src.slingpuck.config import deep_merge

    path = Path(path) if path else GENERAL_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["meta"] = {"config_file": str(path), **git_state()}
    return cfg


@dataclass(frozen=True)
class ObjectShape:
    name: str
    kind: str                 # "box" | "cylinder"
    half: tuple               # box: (sx, sy, sz); cylinder: (r, r, half height)
    mass_kg: float

    @classmethod
    def from_config(cls, name: str, c: dict) -> "ObjectShape":
        if c["kind"] == "box":
            half = tuple(float(v) for v in c["half_m"])
        elif c["kind"] == "cylinder":
            half = (float(c["radius_m"]), float(c["radius_m"]), float(c["half_height_m"]))
        else:
            raise ValueError(f"Unknown object kind {c['kind']!r}")
        return cls(name, c["kind"], half, float(c["mass_kg"]))

    @property
    def half_height(self) -> float:
        return self.half[2]

    @property
    def yaw_symmetry(self) -> int:
        """Order of the shape's symmetry about z: 0 = any angle (cylinder), 4 = square box, 2 = other box."""
        if self.kind == "cylinder":
            return 0
        return 4 if np.isclose(self.half[0], self.half[1]) else 2

    def equivalent_yaw(self, goal_yaw: float, current_yaw: float) -> float:
        """The goal yaw that gives the same physical pose and is nearest to the current yaw."""
        n = self.yaw_symmetry
        if n == 0:
            return float(current_yaw)
        step = 2 * np.pi / n
        k = np.round((current_yaw - goal_yaw) / step)
        return float(wrap_angle(goal_yaw + k * step))

    def width_along(self, u_xy, yaw: float) -> float:
        """Object width along the horizontal unit vector u_xy (world), for an object at this yaw."""
        u = np.asarray(u_xy, float)[:2]
        u = u / np.linalg.norm(u)
        if self.kind == "cylinder":
            return 2.0 * self.half[0]
        c, s = np.cos(yaw), np.sin(yaw)
        ex, ey = np.array([c, s]), np.array([-s, c])
        return 2.0 * (abs(u @ ex) * self.half[0] + abs(u @ ey) * self.half[1])

    def surface_points(self, n: int, seed: int = 0) -> np.ndarray:
        """n points on the object surface, in the object frame. Deterministic for a given seed."""
        rng = np.random.default_rng(seed)
        sx, sy, sz = self.half
        if self.kind == "box":
            areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
            faces = rng.choice(6, size=n, p=areas / areas.sum())
            p = rng.uniform(-1, 1, (n, 3)) * np.array([sx, sy, sz])
            axis, sign = faces // 2, np.where(faces % 2 == 0, 1.0, -1.0)
            p[np.arange(n), axis] = sign * np.array([sx, sy, sz])[axis]
            return p
        r, h = sx, sz
        side, cap = 2 * np.pi * r * 2 * h, np.pi * r * r
        on_side = rng.random(n) < side / (side + 2 * cap)
        a = rng.uniform(0, 2 * np.pi, n)
        rr = np.where(on_side, r, r * np.sqrt(rng.random(n)))
        z = np.where(on_side, rng.uniform(-h, h, n), np.where(rng.random(n) < 0.5, h, -h))
        return np.stack([rr * np.cos(a), rr * np.sin(a), z], axis=1)


def objects_from_config(cfg: dict) -> dict[str, ObjectShape]:
    return {k: ObjectShape.from_config(k, v) for k, v in cfg["objects"].items()}


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_to_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
