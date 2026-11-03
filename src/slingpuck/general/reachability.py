"""Reachable-goal filter: can the SO-101 achieve an object goal pose? (step 1 of the recommended pipeline)

One check, used everywhere goals are made (random sim goals, goals from human demos, real2sim variations):
  (a) IK finds the grasp pose that the goal implies, inside the joint limits, with no collision;
  (b) the approach direction is one that a 5-DoF arm can give: the approach lies in the vertical plane
      through the pan axis (top-down, or tilted away from the base), and the jaw axis is set by wrist_roll;
  (c) the gripper opens wide enough for the object (plus a clearance on each side).

The grasp site is "gripperframe": its x axis is the approach direction, its z axis the jaw opening direction.
A reachability map (a voxel grid x approach tilt) makes the check fast inside many envs:

    python -m src.slingpuck.general.reachability          # build cache/reach_map.npz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mujoco
import numpy as np

from src.slingpuck.config import REPO_ROOT
from src.slingpuck.general.common import ARM_JOINTS, SO101_XML, ObjectShape, load_general_config

REACH_MAP_FILE = REPO_ROOT / "cache" / "reach_map.npz"
SEED_POSES = (                        # IK start poses (5 arm joints)
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -0.8, 0.9, 1.2, 0.0),
    (0.0, 0.6, -0.6, 1.4, 0.0),
    (0.0, -1.2, 1.4, 0.4, 0.0),
)


@dataclass
class Reach:
    ok: bool
    reason: str                      # "ok" | "gripper_width" | "ik" | "collision"
    q: np.ndarray | None = None      # 5 arm joints of the grasp pose
    tilt_deg: float | None = None
    approach: np.ndarray | None = None
    jaw: np.ndarray | None = None
    tried: list = field(default_factory=list)


def approach_dir(pos, tilt_deg: float) -> np.ndarray:
    """Top-down (tilt 0), tilted away from the base in the radial plane (tilt > 0), horizontal at 90."""
    r = np.array([pos[0], pos[1], 0.0])
    r = r / max(np.linalg.norm(r), 1e-9)
    t = np.deg2rad(tilt_deg)
    return np.cos(t) * np.array([0.0, 0.0, -1.0]) + np.sin(t) * r


def tangential_dir(pos) -> np.ndarray:
    """Horizontal direction across the radial plane: the jaw axis of a tilted grasp."""
    n = np.hypot(pos[0], pos[1])
    return np.array([-pos[1], pos[0], 0.0]) / max(n, 1e-9)


class SO101Kinematics:
    """SO-101 model with a table for IK and collision checks (no objects)."""

    def __init__(self, table_z: float = 0.0):
        spec = mujoco.MjSpec.from_file(str(SO101_XML))
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
                                pos=[0, 0, table_z], contype=1, conaffinity=1)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.lo = np.array([self.m.jnt_range[self.m.joint(n).id][0] for n in ARM_JOINTS])
        self.hi = np.array([self.m.jnt_range[self.m.joint(n).id][1] for n in ARM_JOINTS])
        self.site = self.m.site("gripperframe").id
        self.table = self.m.geom("table").id
        g = self.m.joint("gripper")
        self.gripper_qadr = g.qposadr[0]
        self.gripper_range = self.m.jnt_range[g.id].copy()
        self._tips = (self.m.geom("fixed_jaw_sph_tip1").id, self.m.geom("moving_jaw_sph_tip1").id)
        self.max_opening = self.jaw_gap(self.gripper_range[1])

    def set(self, q_arm, gripper: float = 0.0):
        """Always sets the gripper too: the jaw mass changes the Jacobian's rounding, so a leftover gripper
        value from an earlier call would make the IK result depend on the call history."""
        self.d.qpos[self.qadr] = q_arm
        self.d.qpos[self.gripper_qadr] = gripper
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def fk(self, q_arm):
        self.set(q_arm)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jaw_gap(self, q_gripper: float) -> float:
        self.d.qpos[:] = 0.0
        self.d.qpos[self.gripper_qadr] = q_gripper
        mujoco.mj_kinematics(self.m, self.d)
        a, b = self._tips
        return float(np.linalg.norm(self.d.geom_xpos[a] - self.d.geom_xpos[b]))

    def gripper_for_opening(self, width: float) -> float:
        lo, hi = self.gripper_range
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.jaw_gap(mid) < width else (lo, mid)
        return 0.5 * (lo + hi)

    def ik(self, pos, approach, jaw=None, q0=None, iters: int = 120, pos_tol: float = 0.005,
           ang_tol_deg: float = 5.0):
        """Damped least squares on position, approach (site x) and optionally the jaw axis (site z, either sign).

        Returns (ok, q, pos_err, approach_err_deg, jaw_err_deg). Joints stay inside their limits.
        """
        pos, approach = np.asarray(pos, float), np.asarray(approach, float)
        jaw = None if jaw is None else np.asarray(jaw, float)
        q = np.clip(np.asarray(q0 if q0 is not None else SEED_POSES[0], float), self.lo, self.hi)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        w_rot = 0.05                                  # meters per radian
        for _ in range(iters):
            p, R = self.fk(q)
            x, z = R[:, 0], R[:, 2]
            e_rot = np.cross(x, approach)
            if jaw is not None:
                e_rot = e_rot + np.cross(z, jaw * np.sign(z @ jaw or 1.0))
            err = np.concatenate([pos - p, w_rot * e_rot])
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], w_rot * jacr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lo, self.hi)
        p, R = self.fk(q)
        e_pos = float(np.linalg.norm(pos - p))
        e_app = float(np.degrees(np.arccos(np.clip(R[:, 0] @ approach, -1, 1))))
        e_jaw = 0.0 if jaw is None else float(np.degrees(np.arccos(np.clip(abs(R[:, 2] @ jaw), 0, 1))))
        ok = e_pos < pos_tol and e_app < ang_tol_deg and e_jaw < ang_tol_deg
        return ok, q, e_pos, e_app, e_jaw

    def in_collision(self, q_arm, gripper: float, table_penetration: float = 0.003) -> bool:
        """Robot-robot (non-adjacent bodies) or robot-table contact deeper than the allowed penetration."""
        self.d.qpos[:] = 0.0
        self.d.qpos[self.qadr] = q_arm
        self.d.qpos[self.gripper_qadr] = gripper
        mujoco.mj_forward(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            limit = table_penetration if self.table in (c.geom1, c.geom2) else 0.001
            if c.dist < -limit:
                return True
        return False


class ReachabilityChecker:
    def __init__(self, cfg: dict, reach_map: "ReachabilityMap | None" = None):
        self.c = cfg["reach"]
        self.kin = SO101Kinematics(cfg["robot"]["table_z_m"])
        self.table_z = cfg["robot"]["table_z_m"]
        self.map = reach_map

    def grasp_candidates(self, shape: ObjectShape, pos, yaw: float):
        """(tilt, approach, jaw axis, needed opening) for each grasp the 5-DoF arm can give, top-down first."""
        pos = np.asarray(pos, float)
        out = []
        for tilt in self.c["approach_tilt_deg"]:
            a = approach_dir(pos, tilt)
            if tilt == 0:   # Top-down: wrist_roll turns the jaws; close across the narrower horizontal axis
                axes = [np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])]
                axes.sort(key=lambda u: shape.width_along(u, yaw))
            else:           # Tilted: the jaws close across the radial plane
                axes = [tangential_dir(pos)]
            for u in axes:
                out.append((float(tilt), a, u, shape.width_along(u, yaw) + 2 * self.c["jaw_clearance_m"]))
        return out

    def grasp_point(self, shape: ObjectShape, pos) -> np.ndarray:
        p = np.array(pos, float)
        p[2] = max(p[2], self.table_z + self.c["min_grasp_z_m"])
        return p

    def check(self, shape: ObjectShape, pos, yaw: float = 0.0, exact: bool = True) -> Reach:
        """Can the robot hold `shape` with its center at `pos` and this yaw? exact=False uses only the map."""
        gp = self.grasp_point(shape, pos)
        cands = self.grasp_candidates(shape, gp, yaw)
        fits = [c for c in cands if c[3] <= self.kin.max_opening]
        if not fits:
            return Reach(False, "gripper_width", tried=[c[0] for c in cands])
        if not exact:
            if self.map is None:
                raise ValueError("exact=False needs a reachability map")
            for tilt, a, u, _ in fits:
                if self.map.lookup(gp, tilt):
                    return Reach(True, "ok", tilt_deg=tilt, approach=a, jaw=u)
            return Reach(False, "ik", tried=[c[0] for c in fits])
        reason = "ik"
        for tilt, a, u, width in fits:
            if self.map is not None and not self.map.lookup(gp, tilt, margin=1):
                continue
            g = self.kin.gripper_for_opening(width)
            for q0 in SEED_POSES:
                ok, q, *_ = self.kin.ik(gp, a, u, q0=q0, pos_tol=self.c["pos_tol_m"],
                                        ang_tol_deg=self.c["ang_tol_deg"])
                if not ok:
                    continue
                if self.kin.in_collision(q, g, self.c["table_penetration_m"]):
                    reason = "collision"
                    continue
                return Reach(True, "ok", q=q, tilt_deg=tilt, approach=a, jaw=u)
        return Reach(False, reason, tried=[c[0] for c in fits])


class ReachabilityMap:
    """Boolean grid: is a grasp point reachable with each approach tilt (no jaw constraint, no object)."""

    def __init__(self, lo, voxel: float, tilts, grid: np.ndarray):
        self.lo, self.voxel = np.asarray(lo, float), float(voxel)
        self.tilts = np.asarray(tilts, float)
        self.grid = grid                       # (nx, ny, nz, n_tilt) bool

    @classmethod
    def build(cls, cfg: dict, verbose: bool = False) -> "ReachabilityMap":
        c = cfg["reach"]
        kin = SO101Kinematics(cfg["robot"]["table_z_m"])
        lo, hi, v = np.array(c["workspace_min_m"]), np.array(c["workspace_max_m"]), c["voxel_m"]
        n = np.floor((hi - lo) / v).astype(int) + 1
        tilts = list(c["approach_tilt_deg"])
        grid = np.zeros((*n, len(tilts)), bool)
        g_open = kin.gripper_range[1] * 0.3
        for k, tilt in enumerate(tilts):
            for i in range(n[0]):
                q_prev = None
                for j in range(n[1]):
                    for m in range(n[2]):
                        p = lo + v * np.array([i, j, m])
                        p[2] = max(p[2], cfg["robot"]["table_z_m"] + c["min_grasp_z_m"])
                        seeds = ([q_prev] if q_prev is not None else []) + list(SEED_POSES)
                        for q0 in seeds:
                            ok, q, *_ = kin.ik(p, approach_dir(p, tilt), None, q0=q0, iters=60,
                                               pos_tol=c["pos_tol_m"], ang_tol_deg=c["ang_tol_deg"])
                            if ok and not kin.in_collision(q, g_open, c["table_penetration_m"]):
                                grid[i, j, m, k], q_prev = True, q
                                break
            if verbose:
                print(f"tilt {tilt:4.0f} deg: {grid[..., k].mean():.0%} of voxels reachable")
        return cls(lo, v, tilts, grid)

    def index(self, pos):
        return np.round((np.asarray(pos, float) - self.lo) / self.voxel).astype(int)

    def lookup(self, pos, tilt: float, margin: int = 0) -> bool:
        """True if the nearest voxel (or, with margin > 0, any voxel within margin) is reachable."""
        k = int(np.argmin(abs(self.tilts - tilt)))
        idx = self.index(pos)
        n = np.array(self.grid.shape[:3])
        if np.any(idx < -margin) or np.any(idx > n - 1 + margin):
            return False
        a, b = np.clip(idx - margin, 0, n - 1), np.clip(idx + margin, 0, n - 1)
        return bool(self.grid[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1, k].any())

    def save(self, path=REACH_MAP_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, lo=self.lo, voxel=self.voxel, tilts=self.tilts, grid=self.grid)

    @classmethod
    def load(cls, path=REACH_MAP_FILE) -> "ReachabilityMap":
        z = np.load(path)
        return cls(z["lo"], float(z["voxel"]), z["tilts"], z["grid"])

    def matches(self, cfg: dict) -> bool:
        c = cfg["reach"]
        return (np.allclose(self.lo, c["workspace_min_m"]) and np.isclose(self.voxel, c["voxel_m"])
                and np.allclose(self.tilts, c["approach_tilt_deg"]))


def load_or_build_map(cfg: dict, path=REACH_MAP_FILE, verbose: bool = False) -> ReachabilityMap:
    if path.exists():
        rmap = ReachabilityMap.load(path)
        if rmap.matches(cfg):
            return rmap
    rmap = ReachabilityMap.build(cfg, verbose=verbose)
    rmap.save(path)
    return rmap


def main():
    p = argparse.ArgumentParser(description="Build the SO-101 reachability map")
    p.add_argument("--out", default=str(REACH_MAP_FILE))
    args = p.parse_args()
    cfg = load_general_config()
    rmap = ReachabilityMap.build(cfg, verbose=True)
    from pathlib import Path
    rmap.save(Path(args.out))
    print(f"Saved {args.out}: grid {rmap.grid.shape}")


if __name__ == "__main__":
    main()
