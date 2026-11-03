"""Scripted expert for the goal env: a privileged controller that acts through the policy's own actions.

It reads the true sim state (object pose, goal, gripper frame), like the privileged teacher, and outputs the
same 6 end-effector actions as the policy (dx, dy, dz, tilt, roll, gripper). Its episodes are the demos for
the behaviour-cloning warm start (general/bc.py).

Phases (per grasp):
  approach  above the object (object top + 3 cm), jaws open, tilt to the grasp tilt, roll the jaw axis
            across the object's narrow side
  descend   to the grasp point, jaws open
  close     close the jaws for a few steps
  carry     move the object center to the goal; tilt toward the tilt that the goal needs (high goals cannot be
            reached top-down). Chained goals follow without a release.
If the grasp is lost, it starts again from approach.

Limits: rolling the gripper does not turn a held box (it slips), so a box keeps the yaw of its grasp; goals that
fail the reachable-goal filter are not attempted.
"""
from __future__ import annotations

import numpy as np

from src.slingpuck.general.common import wrap_angle

POS_GAIN_M = 0.01          # Error (m) that gives a full-scale move command
ROT_GAIN_RAD = 0.1         # Error (rad) that gives a full-scale tilt or roll command
APPROACH_CLEAR_M = 0.02    # 3 cm was out of top-down reach for pucks 27-33 cm out: the approach never ended
APPROACH_MAX_STEPS = 25    # Then descend anyway
CARRY_SPEED = 0.6          # Fraction of the full move command while carrying (full speed dropped more pucks)
PRESS_CLEAR_M = 0.003      # Carry target height above rest: pressing a held puck onto the table pushed it
                           # out of the jaws (the most common drop)
CLOSE_STEPS = 6
LOST_STEPS = 8             # Steps without a two-jaw grasp in carry before a new approach


class ScriptedExpert:
    def __init__(self, env, noise: float = 0.0, rng: np.random.Generator | None = None):
        """noise: std of Gaussian noise added to the executed action (DART); act() returns the clean action."""
        self.env, self.noise = env, noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.phase, self.count, self.lost = "approach", 0, 0     # count: steps in the current phase
        self.plan = None
        if getattr(self.env, "start", "normal") == "grasped":
            self.phase = "carry"
        elif getattr(self.env, "start", "normal") == "pregrasp":
            self.phase = "close"

    # ------------------------------------------------------------------ geometry
    def _frame(self):
        d, s = self.env.d, self.env.site
        return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()

    def _tilt_deg(self, site_p, R) -> float:
        a = R[:, 0]
        r = np.array([site_p[0], site_p[1], 0.0]) / max(np.hypot(site_p[0], site_p[1]), 1e-6)
        return float(np.degrees(np.arctan2(a @ r, -a[2])))

    def _grasp_plan(self):
        """Grasp point, tilt and jaw axis for the object where it lies (reachable-goal filter candidates)."""
        env = self.env
        pos, yaw = env.object_pose()
        ch = env.checker
        gp = ch.grasp_point(env.shape, pos)
        above = np.array([gp[0], gp[1], pos[2] + env.shape.half_height + APPROACH_CLEAR_M])
        for tilt, approach, jaw, width in ch.grasp_candidates(env.shape, gp, yaw):
            # The tilt must reach the grasp point AND the approach point above the object (a tall tube's
            # approach point is above the top-down reach)
            if width <= ch.kin.max_opening and (ch.map is None or (ch.map.lookup(gp, tilt) and ch.map.lookup(above, tilt))):
                return gp, tilt, jaw
        return gp, 0.0, np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    def _goal_tilt(self) -> float:
        env = self.env
        r = env.checker.check(env.shape, env.goal[:3], env.goal[3], exact=env.checker.map is None)
        return float(r.tilt_deg) if r.ok else 0.0

    # ------------------------------------------------------------------- control
    def _orient(self, tilt_des, jaw_des, site_p, R):
        u3 = np.clip(np.radians(self._tilt_deg(site_p, R) - tilt_des) / ROT_GAIN_RAD, -1, 1)
        u4 = 0.0
        if jaw_des is not None:
            z = R[:, 2]
            cur = np.arctan2(z[1], z[0])
            want = np.arctan2(jaw_des[1], jaw_des[0])
            err = float(wrap_angle(2 * (cur - want)) / 2)            # The jaws are symmetric: modulo pi
            u4 = np.clip(err / ROT_GAIN_RAD, -1, 1)
        return u3, u4

    def act(self, obs=None) -> np.ndarray:
        env = self.env
        site_p, R = self._frame()
        pos, _ = env.object_pose()
        grasped = all(env._jaw_contacts())
        if self.plan is None or self.phase == "approach":
            self.plan = self._grasp_plan()
        gp, g_tilt, jaw = self.plan
        top = pos[2] + env.shape.half_height

        if self.phase == "approach":
            target = np.array([gp[0], gp[1], top + APPROACH_CLEAR_M])
            u3, u4 = self._orient(g_tilt, jaw, site_p, R)
            self.count += 1
            near = np.linalg.norm(target[:2] - site_p[:2]) < 0.006 and abs(target[2] - site_p[2]) < 0.015
            if (near and abs(u3) < 0.5 and abs(u4) < 0.5) or self.count > APPROACH_MAX_STEPS:
                self.phase, self.count = "descend", 0
            return np.r_[np.clip((target - site_p) / POS_GAIN_M, -1, 1), u3, u4, 1.0]
        if self.phase == "descend":
            u3, u4 = self._orient(g_tilt, jaw, site_p, R)
            e = gp - site_p
            if np.linalg.norm(e) < 0.003:
                self.phase, self.count = "close", 0
            return np.r_[np.clip(e / POS_GAIN_M, -1, 1), u3, u4, 1.0]
        if self.phase == "close":
            self.count += 1
            if self.count >= CLOSE_STEPS:
                self.phase, self.lost = "carry", 0
            return np.r_[0, 0, 0, 0, 0, -1.0]
        # carry
        self.lost = 0 if grasped else self.lost + 1
        if self.lost > LOST_STEPS:
            self.phase, self.plan, self.count = "approach", None, 0
            return np.r_[0, 0, 1.0, 0, 0, 1.0]
        target = np.asarray(env.goal[:3], float).copy()
        rest = env.cfg["robot"]["table_z_m"] + env.shape.half_height
        target[2] = max(target[2], rest + PRESS_CLEAR_M)      # Never press a held object into the table
        e = target - pos
        u3, _ = self._orient(self._goal_tilt(), None, site_p, R)
        return np.r_[CARRY_SPEED * np.clip(e / POS_GAIN_M, -1, 1), u3, 0.0, -1.0]

    def act_noisy(self, obs=None):
        """(executed action with noise, clean expert action)."""
        a = self.act(obs)
        if self.noise <= 0:
            return a, a
        x = a.copy()
        x[:5] = np.clip(x[:5] + self.rng.normal(0, self.noise, 5), -1, 1)
        return x, a


def run_expert(env, episodes: int, seed: int = 0, noise: float = 0.0, max_steps: int | None = None,
               record: bool = False):
    """Run the expert. Returns per-episode stats, and (obs, clean action) pairs if record is set."""
    rng = np.random.default_rng(seed)
    stats, data = [], {"points": [], "proprio": [], "privileged": [], "action": []}
    for k in range(episodes):
        obs, info = env.reset(seed=seed + k)
        ex = ScriptedExpert(env, noise=noise, rng=rng)
        goals, steps = 0, 0
        for t in range(max_steps or env.e["max_steps"]):
            a_exec, a_clean = ex.act_noisy(obs)
            if record:
                for key in ("points", "proprio", "privileged"):
                    data[key].append(obs[key])
                data["action"].append(a_clean.astype(np.float32))
            obs, r, te, tr, info = env.step(a_exec)
            goals += info["goal_reached_now"]
            steps += 1
            if te or tr:
                break
        stats.append({"goals": goals, "steps": steps, "lost": bool(info.get("failure")), "start": info["start"]})
    if record:
        data = {k: np.asarray(v) for k, v in data.items()}
        return stats, data
    return stats
