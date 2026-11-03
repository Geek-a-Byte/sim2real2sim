"""Goal-conditioned SO-101 env: move an object from any pose to any reachable pose (AP2AP, Dex4D).

A task is a sequence of object goals. The policy gets the goal as paired points: each object point is
[current xyz, target xyz, valid] (Paired Point Encoding). When the object reaches a goal, the next goal is
sampled near the achieved pose (chained goals), so one episode trains long sequences and recovery. Every
goal passes the reachable-goal filter (general/reachability.py).

Observation (Dict):
  points      (n_points, 7)  paired points; the student mode masks and adds noise (webcam view)
  proprio     (18,)          joint positions, joint velocities / 10, last action
  privileged  (23,)          object pose and velocity, goal, jaw contacts, gripper site (critic / teacher only)
Action: 6 values in [-1, 1]. robot.action_space "ee" (default): gripper motion dx, dy, dz (x ee_pos_step_m),
tilt and roll (x ee_rot_step_rad), mapped to the 5 arm joints by one damped least-squares step of the site
Jacobian; the last value is the gripper target. "joint": 5 arm joint-target changes (x action_scale_rad) and
the gripper target. The gripper site never goes below min_gripper_z_m. The gripper value: robot.gripper_action
"binary" (default: a < 0 closes fully, a >= 0 opens to gripper_open_m) or "continuous" (absolute target).

Reward: reach + grasp (small, per step held) + progress + success bonus, minus action-rate, joint-velocity
and time penalties. reach = reward.reach x (cm that the gripper got closer to the object this step), progress =
reward.progress x (cm that the mean point distance to the goal fell this step). Both are potential-based
shapings: no state pays just for being in it, so holding or hovering earns nothing, a lift toward a lifted
goal pays at once, and a new chained goal restarts the count (no loss for reaching a goal).

Assisted starts (training only), a reverse curriculum over the start state:
  grasped   (env.grasped_start_prob)  the object is already held by a top-down IK grasp: learn lift, carry, place
  pregrasp  (env.pregrasp_start_prob) the open jaws are around the object at grasp height: learn to close
  normal    (the rest)                the gripper starts at home: learn the approach
info["start"] names the start; assisted starts are left out of is_success, the curriculum and the trial-log
success rate (their records carry the start name in `notes`).

Home pose: the gripper starts home_clearance_m above the tallest training object, top-down if the arm can
reach that height, else tilted as little as possible.

Symmetric objects: the goal yaw is replaced by the equivalent yaw nearest to the object's yaw (a cube turned
90 deg, a bar turned 180 deg, a cylinder at any yaw is the same pose), so the point distance measures the pose.

The episode ends (terminated, failure "object_lost") when the object leaves the workspace, or lies on the
table at a pose that fails the reachable-goal filter. It is truncated when no reachable next goal exists.

reset(options={"task": TaskSpec}) starts a given task (the curriculum and rebuilt real failures use it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.slingpuck.general.common import (ARM_JOINTS, SO101_XML, ObjectShape, objects_from_config,
                                          quat_to_yaw, wrap_angle, yaw_matrix, yaw_quat)
from src.slingpuck.general.reachability import SEED_POSES, ReachabilityChecker, ReachabilityMap, approach_dir

HOME_XY = (0.20, 0.0)
HOME_CLEARANCE_M = 0.04                 # Home gripper height above the tallest training object
HOME_TILTS_DEG = (0.0, 15.0, 30.0, 45.0)  # Top-down if reachable (about 8 cm high at most), else the least tilt
ORIENT_GAIN, ORIENT_WEIGHT = 0.5, 0.02  # ee actions: orientation correction per step, and its weight (m/rad)
WINDUP_LIMIT_RAD = 0.1                 # Max joint-target lead over the measured joint (no windup at contacts)
STAGES = ("reach", "grasp", "transport", "place")


def _rotate(v, axis, angle: float) -> np.ndarray:
    """Rodrigues rotation of v about a unit axis."""
    if angle == 0.0:
        return np.asarray(v, float)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    v = np.asarray(v, float)
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * (k @ v) * (1 - np.cos(angle))


@dataclass
class TaskSpec:
    """One task: an object, its start pose on the table, and a goal sequence (empty = sample chained goals)."""
    obj: str
    start_xy: tuple
    start_yaw: float
    goals: list = field(default_factory=list)     # [(x, y, z, yaw), ...]

    def to_dict(self) -> dict:
        return {"obj": self.obj, "start_xy": [float(v) for v in self.start_xy], "start_yaw": float(self.start_yaw),
                "goals": [[float(v) for v in g] for g in self.goals]}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(d["obj"], tuple(d["start_xy"]), float(d["start_yaw"]), [tuple(g) for g in d.get("goals", [])])


class GoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: dict, student: bool = False, reach_map: ReachabilityMap | None = None,
                 objects: list[str] | None = None, render_mode: str | None = None,
                 grasped_start_prob: float | None = None, pregrasp_start_prob: float | None = None):
        """grasped_start_prob, pregrasp_start_prob: None = the config's value (training); 0 for evaluation."""
        super().__init__()
        self.grasped_start_prob = cfg["env"]["grasped_start_prob"] if grasped_start_prob is None \
            else grasped_start_prob
        self.pregrasp_start_prob = cfg["env"]["pregrasp_start_prob"] if pregrasp_start_prob is None \
            else pregrasp_start_prob
        self.cfg, self.e, self.w = cfg, cfg["env"], cfg["reward"]
        self.student = student
        self.render_mode = render_mode
        self.shapes: dict[str, ObjectShape] = objects_from_config(cfg)
        self.object_names = list(objects or self.shapes)
        self.checker = ReachabilityChecker(cfg, reach_map)
        n = self.e["n_points"]
        self.local_points = {k: s.surface_points(n, seed=i) for i, (k, s) in enumerate(self.shapes.items())}
        self._build()

        self.task_pool: list[TaskSpec] | None = None
        self.observation_space = spaces.Dict({
            "points": spaces.Box(-np.inf, np.inf, (n, 7), np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, (18,), np.float32),
            "privileged": spaces.Box(-np.inf, np.inf, (23,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self._renderer = None

    # --------------------------------------------------------------- scene
    def _build(self):
        spec = mujoco.MjSpec.from_file(str(SO101_XML))
        spec.option.timestep = self.cfg["robot"]["sim_dt_s"]
        spec.option.impratio = self.cfg["robot"]["impratio"]     # Stable grasps (see configs/general.yaml)
        spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{self.cfg['robot']['friction_cone'].upper()}")
        w = spec.worldbody
        tz = self.cfg["robot"]["table_z_m"]
        w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, tz],
                   rgba=[0.42, 0.40, 0.37, 1])
        w.add_light(name="overhead", pos=[0, 0, 1.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        w.add_camera(name="overview", pos=[0.55, -0.45, 0.35],      # Looks at (0.25, 0, 0.03)
                     xyaxes=[0.832, 0.555, 0, -0.283, 0.423, 0.860])
        for name, s in self.shapes.items():   # One body per object; unused objects wait far away
            b = w.add_body(name=f"obj_{name}", pos=[2.0, 0, s.half_height])
            b.add_freejoint(name=f"obj_{name}")
            if s.kind == "box":
                b.add_geom(name=f"obj_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(s.half), mass=s.mass_kg,
                           rgba=[0.15, 0.45, 0.85, 1], friction=[1.0, 0.005, 0.0001])
            else:
                b.add_geom(name=f"obj_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[s.half[0], s.half[2], 0],
                           mass=s.mass_kg, rgba=[0.15, 0.45, 0.85, 1], friction=[1.0, 0.005, 0.0001])
            g = w.add_body(name=f"goal_{name}", mocap=True, pos=[2.0, 0.5, s.half_height])
            gtype = mujoco.mjtGeom.mjGEOM_BOX if s.kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
            size = list(s.half) if s.kind == "box" else [s.half[0], s.half[2], 0]
            g.add_geom(type=gtype, size=size, rgba=[0.2, 0.85, 0.3, 0.3], contype=0, conaffinity=0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.n_sub = max(1, int(round(self.cfg["robot"]["control_dt_s"] / self.m.opt.timestep)))
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in (*ARM_JOINTS, "gripper")])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in (*ARM_JOINTS, "gripper")])
        self.ctrl_lo, self.ctrl_hi = self.m.actuator_ctrlrange[:, 0].copy(), self.m.actuator_ctrlrange[:, 1].copy()
        self.site = self.m.site("gripperframe").id
        jaw_bodies = (self.m.geom_bodyid[self.m.geom("fixed_jaw_box1").id],
                      self.m.geom_bodyid[self.m.geom("moving_jaw_box1").id])
        # Jaw collision geoms (priority 1); the gripper body also holds the wrist motor box, which is not a jaw
        self.jaw_geoms = tuple(frozenset(g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b
                                         and self.m.geom_priority[g] == 1 and self.m.geom_contype[g])
                               for b in jaw_bodies)
        self.home = self._home_pose()
        self.g_closed = self.ctrl_lo[5]                                       # Fully closed: full grip force
        self.g_open = self.checker.kin.gripper_for_opening(self.cfg["robot"]["gripper_open_m"])

    def _home_pose(self) -> np.ndarray:
        kin = self.checker.kin
        top = max(2 * self.shapes[n].half_height for n in self.object_names)
        p = (*HOME_XY, self.cfg["robot"]["table_z_m"] + top + HOME_CLEARANCE_M)
        for tilt in HOME_TILTS_DEG:
            for q0 in SEED_POSES:
                ok, q, *_ = kin.ik(p, approach_dir(p, tilt), None, q0=q0)
                if ok and not kin.in_collision(q, 0.8):
                    self.home_point, self.home_tilt = p, tilt
                    return np.concatenate([q, [0.8]])
        raise RuntimeError(f"No IK solution for the home point {p}")

    # --------------------------------------------------------------- reset
    def set_task_pool(self, pool: list[TaskSpec] | None):
        """The curriculum puts its sampled tasks here; reset() draws from them."""
        self.task_pool = pool

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        task = options.get("task")
        if task is None and self.task_pool:
            task = self.task_pool[int(self.np_random.integers(len(self.task_pool)))]
        if task is None:
            task = self.random_task()
        self.task = task if isinstance(task, TaskSpec) else TaskSpec.from_dict(task)
        self.shape = self.shapes[self.task.obj]
        self.points_local = self.local_points[self.task.obj]

        mujoco.mj_resetData(self.m, self.d)
        for name in self.shapes:     # Park the other objects
            self._set_object(name, (2.0 + 0.2 * list(self.shapes).index(name), 0.0, self.shapes[name].half_height),
                             0.0)
        z0 = self.cfg["robot"]["table_z_m"] + self.shape.half_height
        self._set_object(self.task.obj, (*self.task.start_xy, z0), self.task.start_yaw)
        self.d.qpos[self.qadr] = self.home
        self.d.ctrl[:] = self.home
        mujoco.mj_forward(self.m, self.d)
        self.start = "normal"
        u = self.np_random.random()
        if u < self.grasped_start_prob:
            self.start = "grasped" if self._start_assisted(closed=True) else "normal"
        elif u < self.grasped_start_prob + self.pregrasp_start_prob:
            self.start = "pregrasp" if self._start_assisted(closed=False) else "normal"

        R = self.d.site_xmat[self.site].reshape(3, 3)
        self.ee_x, self.ee_z = R[:, 0].copy(), R[:, 2].copy()     # Target gripper orientation (ee actions)
        self.prev_d_reach = float(np.linalg.norm(self.d.site_xpos[self.site] - self.object_pose()[0]))
        self.goal_queue = list(self.task.goals)
        self.goal = None
        self.steps, self.goals_reached = 0, 0
        self.prev_action = np.zeros(6)
        self._next_goal()
        return self._obs(), self._info()

    def _start_assisted(self, closed: bool) -> bool:
        """Put the arm in a top-down IK grasp pose around the object. closed=True: close the jaws and require a
        two-jaw grasp. closed=False: leave the jaws open (the policy must close them) and require that the
        object did not move. False (and back to the home pose) if this fails."""
        pos, yaw = self.object_pose()
        cands = [c for c in self.checker.grasp_candidates(self.shape, self.checker.grasp_point(self.shape, pos), yaw)
                 if c[0] == 0 and c[3] <= self.checker.kin.max_opening]
        kin = self.checker.kin
        for _, approach, jaw, width in cands:
            gp = self.checker.grasp_point(self.shape, pos)
            for q0 in SEED_POSES:
                ok, q, *_ = kin.ik(gp, approach, jaw, q0=q0)
                if not ok:
                    continue
                g_open = max(kin.gripper_for_opening(width), self.g_open)
                self.d.qpos[self.qadr] = np.r_[q, g_open]
                self.d.ctrl[:] = np.r_[q, self.g_closed if closed else self.g_open]
                mujoco.mj_forward(self.m, self.d)
                mujoco.mj_step(self.m, self.d, nstep=10 * self.n_sub)       # Close (or settle) for 0.5 s
                moved = np.linalg.norm(self.object_pose()[0][:2] - pos[:2])
                if (all(self._jaw_contacts()) if closed else moved < 0.005 and not any(self._jaw_contacts())):
                    return True
                break
        mujoco.mj_resetData(self.m, self.d)                                 # No grasp: normal start
        for name in self.shapes:
            self._set_object(name, (2.0 + 0.2 * list(self.shapes).index(name), 0.0, self.shapes[name].half_height),
                             0.0)
        self._set_object(self.task.obj, (*self.task.start_xy, self.cfg["robot"]["table_z_m"] + self.shape.half_height),
                         self.task.start_yaw)
        self.d.qpos[self.qadr] = self.home
        self.d.ctrl[:] = self.home
        mujoco.mj_forward(self.m, self.d)
        return False

    def random_task(self) -> TaskSpec:
        lo, hi = np.array(self.e["start_min_m"]), np.array(self.e["start_max_m"])
        name = self.object_names[int(self.np_random.integers(len(self.object_names)))]
        shape = self.shapes[name]
        z0 = self.cfg["robot"]["table_z_m"] + shape.half_height
        for _ in range(10 * self.e["goal_tries"]):
            xy, yaw = self.np_random.uniform(lo, hi), float(self.np_random.uniform(-np.pi, np.pi))
            if self._reachable(shape, (*xy, z0), yaw):
                return TaskSpec(name, tuple(float(v) for v in xy), yaw)
        raise RuntimeError(f"No reachable start pose for {name!r} in env.start_min_m / start_max_m")

    def _reachable(self, shape, pos, yaw) -> bool:
        return self.checker.check(shape, pos, yaw, exact=self.checker.map is None).ok

    def sample_goal_near(self, pos, yaw):
        """Next goal near the achieved pose: on the table or lifted, reachable. If there is none (the object
        is near the edge of the reach), a reachable goal anywhere in the start region. None if neither exists.
        The current pose is never a fallback: it would count as reached on every step."""
        tz = self.cfg["robot"]["table_z_m"] + self.shape.half_height
        lo, hi = np.array(self.e["start_min_m"]), np.array(self.e["start_max_m"])
        for k in range(2 * self.e["goal_tries"]):
            if k < self.e["goal_tries"]:
                a = self.np_random.uniform(0, 2 * np.pi)
                r = self.e["chain_radius_m"] * np.sqrt(self.np_random.random())
                x, y = pos[0] + r * np.cos(a), pos[1] + r * np.sin(a)
            else:
                x, y = self.np_random.uniform(lo, hi)
            z = tz + (self.np_random.uniform(0.02, self.e["lift_max_m"]) if self.np_random.random() < self.e["lift_prob"]
                      else 0.0)
            g_yaw = float(wrap_angle(yaw + self.np_random.uniform(-1, 1) * self.e["chain_yaw_rad"]))
            if self._reachable(self.shape, (x, y, z), g_yaw):
                return (float(x), float(y), float(z), g_yaw)
        return None

    def _next_goal(self):
        p, yaw = self.object_pose()
        self.no_goal = False
        if self.goal_queue:
            goal = tuple(self.goal_queue.pop(0))
        else:
            goal = self.sample_goal_near(p, yaw)
        if goal is None:          # Keep the old goal for the observation; step() truncates the episode
            self.no_goal = True
            goal = self.goal if self.goal is not None else (float(p[0]), float(p[1]) + 1.0, float(p[2]), float(yaw))
        self.goal = (*goal[:3], self.shape.equivalent_yaw(goal[3], yaw))
        self.goal_steps, self.goal_contact, self.goal_lifted = 0, False, False
        self.prev_d_goal = float(np.linalg.norm(self._object_points() - self._goal_points(), axis=1).mean())
        mid = self.m.body(f"goal_{self.task.obj}").mocapid[0]
        self.d.mocap_pos[mid] = self.goal[:3]
        self.d.mocap_quat[mid] = yaw_quat(self.goal[3])

    def _set_object(self, name, pos, yaw):
        j = self.m.joint(f"obj_{name}")
        a = j.qposadr[0]
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = yaw_quat(yaw)
        self.d.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0.0

    # ---------------------------------------------------------------- step
    def object_pose(self):
        j = self.m.joint(f"obj_{self.task.obj}")
        a = j.qposadr[0]
        return self.d.qpos[a:a + 3].copy(), quat_to_yaw(self.d.qpos[a + 3:a + 7])

    def _object_points(self):
        b = self.m.body(f"obj_{self.task.obj}").id
        R = self.d.xmat[b].reshape(3, 3)
        return self.points_local @ R.T + self.d.xpos[b]

    def _goal_points(self):
        return self.points_local @ yaw_matrix(self.goal[3]).T + np.asarray(self.goal[:3])

    def _jaw_contacts(self):
        obj = self.m.geom(f"obj_{self.task.obj}").id
        touch = [False, False]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            other = c.geom2 if c.geom1 == obj else c.geom1 if c.geom2 == obj else None
            for k, geoms in enumerate(self.jaw_geoms):
                if other in geoms:
                    touch[k] = True
        return touch

    def ee_to_joint(self, u) -> np.ndarray:
        """Joint-target change for a gripper motion u = (dx, dy, dz, tilt, roll) in [-1, 1]."""
        r = self.cfg["robot"]
        site_p = self.d.site_xpos[self.site]
        v = np.asarray(u[:3]) * r["ee_pos_step_m"]
        floor = r["table_z_m"] + r["min_gripper_z_m"]
        v[2] = max(v[2], floor - site_p[2])                    # Never below the floor
        R = self.d.site_xmat[self.site].reshape(3, 3)
        t = np.array([-site_p[1], site_p[0], 0.0]) / max(np.hypot(site_p[0], site_p[1]), 1e-6)  # Tilt axis
        # Target orientation: tilt and roll turn it; every step corrects the gripper toward it (no drift)
        self.ee_x = _rotate(_rotate(self.ee_x, t, u[3] * r["ee_rot_step_rad"]), self.ee_x, 0.0)
        self.ee_z = _rotate(_rotate(self.ee_z, t, u[3] * r["ee_rot_step_rad"]), self.ee_x, u[4] * r["ee_rot_step_rad"])
        w = ORIENT_GAIN * (np.cross(R[:, 0], self.ee_x) + np.cross(R[:, 2], self.ee_z))
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
        w_rot = ORIENT_WEIGHT                                  # Position has priority over orientation
        J = np.vstack([jacp[:, self.dadr[:5]], w_rot * jacr[:, self.dadr[:5]]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([v, w_rot * w]))
        return np.clip(dq, -0.2, 0.2)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        target = self.d.ctrl.copy()
        q = self.d.qpos[self.qadr[:5]]                # Keep the last target (no gravity sag), but no further than
        target[:5] = np.clip(target[:5], q - WINDUP_LIMIT_RAD, q + WINDUP_LIMIT_RAD)   # this from the joints
        if self.cfg["robot"]["action_space"] == "ee":
            target[:5] += self.ee_to_joint(a[:5])
        else:
            target[:5] += a[:5] * self.cfg["robot"]["action_scale_rad"]
        if self.cfg["robot"]["gripper_action"] == "binary":
            target[5] = self.g_closed if a[5] < 0 else self.g_open
        else:
            target[5] = self.ctrl_lo[5] + 0.5 * (a[5] + 1) * (self.ctrl_hi[5] - self.ctrl_lo[5])
        self.d.ctrl[:] = np.clip(target, self.ctrl_lo, self.ctrl_hi)
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        self.steps += 1
        self.goal_steps += 1

        pos, _ = self.object_pose()
        cur, tgt = self._object_points(), self._goal_points()
        d_goal = float(np.linalg.norm(cur - tgt, axis=1).mean())
        d_reach = float(np.linalg.norm(self.d.site_xpos[self.site] - pos))
        contacts = self._jaw_contacts()
        grasped = all(contacts)
        tz = self.cfg["robot"]["table_z_m"] + self.shape.half_height
        lift_needed = self.goal[2] > tz + 0.01
        lifted = pos[2] > tz + 0.01
        self.goal_contact |= any(contacts)
        self.goal_lifted |= lifted

        w = self.w
        progress = (self.prev_d_goal - d_goal) / 0.01                # cm closer to the goal this step
        self.prev_d_goal = d_goal
        reach = (self.prev_d_reach - d_reach) / 0.01                 # cm closer to the object this step
        self.prev_d_reach = d_reach
        reward = (w["reach"] * reach + w["grasp"] * grasped + w["progress"] * progress
                  - w["action_rate"] * float(np.sum((a - self.prev_action) ** 2))
                  - w["joint_vel"] * float(np.sum(self.d.qvel[self.dadr] ** 2)) - w["time"])
        self.prev_action = a
        success = d_goal < self.e["success_dist_m"] and not self.no_goal
        if success:
            reward += w["success"]
            self.goals_reached += 1
            self._next_goal()
        terminated = self.object_lost(pos, grasped)
        truncated = not terminated and (self.steps >= self.e["max_steps"] or self.no_goal)
        info = self._info(d_goal=d_goal, success=success, lift_needed=lift_needed)
        info["goal_reached_now"] = bool(success)
        if terminated or truncated:   # SB3 reads is_success from the last step: did the episode reach a goal?
            info["is_success"] = None if self.start != "normal" else self.goals_reached >= 1   # None: SB3 skips it
        if terminated:
            info["failure"] = "object_lost"
        return self._obs(), float(reward), terminated, truncated, info

    def object_lost(self, pos, grasped: bool) -> bool:
        """Out of the workspace, or resting on the table where the robot cannot grasp it."""
        lo, hi = np.array(self.cfg["reach"]["workspace_min_m"]), np.array(self.cfg["reach"]["workspace_max_m"])
        if np.any(pos[:2] < lo[:2] - 0.1) or np.any(pos[:2] > hi[:2] + 0.1):
            return True
        on_table = pos[2] < self.cfg["robot"]["table_z_m"] + self.shape.half_height + 0.005
        if self.checker.map is not None and on_table and not grasped:
            return not self.checker.check(self.shape, pos, self.object_pose()[1], exact=False).ok
        return False

    def stage(self, d_goal: float, lift_needed: bool) -> str:
        """Where the current goal stands: the failure stage if the episode ends now."""
        if not self.goal_contact:
            return "reach"
        if lift_needed and not self.goal_lifted:
            return "grasp"
        return "transport" if d_goal > 2.5 * self.e["success_dist_m"] else "place"

    # --------------------------------------------------------- observation
    def _obs(self):
        cur, tgt = self._object_points(), self._goal_points()
        valid = np.ones(len(cur))
        if self.student:
            s = self.cfg["student"]
            cur = cur + self.np_random.normal(0, s["point_noise_m"], cur.shape)
            if self.np_random.random() < s["plane_mask_prob"]:
                n = self.np_random.normal(size=3)
                valid *= ((cur - cur.mean(0)) @ n) > 0
            valid *= self.np_random.random(len(cur)) > s["drop_frac"]
            if not valid.any():
                valid[int(self.np_random.integers(len(valid)))] = 1.0
        pts = np.concatenate([cur * valid[:, None], tgt, valid[:, None]], axis=1)   # Targets are always known
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.dadr]
        proprio = np.concatenate([q, 0.1 * qd, self.prev_action])
        j = self.m.joint(f"obj_{self.task.obj}")
        a, v = j.qposadr[0], j.dofadr[0]
        g_yaw = self.goal[3]
        priv = np.concatenate([self.d.qpos[a:a + 7], self.d.qvel[v:v + 6], self.goal[:3],
                               [np.sin(g_yaw), np.cos(g_yaw)], np.array(self._jaw_contacts(), float),
                               self.d.site_xpos[self.site]])
        return {"points": pts.astype(np.float32), "proprio": proprio.astype(np.float32),
                "privileged": priv.astype(np.float32)}

    def _info(self, d_goal=None, success=False, lift_needed=False):
        if d_goal is None:
            d_goal = float(np.linalg.norm(self._object_points() - self._goal_points(), axis=1).mean())
        pos, yaw = self.object_pose()
        return {"task": self.task.to_dict(), "goal": list(self.goal), "goal_index": self.goals_reached,
                "goals_reached": self.goals_reached, "d_goal": d_goal, "is_success": bool(success),
                "stage": self.stage(d_goal, lift_needed), "object_pose": [*map(float, pos), float(yaw)],
                "start": getattr(self, "start", "normal"), "grasped_start": getattr(self, "start", "") == "grasped"}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
        self._renderer.update_scene(self.d, camera="overview")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
