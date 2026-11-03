"""Goal-conditioned SO-101 env: move an object from any pose to any reachable pose (AP2AP, Dex4D).

A task is a sequence of object goals. The policy gets the goal as paired points: each object point is
[current xyz, target xyz, valid] (Paired Point Encoding). When the object reaches a goal, the next goal is
sampled near the achieved pose (chained goals), so one episode trains long sequences and recovery. Every
goal passes the reachable-goal filter (general/reachability.py).

Observation (Dict):
  points      (n_points, 7)  paired points; the student mode masks and adds noise (webcam view)
  proprio     (18,)          joint positions, joint velocities / 10, last action
  privileged  (23,)          object pose and velocity, goal, jaw contacts, gripper site (critic / teacher only)
Action: 6 values in [-1, 1]: arm joint-target changes (x action_scale_rad) and the gripper target.

Reward (staged, SO-101 PPO paper): reach + grasp + lift + track + success bonus, minus action-rate,
joint-velocity and time penalties.

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

HOME_GRASP_POINT = (0.22, 0.0, 0.08)    # Top-down reach ends at about 8 cm height
STAGES = ("reach", "grasp", "transport", "place")


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
                 objects: list[str] | None = None, render_mode: str | None = None):
        super().__init__()
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
        w = spec.worldbody
        tz = self.cfg["robot"]["table_z_m"]
        w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, tz],
                   rgba=[0.42, 0.40, 0.37, 1])
        w.add_light(name="overhead", pos=[0, 0, 1.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        w.add_camera(name="overview", pos=[0.75, -0.55, 0.55], xyaxes=[0.6, 0.8, 0, -0.35, 0.26, 0.9])
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
        self.jaw_geoms = (self.m.geom("fixed_jaw_box1").id, self.m.geom("moving_jaw_box1").id)
        self.jaw_bodies = (self.m.geom_bodyid[self.jaw_geoms[0]], self.m.geom_bodyid[self.jaw_geoms[1]])
        kin = self.checker.kin
        for q0 in SEED_POSES:
            ok, q, *_ = kin.ik(HOME_GRASP_POINT, approach_dir(HOME_GRASP_POINT, 0.0), None, q0=q0)
            if ok:
                break
        if not ok:
            raise RuntimeError(f"No IK solution for the home grasp point {HOME_GRASP_POINT}")
        self.home = np.concatenate([q, [0.8]])

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

        self.goal_queue = list(self.task.goals)
        self.goal = None
        self.steps, self.goals_reached = 0, 0
        self.prev_action = np.zeros(6)
        self._next_goal()
        return self._obs(), self._info()

    def random_task(self) -> TaskSpec:
        lo, hi = np.array(self.e["start_min_m"]), np.array(self.e["start_max_m"])
        name = self.object_names[int(self.np_random.integers(len(self.object_names)))]
        shape = self.shapes[name]
        z0 = self.cfg["robot"]["table_z_m"] + shape.half_height
        for _ in range(self.e["goal_tries"]):
            xy, yaw = self.np_random.uniform(lo, hi), float(self.np_random.uniform(-np.pi, np.pi))
            if self._reachable(shape, (*xy, z0), yaw):
                break
        return TaskSpec(name, tuple(float(v) for v in xy), yaw)

    def _reachable(self, shape, pos, yaw) -> bool:
        return self.checker.check(shape, pos, yaw, exact=self.checker.map is None).ok

    def sample_goal_near(self, pos, yaw):
        """Next goal near the achieved pose: on the table or lifted, reachable."""
        tz = self.cfg["robot"]["table_z_m"] + self.shape.half_height
        for _ in range(self.e["goal_tries"]):
            a = self.np_random.uniform(0, 2 * np.pi)
            r = self.e["chain_radius_m"] * np.sqrt(self.np_random.random())
            x, y = pos[0] + r * np.cos(a), pos[1] + r * np.sin(a)
            z = tz + (self.np_random.uniform(0.02, self.e["lift_max_m"]) if self.np_random.random() < self.e["lift_prob"]
                      else 0.0)
            g_yaw = float(wrap_angle(yaw + self.np_random.uniform(-1, 1) * self.e["chain_yaw_rad"]))
            if self._reachable(self.shape, (x, y, z), g_yaw):
                return (float(x), float(y), float(z), g_yaw)
        return (float(pos[0]), float(pos[1]), float(tz), float(yaw))   # Fallback: hold the current pose

    def _next_goal(self):
        if self.goal_queue:
            self.goal = tuple(self.goal_queue.pop(0))
        else:
            p, yaw = self.object_pose()
            self.goal = self.sample_goal_near(p, yaw)
        self.goal_steps, self.goal_contact, self.goal_lifted = 0, False, False
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
        obj = self.m.body(f"obj_{self.task.obj}").id
        touch = [False, False]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for k, jb in enumerate(self.jaw_bodies):
                if {b1, b2} == {jb, obj}:
                    touch[k] = True
        return touch

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        target = self.d.ctrl.copy()
        target[:5] += a[:5] * self.cfg["robot"]["action_scale_rad"]
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
        reward = (w["reach"] * (1 - np.tanh(d_reach / 0.1)) + w["grasp"] * grasped
                  + w["lift"] * (lifted and lift_needed) + w["track"] * (1 - np.tanh(d_goal / 0.05))
                  - w["action_rate"] * float(np.sum((a - self.prev_action) ** 2))
                  - w["joint_vel"] * float(np.sum(self.d.qvel[self.dadr] ** 2)) - w["time"])
        self.prev_action = a
        success = d_goal < self.e["success_dist_m"]
        if success:
            reward += w["success"]
            self.goals_reached += 1
            self._next_goal()
        lo, hi = np.array(self.cfg["reach"]["workspace_min_m"]), np.array(self.cfg["reach"]["workspace_max_m"])
        off = pos[2] < self.cfg["robot"]["table_z_m"] - 0.05 or np.any(pos[:2] < lo[:2] - 0.1) or \
            np.any(pos[:2] > hi[:2] + 0.1)
        terminated = bool(off)
        truncated = not terminated and self.steps >= self.e["max_steps"]
        info = self._info(d_goal=d_goal, success=success, lift_needed=lift_needed)
        if terminated:
            info["failure"] = "object_lost"
        return self._obs(), float(reward), terminated, truncated, info

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
                "stage": self.stage(d_goal, lift_needed), "object_pose": [*map(float, pos), float(yaw)]}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
        self._renderer.update_scene(self.d, camera="overview")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
