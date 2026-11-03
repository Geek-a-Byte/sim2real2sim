"""Task curricula for the goal-conditioned policy (the three arms of the main experiment).

  uniform   (A) sample tasks uniformly
  atr       (B) ATR, sim only: score = feasibility + w_div * diversity
  failure   (C) this project: score = feasibility + w_div * diversity + w_fail * closeness to real failures

feasibility  predicted success of the current policy on the task: the mean success of the k nearest tasks
             already trained on (ATR learns a value function for this; k-NN is the simple stand-in).
diversity    how rare the task is among the tasks already trained on: log of the k-th neighbour distance
             (a k-NN negative log density, as in ATR).
closeness    exp(-d^2 / (2 sigma^2)), d = distance to the nearest real policy failure.

Candidates are random tasks plus variations of the rebuilt real failures (failure arm only). Every candidate
passes the reachable-goal filter. Tasks are drawn with a softmax over the scores, plus uniform_frac uniform.
"""
from __future__ import annotations

import numpy as np

from src.slingpuck.general.common import wrap_angle
from src.slingpuck.general.failures import TrialRecord, rebuild_task
from src.slingpuck.general.goal_env import TaskSpec

MODES = ("uniform", "atr", "failure")


def task_vector(task: TaskSpec, object_names: list[str]) -> np.ndarray:
    """Task embedding: object one-hot, start (x, y, yaw), first goal (x, y, z, yaw). Lengths in 10 cm units."""
    onehot = np.array([float(task.obj == n) for n in object_names])
    gx, gy, gz, gyaw = task.goals[0] if task.goals else (*task.start_xy, 0.0, task.start_yaw)
    return np.concatenate([onehot, np.array(task.start_xy) / 0.1, [np.sin(task.start_yaw), np.cos(task.start_yaw)],
                           np.array([gx, gy, gz]) / 0.1, [np.sin(gyaw), np.cos(gyaw)]])


class Curriculum:
    def __init__(self, cfg: dict, checker, shapes: dict, mode: str = "failure",
                 failures: list[TrialRecord] | None = None, object_names: list[str] | None = None, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg, self.c, self.e = cfg, cfg["curriculum"], cfg["env"]
        self.mode, self.checker, self.shapes = mode, checker, shapes
        self.names = list(object_names or shapes)
        self.rng = np.random.default_rng(seed)
        self.exact = checker.map is None
        failures = [f for f in (failures or []) if f.is_policy_failure and f.task["obj"] in self.names]
        tz = cfg["robot"]["table_z_m"]
        self.failure_tasks = [rebuild_task(f, tz, shapes[f.task["obj"]].half_height) for f in failures]
        dim = len(self.vec(TaskSpec(self.names[0], (0.0, 0.0), 0.0)))
        self.failure_vecs = np.array([self.vec(t) for t in self.failure_tasks]).reshape(-1, dim)
        self.hist_vecs: list[np.ndarray] = []
        self.hist_success: list[float] = []

    def vec(self, task: TaskSpec) -> np.ndarray:
        return task_vector(task, self.names)

    # ------------------------------------------------------------ candidates
    def reachable(self, task: TaskSpec) -> bool:
        s = self.shapes[task.obj]
        z0 = self.cfg["robot"]["table_z_m"] + s.half_height
        r_lo, r_hi = self.e["min_radius_m"], self.e["max_radius_m"]
        if not all(r_lo <= np.hypot(x, y) <= r_hi for x, y in [task.start_xy, *[g[:2] for g in task.goals]]):
            return False
        ok = self.checker.check(s, (*task.start_xy, z0), task.start_yaw, exact=self.exact).ok
        return ok and all(self.checker.check(s, g[:3], g[3], exact=self.exact).ok for g in task.goals)

    def random_task(self) -> TaskSpec:
        e = self.e
        name = self.names[int(self.rng.integers(len(self.names)))]
        s = self.shapes[name]
        tz = self.cfg["robot"]["table_z_m"] + s.half_height
        xy = self.rng.uniform(e["start_min_m"], e["start_max_m"])
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        a, r = self.rng.uniform(0, 2 * np.pi), e["chain_radius_m"] * np.sqrt(self.rng.random())
        z = tz + (self.rng.uniform(0.02, e["lift_max_m"]) if self.rng.random() < e["lift_prob"] else 0.0)
        goal = (xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), z,
                float(wrap_angle(yaw + self.rng.uniform(-1, 1) * e["chain_yaw_rad"])))
        return TaskSpec(name, (float(xy[0]), float(xy[1])), yaw, [tuple(float(v) for v in goal)])

    def vary(self, task: TaskSpec) -> TaskSpec:
        """A nearby variation of a rebuilt failure (procedural generator around the failure)."""
        dm, dy = self.c["variation_m"], self.c["variation_yaw_rad"]
        sxy = np.array(task.start_xy) + self.rng.normal(0, dm, 2)
        goals = [(g[0] + self.rng.normal(0, dm), g[1] + self.rng.normal(0, dm), g[2],
                  float(wrap_angle(g[3] + self.rng.normal(0, dy)))) for g in task.goals[:1]]
        return TaskSpec(task.obj, (float(sxy[0]), float(sxy[1])),
                        float(wrap_angle(task.start_yaw + self.rng.normal(0, dy))),
                        [tuple(float(v) for v in g) for g in goals])

    def candidates(self, n: int) -> list[TaskSpec]:
        out, tries = [], 0
        use_failures = self.mode == "failure" and self.failure_tasks
        while len(out) < n and tries < 20 * n:
            tries += 1
            if use_failures and self.rng.random() < 0.5:
                t = self.vary(self.failure_tasks[int(self.rng.integers(len(self.failure_tasks)))])
            else:
                t = self.random_task()
            if self.reachable(t):
                out.append(t)
        return out

    # ---------------------------------------------------------------- scores
    def feasibility(self, V: np.ndarray) -> np.ndarray:
        if len(self.hist_vecs) < self.c["knn"]:
            return np.full(len(V), 0.5)
        H, s = np.array(self.hist_vecs), np.array(self.hist_success)
        d = np.linalg.norm(V[:, None] - H[None], axis=2)
        idx = np.argsort(d, axis=1)[:, :self.c["knn"]]
        return s[idx].mean(1)

    def diversity(self, V: np.ndarray) -> np.ndarray:
        if len(self.hist_vecs) < self.c["knn"]:
            return np.zeros(len(V))
        H = np.array(self.hist_vecs)
        d = np.sort(np.linalg.norm(V[:, None] - H[None], axis=2), axis=1)[:, self.c["knn"] - 1]
        return np.log(d + 1e-3)

    def closeness(self, V: np.ndarray) -> np.ndarray:
        if not len(self.failure_vecs):
            return np.zeros(len(V))
        d = np.linalg.norm(V[:, None] - self.failure_vecs[None], axis=2).min(1)
        return np.exp(-d ** 2 / (2 * self.c["failure_sigma"] ** 2))

    def score(self, V: np.ndarray) -> np.ndarray:
        s = self.feasibility(V) + self.c["w_diversity"] * self.diversity(V)
        if self.mode == "failure":
            s = s + self.c["w_failure"] * self.closeness(V)
        return s

    # -------------------------------------------------------------- sampling
    def sample_pool(self, size: int | None = None) -> list[TaskSpec]:
        size = size or self.c["pool_size"]
        cands = self.candidates(2 * size)
        if self.mode == "uniform" or not cands:
            return cands[:size]
        V = np.array([self.vec(t) for t in cands])
        z = self.score(V) / self.c["temperature"]
        p = np.exp(z - z.max())
        p = (1 - self.c["uniform_frac"]) * p / p.sum() + self.c["uniform_frac"] / len(cands)
        idx = self.rng.choice(len(cands), size=size, p=p / p.sum())
        return [cands[i] for i in idx]

    def update(self, results):
        """results: [(TaskSpec, success)] from finished training episodes."""
        for task, success in results:
            self.hist_vecs.append(self.vec(task))
            self.hist_success.append(float(success))
        keep = 5000
        self.hist_vecs, self.hist_success = self.hist_vecs[-keep:], self.hist_success[-keep:]
