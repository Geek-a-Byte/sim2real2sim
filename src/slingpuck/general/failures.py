"""Structured trial logs (sim and real): the input of the real2sim stage.

Every trial, real or sim, is one JSON line with the same fields, so a real failure can be rebuilt in sim:
  trial_id, source ("real" | "sim"), task (TaskSpec), goal_index (the goal step where it ended), goal,
  stage, object_pose (x, y, z, yaw at the end), success, notes, time.

stage is where the trial ended: "success", or the failure stage "reach" (never touched the object),
"grasp" (touched, but did not lift a lifted goal), "transport" (far from the goal), "place" (near the
goal, not within the success distance), "object_lost" (off the table or out of the workspace),
"tracking_lost" (real only: the point tracker lost the object), or "unreachable" (the goal fails the
reachable-goal filter; not a policy failure, so the curriculum does not count it).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.slingpuck.general.goal_env import TaskSpec

FAILURE_STAGES = ("reach", "grasp", "transport", "place", "object_lost", "tracking_lost")
STAGES = ("success", *FAILURE_STAGES, "unreachable")


@dataclass
class TrialRecord:
    trial_id: str
    source: str
    task: dict
    goal_index: int
    goal: list
    stage: str
    object_pose: list
    success: bool
    notes: str = ""
    time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.source not in ("real", "sim"):
            raise ValueError(f"source must be 'real' or 'sim', not {self.source!r}")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {self.stage!r}")

    @property
    def is_policy_failure(self) -> bool:
        return not self.success and self.stage != "unreachable"

    def task_spec(self) -> TaskSpec:
        return TaskSpec.from_dict(self.task)


def record_from_info(info: dict, trial_id: str, source: str = "sim", n_goals: int = 1) -> TrialRecord:
    """A trial record from the last info of a GoalEnv episode. Success = reached n_goals goals."""
    success = info["goals_reached"] >= n_goals
    stage = "success" if success else info.get("failure", info["stage"])
    return TrialRecord(trial_id, source, info["task"], int(info["goal_index"]), list(info["goal"]), stage,
                       list(info["object_pose"]), bool(success),
                       notes="grasped_start" if info.get("grasped_start") else "")


def tag_unreachable(rec: TrialRecord, checker, shapes) -> TrialRecord:
    """Mark a failure whose goal fails the reachable-goal filter, so it does not count as a policy failure."""
    if rec.success:
        return rec
    x, y, z, yaw = rec.goal
    if not checker.check(shapes[rec.task["obj"]], (x, y, z), yaw).ok:
        rec.stage = "unreachable"
    return rec


def rebuild_task(rec: TrialRecord, table_z: float, half_height: float) -> TaskSpec:
    """The sim task that reproduces a failure: start where the object was (if it lay on the table), and the
    goal that failed, followed by the rest of the original goal sequence."""
    task = rec.task_spec()
    x, y, z, yaw = rec.object_pose
    on_table = abs(z - (table_z + half_height)) < 0.01
    start_xy, start_yaw = ((x, y), yaw) if on_table else (task.start_xy, task.start_yaw)
    rest = task.goals[rec.goal_index + 1:] if task.goals else []
    return TaskSpec(task.obj, tuple(start_xy), float(start_yaw), [tuple(rec.goal), *rest])


def save_jsonl(records, path, append: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=_json_default) + "\n")


def load_jsonl(path) -> list[TrialRecord]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(TrialRecord(**json.loads(line)))
    return out


def summarize(records) -> dict:
    n = len(records)
    counts = {s: sum(r.stage == s for r in records) for s in STAGES}
    policy = [r for r in records if r.stage != "unreachable"]
    return {"trials": n, "success_rate": (sum(r.success for r in policy) / len(policy)) if policy else None,
            "stages": counts}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
