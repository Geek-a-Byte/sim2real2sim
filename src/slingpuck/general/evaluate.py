"""Evaluate a goal-conditioned policy in sim and write one trial record per episode.

    python -m src.slingpuck.general.evaluate --model logs/general/<run>/final_model.zip --episodes 100
    python -m src.slingpuck.general.evaluate --model ... --tasks tasks.jsonl --student

The output has the same format as the real trial log (general/failures.py), so the success rate per
stage can be compared between sim and real, and the sim failures can feed the curriculum too.
--tasks: a JSONL file of TaskSpec dicts or of trial records (their tasks are replayed).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.slingpuck.config import REPO_ROOT
from src.slingpuck.eval.stats import wilson_ci
from src.slingpuck.general.common import load_general_config, objects_from_config
from src.slingpuck.general.failures import record_from_info, save_jsonl, summarize, tag_unreachable
from src.slingpuck.general.goal_env import GoalEnv, TaskSpec


def load_tasks(path) -> list[TaskSpec]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(TaskSpec.from_dict(d.get("task", d)))
    return out


def run(policy, env: GoalEnv, episodes: int, tasks=None, seed: int = 0, n_goals: int = 1):
    records = []
    shapes = objects_from_config(env.cfg)
    for k in range(episodes):
        options = {"task": tasks[k % len(tasks)]} if tasks else None
        obs, info = env.reset(seed=seed + k, options=options)
        while True:
            action = policy.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            if info["goals_reached"] >= n_goals or terminated or truncated:
                break
        rec = record_from_info(info, f"sim-eval-{k}", n_goals=n_goals)
        records.append(tag_unreachable(rec, env.checker, shapes))
    return records


def main():
    from stable_baselines3 import PPO

    from src.slingpuck.general.policy import PairedPointPolicy

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--tasks")
    p.add_argument("--student", action="store_true", help="Masked, noisy points (webcam view)")
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "general" / "sim_eval.jsonl"))
    args = p.parse_args()

    cfg = load_general_config()
    model = PPO.load(args.model, custom_objects={"policy_class": PairedPointPolicy})
    env = GoalEnv(cfg, student=args.student)
    tasks = load_tasks(args.tasks) if args.tasks else None
    records = run(model, env, args.episodes, tasks, args.seed)
    save_jsonl(records, args.out, append=False)
    s = summarize(records)
    ok = sum(r.success for r in records if r.stage != "unreachable")
    n = sum(r.stage != "unreachable" for r in records)
    lo, hi = wilson_ci(ok, n) if n else (np.nan, np.nan)
    print(f"Success {ok}/{n} [{lo:.0%}, {hi:.0%}]  stages {s['stages']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
