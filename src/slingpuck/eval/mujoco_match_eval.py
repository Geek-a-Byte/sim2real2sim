"""Full-physics MuJoCo matches: the scripted arm skills against the scripted human. No training.

    python -m src.slingpuck.eval.mujoco_match_eval [--matches 30] [--speeds 1 2] [--rules reload greedy_sling]

The event-level MatchEnv uses placeholder times; here the times and the results come from the skills themselves.
The human's timings are still placeholders. A match ends when one side is empty, or at match.max_time_s.
Writes results/match/mujoco_<time>/matches.csv.
"""
import argparse
import csv
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.slingpuck.config import REPO_ROOT, load_config
from src.slingpuck.eval.stats import wilson_ci
from src.slingpuck.physics.mujoco_match import MatchSim
from src.slingpuck.primitives.sling import ShotSelector
from src.slingpuck.shot.model import load_for

_SIMS = {}


def play(job) -> dict:
    speed, rule, seed = job
    key = (speed, rule)
    if key not in _SIMS:
        cfg = load_config("latest", overrides={"skills": {"speed_scale": speed, "game_rule": rule}})
        _SIMS[key] = MatchSim(cfg, selector=ShotSelector(cfg, load_for(cfg)))
    sim = _SIMS[key]
    sim.rng = np.random.default_rng(seed)
    sim.see_noise_m = sim.cfg["match"]["puck_track_noise_m"]
    sim.reset()
    max_t = sim.cfg["match"]["max_time_s"]
    while sim.t < max_t and sim.winner is None:
        sim.step()
    logs = sim.fetch_log
    c = sim.counts()
    return {"speed": speed, "rule": rule, "seed": seed, "winner": sim.winner or "time", "time_s": sim.t, "robot_left": c["robot"],
            "human_left": c["human"], "tries": len(logs), "through": sum(x["outcome"] == "through" for x in logs),
            "missed_grasps": sum(x["outcome"] == "missed_grasp" for x in logs), "slips": sum(x["slip"] for x in logs),
            "robot_shots": sim.shots["robot"], "human_shots": sim.shots["human"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=30)
    p.add_argument("--speeds", type=float, nargs="*", default=[1.0, 2.0])
    p.add_argument("--rules", nargs="*", default=["reload", "greedy_sling"])
    p.add_argument("--seed", type=int, default=80_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out")
    a = p.parse_args()
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "match" / f"mujoco_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r, a.seed + i) for s in a.speeds for r in a.rules for i in range(a.matches)]
    with Pool(a.workers) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    with open(out / "matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in a.speeds:
        for r in a.rules:
            m = [x for x in rows if x["speed"] == s and x["rule"] == r]
            wins, n = sum(x["winner"] == "robot" for x in m), len(m)
            lo, hi = wilson_ci(wins, n)
            print(f"speed x{s:g} {r:>12}: win {wins}/{n} [{lo:.0%}, {hi:.0%}]  loss {sum(x['winner'] == 'human' for x in m)}  "
                  f"time-out {sum(x['winner'] == 'time' for x in m)}  time {np.mean([x['time_s'] for x in m]):.1f} s  "
                  f"robot shots {np.mean([x['robot_shots'] for x in m]):.1f}  human shots {np.mean([x['human_shots'] for x in m]):.1f}  "
                  f"pass {np.mean([x['through'] for x in m]):.1f} / {np.mean([x['tries'] for x in m]):.1f} tries  "
                  f"pucks left robot {np.mean([x['robot_left'] for x in m]):.1f}, human {np.mean([x['human_left'] for x in m]):.1f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
