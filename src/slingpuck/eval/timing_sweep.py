"""Timing sweep: when does blocking start to matter? Selectors over a range of robot timings. No training.

    python -m src.slingpuck.eval.timing_sweep [--matches 300] [--sling 0.5 1 2 3] [--move 1 0.5 0.25] [--threat 0.5 0.8]

sling: multiplies the time of a shot (pick, carry, seat, draw). move: multiplies the arm's trip between the
gate and the band. threat: fraction of opponent shots aimed at the gate. Every selector plays the same
matches (paired seeds). All base timings are placeholders, so the sweep shows where the ranking changes; the
real timings say where the real arm lies. Writes results/match/sweep_<time>/{sweep.csv, sweep.png}.
"""
import argparse
import copy
import csv
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.slingpuck.config import REPO_ROOT, load_config  # noqa: E402
from src.slingpuck.envs.match_env import MatchEnv  # noqa: E402
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper  # noqa: E402
from src.slingpuck.eval.match_eval import SCRIPTED, play, summarize  # noqa: E402

POLICIES = ("greedy_sling", "greedy_rule", "tell_rule", "reload_rule")
COLORS = {"greedy_sling": "#2a78d6", "greedy_rule": "#eb6834", "tell_rule": "#1baf7a", "reload_rule": "#eda100"}


def scaled(cfg: dict, sling: float, move: float, threat: float) -> dict:
    c = copy.deepcopy(cfg)
    m, s = c["match"], c["shot"]
    m["fetch_time_s"] *= sling
    m["sling_core_s"] *= sling
    s["pick_time_s"] *= sling
    s["seat_release_s"] *= sling
    s["arm_speed_m_s"] /= sling
    s["draw_speed_m_s"] /= sling
    m["move_gate_to_band_s"] *= move
    m["move_band_to_gate_s"] *= move
    c["opponent"]["threat_prob"] = threat
    return c


def run_point(job):
    sling, move, threat, name, matches, seed = job
    cfg = scaled(load_config("latest"), sling, move, threat)
    env = PrivilegedObsWrapper(MatchEnv(cfg))
    r = summarize(name, cfg["opponent"]["tell_strength"], play(SCRIPTED[name](cfg), env, matches, seed, None))
    return {"sling_scale": sling, "move_scale": move, "threat_prob": threat, **r}


def plot(rows, out_path, threats, moves):
    fig, axes = plt.subplots(len(threats), len(moves), figsize=(4 * len(moves), 3 * len(threats)), squeeze=False,
                             sharex=True, sharey=True)
    for i, threat in enumerate(threats):
        for j, move in enumerate(moves):
            ax = axes[i][j]
            for name in POLICIES:
                sel = sorted((r for r in rows if r["policy"] == name and r["threat_prob"] == threat and r["move_scale"] == move),
                             key=lambda r: r["sling_scale"])
                x = [r["sling_scale"] for r in sel]
                ax.fill_between(x, [r["win_ci_lo"] for r in sel], [r["win_ci_hi"] for r in sel], color=COLORS[name], alpha=0.12,
                                linewidth=0)
                ax.plot(x, [r["win_rate"] for r in sel], color=COLORS[name], marker="o", markersize=4, linewidth=1.8, label=name)
            ax.set_title(f"threat {threat:g}, gate trip x{move:g}", fontsize=10, loc="left")
            ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Shot time (x placeholder)")
    for row in axes:
        row[0].set_ylabel("Win rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=300)
    p.add_argument("--sling", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--move", type=float, nargs="*", default=[1.0, 0.5, 0.25])
    p.add_argument("--threat", type=float, nargs="*", default=[0.5, 0.8])
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out")
    a = p.parse_args()
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "match" / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, t, n, a.matches, a.seed) for t in a.threat for m in a.move for s in a.sling for n in POLICIES]
    with Pool(a.workers) as pool:
        rows = pool.map(run_point, jobs)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out / "sweep.png", a.threat, a.move)
    for t in a.threat:
        for m in a.move:
            print(f"\nthreat {t:g}, gate trip x{m:g}:  win % / net per min / saves %")
            for s in a.sling:
                cells = []
                for n in POLICIES:
                    r = next(r for r in rows if (r["sling_scale"], r["move_scale"], r["threat_prob"], r["policy"]) == (s, m, t, n))
                    cells.append(f"{n} {100 * r['win_rate']:3.0f} {r['net_clearing_per_min']:+5.1f} {100 * r['save_rate']:3.0f}")
                print(f"  shot x{s:<4g} " + " | ".join(cells))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
