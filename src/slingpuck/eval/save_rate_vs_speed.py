"""Phase 1 eval: goalkeeper save rate vs incoming puck speed (and vs start offset).

Evaluates trained PPO runs (one per training seed) and scripted baselines on the
same shots (paired eval seeds). Only threatening shots count (shots that would
score with no paddle). Speeds are stratified: each bin gets the same number of
episodes with speed uniform in the bin.

CIs: PPO uses a 95% Student-t interval across training seeds. Scripted baselines
(deterministic) use a 95% Wilson interval over episodes.

Example:
    python -m src.slingpuck.eval.save_rate_vs_speed --runs logs/goalkeeper/goalkeeper_v0_s0_* \
        --episodes-per-bin 300
Outputs CSV tables, a PNG figure and eval_meta.yaml in results/goalkeeper/<timestamp>/.
save_rate_grid.csv (speed x start offset) is the time-to-cover result that Phase 3
uses for the vulnerability window. --release-delay sets how long after the launch
the paddle becomes free.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.slingpuck.config import REPO_ROOT, git_state, load_config  # noqa: E402
from src.slingpuck.eval.stats import t_ci, wilson_ci  # noqa: E402
from src.slingpuck.policies.scripted import CenterBlocker, HoldStart  # noqa: E402
from src.slingpuck.train.common import load_trained_run, make_goalkeeper_env  # noqa: E402

N_START_BINS = 4


def run_episodes(policy, config, asymmetric, speed_bins, episodes_per_bin, base_seed, randomize, release_delay):
    """Return list of dicts: speed, start_offset (fraction of pan range), saved."""
    env = make_goalkeeper_env(config, asymmetric)
    gk = env.unwrapped
    rng = np.random.default_rng(base_seed)
    rows = []
    for b, (lo, hi) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        for k in range(episodes_per_bin):
            seed = base_seed + 100_000 * b + k
            obs, info = env.reset(seed=seed, options={"speed": rng.uniform(lo, hi), "randomize": randomize,
                                                      "release_delay": release_delay})
            if not info["threat"]:
                continue
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            rows.append({"speed": info["shot_speed"], "speed_bin": b,
                         "start_offset": abs(info["start_pan"]) / gk.geom.pan_max,
                         "saved": info["outcome"] == "save"})
    return rows


def in_bin(value, bins, b):
    lo, hi = bins[b], bins[b + 1]
    return lo <= value < hi or (b == len(bins) - 2 and value == hi)


def rate_with_ci(groups, select):
    """Save rate and 95% CI over the rows that `select` accepts, for one policy."""
    per_group, k_all, n_all = [], 0, 0
    for rows in groups:
        saved = [r["saved"] for r in rows if select(r)]
        per_group.append(np.mean(saved) if saved else np.nan)
        k_all += sum(saved)
        n_all += len(saved)
    if len(groups) > 1:
        mean, lo_ci, hi_ci = t_ci(per_group)
        method = f"t across {len(groups)} training seeds"
    else:
        mean = k_all / n_all if n_all else np.nan
        lo_ci, hi_ci = wilson_ci(k_all, n_all)
        method = "Wilson over episodes"
    return {"save_rate": mean, "ci_lo": lo_ci, "ci_hi": hi_ci, "episodes": n_all, "ci_method": method}


def summarize(results, key, bins):
    """results: {policy: [rows per group]}. Returns table rows with mean and CI per bin."""
    table = []
    for name, groups in results.items():
        for b in range(len(bins) - 1):
            stats_row = rate_with_ci(groups, lambda r: in_bin(r[key], bins, b))
            table.append({"policy": name, "bin_lo": bins[b], "bin_hi": bins[b + 1],
                          "bin_center": 0.5 * (bins[b] + bins[b + 1]), **stats_row})
    return table


def summarize_grid(results, speed_bins, start_bins):
    """Save rate per (speed bin, start-offset bin) for each policy."""
    table = []
    for name, groups in results.items():
        for i in range(len(speed_bins) - 1):
            for j in range(len(start_bins) - 1):
                stats_row = rate_with_ci(groups, lambda r: in_bin(r["speed"], speed_bins, i)
                                         and in_bin(r["start_offset"], start_bins, j))
                table.append({"policy": name, "speed_lo": speed_bins[i], "speed_hi": speed_bins[i + 1],
                              "start_offset_lo": start_bins[j], "start_offset_hi": start_bins[j + 1],
                              **stats_row})
    return table


# Reference palette (dataviz skill), light mode; first three categorical slots, validated.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"ppo": "#2a78d6", "center": "#eb6834", "hold": "#1baf7a"}
LABELS = {"ppo": "PPO (tracked obs)", "center": "Scripted: go to center", "hold": "Scripted: never move"}
SHORT = {"ppo": "PPO", "center": "center", "hold": "never move"}
DRAW_ORDER = ("hold", "center", "ppo")  # PPO last, so it stays visible when lines coincide
LABEL_GAP = 0.07  # Minimum vertical gap between end labels, in save-rate units
MARKERS = {"ppo": "o", "center": "s", "hold": "^"}


def plot(speed_table, start_table, out_path, subtitle):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    panels = [(axes[0], speed_table, "Incoming puck speed (m/s)", "Save rate vs puck speed"),
              (axes[1], start_table, "Paddle start offset (fraction of pan range)", "Save rate vs start offset")]
    for ax, table, xlabel, title in panels:
        ax.set_facecolor(SURFACE)
        ends = []
        for name in DRAW_ORDER:
            rows = [r for r in table if r["policy"] == name]
            if not rows:
                continue
            x = [r["bin_center"] for r in rows]
            y = [r["save_rate"] for r in rows]
            ax.fill_between(x, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows],
                            color=SERIES[name], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=SERIES[name], linewidth=2, marker=MARKERS[name], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[name])
            ends.append((y[-1], x[-1], f"{SHORT[name]} {y[-1]:.0%}"))
        # Direct end labels in text ink. Top to bottom, each label sits at its value
        # or LABEL_GAP below the previous label, whichever is lower.
        label_y = np.inf
        for y_end, x_end, text in sorted(ends, reverse=True):
            label_y = min(y_end, label_y - LABEL_GAP)
            ax.annotate(text, (x_end, label_y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2, annotation_clip=False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Save rate (threatening shots)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(LABELS[n]) for n in SERIES if LABELS[n] in labels]  # Palette slot order
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"Goalkeeper save rate, bands = 95% CI. {subtitle}", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_csv(path, table):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="Training run dirs (one per seed)")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--episodes-per-bin", type=int, default=300)
    parser.add_argument("--n-speed-bins", type=int, default=6)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--nominal", action="store_true", help="Disable domain randomization in eval")
    parser.add_argument("--release-delay", type=float, default=0.0,
                        help="Seconds after the launch until the paddle is free")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.params_version)
    lo, hi = config["goalkeeper"]["speed_range_m_s"]
    speed_bins = np.linspace(lo, hi, args.n_speed_bins + 1)
    randomize = not args.nominal
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "goalkeeper" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    results = defaultdict(list)
    run_meta = []
    for run in args.runs:
        run = Path(run)
        model, asymmetric, run_cfg = load_trained_run(run)
        print(f"Evaluating {run.name}")
        results["ppo"].append(run_episodes(model, config, asymmetric, speed_bins, args.episodes_per_bin,
                                           args.eval_seed, randomize, args.release_delay))
        run_meta.append({"run": str(run), "seed": run_cfg["meta"]["seed"],
                         "trained_params_version": run_cfg["meta"]["params_version"],
                         "git_hash": run_cfg["meta"]["git_hash"]})
    for policy in (CenterBlocker(), HoldStart()):
        print(f"Evaluating scripted baseline: {policy.name}")
        results[policy.name].append(run_episodes(policy, config, False, speed_bins, args.episodes_per_bin,
                                                 args.eval_seed, randomize, args.release_delay))

    speed_table = summarize(results, "speed", speed_bins)
    start_bins = np.linspace(0.0, 1.0, N_START_BINS + 1)
    start_table = summarize(results, "start_offset", start_bins)
    write_csv(out / "save_rate_vs_speed.csv", speed_table)
    write_csv(out / "save_rate_vs_start_offset.csv", start_table)
    write_csv(out / "save_rate_grid.csv", summarize_grid(results, speed_bins, start_bins))
    subtitle = (f"params {config['meta']['params_version']}, "
                f"{'domain-randomized' if randomize else 'nominal'} physics, "
                f"{args.episodes_per_bin} shots per speed bin, release delay {args.release_delay * 1000:.0f} ms")
    plot(speed_table, start_table, out / "save_rate.png", subtitle)
    meta = {"eval_params_version": config["meta"]["params_version"], "randomize": randomize,
            "episodes_per_bin": args.episodes_per_bin, "eval_seed": args.eval_seed,
            "release_delay_s": args.release_delay,
            "speed_bins": speed_bins.tolist(), "runs": run_meta, **git_state()}
    (out / "eval_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"\nSave rate vs speed ({out})")
    for r in speed_table:
        print(f"  {r['policy']:7s} {r['bin_lo']:.2f}-{r['bin_hi']:.2f} m/s: {r['save_rate']:6.1%} "
              f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]  n={r['episodes']}")


if __name__ == "__main__":
    main()
