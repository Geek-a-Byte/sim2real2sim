"""Phase 2 eval: gate ("hole") success rate.

Policy mode: success rate vs shot count (running rate with a 95% Wilson band) and
by puck start position, for trained runs and the scripted baselines, on the same
shots (paired eval seeds).

    python -m src.slingpuck.eval.hole_rate --runs logs/sling/sling_v0_s*_* --shots 1000

Parity mode (sim vs real): replay every logged shot (same rest position and
commands) many times in the sim and compare the predicted success probability
with the logged outcome. Run it for several params versions to see whether a
sysid fit makes the sim agree with reality. Two measures:
- success: Brier score and a reliability table. The outcome depends mostly on
  the lateral aim, so it is not sensitive to friction or launch energy.
- transit time: time from 2 cm past the band line to y = -4 cm, from the camera
  frames. It depends directly on the launch speed and friction.

    python -m src.slingpuck.eval.hole_rate --parity data/real_logs/<session>/shots.csv \\
        --versions v0 v1 [--cfg-dir <dir with the params files>]

Outputs CSV tables, a PNG and eval_meta.yaml in results/sling/<timestamp>/.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.slingpuck.config import REPO_ROOT, git_state, load_config  # noqa: E402
from src.slingpuck.envs.sling_env import launch_and_fly, realize_pull  # noqa: E402
from src.slingpuck.eval.stats import t_ci, wilson_ci  # noqa: E402
from src.slingpuck.physics.band_model import BandModel  # noqa: E402
from src.slingpuck.policies.sling_scripted import SlideCenter, SlideCenterCorrect  # noqa: E402
from src.slingpuck.train.common import load_trained_run, make_sling_env  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"ppo": "#2a78d6", "center": "#eb6834", "center_correct": "#1baf7a"}
LABELS = {"ppo": "PPO (tracked obs)", "center": "Scripted: slide to center",
          "center_correct": "Scripted: slide + camera correction"}
MARKERS = {"ppo": "o", "center": "s", "center_correct": "^"}
VERSION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# ------------------------------------------------------------ policy mode
def run_shots(policy, config, asymmetric, n_shots, base_seed):
    env = make_sling_env(config, asymmetric)
    rows = []
    for k in range(n_shots):
        obs, _ = env.reset(seed=base_seed + k)
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rows.append({"shot": k, "x0": info["x0"], "success": bool(info["success"]), "miss_m": info["miss_m"]})
    return rows


def summarize(results, x_bins):
    table = []
    for name, groups in results.items():
        g = len(groups)
        for lo, hi in [(None, None)] + list(zip(x_bins[:-1], x_bins[1:])):
            per = []
            k_all = n_all = 0
            for rows in groups:
                sel = [r["success"] for r in rows if lo is None or lo <= abs(r["x0"]) < hi]
                per.append(np.mean(sel) if sel else np.nan)
                k_all += sum(sel)
                n_all += len(sel)
            w_lo, w_hi = wilson_ci(int(round(k_all / g)), int(round(n_all / g)))
            if g > 1:
                mean, t_lo, t_hi = t_ci(per)
                lo_ci, hi_ci = np.nanmin([t_lo, w_lo]), np.nanmax([t_hi, w_hi])
            else:
                mean, lo_ci, hi_ci = k_all / n_all, w_lo, w_hi
            table.append({"policy": name, "abs_x0_lo": "all" if lo is None else lo, "abs_x0_hi": "all" if hi is None else hi,
                          "success_rate": mean, "ci_lo": lo_ci, "ci_hi": hi_ci, "shots_per_seed": n_all // g,
                          "seeds": g})
    return table


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def plot_policies(results, table, x_bins, out_path, subtitle):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ends = []
    for name in ("center", "center_correct", "ppo"):
        if name not in results:
            continue
        succ = np.mean([[r["success"] for r in rows] for rows in results[name]], axis=0)
        n = np.arange(1, len(succ) + 1)
        run = np.cumsum(succ) / n
        k = np.cumsum(succ)
        band = np.array([wilson_ci(int(round(ki)), int(ni)) for ki, ni in zip(k, n)])
        ax.fill_between(n, band[:, 0], band[:, 1], color=SERIES[name], alpha=0.12, linewidth=0)
        ax.plot(n, run, color=SERIES[name], linewidth=2, label=LABELS[name])
        ends.append((run[-1], n[-1], f"{name} {run[-1]:.0%}"))
    label_y = np.inf
    for y_end, x_end, text in sorted(ends, reverse=True):
        label_y = min(y_end, label_y - 0.05)
        ax.annotate(text, (x_end, label_y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2, annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(10, max(len(r) for g in results.values() for r in g))  # Fewer shots carry little information
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("Shot count")
    ax.set_ylabel("Running gate success rate")
    ax.set_title("Success rate vs shot count", loc="left", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes[1]
    _style(ax)
    centers = 0.5 * (np.array(x_bins[:-1]) + np.array(x_bins[1:])) * 1000
    width = (x_bins[1] - x_bins[0]) * 1000 / 4
    for j, name in enumerate([n for n in ("ppo", "center", "center_correct") if n in results]):
        rows = [r for r in table if r["policy"] == name and r["abs_x0_lo"] != "all"]
        y = np.array([r["success_rate"] for r in rows])
        err = np.array([[y_ - r["ci_lo"] for y_, r in zip(y, rows)], [r["ci_hi"] - y_ for y_, r in zip(y, rows)]])
        ax.bar(centers + (j - 1) * width * 1.05, y, width=width, color=SERIES[name], yerr=err,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 0}, label=LABELS[name])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Puck start |x| on the band (mm)")
    ax.set_title("Success by puck start position", loc="left", fontsize=11)
    fig.suptitle(f"Phase 2 gate success, 95% CI. {subtitle}", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------ parity mode
TRANSIT_FROM_BAND_M = 0.02
TRANSIT_TO_Y_M = -0.04


def _crossing_time(t, y, y_level):
    """First time y reaches y_level (linear interpolation between frames), or None."""
    above = np.flatnonzero(y >= y_level)
    if len(above) == 0 or above[0] == 0:
        return None
    k = above[0]
    return t[k - 1] + (y_level - y[k - 1]) * (t[k] - t[k - 1]) / (y[k] - y[k - 1])


def _transit(t, y, band_y):
    t1 = _crossing_time(t, y, band_y + TRANSIT_FROM_BAND_M)
    t2 = _crossing_time(t, y, TRANSIT_TO_Y_M)
    return None if t1 is None or t2 is None else t2 - t1


def parity(shots_csv, versions, cfg_dir, repeats, seed):
    from src.slingpuck.sysid.fit import _shots
    from src.slingpuck.sysid.schema import load_log
    df = load_log(shots_csv, "shots")
    shots = _shots(df)
    observed = df.groupby("shot_id").success.first().loc[[s.sid for s in shots]].to_numpy().astype(bool)
    band_y = float(np.mean([s.rest[1] for s in shots]))
    obs_transit = np.array([_transit(s.flight.t_s.to_numpy(), s.flight.puck_y_m.to_numpy(), band_y) or np.nan
                            for s in shots])
    out = {}
    for version in versions:
        cfg = load_config(version, cfg_dir=cfg_dir)
        band = BandModel.from_config(cfg)
        rng = np.random.default_rng(seed)
        p, transit = [], []
        for s in shots:
            hits, times = 0, []
            for _ in range(repeats):
                slide = s.slide + rng.normal(0.0, cfg["sling"]["place_noise_m"])
                pull = realize_pull(cfg, band, slide, s.angle, s.pull, rng)
                fl = launch_and_fly(cfg, band, pull.pos, rng, record=True)
                hits += fl.success
                if len(fl.trajectory):
                    tt = _transit(fl.trajectory[:, 0], fl.trajectory[:, 2], band.band_y)
                    if tt is not None:
                        times.append(tt)
            p.append(hits / repeats)
            transit.append(np.mean(times) if times else np.nan)
        out[version] = (np.array(p), np.array(transit))
    return observed, obs_transit, out


def parity_report(observed, obs_transit, preds, out_dir):
    rows = []
    bins = np.linspace(0, 1, 6)
    for version, (p, transit) in preds.items():
        brier = float(np.mean((p - observed) ** 2))
        k, n = int(observed.sum()), len(observed)
        o_lo, o_hi = wilson_ci(k, n)
        diff = (transit - obs_transit) * 1000
        diff = diff[np.isfinite(diff)]
        bias, b_lo, b_hi = t_ci(diff)
        rows.append({"version": version, "shots": n, "observed_rate": k / n, "observed_ci_lo": o_lo,
                     "observed_ci_hi": o_hi, "predicted_rate": float(p.mean()), "brier": brier,
                     "transit_shots": len(diff), "transit_bias_ms": bias, "transit_bias_ci_lo": b_lo,
                     "transit_bias_ci_hi": b_hi, "transit_rmse_ms": float(np.sqrt(np.mean(diff ** 2)))})
    preds = {v: pt[0] for v, pt in preds.items()}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.98, 0.93, "perfect agreement", ha="right", fontsize=8, color=MUTED, rotation=38)
    rel_rows = []
    for j, (version, p) in enumerate(preds.items()):
        xs, ys, los, his = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p >= lo) & ((p < hi) | (hi == 1.0))
            if sel.sum() == 0:
                continue
            k, n = int(observed[sel].sum()), int(sel.sum())
            c_lo, c_hi = wilson_ci(k, n)
            xs.append(p[sel].mean()); ys.append(k / n); los.append(c_lo); his.append(c_hi)
            rel_rows.append({"version": version, "p_bin_lo": lo, "p_bin_hi": hi, "mean_predicted": p[sel].mean(),
                             "observed_rate": k / n, "ci_lo": c_lo, "ci_hi": c_hi, "shots": n})
        color = VERSION_COLORS[j % len(VERSION_COLORS)]
        ys, los, his = np.array(ys), np.array(los), np.array(his)
        ax.errorbar(xs, ys, yerr=[ys - los, his - ys], color=color, marker="o", markersize=6, linewidth=2,
                    capsize=0, markeredgecolor=SURFACE, label=f"{version} (Brier {rows[j]['brier']:.3f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Sim-predicted success probability")
    ax.set_ylabel("Logged success rate")
    ax.set_title("Sim vs logged shots (reliability)", loc="left", fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # Transit-time bias: sensitive to launch speed and friction.
    ax_t.set_facecolor(SURFACE)
    names = [r["version"] for r in rows]
    for j, r in enumerate(rows):
        color = VERSION_COLORS[j % len(VERSION_COLORS)]
        ax_t.errorbar(r["transit_bias_ms"], j, xerr=[[r["transit_bias_ms"] - r["transit_bias_ci_lo"]],
                                                     [r["transit_bias_ci_hi"] - r["transit_bias_ms"]]],
                      color=color, marker="o", markersize=8, linewidth=2, capsize=0, markeredgecolor=SURFACE)
        ax_t.annotate(f"{r['transit_bias_ms']:+.1f} ms", (r["transit_bias_ci_hi"], j), xytext=(6, 0),
                      textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax_t.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_t.set_yticks(range(len(names)), names)
    ax_t.set_ylim(-0.6, len(names) - 0.4)
    ax_t.invert_yaxis()
    ax_t.set_xlabel("Transit time bias, sim - logged (ms), 95% CI")
    ax_t.set_title("Launch speed check (transit time)", loc="left", fontsize=11)
    ax_t.grid(axis="x", color=GRID, linewidth=0.8)
    ax_t.set_axisbelow(True)
    for side in ("top", "right"):
        ax_t.spines[side].set_visible(False)
    lim = max(abs(r[k]) for r in rows for k in ("transit_bias_ci_lo", "transit_bias_ci_hi")) * 1.4 + 1
    ax_t.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out_dir / "parity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return rows, rel_rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="Training run dirs (one per seed)")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--parity", help="shots CSV for sim-vs-real parity")
    parser.add_argument("--versions", nargs="*", default=["latest"], help="Params versions for parity")
    parser.add_argument("--cfg-dir", help="Config dir with the params files (parity)")
    parser.add_argument("--repeats", type=int, default=100, help="Sim repeats per logged shot (parity)")
    parser.add_argument("--out")
    args = parser.parse_args()

    out = Path(args.out) if args.out else REPO_ROOT / "results" / "sling" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"eval_seed": args.eval_seed, **git_state()}

    if args.parity:
        observed, obs_transit, preds = parity(args.parity, args.versions, args.cfg_dir, args.repeats,
                                              args.eval_seed)
        rows, rel = parity_report(observed, obs_transit, preds, out)
        write_csv(out / "parity_summary.csv", rows)
        write_csv(out / "parity_reliability.csv", rel)
        meta.update(mode="parity", shots_csv=str(args.parity), versions=args.versions, repeats=args.repeats)
        print(f"Parity ({out})")
        for r in rows:
            print(f"  {r['version']:>6}: success logged {r['observed_rate']:.1%} [{r['observed_ci_lo']:.1%}, "
                  f"{r['observed_ci_hi']:.1%}], sim {r['predicted_rate']:.1%}, Brier {r['brier']:.3f} | "
                  f"transit bias (sim - logged) {r['transit_bias_ms']:+.1f} ms [{r['transit_bias_ci_lo']:+.1f}, "
                  f"{r['transit_bias_ci_hi']:+.1f}], RMSE {r['transit_rmse_ms']:.1f} ms, n={r['transit_shots']}")
    else:
        config = load_config(args.params_version, cfg_dir=args.cfg_dir)
        results = {}
        runs_meta = []
        for run in args.runs:
            model, asymmetric, run_cfg = load_trained_run(run)
            print(f"Evaluating {Path(run).name}")
            results.setdefault("ppo", []).append(run_shots(model, config, asymmetric, args.shots, args.eval_seed))
            runs_meta.append({"run": str(run), "seed": run_cfg["meta"]["seed"],
                              "trained_params_version": run_cfg["meta"]["params_version"]})
        for policy in (SlideCenter(), SlideCenterCorrect(config)):
            print(f"Evaluating scripted baseline: {policy.name}")
            results[policy.name] = [run_shots(policy, config, False, args.shots, args.eval_seed)]
        x_bins = [0.0, 0.015, 0.03, 0.045, 0.06]
        table = summarize(results, x_bins)
        write_csv(out / "success_summary.csv", table)
        write_csv(out / "shots.csv", [{"policy": n, "group": g, **r} for n, groups in results.items()
                                      for g, rows in enumerate(groups) for r in rows])
        plot_policies(results, table, x_bins, out / "hole_rate.png",
                      f"params {config['meta']['params_version']}, domain-randomized, {args.shots} shots per policy/seed")
        meta.update(mode="policy", params_version=config["meta"]["params_version"], shots=args.shots,
                    runs=runs_meta)
        print(f"Success ({out})")
        for r in table:
            if r["abs_x0_lo"] == "all":
                print(f"  {r['policy']:>15}: {r['success_rate']:.1%} [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] "
                      f"({r['seeds']} seed(s) x {r['shots_per_seed']} shots)")
    (out / "eval_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


if __name__ == "__main__":
    main()
