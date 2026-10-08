"""Phase 3 eval: match results of high-level selectors against the scripted opponent.

    python -m src.slingpuck.eval.match_eval [--policies reload_rule tell_rule greedy_sling] \\
        [--runs logs/selector/<run> ...] [--matches 300] [--tell 0 0.5 1]

Every policy plays the same matches (paired seeds). Per policy (and tell_strength):
win / loss / time-out rates (95% Wilson), mean puck difference sent - received
(95% t), slings, pucks conceded while the arm was away, and the save rate.
Outputs match_summary.csv, match_results.png and eval_meta.yaml in
results/match/<timestamp>/.
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
from src.slingpuck.envs.match_env import MatchEnv  # noqa: E402
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper  # noqa: E402
from src.slingpuck.eval.stats import t_ci, wilson_ci  # noqa: E402
from src.slingpuck.policies import match_scripted as ms  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SCRIPTED = {
    "always_block": lambda cfg: ms.AlwaysBlock(),
    "greedy_sling": lambda cfg: ms.GreedySling(),
    "reload_rule": lambda cfg: ms.ReloadRule(cfg),
    "tell_rule": lambda cfg: ms.TellRule(cfg),
    "oracle_reload": lambda cfg: ms.OracleReload(cfg),
}


def play(policy, env, n_matches, base_seed, tell):
    rows = []
    for k in range(n_matches):
        options = {} if tell is None else {"tell_strength": tell}
        obs, info = env.reset(seed=base_seed + k, options=options)
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(np.asarray(action).reshape(-1)[0]))
            if terminated or truncated:
                break
        rows.append({"outcome": info["outcome"], **info["stats"]})
    return rows


def summarize(name, tell, rows):
    n = len(rows)
    out = {"policy": name, "tell_strength": tell, "matches": n}
    for key in ("win", "loss", "time"):
        k = sum(r["outcome"] == key for r in rows)
        lo, hi = wilson_ci(k, n)
        out.update({f"{key}_rate": k / n, f"{key}_ci_lo": lo, f"{key}_ci_hi": hi})
    diff = [r["sent"] - r["received"] for r in rows]
    mean, lo, hi = t_ci(diff)
    out.update(diff_mean=mean, diff_ci_lo=lo, diff_ci_hi=hi)
    for key in ("slings", "sent", "received", "conceded_while_away"):
        out[f"{key}_per_match"] = float(np.mean([r[key] for r in rows]))
    threats = sum(r["threats"] for r in rows)
    out["save_rate"] = sum(r["saves"] for r in rows) / threats if threats else float("nan")
    return out


def color_for(name, names):
    """Fixed color per policy: scripted policies keep their slot in every chart; runs follow them."""
    order = list(SCRIPTED) + [n for n in names if n not in SCRIPTED]
    return PALETTE[order.index(name) % len(PALETTE)]


def plot(table, out_path, subtitle):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    names = list(dict.fromkeys(r["policy"] for r in table))
    tells = sorted({r["tell_strength"] for r in table})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, key, label in ((axes[0], "win", "Win rate"), (axes[1], "diff", "Pucks sent - received per match")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        if len(tells) > 1:  # Sweep: one line per policy
            for j, name in enumerate(names):
                rows = sorted([r for r in table if r["policy"] == name], key=lambda r: r["tell_strength"])
                x = [r["tell_strength"] for r in rows]
                y = [r[f"{key}_rate" if key == "win" else "diff_mean"] for r in rows]
                lo = [r[f"{key}_ci_lo"] for r in rows]
                hi = [r[f"{key}_ci_hi"] for r in rows]
                c = color_for(name, names)
                ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
                ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=name)
            ax.set_xlabel("Opponent tell_strength")
            ax.grid(axis="y", color=GRID, linewidth=0.8)
        else:  # One condition: a dot per policy with its CI
            for j, name in enumerate(names):
                r = next(r for r in table if r["policy"] == name)
                y = r[f"{key}_rate" if key == "win" else "diff_mean"]
                c = color_for(name, names)
                err = [[max(y - r[f"{key}_ci_lo"], 0.0)], [max(r[f"{key}_ci_hi"] - y, 0.0)]]
                ax.errorbar(y, j, xerr=err, color=c, marker="o",
                            markersize=8, linewidth=2, capsize=0, markeredgecolor=SURFACE)
                ax.annotate(f"{y:.0%}" if key == "win" else f"{y:+.1f}", (r[f"{key}_ci_hi"], j), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, color=INK_2)
            ax.set_yticks(range(len(names)), names)
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            if key == "diff":
                ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(label, loc="left", fontsize=11)
        if key == "win":
            fmt = matplotlib.ticker.PercentFormatter(1.0)
            (ax.yaxis if len(tells) > 1 else ax.xaxis).set_major_formatter(fmt)
    if len(tells) > 1:
        axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Phase 3 match results, 95% CI. {subtitle}", x=0.01, ha="left", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="*", default=list(SCRIPTED))
    parser.add_argument("--runs", nargs="*", default=[], help="Trained selector run dirs")
    parser.add_argument("--params-version", default="latest")
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--eval-seed", type=int, default=30_000)
    parser.add_argument("--tell", type=float, nargs="*", help="tell_strength values (default: config value)")
    parser.add_argument("--no-hand", action="store_true", help="Ablation: the policy does not see the hand")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(args.params_version)
    if args.no_hand:
        config["match"]["observe_hand"] = False
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "match" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    table_env = MatchEnv(config)
    tells = args.tell if args.tell else [None]

    policies = [(name, SCRIPTED[name](config), True) for name in args.policies]
    from src.slingpuck.train.common import load_trained_run
    for run in args.runs:
        model, asymmetric, _ = load_trained_run(run)
        policies.append((Path(run).name, model, asymmetric))

    table = []
    for name, policy, wrap in policies:
        for tell in tells:
            env = MatchEnv(config, table_env.block)
            env = PrivilegedObsWrapper(env) if wrap else env
            rows = play(policy, env, args.matches, args.eval_seed, tell)
            t_val = config["opponent"]["tell_strength"] if tell is None else tell
            table.append(summarize(name, t_val, rows))
            r = table[-1]
            print(f"{name:>16} tell {t_val:.2f}: win {r['win_rate']:.1%} [{r['win_ci_lo']:.1%}, {r['win_ci_hi']:.1%}]"
                  f"  loss {r['loss_rate']:.1%}  diff {r['diff_mean']:+.2f} [{r['diff_ci_lo']:+.2f}, {r['diff_ci_hi']:+.2f}]"
                  f"  conceded away {r['conceded_while_away_per_match']:.1f}/match  saves {r['save_rate']:.0%}")
    with open(out / "match_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    subtitle = (f"params {config['meta']['params_version']}, {args.matches} matches per point"
                f"{', no hand input' if args.no_hand else ''}")
    plot(table, out / "match_results.png", subtitle)
    (out / "eval_meta.yaml").write_text(yaml.safe_dump(
        {"params_version": config["meta"]["params_version"], "matches": args.matches, "eval_seed": args.eval_seed,
         "tell": tells, "observe_hand": config["match"]["observe_hand"], "runs": args.runs, **git_state()},
        sort_keys=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
