"""Check the twin on a synthetic session (needs the ground truth in <session>/truth).

    python -m src.slingpuck.twin.evaluate --session sessions/<name> [--fitted <session>/twin/fit/physics_params_v1.yaml]

1. Tracking: measurement error against the true puck positions, identity errors,
   and how many frames of the shot puck the camera saw.
2. Start from a real moment: for each shot, take the twin state at the
   settle_frames-th frame where the shot puck is seen moving after the release
   (the filter restarts when the puck comes out from under the arm, and needs a
   few frames for a good speed), start the 2D sim
   there, and compare its prediction with the true path `horizon_s` later. Done
   with the v0 physics and (if given) the sysid fit, to show what the fit buys.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.slingpuck.config import load_config
from src.slingpuck.physics.puck_dynamics import Fast2DPuckSim


def tracking_metrics(session) -> dict:
    session = Path(session)
    tr = pd.read_csv(session / "twin" / "tracks.csv").dropna(subset=["meas_x_m"])
    gt = pd.read_csv(session / "truth" / "pucks.csv")
    tr["t_r"], gt["t_r"] = tr.t_s.round(4), gt.t_s.round(4)
    m = tr.merge(gt, on="t_r", suffixes=("", "_gt"))
    m["d"] = np.hypot(m.meas_x_m - m.x_m_gt, m.meas_y_m - m.y_m_gt)
    best = m.loc[m.groupby(["t_r", "puck_id"]).d.idxmin()]
    idmap = best.groupby("puck_id").puck.agg(lambda s: s.mode()[0])
    ok = best.puck == best.puck_id.map(idmap)
    vis = best[ok]
    shot = vis[vis.puck == 0]
    return {"puck_frames": int(len(best)), "identity_errors": int((~ok).sum()),
            "error_median_mm": float(1000 * vis.d.median()), "error_p95_mm": float(1000 * vis.d.quantile(0.95)),
            "shot_puck_frames_seen": int(len(shot)), "shot_puck_frames_total": int((gt.puck == 0).sum()),
            "track_to_truth": {int(k): int(v) for k, v in idmap.items()}}


def _with_params(cfg: dict, params_file) -> dict:
    fitted = yaml.safe_load(Path(params_file).read_text())
    cfg = copy.deepcopy(cfg)
    for section in ("puck", "board", "band", "arm"):
        cfg[section].update({k: v for k, v in fitted.get(section, {}).items() if k in cfg[section]})
    return cfg


def restart_errors(session, cfg: dict, horizon_s: float = 0.15, min_speed: float = 0.3,
                   settle_frames: int = 6) -> list:
    """Position error of the 2D sim started from the twin state, horizon_s later, per shot."""
    session = Path(session)
    tr = pd.read_csv(session / "twin" / "tracks.csv")
    gt = pd.read_csv(session / "truth" / "pucks.csv")
    gt = gt[gt.puck == 0]
    shots = pd.read_csv(session / "robot_shots.csv")
    lim = np.array([0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]]) - cfg["puck"]["radius_m"]
    out = []
    for _, s in shots.iterrows():
        after = tr[(tr.t_s > s.t_release_s) & (tr.hidden == 0) & (np.hypot(tr.vx_m_s, tr.vy_m_s) > min_speed)]
        if after.empty:
            continue
        after = after[after.puck_id == after.iloc[0].puck_id]
        if len(after) < settle_frames:
            continue
        row = after.iloc[settle_frames - 1]
        sim = Fast2DPuckSim.from_config(cfg)
        sim.reset(np.clip([row.x_m, row.y_m], -lim, lim)[None], np.array([[row.vx_m_s, row.vy_m_s]]))
        dt = 0.001
        for _ in range(int(round(horizon_s / dt))):
            sim.step(dt)
        k = (gt.t_s - (row.t_s + horizon_s)).abs().idxmin()
        out.append(float(np.hypot(*(sim.pos[0] - gt.loc[k, ["x_m", "y_m"]].to_numpy(float)))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--fitted", help="physics_params file written by run_sysid from <session>/twin")
    p.add_argument("--horizon", type=float, default=0.15)
    args = p.parse_args()
    print("tracking:", yaml.safe_dump(tracking_metrics(args.session), sort_keys=False), sep="\n")
    base = load_config("v0")
    sets = [("v0", base)] + ([("fitted", _with_params(base, args.fitted))] if args.fitted else [])
    for name, cfg in sets:
        e = np.array(restart_errors(args.session, cfg, args.horizon)) * 1000
        print(f"start from a real moment, {name} physics (mu {cfg['puck']['friction_kinetic']:.3f}): "
              f"position error {args.horizon:.2f} s later: median {np.median(e):.1f} mm, max {e.max():.1f} mm "
              f"({len(e)} shots)")


if __name__ == "__main__":
    main()
