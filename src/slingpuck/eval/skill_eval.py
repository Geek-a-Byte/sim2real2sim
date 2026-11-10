"""S3: measure the scripted arm skills in the MuJoCo match (friction grip, one puck per trial). No training.

    python -m src.slingpuck.eval.skill_eval [--trials 150] [--workers 8] [--out results/skills/<name>]

Tests:
  reach      IK error and gripper tilt over the agent half (top-down grasp at each point)
  shots      random shots from the shot model's range: is the model's P(through) right for the real skill?
  best       the model's best shot under joint calibration errors (0 to 1 deg per joint)
  block      opponent shots into the closed gripper, at several gripper offsets from the gate center
  hold       puck drift while the arm holds it seated and pulled back
  row        the arm alone shoots all 5 pucks from the touching start row, with a fixed 2.5 cm pull
  clear      the same with the shot model's choices (shots per minute)

The camera error on the puck position is match.puck_track_noise_m. All arm speeds are script choices
(skills.speed_scale), so the times are what the script takes in sim, not measured arm times.
Writes summary.yaml and trials.csv.
"""
import argparse
import csv
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import yaml

from src.slingpuck.config import REPO_ROOT, git_state, load_config
from src.slingpuck.eval.stats import wilson_ci
from src.slingpuck.opponent.scripted_opponent import threat_half_width
from src.slingpuck.physics.mujoco_match import MatchSim
from src.slingpuck.primitives.sling import ShotSelector
from src.slingpuck.shot.model import load_for, shot_bounds

CALIBRATION_DEG = (0.0, 0.25, 0.5, 1.0)
BLOCK_OFFSETS_M = (0.0, 0.01, 0.02, 0.03, 0.06)  # 6 cm: the jaws are clear of the gate (the control)
SHOT_TIMEOUT_S = 8.0


def park_others(sim: MatchSim, keep):
    """Move every puck except `keep` to the sides of the opponent half, away from the gate's line."""
    spots = [np.array([sx * x, y]) for y in (0.14, 0.10, 0.06) for x in (0.05, 0.08) for sx in (-1, 1)]
    for p, spot in zip((q for q in sim.pucks if q is not keep), spots):
        sim.data.qpos[p["q"]] = spot - p["rest"]
        sim.data.qvel[p["v"]] = 0.0


def place(sim: MatchSim, p, xy):
    sim.data.qpos[p["q"]] = np.asarray(xy) - p["rest"]
    sim.data.qvel[p["v"]] = 0.0


def run_shot(sim: MatchSim, shot, rng, calibration_rad=0.0) -> dict:
    """One trial: a puck at a random spot, then pick, seat, draw, release, back to the gate."""
    sim.joint_offset = rng.uniform(-calibration_rad, calibration_rad, 5)
    sim.reset()
    sim.auto_fetch = False
    p = next(q for q in sim.pucks if q["colour"] == "black")
    park_others(sim, p)
    xy = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.09, -0.04)])
    place(sim, p, xy)
    seen = xy + rng.normal(0.0, sim.cfg["match"]["puck_track_noise_m"], 2)
    sim._robot_fetch(p, shot, seen)
    log = sim.fetch_log[-1]
    min_tip_z = np.inf
    while sim.t < SHOT_TIMEOUT_S and not (log["outcome"] and not sim.robot["moves"]):
        sim.step()
        min_tip_z = min(min_tip_z, min(sim.to_board(sim.data.geom_xpos[g])[2] for g in sim._tips))
    out = {"launch_x": shot[0], "aim": shot[1], "pull": shot[2], "calibration_rad": calibration_rad,
           "min_tip_z_m": float(min_tip_z), "start_x": float(xy[0]), "start_y": float(xy[1]), "seen_err_x": float(seen[0] - xy[0]), "seen_err_y": float(seen[1] - xy[1]),
           "grasped": log["grasped"], "slip": log["slip"], "outcome": log["outcome"] or "timeout",
           "t_pick": log.get("t_grasp", np.nan) - log["t0"], "t_seat": log.get("t_seated", np.nan) - log.get("t_grasp", np.nan),
           "t_draw": log.get("t_drawn", np.nan) - log.get("t_seated", np.nan),
           "t_to_release": log.get("t_release", np.nan) - log["t0"], "t_cycle": sim.t if not sim.robot["moves"] else np.nan}
    if "pos_drawn" in log:
        out["draw_err_m"] = float(np.linalg.norm(log["pos_drawn"] - sim.pull_point(*shot)[:2]))
    if "exit_pos" in log and "t_exit" in log:
        v = log["exit_vel"]
        out.update(exit_x=float(log["exit_pos"][0]), exit_speed=float(np.linalg.norm(v)),
                   exit_angle=float(np.arctan2(v[0], v[1])), launch_err_m=float(log["exit_pos"][0] - shot[0]))
        ideal = sim.band.release(sim.pull_point(*shot)[:2], sim.cfg["puck"]["mass_kg"], sim.cfg["puck"]["friction_kinetic"])
        out.update(ideal_exit_speed=float(np.linalg.norm(ideal.exit_vel)), ideal_exit_x=float(ideal.exit_pos[0]),
                   ideal_exit_angle=float(np.arctan2(ideal.exit_vel[0], ideal.exit_vel[1])))
    return out


def run_block(sim: MatchSim, offset, x, speed, settle_s=0.4) -> dict:
    """The closed gripper stands `offset` from the gate center; one opponent shot comes at launch x."""
    sim.joint_offset = np.zeros(5)
    sim.reset()
    sim.auto_fetch = False
    sim.robot.update(state="guard", k=0, t0=0.0, moves=sim.skill_block(sim.robot["target"], offset))
    while sim.t < settle_s or sim.robot["moves"]:
        sim.step()
    p = next(q for q in sim.pucks if q["colour"] == "white")
    park_others(sim, p)
    place(sim, p, (x, -sim.band.band_y))
    sim.data.qvel[p["v"]] = (0.0, -speed)
    t0, passed = sim.t, False
    half_t = 0.5 * sim.cfg["board"]["divider_thickness_m"]
    limit = -(half_t + 2 * sim.r + 0.03)           # Past the gripper, on the robot's half
    while sim.t - t0 < 1.0:
        sim.step()
        y = sim.puck_pos(p)[1]
        passed |= y < limit
        if np.linalg.norm(sim.puck_vel(p)) < 0.01:
            break
    return {"offset": offset, "x": x, "speed": speed, "passed": bool(passed), "end_y": float(sim.puck_pos(p)[1])}


def run_hold(sim: MatchSim, pull, seconds=2.0) -> dict:
    """Grip a puck, seat it, draw it back by `pull`, then hold. Drift of the puck during the hold."""
    sim.joint_offset = np.zeros(5)
    sim.reset()
    sim.auto_fetch = False
    p = next(q for q in sim.pucks if q["colour"] == "black")
    park_others(sim, p)
    place(sim, p, (0.0, -0.07))
    here = sim.robot["target"]
    seat = np.array([0.0, sim.band.band_y, sim.z_mid])
    at = np.array([0.0, -0.07, sim.z_mid])
    pulled = sim.pull_point(0.0, 0.0, pull)
    sim.fetch_log.append({"puck": p["name"], "t0": 0.0, "grasped": False, "slip": False, "outcome": None})
    sim.robot.update(state="fetch", k=0, t0=0.0, puck=p, moves=[
        *sim.skill_pick(here, np.array([0.0, -0.07])), *sim.skill_seat(at, 0.0), *sim.skill_draw(seat, 0.0, 0.0, pull),
        *sim.skill_hold(pulled, seconds)])
    start = None
    while sim.robot["moves"]:
        sim.step()
        if start is None and sim.robot["moves"] and sim.robot["moves"][sim.robot["k"]].name == "hold":
            start = sim.puck_pos(p).copy()
    log = sim.fetch_log[-1]
    drift = float(np.linalg.norm(sim.puck_pos(p) - start)) if start is not None else np.nan
    return {"pull": pull, "grasped": log["grasped"], "slip": log["slip"], "drift_m": drift, "held": sim.robot["held"] is not None}


def run_clear(sim: MatchSim, max_t=90.0) -> dict:
    """The arm alone clears its pucks, with the shot model's choices. Rack start."""
    sim.joint_offset = np.zeros(5)
    sim.reset()
    while sim.t < max_t and sim.counts()["robot"] > 0:
        sim.step()
    logs = sim.fetch_log
    return {"cleared": 5 - sim.counts()["robot"], "time_s": sim.t, "fetches": len(logs),
            "through": sum(x["outcome"] == "through" for x in logs),
            "missed_grasps": sum(x["outcome"] == "missed_grasp" for x in logs), "slips": sum(x["slip"] for x in logs)}


def run_row(sim: MatchSim, seed: int, park_white: bool = False, max_t=60.0) -> dict:
    """The touching start row, fixed straight shots: does each puck come out, get gripped and go through?
    park_white: the opponent's row is off the gate's line (else the shots bounce off it, back into the gate)."""
    sim.rng = np.random.default_rng(seed)
    sim.joint_offset = np.zeros(5)
    sim.see_noise_m = sim.cfg["match"]["puck_track_noise_m"]
    sim.reset()
    blacks = [p for p in sim.pucks if p["colour"] == "black"]
    if park_white:
        spots = [np.array([sx * 0.07, y]) for y in (0.14, 0.11, 0.08) for sx in (-1, 1)]
        for p, spot in zip((q for q in sim.pucks if q["colour"] == "white"), spots):
            place(sim, p, spot)
    first, min_tip_z = None, np.inf
    while sim.t < max_t and sim.counts()["robot"] > 0:
        sim.step()
        min_tip_z = min(min_tip_z, min(sim.to_board(sim.data.geom_xpos[g])[2] for g in sim._tips))
        if first is None and sim.fetch_log and sim.fetch_log[0].get("freed"):
            first = {p["name"]: float(np.linalg.norm(sim.puck_pos(p) - p["rest"])) for p in blacks
                     if p["name"] != sim.fetch_log[0]["puck"]}
    logs = sim.fetch_log
    return {"cleared": 5 - sim.counts()["robot"], "time_s": sim.t, "fetches": len(logs), "freed": sum(x["freed"] for x in logs),
            "grasped": sum(x["grasped"] for x in logs), "through": sum(x["outcome"] == "through" for x in logs),
            "missed_grasps": sum(x["outcome"] == "missed_grasp" for x in logs), "slips": sum(x["slip"] for x in logs),
            "neighbour_move_m": max(first.values()) if first else float("nan"), "min_tip_z_m": float(min_tip_z),
            "t_first_release": logs[0].get("t_release", float("nan")) - logs[0]["t0"] if logs else float("nan")}


def reach_map(sim: MatchSim) -> dict:
    """Top-down reach over the agent half: IK from the guard pose, position error below 1 mm, tilt below 5 deg."""
    b = sim.cfg["board"]
    xs = np.linspace(-0.5 * b["width_m"] + sim.r, 0.5 * b["width_m"] - sim.r, 13)
    ys = np.linspace(-0.5 * b["length_m"] + sim.r, -sim.face_y, 13)
    d, rows = sim._ik, []
    for y in ys:
        for x in xs:
            target = np.array([x, y, sim.z_mid])
            q = sim.solve_ik(target, sim._guard_q)
            d.qpos[sim._arm_q] = q
            mujoco.mj_kinematics(sim.model, d)
            err = float(np.linalg.norm(d.site_xpos[sim._site] - sim.to_world(target)))
            tilt = float(np.degrees(np.arccos(np.clip(-d.site_xmat[sim._site].reshape(3, 3)[2, 0], -1, 1))))
            rows.append({"x": x, "y": y, "err_m": err, "tilt_deg": tilt, "ok": err < 1e-3 and tilt < 5.0})
    return {"rows": rows, "fraction_ok": float(np.mean([r["ok"] for r in rows])),
            "y_min_ok": min((r["y"] for r in rows if r["ok"]), default=None)}


def _worker(job):
    kind, args, seed = job
    cfg = load_config("latest")
    global _SIM
    if _SIM is None:
        _SIM = MatchSim(cfg, opponent=False)
    rng = np.random.default_rng(seed)
    if kind == "shot":
        return run_shot(_SIM, args[0], rng, args[1])
    if kind == "block":
        return run_block(_SIM, *args)
    if kind == "hold":
        return run_hold(_SIM, *args)
    if kind == "row":
        return run_row(_SIM, seed, *args)
    raise ValueError(kind)


_SIM = None


def rate(rows, key, value=True):
    k, n = sum(r[key] == value for r in rows), len(rows)
    lo, hi = wilson_ci(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": n, "rate": k / n if n else float("nan"), "ci": [float(lo), float(hi)]}


def stat(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p90_abs": float(np.percentile(np.abs(v), 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=150, help="Trials per shot test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=40_000)
    p.add_argument("--out")
    a = p.parse_args()
    cfg = load_config("latest")
    model = load_for(cfg)
    selector = ShotSelector({**cfg, "shot": {**cfg["shot"], "selection": "greedy"}}, model)
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "skills" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    bounds = shot_bounds(cfg)
    k_best = int(np.argmax(selector.p))
    best = selector.shots[k_best]
    half = threat_half_width(cfg)

    shots = rng.uniform(bounds[:, 0], bounds[:, 1], (2 * a.trials, 3))
    shots[:, 0] = rng.uniform(-0.01, 0.01, len(shots))   # Launch x near the gate: the rest is a sure miss
    jobs = [("shot", (tuple(s), 0.0), a.seed + i) for i, s in enumerate(shots)]
    n_rand = len(jobs)
    for j, deg in enumerate(CALIBRATION_DEG):
        jobs += [("shot", (tuple(best), np.radians(deg)), a.seed + 10_000 * (j + 1) + i) for i in range(a.trials)]
    n_shots = len(jobs)
    for offset in BLOCK_OFFSETS_M:
        for x in np.linspace(-half, half, 5):
            for speed in (1.0, 2.0, 3.0):
                jobs.append(("block", (offset, float(x), speed), 0))
    n_block = len(jobs)
    n_hold = n_block + 4
    jobs += [("hold", (pull,), 0) for pull in (0.0, 0.01, 0.02, 0.03)]
    n_row = n_hold + a.trials // 5
    jobs += [("row", (False,), a.seed + 90_000 + i) for i in range(a.trials // 5)]
    jobs += [("row", (True,), a.seed + 90_000 + i) for i in range(a.trials // 5)]

    with Pool(a.workers) as pool:
        res = pool.map(_worker, jobs, chunksize=4)
    rand_rows = res[:n_rand]
    best_rows = res[n_rand:n_shots]
    block_rows = res[n_shots:n_block]
    hold_rows = res[n_block:n_hold]
    row_rows, row_empty_rows = res[n_hold:n_row], res[n_row:]

    pred = model.predict(np.array([[r["launch_x"], r["aim"], r["pull"]] for r in rand_rows]))[0]
    bins = []
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        idx = [i for i, q in enumerate(pred) if lo <= q < hi and rand_rows[i]["grasped"]]
        rows = [rand_rows[i] for i in idx]
        bins.append({"model_p": [lo, min(hi, 1.0)], "mean_model_p": float(pred[idx].mean()) if idx else None,
                     "through": rate(rows, "outcome", "through")})
    summary = {
        "params_version": cfg["meta"]["params_version"], "speed_scale": cfg["skills"]["speed_scale"], **git_state(),
        "best_shot": {"launch_x": float(best[0]), "aim": float(best[1]), "pull": float(best[2]), "model_p": float(selector.p[k_best])},
        "random_shots": {"grasped": rate(rand_rows, "grasped"), "slip": rate(rand_rows, "slip"),
                         "launch_err_m": stat(rand_rows, "launch_err_m"), "draw_err_m": stat(rand_rows, "draw_err_m"),
                         "exit_speed_ratio": stat([{"v": r["exit_speed"] / r["ideal_exit_speed"]} for r in rand_rows
                                                   if r.get("ideal_exit_speed", 0) > 0 and "exit_speed" in r], "v"),
                         "exit_angle_err_rad": stat([{"v": r["exit_angle"] - r["ideal_exit_angle"]} for r in rand_rows
                                                     if "exit_angle" in r and "ideal_exit_angle" in r], "v"),
                         "calibration_of_model": bins, "outcomes": _counts(rand_rows)},
        "best_shot_by_calibration": [], "times_s": {k: stat(best_rows[:a.trials], k) for k in
                                                    ("t_pick", "t_seat", "t_draw", "t_to_release", "t_cycle")},
        "block": [{"offset_m": o, "saved": rate([r for r in block_rows if r["offset"] == o], "passed", False)}
                  for o in BLOCK_OFFSETS_M],
        "hold": hold_rows,
        "row": {name: {"trials": len(rows), **{k: stat(rows, k) for k in
                ("cleared", "through", "grasped", "freed", "fetches", "missed_grasps", "slips", "time_s", "neighbour_move_m",
                 "t_first_release", "min_tip_z_m")}} for name, rows in (("opponent_row", row_rows), ("no_opponent_row", row_empty_rows))}}
    for j, deg in enumerate(CALIBRATION_DEG):
        rows = best_rows[j * a.trials:(j + 1) * a.trials]
        summary["best_shot_by_calibration"].append({
            "calibration_deg": deg, "grasped": rate(rows, "grasped"), "through": rate(rows, "outcome", "through"),
            "launch_err_m": stat(rows, "launch_err_m"), "exit_angle_rad": stat(rows, "exit_angle"),
            "exit_speed_m_s": stat(rows, "exit_speed"), "ideal_exit_speed_m_s": stat(rows, "ideal_exit_speed"),
            "outcomes": _counts(rows)})

    solo = MatchSim(cfg, opponent=False, selector=selector)
    summary["clear"] = run_clear(solo)
    summary["clear"]["shots_per_min"] = summary["clear"]["fetches"] / (summary["clear"]["time_s"] / 60.0)
    rm = reach_map(solo)
    summary["reach"] = {k: rm[k] for k in ("fraction_ok", "y_min_ok")}

    (out / "summary.yaml").write_text(yaml.safe_dump(_plain(summary), sort_keys=False))
    keys = sorted({k for r in rand_rows + best_rows for k in r})
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rand_rows + best_rows)
    print(yaml.safe_dump(_plain(summary), sort_keys=False))
    print(f"Wrote {out}")


def _counts(rows):
    c = {}
    for r in rows:
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    return c


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


if __name__ == "__main__":
    main()
