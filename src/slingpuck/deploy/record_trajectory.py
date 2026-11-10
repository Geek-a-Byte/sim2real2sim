"""Step 4a: record one sim shot (pick, seat, draw, release) as joint angles, for replay on the real arm.

Run it in the project env (MuJoCo), after the touch test:

    python3 -m src.slingpuck.deploy.record_trajectory --hover-mm 30     # first test: 3 cm above the board
    python3 -m src.slingpuck.deploy.record_trajectory                   # then: on the board, puck on the mid_center mark
    python3 -m src.slingpuck.deploy.record_trajectory --camera 0        # the puck where the overhead camera sees it

The sim is placed like the real setup: robot.base_xy_m and the surface height come from
configs/real_arm_board.yaml (touch_test.py). --hover-mm raises the sim board, so the real arm moves that much
above the real board, with the same sim shot. The puck starts on a touch mark (--mark, default mid_center), or at
--puck X Y in the board frame (m): x = the robot's right, y < 0 on the robot's half.

--camera SOURCE: the overhead camera (sensing/overhead_camera.py, calibrated first) measures the resting pucks.
Every puck on the robot half goes into the sim at its seen position (so the sim sees the neighbours, and pushes a
puck out of a row first when the jaws do not fit). --pick chooses the puck to shoot: greedy (the shot model's best
P(through) / T, as in the match; falls back to nearest when the shot model does not load), nearest (shortest
carry from the gate), or an index into the printed list. --auto-shot also takes the shot of the greedy choice.

Writes results/real/traj_<time>/traj.npz (t, q (sim rad, ARM_JOINTS order), grip (sim rad), move names) and
meta.yaml (the setup and the sim outcome). replay_trajectory.py plays it.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.config import REPO_ROOT, git_state, load_config
from src.slingpuck.deploy import board_frame as bf

OUT_DIR = REPO_ROOT / "results" / "real"
TIMEOUT_S = 8.0


def setup_config(board_file: Path | None, hover_m: float, speed_scale: float | None) -> tuple[dict, dict]:
    """Physics config placed like the real setup. Returns (cfg, notes)."""
    cfg = load_config("latest")
    notes = {"board_file": None, "hover_m": hover_m}
    surface = cfg["board"]["outer_height_m"] - cfg["board"]["wall_height_m"]
    if board_file is not None and Path(board_file).is_file():
        b = bf.load(board_file)
        cfg["robot"]["base_xy_m"] = list(b["robot"]["base_xy_m"])
        surface = float(b["board"]["surface_z_m"])
        notes.update(board_file=str(board_file), yaw_deg=b["fit"]["yaw_deg"], fit_rms_mm=b["fit"]["rms_mm"])
    else:
        print(f"WARNING: no {board_file}: the arm base position is the PLACEHOLDER {cfg['robot']['base_xy_m']}. "
              f"Run touch_test.py first, or use only --hover-mm runs with no board under the arm.")
    cfg["board"]["outer_height_m"] = surface + hover_m + cfg["board"]["wall_height_m"]   # Sim surface height
    if speed_scale is not None:
        cfg["skills"]["speed_scale"] = speed_scale
    notes.update(base_xy_m=cfg["robot"]["base_xy_m"], surface_z_m=surface, sim_surface_z_m=surface + hover_m,
                 speed_scale=cfg["skills"]["speed_scale"])
    return cfg, notes


def guard_xy(cfg: dict) -> np.ndarray:
    """Board (x, y) of the jaws in the guard pose (as MatchSim.reset): the arm starts every shot there."""
    return np.array([0.0, -(0.5 * cfg["board"]["divider_thickness_m"] + cfg["puck"]["radius_m"] + 0.004)])


def choose_puck(cfg: dict, pucks_xy: np.ndarray, pick: str, model=None) -> tuple[int, np.ndarray | None, str]:
    """Index of the puck to shoot, the greedy shot (or None), and how it was chosen. model: a ShotModel (default:
    the file in shot.model_file)."""
    if pick not in ("greedy", "nearest"):
        i = int(pick)
        if not 0 <= i < len(pucks_xy):
            raise IndexError(f"--pick {i}: the camera sees {len(pucks_xy)} pucks on the robot half")
        return i, None, "index"
    if pick == "greedy":
        try:
            from src.slingpuck.primitives.sling import ShotSelector
            from src.slingpuck.shot.model import load_for
            sel_cfg = {**cfg, "shot": {**cfg["shot"], "selection": "greedy"}}
            if model is None:
                # Check the fingerprint against the config before setup_config(): that only moves the board up
                # (board.outer_height_m: real surface height, hover), which the board-only shot sim does not use.
                model = load_for(load_config(cfg.get("meta", {}).get("params_version", "latest")))
            c = ShotSelector(sel_cfg, model).choose(pucks_xy, guard_xy(cfg))
            return c.index, c.shot, f"greedy (P(through) {c.p:.2f}, T {c.time_s:.2f} s)"
        except (FileNotFoundError, ValueError) as e:
            print(f"WARNING: no greedy choice ({e}). The nearest puck is used.")
    d = np.linalg.norm(pucks_xy - guard_xy(cfg), axis=1)
    return int(np.argmin(d)), None, "nearest"


def from_camera(cfg: dict, source: str, frames: int, calib: str, pick: str):
    """(puck_xy, others_xy, greedy shot or None, notes) from the overhead camera."""
    from src.slingpuck.sensing.overhead_camera import locate_pucks, robot_half
    seen = locate_pucks(source, frames, calib, cfg)
    mine = robot_half(seen, cfg)
    print(f"Camera: {len(seen)} resting pucks, {len(mine)} on the robot half (board frame, mm):")
    for k, q in enumerate(mine):
        print(f"  [{k}] {q['colour']:5s} x {1000 * q['xy'][0]:+6.1f}  y {1000 * q['xy'][1]:+6.1f}")
    if not mine:
        sys.exit("The camera sees no puck on the robot half. Check the light and the calibration "
                 "(python -m src.slingpuck.sensing.overhead_camera live).")
    xy = np.array([q["xy"] for q in mine])
    i, shot, how = choose_puck(cfg, xy, pick)
    print(f"Puck [{i}] chosen: {how}")
    notes = {"camera": {"source": str(source), "calibration": str(calib), "frames": frames, "pick": how,
                        "robot_half_xy_m": [[round(float(v), 4) for v in q] for q in xy],
                        "colours": [q["colour"] for q in mine]}}
    return xy[i], np.delete(xy, i, axis=0), shot, notes


def record(cfg: dict, puck_xy, shot, others_xy=()) -> dict:
    from src.slingpuck.eval.skill_eval import park_others, place
    from src.slingpuck.physics.mujoco_match import CONTROL_DT_S, MatchSim

    sim = MatchSim(cfg, opponent=False)
    sim.reset()
    sim.auto_fetch = False
    p = next(q for q in sim.pucks if q["colour"] == "black")
    park_others(sim, p)
    place(sim, p, puck_xy)
    others_xy = np.asarray(others_xy, float).reshape(-1, 2)
    rest = [q for q in sim.pucks if q is not p]
    if len(others_xy) > len(rest):
        raise ValueError(f"{len(others_xy) + 1} pucks on the robot half, but the sim has {len(sim.pucks)}")
    for q, xy in zip(rest, others_xy):       # The other pucks the camera sees on the robot half
        place(sim, q, xy)
    sim._robot_fetch(p, shot, np.asarray(puck_xy, float))
    log = sim.fetch_log[-1]
    rows, k_last = [], -1
    while sim.t < TIMEOUT_S and not (log["outcome"] and not sim.robot["moves"]):
        sim.step()
        k = int(sim.t / CONTROL_DT_S)
        if k != k_last:                      # One row per control step (the IK update)
            k_last = k
            rb = sim.robot
            name = rb["moves"][rb["k"]].name if rb["moves"] else "done"
            rows.append((sim.t, sim.q.copy(), float(sim.data.ctrl[sim._grip_act]), name))
    t = np.array([r[0] for r in rows])
    return {"t": t - t[0], "q": np.array([r[1] for r in rows]), "grip": np.array([r[2] for r in rows]),
            "move": np.array([r[3] for r in rows]),
            "outcome": log["outcome"] or "timeout", "grasped": bool(log["grasped"]), "slip": bool(log["slip"])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board", default=str(bf.DEFAULT_PATH))
    p.add_argument("--mark", default="mid_center", help="Start the puck on this touch mark (touch_test.py)")
    p.add_argument("--puck", nargs=2, type=float, default=None, metavar=("X", "Y"), help="Puck start (board frame, m)")
    p.add_argument("--camera", nargs="?", const="0", default=None, metavar="SOURCE",
                   help="Puck positions from the overhead camera (index, default 0, or a video/image file)")
    p.add_argument("--camera-frames", type=int, default=15, help="Frames for the median puck position")
    p.add_argument("--camera-calib", default=None, help="Camera calibration (default configs/overhead_camera.yaml)")
    p.add_argument("--pick", default="greedy", help="With --camera: greedy, nearest, or an index into the list")
    p.add_argument("--auto-shot", action="store_true", help="With --camera --pick greedy: use the greedy shot")
    p.add_argument("--shot", nargs=3, type=float, default=[0.0, 0.0, 0.025], metavar=("LAUNCH_X", "AIM", "PULL"),
                   help="Launch x (m), aim (rad), pull (m). Default: straight, 2.5 cm")
    p.add_argument("--hover-mm", type=float, default=0.0)
    p.add_argument("--speed-scale", type=float, default=None, help="skills.speed_scale (default: env.yaml)")
    a = p.parse_args()

    cfg, notes = setup_config(Path(a.board), a.hover_mm / 1000, a.speed_scale)
    others = np.zeros((0, 2))
    if a.camera is not None:
        if a.puck is not None:
            sys.exit("Give --camera or --puck, not both.")
        from src.slingpuck.sensing.board_view import DEFAULT_PATH as CAM_PATH
        puck, others, greedy_shot, cam_notes = from_camera(cfg, a.camera, a.camera_frames,
                                                           a.camera_calib or str(CAM_PATH), a.pick)
        puck = [float(v) for v in puck]
        notes.update(cam_notes)
        if a.auto_shot:
            if greedy_shot is None:
                sys.exit("--auto-shot needs a greedy choice (--pick greedy and a shot model).")
            a.shot = [round(float(v), 4) for v in greedy_shot]
            print(f"Greedy shot: launch x {1000 * a.shot[0]:+.1f} mm, aim {a.shot[1]:+.3f} rad, pull {1000 * a.shot[2]:.1f} mm")
    else:
        puck = a.puck if a.puck is not None else bf.mark_xy(a.mark, cfg["board"]["divider_thickness_m"]).tolist()
    tr = record(cfg, puck, a.shot, others)
    moves = list(dict.fromkeys(tr["move"].tolist()))
    out = OUT_DIR / f"traj_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "traj.npz", t=tr["t"], q=tr["q"], grip=tr["grip"], move=tr["move"])
    meta = {"puck_xy_m": [float(v) for v in puck], "mark": None if (a.puck is not None or a.camera is not None) else a.mark,
            "shot": a.shot, **notes, "sim_outcome": tr["outcome"], "sim_grasped": tr["grasped"],
            "sim_slip": tr["slip"], "duration_s": float(tr["t"][-1]), "frames": int(len(tr["t"])), "moves": moves,
            "date": datetime.datetime.now().isoformat(timespec="seconds"), **git_state()}
    with open(out / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"Sim outcome: {tr['outcome']} (grasped {tr['grasped']}, slip {tr['slip']}); {meta['duration_s']:.2f} s, "
          f"{meta['frames']} frames")
    print(f"Moves: {' -> '.join(moves)}")
    if a.hover_mm:
        print(f"The sim board is {a.hover_mm:.0f} mm higher than the real one: on the arm, the jaws stay above the board.")
    print(f"Saved {out}")
    if tr["outcome"] == "timeout":
        sys.exit("The sim shot timed out: do not replay this trajectory.")


if __name__ == "__main__":
    main()
