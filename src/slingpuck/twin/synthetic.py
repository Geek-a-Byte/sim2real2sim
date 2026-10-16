"""Synthetic "real" session for testing the twin before the hardware exists.

MuJoCo plays robot sling shots with physics values that the twin does NOT know
(the "truth"), and renders what an overhead camera would record. The output
folder has exactly what a real session will have, plus the ground truth:

  camera.mp4        overhead video (--fps, default 60; mp4 compression, pixel noise)
  calibration.yaml  board corners as a person would click them (about 0.7 px error),
                    camera height and nadir as measured with a tape (a few mm error)
  joints.csv        joint readings (encoder quantization 0.0015 rad)
  robot_shots.csv   commands and event times of each shot
  truth/            true puck positions per frame, truth.yaml (true physics values)

Scene: 5 pucks (black = robot's, white = human's). One black puck is shot again and again: between shots a hand picks
it up (it is out of view for gap_s) and puts it back on the band at a new x.
Four pucks stay still away from the shot lanes. Some shots aim 3-5 cm off
center so they bounce back from the divider (sysid needs those for the wall
restitution).

    python -m src.slingpuck.twin.synthetic --out sessions/synthetic_001
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
import pandas as pd
import yaml

from src.slingpuck.config import load_config
from src.slingpuck.physics.arm_deflection import pull_direction
from src.slingpuck.physics.mujoco_sling import MujocoSlingSim, ShotPlan
from src.slingpuck.sysid.fit import with_values
from src.slingpuck.twin.calibration import BoardCalibration, board_corners
from src.slingpuck.twin.logs import ROBOT_SHOT_COLUMNS
from src.slingpuck.twin.scene import JOINTS, TwinScene, project

TRUTH = {"puck.friction_kinetic": 0.25, "band.energy_transfer": 0.62, "board.restitution": 0.80}
STATIC_PUCKS = [(-0.07, 0.10, "white"), (0.07, 0.13, "white"), (-0.075, -0.035, "black"), (0.065, 0.06, "white")]
ENCODER_RAD = 2 * np.pi / 4096


def shot_plans(rng, band_y, n_gate=6, n_divider=4):
    plans = []
    for k in range(n_gate + n_divider):
        divider = k >= n_gate
        slide = rng.choice([-1, 1]) * rng.uniform(0.03, 0.045) if divider else rng.uniform(-0.002, 0.002)
        angle = rng.uniform(-0.08, 0.08)
        pull = rng.uniform(0.02, 0.028) if divider else rng.uniform(0.015, 0.028)
        x0 = rng.uniform(-0.05, 0.05)
        target = np.array([slide, band_y]) + pull * pull_direction(angle)
        plans.append((ShotPlan(x0, slide, target), slide, angle, pull))
    rng.shuffle(plans)
    return plans


def generate(out_dir, base_version: str = "v0", truth: dict | None = None, seed: int = 0, n_gate: int = 6,
             n_divider: int = 4, gap_s: float = 0.5, width: int = 640, height: int = 480,
             camera_height_m: float = 0.6, fps: int = 60) -> Path:
    out = Path(out_dir)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    truth = dict(TRUTH if truth is None else truth)
    cfg = with_values(load_config(base_version), truth)
    rng = np.random.default_rng(seed)
    sim = MujocoSlingSim(cfg)
    n_pucks = 1 + len(STATIC_PUCKS)
    scene = TwinScene(cfg, n_pucks, overhead_height_m=camera_height_m)
    renderer = mujoco.Renderer(scene.model, height, width)
    writer = cv2.VideoWriter(str(out / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    statics = [SimpleNamespace(id=k + 1, pos=np.array(p[:2]), hidden=False, colour=p[2])
               for k, p in enumerate(STATIC_PUCKS)]
    frames_truth, joint_rows, shot_rows = [], [], []
    clock = {"t": 0.0, "k": 0}

    def emit(q, shot_xy):
        """Render and record one camera frame at the current global time."""
        t = clock["k"] / fps
        pucks = list(statics)
        if shot_xy is not None:
            pucks.append(SimpleNamespace(id=0, pos=np.asarray(shot_xy), hidden=False, colour="black"))
        scene.set_arm(q)
        scene.set_pucks(pucks)
        renderer.update_scene(scene.data, camera="overhead")
        img = renderer.render()[:, :, ::-1].astype(np.int16)
        img = np.clip(img + rng.normal(0.0, 2.0, img.shape), 0, 255).astype(np.uint8)
        writer.write(img)
        for p in pucks:
            frames_truth.append((t, p.id, p.colour, *p.pos))
        q_meas = np.round((np.asarray(q) + rng.normal(0.0, 0.3 * ENCODER_RAD, len(q))) / ENCODER_RAD) * ENCODER_RAD
        joint_rows.append((t, *q_meas))
        clock["k"] += 1

    def arm_q(s):
        return np.r_[s.data.qpos[s._arm_qadr], s.data.qpos[s._grip_qadr]]

    plans = shot_plans(rng, sim.band.band_y, n_gate, n_divider)
    for sid, (plan, slide, angle, pull) in enumerate(plans):
        # Hand puts the puck back on the band: out of view for gap_s, arm at home.
        sim.reset(plan.x0)
        for _ in range(int(gap_s * fps)):
            emit(arm_q(sim), None)
        t_start = clock["k"] / fps
        frame_t0 = {}

        def on_frame(s):
            frame_t0.setdefault("sim0", s.t)
            emit(arm_q(s), s.puck_pos)

        res = sim.run(plan, on_frame, frame_dt=1.0 / fps, observe_s=cfg["sling"]["observe_window_s"],
                      max_flight_s=1.2, after_gate_s=0.5, rest_s=0.5)
        ev = {k: t_start + v for k, v in res.events.items()}
        shot_rows.append((sid, ev["idle"], ev["approach"], ev["slide"], ev["release"], slide, angle, pull))
        print(f"shot {sid}: slide {1000 * slide:+.1f} mm, pull {1000 * pull:.1f} mm, "
              f"{'through' if res.success else 'miss'} (truth)")
    writer.release()

    pd.DataFrame(joint_rows, columns=["t_s", *JOINTS]).to_csv(out / "joints.csv", index=False)
    pd.DataFrame(shot_rows, columns=ROBOT_SHOT_COLUMNS).to_csv(out / "robot_shots.csv", index=False)
    pd.DataFrame(frames_truth, columns=["t_s", "puck", "colour", "x_m", "y_m"]).to_csv(out / "truth" / "pucks.csv", index=False)

    # Calibration as a person would make it: clicked corners, taped camera height and nadir.
    scene.set_arm(arm_q(sim))
    corners = project(scene.model, scene.data, "overhead", np.c_[board_corners(cfg), np.zeros(4)], width, height)
    corners += rng.normal(0.0, 0.7, corners.shape)
    BoardCalibration.from_corners(cfg, corners, camera_height_m + rng.normal(0.0, 0.003),
                                  rng.normal(0.0, 0.003, 2)).save(out / "calibration.yaml")
    (out / "truth" / "truth.yaml").write_text(yaml.safe_dump(
        {"synthetic": True, "base_version": base_version, "seed": seed, "truth": truth,
         "n_pucks": n_pucks, "fps": fps}, sort_keys=False))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="sessions/synthetic_001")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-shots", type=int, default=6)
    p.add_argument("--divider-shots", type=int, default=4)
    p.add_argument("--fps", type=int, default=60, help="Camera frame rate")
    args = p.parse_args()
    out = generate(args.out, seed=args.seed, n_gate=args.gate_shots, n_divider=args.divider_shots, fps=args.fps)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
