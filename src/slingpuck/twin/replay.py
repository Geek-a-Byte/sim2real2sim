"""Run the digital twin on a recorded session (offline twin).

    python -m src.slingpuck.twin.replay --session sessions/<name> [--save twin.mp4] [--camera sling]

Session folder: camera.mp4, calibration.yaml, joints.csv, robot_shots.csv
(see data/real_logs/schema.md, section 4). Each frame:
1. Set the twin arm from the joint readings at the frame time.
2. Render the arm mask in the overhead view; detect pucks outside it.
3. Track the pucks; put them in the twin; predict the path of moving pucks.
4. Optionally draw: camera frame with detections | twin view.

Output in <session>/twin/: tracks.csv, shots.csv (sysid format), summary.yaml,
and the side-by-side video if --save is given. Then:

    python -m src.slingpuck.sysid.run_sysid --logs <session>/twin
"""
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
import yaml

from src.slingpuck.config import load_config
from src.slingpuck.physics.puck_dynamics import Fast2DPuckSim
from src.slingpuck.twin.calibration import BoardCalibration
from src.slingpuck.twin.detector import PuckDetector
from src.slingpuck.twin.logs import build_shots, tracks_frame
from src.slingpuck.twin.scene import JOINTS, TwinScene
from src.slingpuck.twin.tracker import MultiPuckTracker


def joints_at(joints: pd.DataFrame, t: float) -> np.ndarray:
    """Joint angles at time t, linear interpolation between readings (time sync of camera and robot)."""
    ts = joints["t_s"].to_numpy()
    return np.array([np.interp(t, ts, joints[j].to_numpy()) for j in JOINTS])


def start_sim_from(states, cfg: dict, include_hidden: bool = False) -> Fast2DPuckSim:
    """A 2D sim that starts from a measured moment of the real game (all tracked pucks)."""
    use = [s for s in states if include_hidden or not s.hidden]
    lim = np.array([0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]]) - cfg["puck"]["radius_m"]
    sim = Fast2DPuckSim.from_config(cfg, n_pucks=len(use))
    sim.reset(np.array([np.clip(s.pos, -lim, lim) for s in use]), np.array([s.vel for s in use]))
    return sim


class Twin:
    def __init__(self, cfg: dict, calib: BoardCalibration, n_pucks: int, width: int, height: int,
                 fps: float = 60.0):
        self.cfg, self.calib = cfg, calib
        nadir = calib.camera_nadir_xy if calib.camera_nadir_xy is not None else (0.0, 0.0)
        self.scene = TwinScene(cfg, n_pucks, overhead_height_m=calib.camera_height_m or 0.6, overhead_nadir=nadir)
        self.detector = PuckDetector(cfg, calib)
        self.tracker = MultiPuckTracker(cfg, n_pucks, meas_noise_m=0.0015)
        self.fps = fps
        self.last_detections = []

    def step(self, t: float, frame: np.ndarray, q: np.ndarray | None):
        if q is not None:
            self.scene.set_arm(q)
        mask = None
        if q is not None and self.calib.camera_height_m:
            mask = self.scene.arm_mask(self.calib, frame.shape[0], frame.shape[1])
        self.last_detections = self.detector.detect(frame, mask)
        states = self.tracker.update(t, [d.xy for d in self.last_detections])
        self.scene.set_pucks(states)
        self.scene.predict()
        return states


def annotate(frame, twin: Twin, states, t):
    img = frame.copy()
    for d in twin.last_detections:
        u, v = twin.calib.board_to_pixel(d.xy, 0.5 * twin.cfg["puck"]["thickness_m"])[0]
        cv2.circle(img, (int(u), int(v)), 4, (0, 200, 0), -1)
    for s in states:
        u, v = twin.calib.board_to_pixel(s.pos, twin.cfg["puck"]["thickness_m"])[0]
        color = (160, 160, 160) if s.hidden else (255, 120, 0)
        cv2.putText(img, str(s.id), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"camera  t {t:6.2f} s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def run(session, params_version: str = "v0", save: str | None = None, camera: str = "sling",
        view_size=(640, 480), max_frames: int | None = None, quiet: bool = False) -> dict:
    session = Path(session)
    out = session / "twin"
    out.mkdir(exist_ok=True)
    cfg = load_config(params_version)
    calib = BoardCalibration.load(session / "calibration.yaml")
    joints = pd.read_csv(session / "joints.csv") if (session / "joints.csv").exists() else None
    meta = yaml.safe_load((session / "truth" / "truth.yaml").read_text()) if (session / "truth" / "truth.yaml").exists() \
        else {}
    n_pucks = int(meta.get("n_pucks", cfg["match"]["pucks_per_side"] * 2))
    cap = cv2.VideoCapture(str(session / "camera.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    twin = Twin(cfg, calib, n_pucks, w, h, fps)
    view = mujoco.Renderer(twin.scene.model, view_size[1], view_size[0]) if save else None
    writer = None
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and k >= max_frames):
            break
        t = k / fps
        q = joints_at(joints, t) if joints is not None else None
        states = twin.step(t, frame, q)
        if save:
            view.update_scene(twin.scene.data, camera=camera)
            twin.scene.add_overlays(view.scene)
            right = view.render()[:, :, ::-1].copy()
            cv2.putText(right, f"MuJoCo twin  ({len(states)} pucks)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (20, 20, 20), 1, cv2.LINE_AA)
            left = cv2.resize(annotate(frame, twin, states, t), (view_size[0], view_size[1]))
            both = np.hstack([left, right])
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*("mp4v" if save.endswith(".mp4") else "MJPG"))
                writer = cv2.VideoWriter(save, fourcc, fps, (both.shape[1], both.shape[0]))
            writer.write(both)
        k += 1
    if writer is not None:
        writer.release()

    tracks = tracks_frame(twin.tracker)
    tracks.to_csv(out / "tracks.csv", index=False)
    summary = {"frames": k, "fps": float(fps), "tracks": int(tracks.puck_id.nunique()) if len(tracks) else 0}
    if joints is not None and (session / "robot_shots.csv").exists():
        def grasp_fk(q):
            twin.scene.set_arm(q)
            return twin.scene.grasp_xy()

        shots, matches = build_shots(tracks, pd.read_csv(session / "robot_shots.csv"), joints, grasp_fk)
        shots.to_csv(out / "shots.csv", index=False)
        summary["shots"] = [m.__dict__ for m in matches]
    if meta.get("synthetic"):
        (out / "truth.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))  # Marks the logs as synthetic for sysid
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    if not quiet:
        print(f"{k} frames, {summary['tracks']} puck tracks -> {out}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--params-version", default="v0")
    p.add_argument("--save", help="Side-by-side video: camera with detections | MuJoCo twin (.mp4 or .avi)")
    p.add_argument("--camera", default="sling", choices=["sling", "band", "shot", "gate", "side", "top", "overhead"])
    p.add_argument("--max-frames", type=int)
    args = p.parse_args()
    run(args.session, args.params_version, args.save, args.camera, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
