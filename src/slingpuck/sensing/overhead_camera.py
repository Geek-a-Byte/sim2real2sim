"""Overhead camera: calibrate it to the board, watch the pucks live, or measure where the pucks rest.

Needs OpenCV (pip install -e '.[vision]'). --source is a camera index (0, 1, ...), a video file or an image file.

1. Calibrate once per camera position (writes configs/overhead_camera.yaml):

    python -m src.slingpuck.sensing.overhead_camera calibrate --source 0
    python -m src.slingpuck.sensing.overhead_camera calibrate --source frame.jpg --corners U1 V1 U2 V2 U3 V3 U4 V4

   Click the 4 inner corners of the play area in this order (the window says which one is next):
     1 robot end, robot's left   2 robot end, robot's right   3 opponent end, robot's right   4 opponent end, robot's left
   Keys: r = start again, Enter = save, q = quit without a save. Then the window shows the rectified view: the robot
   end must be at the bottom and the gate line across the middle.
   Optional parallax correction: --camera-height-m (lens above the play surface) and --camera-xy X Y (board (x, y)
   under the lens, m).

2. Watch the pucks (board (x, y) in mm on each puck; tracked speed with --print):

    python -m src.slingpuck.sensing.overhead_camera live --source 0 [--print] [--save out.mp4]

3. Measure the resting pucks (median over --frames frames; the arm scripts use this):

    python -m src.slingpuck.sensing.overhead_camera snapshot --source 0 [--frames 15] [--out pucks.yaml]

Board frame: origin at the board center, x = the robot's right, y toward the opponent (robot half y < 0), meters.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.config import REPO_ROOT, load_config
from src.slingpuck.sensing.board_view import CORNER_ORDER, DEFAULT_PATH, BoardView, _cv2
from src.slingpuck.sensing.puck_detector import DEFAULTS as DETECT_DEFAULTS
from src.slingpuck.sensing.puck_detector import PuckDetector, draw
from src.slingpuck.sensing.puck_tracker import PuckTracker

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REST_SPREAD_M = 0.004     # A snapshot puck: 90% of its positions within this of their median (else it moves)


class FrameSource:
    """Frames with their capture time (s) from a camera, a video file or one image (repeated)."""

    def __init__(self, source: str, width: int | None = None, height: int | None = None, fps: float | None = None):
        cv2 = _cv2()
        self.source = str(source)
        self.image = None
        self.cap = None
        self.t0 = time.perf_counter()
        self.k = 0
        if Path(self.source).suffix.lower() in IMAGE_EXT:
            self.image = cv2.imread(self.source)
            if self.image is None:
                raise FileNotFoundError(f"Cannot read the image {self.source}")
            self.fps = 30.0
            self.is_file = True
            return
        self.cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open the camera or video {self.source!r}. On macOS, give the terminal "
                               f"camera access (System Settings > Privacy & Security > Camera).")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.is_file = not self.source.isdigit()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        """(ok, frame, t). For a file, t is the frame time in the video; for a camera, the wall time."""
        self.k += 1
        if self.image is not None:
            return True, self.image.copy(), self.k / self.fps
        ok, frame = self.cap.read()
        t = self.k / self.fps if self.is_file else time.perf_counter() - self.t0
        return ok, frame, t

    def release(self):
        if self.cap is not None:
            self.cap.release()


def build(cfg: dict | None = None, calib=DEFAULT_PATH) -> tuple[dict, BoardView, PuckDetector]:
    cfg = cfg or load_config("latest")
    view = BoardView.from_config(cfg, calib)
    return cfg, view, PuckDetector.from_config(cfg, view)


def locate_pucks(source: str = "0", frames: int = 15, calib=DEFAULT_PATH, cfg: dict | None = None,
                 min_seen_frac: float = 0.6) -> list[dict]:
    """Resting pucks: board (x, y) as the median over `frames` frames. Each: {xy, colour, seen, side}.

    A puck must be found in min_seen_frac of the frames, and its positions must stay within REST_SPREAD_M
    (the tracked speed is not used: 1 mm of camera noise gives a resting puck about 0.1 m/s).
    """
    cfg, view, det = build(cfg, calib)
    src = FrameSource(source)
    tracker = PuckTracker(gate_m=cfg["puck"]["radius_m"])
    seen: dict[int, list] = {}
    try:
        for _ in range(3 if src.cap is not None and not src.is_file else 0):      # Let the camera exposure settle
            src.read()
        n = 0
        while n < frames:
            ok, frame, t = src.read()
            if not ok:
                break
            n += 1
            tracker.update(t, det.detect(frame))
            for s in tracker.estimate(confirmed=False):
                if s.age_s == 0.0:
                    seen.setdefault(s.id, []).append((s.xy.copy(), s.colour))
    finally:
        src.release()
    if n == 0:
        raise RuntimeError(f"No frames from {source!r}.")
    out = []
    for tid, rows in seen.items():
        if len(rows) < min_seen_frac * n:
            continue
        pts = np.array([r[0] for r in rows])
        xy = np.median(pts, axis=0)
        if np.percentile(np.linalg.norm(pts - xy, axis=1), 90) > REST_SPREAD_M:
            continue
        colours = [r[1] for r in rows]
        colour = max(set(colours), key=colours.count)
        out.append({"xy": xy, "colour": colour, "seen": len(rows) / n, "side": "robot" if xy[1] < 0 else "opponent"})
    return sorted(out, key=lambda d: (d["side"] != "robot", d["xy"][1], d["xy"][0]))


def robot_half(pucks: list[dict], cfg: dict) -> list[dict]:
    """Pucks the robot must shoot: on its half, past the divider face."""
    face = 0.5 * cfg["board"]["divider_thickness_m"]
    return [p for p in pucks if p["xy"][1] < -face]


# ------------------------------------------------------------------ calibrate
def click_corners(frame) -> np.ndarray | None:
    cv2 = _cv2()
    pts: list[tuple[float, float]] = []
    win = "calibrate: click the inner play-area corners"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) == 4, (255, 160, 0), 2)
        msg = (f"Click corner {len(pts) + 1}: {CORNER_ORDER[len(pts)][0]}" if len(pts) < 4
               else "Enter = use these corners, r = start again, q = quit")
        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            cv2.destroyWindow(win)
            return None
        elif k in (13, 10) and len(pts) == 4:
            cv2.destroyWindow(win)
            return np.array(pts)


def cmd_calibrate(a, cfg):
    cv2 = _cv2()
    src = FrameSource(a.source, a.width, a.height)
    for _ in range(5 if src.cap is not None and not src.is_file else 1):     # Exposure settles on a live camera
        ok, frame, _t = src.read()
    src.release()
    if not ok:
        sys.exit(f"No frame from {a.source!r}.")
    if a.corners:
        corners = np.array(a.corners, float).reshape(4, 2)
    else:
        corners = click_corners(frame)
        if corners is None:
            sys.exit("Not saved.")
    view = BoardView(corners, cfg["board"]["width_m"], cfg["board"]["length_m"], px_per_mm=a.px_per_mm,
                     camera_height_m=a.camera_height_m, camera_xy_m=a.camera_xy,
                     image_size=(frame.shape[1], frame.shape[0]))
    det = PuckDetector(view, cfg["puck"]["radius_m"], cfg["puck"]["thickness_m"])
    dets = det.detect(frame)
    print(f"Camera scale at the board center: {view.camera_px_per_m() / 1000:.2f} px/mm "
          f"(a puck is {2 * cfg['puck']['radius_m'] * view.camera_px_per_m():.0f} px across).")
    if view.camera_px_per_m() / 1000 < 1.0:
        print("WARNING: less than 1 px/mm. Move the camera closer or use a higher resolution.")
    print(f"Pucks found in this frame: {len(dets)} "
          f"({sum(d.side == 'robot' for d in dets)} on the robot half).")
    path = view.save(a.calib, detect={k: v for k, v in DETECT_DEFAULTS.items()}, source=str(a.source))
    print(f"Saved {path}")
    cv2.imwrite(str(Path(path).with_suffix(".jpg")), draw(frame, view, dets))
    if not a.no_window:
        cv2.imshow("rectified (robot end at the bottom): any key closes", draw(view.rectify(frame), view, dets, True))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------- live
def cmd_live(a, cfg):
    cv2 = _cv2()
    cfg, view, det = build(cfg, a.calib)
    src = FrameSource(a.source, a.width, a.height)
    tracker = PuckTracker.from_config(cfg)
    latency = float(cfg["camera"]["latency_s"])
    writer = None
    n, t_proc = 0, 0.0
    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                break
            t0 = time.perf_counter()
            rect = view.rectify(frame)
            dets = det.detect_rectified(rect)
            tracker.update(t, dets)
            t_proc += time.perf_counter() - t0
            n += 1
            states = tracker.estimate(t + latency)            # What the robot should act on now
            if a.print:
                rows = ", ".join(f"#{s.id} {s.colour[0]} ({1000 * s.xy[0]:+4.0f},{1000 * s.xy[1]:+4.0f}) mm "
                                 f"{s.speed:4.2f} m/s" for s in states)
                print(f"t {t:7.2f} s  {len(states)} pucks  {rows}")
            view_img = draw(rect, view, dets, rectified=True)
            for s in states:
                if s.speed > 0.1:                               # Draw the velocity of a moving puck (0.1 s ahead)
                    p0, p1 = view.board_to_rect(s.xy), view.board_to_rect(s.xy + 0.1 * s.vel)
                    cv2.arrowedLine(view_img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (255, 0, 255), 2)
            robot_n = sum(d.side == "robot" for d in dets)
            cv2.putText(view_img, f"robot half {robot_n}  opponent half {len(dets) - robot_n}  "
                                  f"{1000 * t_proc / n:.1f} ms/frame", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
            if a.save:
                if writer is None:
                    h, w = view_img.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
                writer.write(view_img)
            if not a.no_window:
                cv2.imshow("camera", draw(frame, view, dets))
                cv2.imshow("board (robot end at the bottom)", view_img)
                if cv2.waitKey(1 if src.image is None else 0) & 0xFF == ord("q"):
                    break
            elif src.image is not None:
                break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if n:
        print(f"{n} frames, {1000 * t_proc / n:.1f} ms per frame (rectify + detect + track).")


# ------------------------------------------------------------------- snapshot
def cmd_snapshot(a, cfg):
    pucks = locate_pucks(a.source, a.frames, a.calib, cfg)
    print(f"{len(pucks)} resting pucks ({len(robot_half(pucks, cfg))} on the robot half). Board frame, mm:")
    for p in pucks:
        print(f"  {p['side']:8s} {p['colour']:5s} x {1000 * p['xy'][0]:+6.1f}  y {1000 * p['xy'][1]:+6.1f}  "
              f"(seen in {100 * p['seen']:.0f}% of frames)")
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "real" / f"pucks_{datetime.datetime.now():%Y%m%d_%H%M%S}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "source": str(a.source),
           "calibration": str(a.calib), "frames": a.frames,
           "pucks": [{"xy_m": [round(float(v), 4) for v in p["xy"]], "colour": p["colour"], "side": p["side"],
                      "seen": round(p["seen"], 2)} for p in pucks]}
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "live", "snapshot"):
        s = sub.add_parser(name)
        s.add_argument("--source", default="0", help="Camera index, video file or image file (default 0)")
        s.add_argument("--calib", default=str(DEFAULT_PATH))
        s.add_argument("--params-version", default="latest")
        if name != "snapshot":
            s.add_argument("--width", type=int, default=None)
            s.add_argument("--height", type=int, default=None)
            s.add_argument("--no-window", action="store_true")
        if name == "calibrate":
            s.add_argument("--corners", nargs=8, type=float, default=None, metavar="UV",
                           help="Corner pixels u1 v1 ... u4 v4 (no clicks)")
            s.add_argument("--px-per-mm", type=float, default=2.0, help="Scale of the rectified view")
            s.add_argument("--camera-height-m", type=float, default=None)
            s.add_argument("--camera-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
        if name == "live":
            s.add_argument("--print", action="store_true", help="Print the tracked pucks each frame")
            s.add_argument("--save", default=None, help="Write the board view to this video file")
        if name == "snapshot":
            s.add_argument("--frames", type=int, default=15)
            s.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load_config(a.params_version)
    {"calibrate": cmd_calibrate, "live": cmd_live, "snapshot": cmd_snapshot}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
