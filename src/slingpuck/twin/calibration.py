"""Overhead camera calibration: image pixels <-> board frame.

The four inner corners of the play area are clicked once in the camera image
(or found from markers). A homography maps the board plane (z = 0) to pixels.

The camera sees the TOP face of the puck, which is puck.thickness_m above the
board. Seen from a camera at height H, a point at height h appears pushed away
from the camera nadir by a factor H / (H - h). With camera_height_m and
camera_nadir_xy set (measure them once with a tape), pixel_to_board corrects
this. Without them, the error grows with the distance from the nadir: about
2.5 mm at 15 cm for a 10 mm puck and a camera 0.6 m above the board.

Corner order (board frame, agent half y < 0):
    0 agent-left (-x, -y), 1 agent-right (+x, -y), 2 opponent-right (+x, +y), 3 opponent-left (-x, +y)
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


def board_corners(cfg: dict) -> np.ndarray:
    hw, hl = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]
    return np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])


@dataclass
class BoardCalibration:
    corners_px: np.ndarray          # (4, 2) image pixels of the inner corners, in the order above
    corners_board: np.ndarray       # (4, 2) the same corners in the board frame, m
    camera_height_m: float | None = None
    camera_nadir_xy: np.ndarray | None = None  # Board point straight below the camera lens

    def __post_init__(self):
        self.corners_px = np.asarray(self.corners_px, float)
        self.corners_board = np.asarray(self.corners_board, float)
        self.H_board_to_px = cv2.getPerspectiveTransform(self.corners_board.astype(np.float32),
                                                         self.corners_px.astype(np.float32))
        self.H_px_to_board = np.linalg.inv(self.H_board_to_px)
        if self.camera_nadir_xy is not None:
            self.camera_nadir_xy = np.asarray(self.camera_nadir_xy, float)

    @classmethod
    def from_corners(cls, cfg: dict, corners_px, camera_height_m=None, camera_nadir_xy=None) -> "BoardCalibration":
        return cls(np.asarray(corners_px, float), board_corners(cfg), camera_height_m, camera_nadir_xy)

    # ---------------------------------------------------------------- maps
    def plane_to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return cv2.perspectiveTransform(xy[None].astype(np.float64), self.H_board_to_px)[0]

    def px_to_plane(self, px) -> np.ndarray:
        px = np.asarray(px, float).reshape(-1, 2)
        return cv2.perspectiveTransform(px[None].astype(np.float64), self.H_px_to_board)[0]

    def pixel_to_board(self, px, height_m: float = 0.0) -> np.ndarray:
        """Board xy of a point seen at pixel px that is height_m above the board."""
        xy = self.px_to_plane(px)
        if height_m and self.camera_height_m:
            k = (self.camera_height_m - height_m) / self.camera_height_m
            xy = self.camera_nadir_xy + (xy - self.camera_nadir_xy) * k
        return xy

    def board_to_pixel(self, xy, height_m: float = 0.0) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        if height_m and self.camera_height_m:
            k = self.camera_height_m / (self.camera_height_m - height_m)
            xy = self.camera_nadir_xy + (xy - self.camera_nadir_xy) * k
        return self.plane_to_px(xy)

    def px_per_m(self) -> float:
        """Mean image scale on the board plane."""
        d_px = np.linalg.norm(np.diff(np.vstack([self.corners_px, self.corners_px[:1]]), axis=0), axis=1)
        d_m = np.linalg.norm(np.diff(np.vstack([self.corners_board, self.corners_board[:1]]), axis=0), axis=1)
        return float(np.mean(d_px / d_m))

    # ----------------------------------------------------------------- I/O
    def save(self, path):
        data = {"corners_px": self.corners_px.tolist(), "corners_board_m": self.corners_board.tolist(),
                "camera_height_m": self.camera_height_m,
                "camera_nadir_xy_m": None if self.camera_nadir_xy is None else self.camera_nadir_xy.tolist()}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path) -> "BoardCalibration":
        d = yaml.safe_load(Path(path).read_text())
        return cls(np.array(d["corners_px"]), np.array(d["corners_board_m"]), d.get("camera_height_m"),
                   None if d.get("camera_nadir_xy_m") is None else np.array(d["camera_nadir_xy_m"]))


def click_corners(image: np.ndarray) -> np.ndarray:
    """Let the user click the 4 inner corners in a window (order in the module docstring)."""
    names = ("agent-left", "agent-right", "opponent-right", "opponent-left")
    pts = []
    shown = image.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(shown, (x, y), 4, (0, 0, 255), -1)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while len(pts) < 4:
        view = shown.copy()
        cv2.putText(view, f"click the {names[len(pts)]} inner corner", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.imshow("calibrate", view)
        if cv2.waitKey(20) == 27:
            raise KeyboardInterrupt("calibration cancelled")
    cv2.destroyWindow("calibrate")
    return np.array(pts, float)


def main():
    import argparse

    from src.slingpuck.config import load_config
    parser = argparse.ArgumentParser(description="Click the 4 inner board corners in the first video frame")
    parser.add_argument("video")
    parser.add_argument("--out", default="calibration.yaml")
    parser.add_argument("--camera-height", type=float, help="Lens height above the board, m")
    parser.add_argument("--camera-nadir", type=float, nargs=2, help="Board xy straight below the lens, m")
    parser.add_argument("--params-version", default="latest")
    args = parser.parse_args()
    ok, frame = cv2.VideoCapture(args.video).read()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    cal = BoardCalibration.from_corners(load_config(args.params_version), click_corners(frame),
                                        args.camera_height, args.camera_nadir)
    cal.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
