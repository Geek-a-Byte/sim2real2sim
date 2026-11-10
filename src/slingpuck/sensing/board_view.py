"""Overhead camera image <-> board frame: a homography from the 4 inner corners of the play area.

Board frame (as in board_frame.py and the README): origin at the board center on the play surface, x = the
robot's right, y toward the opponent (robot half y < 0). Meters.

Calibration: click (or give) the 4 inner corners of the play area in the camera image, in CORNER_ORDER. The
homography maps every image pixel on the play surface to the board frame. The camera may look at the board at any
angle and in any image orientation (the corner order fixes the orientation).

Rectified view: rectify() warps a camera frame to a top-down image of the play area at px_per_mm, as the robot sees
the board from behind: the opponent end at the top, the robot end at the bottom, the robot's right at the right.
In this image a puck is a circle of a known radius, so the detector needs no per-camera tuning of the radius.

Parallax: the camera sees the TOP face of a puck, puck.thickness_m above the play surface. With the camera height
above the surface (camera_height_m) and the board (x, y) under the camera lens (camera_xy_m), correct_parallax()
moves the seen center to the true center. Without them, the error is thickness / height x the distance from the
point under the lens (about 2.5 mm at 15 cm for a 10 mm puck and a 60 cm camera height).

The file configs/overhead_camera.yaml holds the corners and the camera values (overhead_camera.py calibrate writes
it). The board size comes from the physics config, so a new board measurement does not need a new calibration.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.config import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "configs" / "overhead_camera.yaml"

# Inner corners of the play area, in the order to click them: (name, sign of board x, sign of board y)
CORNER_ORDER = (
    ("robot end, robot's left", -1, -1),
    ("robot end, robot's right", +1, -1),
    ("opponent end, robot's right", +1, +1),
    ("opponent end, robot's left", -1, +1),
)
RECT_MARGIN_M = 0.005     # The rectified view shows this much past the inner corners (calibration error)


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the env
        raise ImportError("The overhead camera needs OpenCV: pip install -e '.[vision]'") from e
    return cv2


def corner_board_xy(width_m: float, length_m: float) -> np.ndarray:
    """Board (x, y) of the 4 inner corners in CORNER_ORDER."""
    return np.array([[sx * 0.5 * width_m, sy * 0.5 * length_m] for _, sx, sy in CORNER_ORDER])


def apply_h(h: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to (n, 2) points (or one (2,) point)."""
    p = np.asarray(pts, float)
    one = p.ndim == 1
    p = np.atleast_2d(p)
    q = np.c_[p, np.ones(len(p))] @ h.T
    q = q[:, :2] / q[:, 2:3]
    return q[0] if one else q


def fit_homography(src, dst) -> np.ndarray:
    """Homography that maps src -> dst (>= 4 point pairs, least squares). Pure numpy (DLT with normalization)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("A homography needs at least 4 point pairs.")

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    ts, td = norm(src), norm(dst)
    a, b = apply_h(ts, src), apply_h(td, dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


@dataclass
class BoardView:
    corners_px: np.ndarray            # (4, 2) image pixels of the inner corners, CORNER_ORDER
    width_m: float                    # Inner play width (board.width_m)
    length_m: float                   # Inner play length (board.length_m)
    px_per_mm: float = 2.0            # Scale of the rectified view
    camera_height_m: float | None = None    # Lens height above the play surface (parallax correction)
    camera_xy_m: np.ndarray | None = None   # Board (x, y) under the lens
    image_size: tuple | None = None   # (w, h) of the calibration image
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.corners_px = np.asarray(self.corners_px, float).reshape(4, 2)
        if self.camera_xy_m is not None:
            self.camera_xy_m = np.asarray(self.camera_xy_m, float)
        u, v = self.corners_px.T
        area = 0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))     # Shoelace
        if area < 100:
            raise ValueError("The corners are almost on one line. Click the 4 inner corners of the play area.")
        board = corner_board_xy(self.width_m, self.length_m)
        self.H_img_to_board = fit_homography(self.corners_px, board)
        self.H_board_to_img = np.linalg.inv(self.H_img_to_board)
        s = 1000.0 * self.px_per_mm
        m = RECT_MARGIN_M
        # Board -> rectified pixels: u = (x + W/2 + m) s, v = (L/2 + m - y) s
        self.H_board_to_rect = np.array([[s, 0, (0.5 * self.width_m + m) * s],
                                         [0, -s, (0.5 * self.length_m + m) * s],
                                         [0, 0, 1.0]])
        self.H_rect_to_board = np.linalg.inv(self.H_board_to_rect)
        self.H_img_to_rect = self.H_board_to_rect @ self.H_img_to_board
        self.rect_size = (int(round((self.width_m + 2 * m) * s)), int(round((self.length_m + 2 * m) * s)))

    # -------------------------------------------------------------- maps
    def img_to_board(self, px) -> np.ndarray:
        return apply_h(self.H_img_to_board, px)

    def board_to_img(self, xy) -> np.ndarray:
        return apply_h(self.H_board_to_img, xy)

    def rect_to_board(self, px) -> np.ndarray:
        return apply_h(self.H_rect_to_board, px)

    def board_to_rect(self, xy) -> np.ndarray:
        return apply_h(self.H_board_to_rect, xy)

    @property
    def rect_px_per_m(self) -> float:
        return 1000.0 * self.px_per_mm

    def camera_px_per_m(self) -> float:
        """Camera pixels per meter at the board center (sqrt of the local area scale)."""
        j = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = 1e-3
            j[:, k] = (self.board_to_img(d) - self.board_to_img(-d)) / 2e-3
        return float(np.sqrt(abs(np.linalg.det(j))))

    def rectify(self, frame):
        cv2 = _cv2()
        return cv2.warpPerspective(frame, self.H_img_to_rect, self.rect_size, flags=cv2.INTER_LINEAR)

    def correct_parallax(self, xy, height_m: float) -> np.ndarray:
        """Seen board (x, y) of a point height_m above the surface -> its true board (x, y) (needs the camera)."""
        xy = np.asarray(xy, float)
        if self.camera_height_m is None or self.camera_xy_m is None or height_m <= 0:
            return xy
        if height_m >= self.camera_height_m:
            raise ValueError("camera_height_m must be larger than the puck thickness.")
        return self.camera_xy_m + (xy - self.camera_xy_m) * (1.0 - height_m / self.camera_height_m)

    def on_board(self, xy, margin_m: float = 0.0) -> np.ndarray:
        """True where (x, y) is inside the play area grown by margin_m."""
        xy = np.atleast_2d(np.asarray(xy, float))
        return (np.abs(xy[:, 0]) <= 0.5 * self.width_m + margin_m) & (np.abs(xy[:, 1]) <= 0.5 * self.length_m + margin_m)

    # -------------------------------------------------------------- file
    @classmethod
    def from_config(cls, cfg: dict, path=DEFAULT_PATH) -> "BoardView":
        """The calibration file plus the board size of the physics config."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No camera calibration at {path}. "
                                    f"Run: python -m src.slingpuck.sensing.overhead_camera calibrate --source 0")
        doc = yaml.safe_load(path.read_text())
        cam = doc.get("camera") or {}
        return cls(corners_px=doc["corners_px"], width_m=cfg["board"]["width_m"], length_m=cfg["board"]["length_m"],
                   px_per_mm=float(doc.get("detect", {}).get("px_per_mm", 2.0)),
                   camera_height_m=cam.get("height_m"), camera_xy_m=cam.get("xy_m"),
                   image_size=tuple(doc["meta"]["image_size"]) if doc.get("meta", {}).get("image_size") else None,
                   meta=doc)

    def save(self, path=DEFAULT_PATH, detect: dict | None = None, source: str | None = None) -> Path:
        path = Path(path)
        doc = {
            "meta": {"date": datetime.datetime.now().isoformat(timespec="seconds"), "source": source,
                     "image_size": list(self.image_size) if self.image_size else None,
                     "board_size_at_calibration_m": [self.width_m, self.length_m],
                     "corner_order": [n for n, _, _ in CORNER_ORDER]},
            "corners_px": [[round(float(u), 2), round(float(v), 2)] for u, v in self.corners_px],
            "camera": {"height_m": self.camera_height_m,
                       "xy_m": None if self.camera_xy_m is None else [float(v) for v in self.camera_xy_m]},
            "detect": {"px_per_mm": self.px_per_mm, **(detect or {})},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Overhead camera calibration. Written by: python -m src.slingpuck.sensing.overhead_camera calibrate\n"
                    "# corners_px: the inner corners of the play area in the camera image, in meta.corner_order.\n"
                    "# camera: lens height above the play surface and the board (x, y) under the lens (m); null = no\n"
                    "# parallax correction. detect: puck detector settings (sensing/puck_detector.py DEFAULTS).\n")
            yaml.safe_dump(doc, f, sort_keys=False)
        return path
