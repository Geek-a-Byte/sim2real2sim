"""Find the pucks in an overhead camera frame and give their board (x, y) and colour.

From the python-opencv-messaround prototype (CS6341-Robotics-Group5): median blur, then the Hough circle
transform. Changes for the robot:

- The search runs on the rectified top-down view (board_view.BoardView.rectify), not on the raw frame. There the
  puck radius in pixels is puck.radius_m x px_per_mm, so minRadius / maxRadius come from the config and not from a
  hand tuning per image (the prototype used 60-80 px for one photo and 30-40 px for another).
- The search runs on two channels: the blue channel (black puck and white puck both differ much from the yellow
  wood) and the gray image. Circles found twice are merged.
- Each circle is checked inside a disk of inner_frac x radius: a puck face is one colour. Black: dark compared to
  the board. White: bright and grey (low saturation) compared to the board. Anything else (printed graphics, the
  band, a hand) is rejected. Then min_fill of the disk of fill_frac x radius must have the puck colour: a printed
  graphic (the boxing gloves on the moopok board) is dark in the middle but has light gaps near its edge.
  The thresholds are relative to the median of the board, so they follow the light.
- The rectified center goes to the board frame, then the parallax correction (the camera sees the top face).

In sling puck every puck on your half is yours to shoot, so the colour is information only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.slingpuck.sensing.board_view import BoardView, _cv2

DEFAULTS = {
    "blur_ksize": 5,             # Median blur (prototype: 5)
    "hough_dp": 1.0,
    "hough_param1": 60,          # Upper Canny threshold (prototype: 50)
    "hough_param2": 18,          # Accumulator threshold: lower finds more (and more false) circles (prototype: 30)
    "radius_tol": 0.25,          # Search radius range: (1 -/+ tol) x the expected radius in pixels
    "min_dist_frac": 1.6,        # Minimum distance between centers, x radius (touching pucks: 2 x radius)
    "inner_frac": 0.6,           # The colour check uses the disk of this fraction of the radius
    "black_v_max_frac": 0.55,    # Black: mean value (HSV V) below this fraction of the board's median V
    "white_v_min_frac": 0.95,    # White: mean V above this fraction of the board's median V ...
    "white_s_max_frac": 0.6,     # ... and mean saturation below this fraction of the board's median S
    "max_v_std": 40.0,           # Puck face is one colour: V standard deviation in the disk below this
    "fill_frac": 0.9,            # The fill check uses the disk of this fraction of the radius ...
    "min_fill": 0.95,            # ... and this fraction of it must have the puck colour (a printed graphic: ~0.87)
    "white_fill_v_frac": 0.85,   # A white pixel for the fill check: V above this fraction of the board's median V
    "on_board_margin_m": 0.003,  # A center may be this far outside the play area (calibration error)
}


@dataclass
class Detection:
    xy: np.ndarray        # Board (x, y), m, parallax corrected
    colour: str           # "black" or "white"
    radius_m: float       # Seen radius
    rect_px: np.ndarray   # Center in the rectified view (pixels)
    v_mean: float         # Mean V in the inner disk (0-255)
    s_mean: float         # Mean S in the inner disk (0-255)

    @property
    def side(self) -> str:
        return "robot" if self.xy[1] < 0.0 else "opponent"


class PuckDetector:
    def __init__(self, view: BoardView, puck_radius_m: float, puck_thickness_m: float = 0.0, params: dict | None = None):
        self.view = view
        self.r_m = float(puck_radius_m)
        self.thickness_m = float(puck_thickness_m)
        unknown = set(params or {}) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown detector settings {sorted(unknown)}. Known: {sorted(DEFAULTS)}")
        self.p = {**DEFAULTS, **(params or {})}
        self.r_px = self.r_m * view.rect_px_per_m
        if self.r_px < 6:
            raise ValueError(f"The puck is {self.r_px:.1f} px in the rectified view. Use a larger px_per_mm.")

    @classmethod
    def from_config(cls, cfg: dict, view: BoardView) -> "PuckDetector":
        params = {k: v for k, v in (view.meta.get("detect") or {}).items() if k in DEFAULTS}
        return cls(view, cfg["puck"]["radius_m"], cfg["puck"]["thickness_m"], params)

    # ----------------------------------------------------------- circles
    def _circles(self, chan) -> np.ndarray:
        cv2, p = _cv2(), self.p
        c = cv2.HoughCircles(chan, cv2.HOUGH_GRADIENT, dp=p["hough_dp"], minDist=p["min_dist_frac"] * self.r_px,
                             param1=p["hough_param1"], param2=p["hough_param2"],
                             minRadius=max(1, int(np.floor((1 - p["radius_tol"]) * self.r_px))),
                             maxRadius=int(np.ceil((1 + p["radius_tol"]) * self.r_px)))
        return np.zeros((0, 3)) if c is None else c[0].astype(float)

    def candidates(self, rect_bgr) -> np.ndarray:
        """(n, 3) circles (u, v, r) in the rectified view from both channels, duplicates merged."""
        cv2 = _cv2()
        k = int(self.p["blur_ksize"]) | 1
        blue = cv2.medianBlur(np.ascontiguousarray(rect_bgr[..., 0]), k)
        gray = cv2.medianBlur(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY), k)
        found = np.vstack([self._circles(blue), self._circles(gray)])
        keep = []
        for c in found:                       # Merge: the first of two centers closer than a radius
            if all(np.hypot(*(c[:2] - q[:2])) > 0.8 * self.r_px for q in keep):
                keep.append(c)
        return np.array(keep).reshape(-1, 3)

    # ------------------------------------------------------------ colour
    def board_stats(self, hsv) -> tuple[float, float]:
        """Median V and S of the play area (most of it is bare board)."""
        flat = hsv.reshape(-1, 3)
        return float(np.median(flat[:, 2])), float(np.median(flat[:, 1]))

    @staticmethod
    def _disk(hsv, u0: float, v0: float, rr: float) -> tuple[np.ndarray, np.ndarray]:
        """S and V values (float) of the pixels inside the disk of radius rr at (u0, v0)."""
        h, w = hsv.shape[:2]
        x0, x1 = max(0, int(u0 - rr)), min(w, int(u0 + rr) + 2)       # Crop around the disk
        y0, y1 = max(0, int(v0 - rr)), min(h, int(v0 + rr) + 2)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask = (xs - u0) ** 2 + (ys - v0) ** 2 <= rr ** 2
        crop = hsv[y0:y1, x0:x1]
        return crop[..., 1][mask].astype(float), crop[..., 2][mask].astype(float)

    def classify(self, hsv, c, board_v: float, board_s: float) -> tuple[str | None, float, float]:
        """Colour of the circle c = (u, v, r): "black", "white" or None (not a puck), and the inner mean V and S."""
        p = self.p
        s_vals, v_vals = self._disk(hsv, c[0], c[1], p["inner_frac"] * self.r_px)
        if len(v_vals) < 10:
            return None, 0.0, 0.0
        v_mean, s_mean, v_std = float(v_vals.mean()), float(s_vals.mean()), float(v_vals.std())
        if v_std > p["max_v_std"]:
            return None, v_mean, s_mean
        white_s = p["white_s_max_frac"] * max(board_s, 1.0)
        if v_mean < p["black_v_max_frac"] * board_v:
            colour = "black"
        elif v_mean > p["white_v_min_frac"] * board_v and s_mean < white_s:
            colour = "white"
        else:
            return None, v_mean, s_mean
        # Fill: almost all of the disk (fill_frac x radius) has the puck colour. A printed graphic has gaps.
        s_all, v_all = self._disk(hsv, c[0], c[1], p["fill_frac"] * self.r_px)
        if colour == "black":
            fill = np.mean(v_all < p["black_v_max_frac"] * board_v)
        else:
            fill = np.mean((v_all > p["white_fill_v_frac"] * board_v) & (s_all < white_s))
        return (colour if fill >= p["min_fill"] else None), v_mean, s_mean

    # ------------------------------------------------------------ detect
    def detect_rectified(self, rect_bgr) -> list[Detection]:
        cv2 = _cv2()
        hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
        board_v, board_s = self.board_stats(hsv)
        out = []
        for c in self.candidates(rect_bgr):
            colour, v_mean, s_mean = self.classify(hsv, c, board_v, board_s)
            if colour is None:
                continue
            xy = self.view.correct_parallax(self.view.rect_to_board(c[:2]), self.thickness_m)
            if not self.view.on_board(xy, self.p["on_board_margin_m"])[0]:
                continue
            out.append(Detection(xy=xy, colour=colour, radius_m=float(c[2] / self.view.rect_px_per_m),
                                 rect_px=c[:2].copy(), v_mean=v_mean, s_mean=s_mean))
        return out

    def detect(self, frame_bgr) -> list[Detection]:
        """Pucks in one camera frame (BGR, as OpenCV reads it)."""
        return self.detect_rectified(self.view.rectify(frame_bgr))


def draw(frame_bgr, view: BoardView, dets: list[Detection], rectified: bool = False):
    """A copy of the frame (camera or rectified) with the play area outline, the circles and the board (x, y) in mm."""
    cv2 = _cv2()
    out = frame_bgr.copy()
    to_px = view.board_to_rect if rectified else view.board_to_img
    w, l = 0.5 * view.width_m, 0.5 * view.length_m
    outline = to_px(np.array([[-w, -l], [w, -l], [w, l], [-w, l]]))
    cv2.polylines(out, [np.round(outline).astype(np.int32)], True, (255, 160, 0), 2)
    gate = to_px(np.array([[-w, 0.0], [w, 0.0]]))
    cv2.line(out, *[tuple(int(round(v)) for v in q) for q in gate], (255, 160, 0), 1)
    for d in dets:
        c = to_px(d.xy)
        edge = to_px(d.xy + [d.radius_m, 0.0])
        r = max(2, int(round(np.hypot(*(edge - c)))))
        cu, cv_ = int(round(c[0])), int(round(c[1]))
        col = (0, 200, 0) if d.colour == "black" else (0, 0, 255)
        cv2.circle(out, (cu, cv_), r, col, 2)
        cv2.circle(out, (cu, cv_), 2, (0, 0, 255), 3)
        cv2.putText(out, f"{1000 * d.xy[0]:+.0f},{1000 * d.xy[1]:+.0f}", (cu - r, cv_ - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return out
