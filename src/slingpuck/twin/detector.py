"""Puck detector for the overhead camera: dark round blobs of puck size on the light board.

1. Warp the frame to a top-down image of the play area (px_per_m pixels per m),
   with the calibration homography of the board plane.
2. Threshold dark pixels: Otsu's limit on the play-area image, never above
   dark_frac x its median brightness (so the limit follows the exposure, and a
   board with no pucks gives no blobs). Clean with a small opening.
3. Each blob of about one puck area and round enough is one puck. A blob of
   two or more puck areas (touching pucks) is split by k-means on its pixels,
   k = blob area / one puck area.
4. Blob centers go back to the board frame. The blob is the puck top face (seen
   pushed away from the camera nadir) joined to the visible side wall, so its
   center is taken at half the puck height for the parallax correction.

Dark parts of the arm (servo cases) can also pass the threshold. The twin knows
the arm pose from the joint readings, so it renders an arm mask in the camera
image (TwinScene.arm_mask, from the calibration) and detect() ignores those pixels. A human hand
gets no mask; the size and roundness checks and the tracker handle it.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.slingpuck.twin.calibration import BoardCalibration


@dataclass(frozen=True)
class Detection:
    xy: np.ndarray       # Board frame, m
    score: float         # Roundness in [0, 1] (1 = perfect disk); 0.5 for split blobs


class PuckDetector:
    def __init__(self, cfg: dict, calib: BoardCalibration, px_per_m: float = 2000.0, dark_frac: float = 0.8,
                 min_roundness: float = 0.7):
        self.cfg, self.calib, self.s = cfg, calib, px_per_m
        self.dark_frac, self.min_roundness = dark_frac, min_roundness
        self.hw, self.hl = 0.5 * cfg["board"]["width_m"], 0.5 * cfg["board"]["length_m"]
        self.r_px = cfg["puck"]["radius_m"] * px_per_m
        self.half_height = 0.5 * cfg["puck"]["thickness_m"]
        self.size = (int(round(2 * self.hw * px_per_m)), int(round(2 * self.hl * px_per_m)))  # (w, h)
        # Top-down pixel (u, v) = ((x + hw) * s, (hl - y) * s): opponent end at the top of the image.
        T = np.array([[px_per_m, 0.0, self.hw * px_per_m], [0.0, -px_per_m, self.hl * px_per_m], [0.0, 0.0, 1.0]])
        self.M = T @ calib.H_px_to_board
        self.T_inv = np.linalg.inv(T)
        self.last_mask = None

    def top_down(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, self.size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    def _to_board(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        xy = (self.T_inv @ np.c_[uv, np.ones(len(uv))].T).T[:, :2]
        if self.calib.camera_height_m:
            k = (self.calib.camera_height_m - self.half_height) / self.calib.camera_height_m
            xy = self.calib.camera_nadir_xy + (xy - self.calib.camera_nadir_xy) * k
        return xy

    def detect(self, frame: np.ndarray, ignore_mask: np.ndarray | None = None) -> list[Detection]:
        """ignore_mask: image-size bool array of pixels to ignore (the arm)."""
        top = self.top_down(frame)
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (gray < min(otsu, self.dark_frac * float(np.median(gray)))).astype(np.uint8)
        if ignore_mask is not None:
            ign = cv2.warpPerspective(ignore_mask.astype(np.uint8), self.M, self.size, flags=cv2.INTER_NEAREST)
            mask[ign > 0] = 0
        k = max(3, int(round(0.15 * self.r_px)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        self.last_mask = mask
        area1 = np.pi * self.r_px ** 2
        out = []
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.5 * area1:
                continue
            blob = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perim = cv2.arcLength(contours[0], True)
            roundness = float(min(1.0, 4 * np.pi * area / max(perim, 1e-9) ** 2))
            if area <= 1.7 * area1:
                if roundness >= self.min_roundness:
                    out.append(Detection(self._to_board(cents[i])[0], roundness))
                continue
            # Touching pucks: k-means on the blob pixels, k = blob area / one puck area.
            k = int(round(area / area1))
            vv, uu = np.nonzero(blob)
            pts = np.c_[uu, vv].astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, centers = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
            for uv in centers:
                out.append(Detection(self._to_board(uv)[0], 0.5))
        return out

