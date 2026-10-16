"""Puck detector for the overhead camera: black and white round blobs of puck size on the wood board.

1. Warp the frame to a top-down image of the play area (px_per_m pixels per m),
   with the calibration homography of the board plane.
2. Compare each pixel with the local board background (a wide median blur), so
   uneven light cancels out. Black pucks: darker than the background. White
   pucks: less colour-saturated than the background (wood has some saturation, a
   white puck almost none) and not darker. Clean both masks with a small opening.
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
    colour: str = "black"


class PuckDetector:
    def __init__(self, cfg: dict, calib: BoardCalibration, px_per_m: float = 2000.0, dark_ratio: float = 0.55,
                 sat_delta: int = 12, white_dark_margin: int = 25, min_roundness: float = 0.7):
        """Limits are against the local board background (grey levels 0-255); tune them on real video."""
        self.cfg, self.calib, self.s = cfg, calib, px_per_m
        self.dark_ratio, self.sat_delta, self.white_dark_margin = dark_ratio, sat_delta, white_dark_margin
        self.min_roundness = min_roundness
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

    def masks(self, frame: np.ndarray, ignore_mask: np.ndarray | None = None) -> dict:
        """Top-down candidate masks, each against the local board background (a median blur about 4 puck
        radii wide), so smooth changes of the light cancel out:
          "black": darker than dark_ratio x the background (a puck shadow is about 0.7 x, a black puck
                   under 0.45 x; a shadow in the mask moves the blob centre away from the light)
          "white": greyer than the background by sat_delta (wood has some colour saturation, a white
                   puck almost none) AND not darker than it by white_dark_margin (keeps out black pucks
                   and shadows). Brightness alone fails: in the lit area board and puck both saturate."""
        top = self.top_down(frame)
        if top.ndim == 2:
            top = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        k_bg = int(4 * self.r_px) | 1
        bg_gray, bg_sat, bg_val = (cv2.medianBlur(c, k_bg).astype(np.int16) for c in (gray, sat, val))
        out = {"black": (gray.astype(np.int16) < self.dark_ratio * bg_gray).astype(np.uint8),
               "white": ((val.astype(np.int16) > bg_val - self.white_dark_margin)
                         & (bg_sat - sat.astype(np.int16) > self.sat_delta)).astype(np.uint8)}
        ign = None if ignore_mask is None else \
            cv2.warpPerspective(ignore_mask.astype(np.uint8), self.M, self.size, flags=cv2.INTER_NEAREST)
        k = max(3, int(round(0.15 * self.r_px)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        for c, mask in out.items():
            if ign is not None:
                mask[ign > 0] = 0
            out[c] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self.last_mask = out["black"] | out["white"]
        return out

    def detect(self, frame: np.ndarray, ignore_mask: np.ndarray | None = None) -> list[Detection]:
        """ignore_mask: image-size bool array of pixels to ignore (the arm)."""
        return [d for colour, mask in self.masks(frame, ignore_mask).items() for d in self._blobs(mask, colour)]

    def _blobs(self, mask: np.ndarray, colour: str) -> list[Detection]:
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
                    out.append(Detection(self._to_board(cents[i])[0], roundness, colour))
                continue
            if area > 6.5 * area1:
                continue  # Not pucks (a lit patch of board, a hand)
            # Touching pucks: k-means on the blob pixels, k = blob area / one puck area.
            k = int(round(area / area1))
            vv, uu = np.nonzero(blob)
            pts = np.c_[uu, vv].astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, centers = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
            for uv in centers:
                out.append(Detection(self._to_board(uv)[0], 0.5, colour))
        return out
