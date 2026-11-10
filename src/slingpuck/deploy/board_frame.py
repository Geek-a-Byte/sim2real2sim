"""Where the real board lies relative to the real arm (step 3), and the touch marks to measure it.

Frames (as so101_slingpuck.py):
  robot  origin at the SO-101 base, x forward, y the robot's left, z up
  board  origin at the board center on the play surface, y toward the opponent (agent half y < 0), x = the
         robot's right. The sim puts the board square to the arm: robot (x, y) = (y_b, -x_b) + offset.

The touch test gives pairs (mark in the board frame, jaw tip in the robot frame). fit_board() finds the
yaw, the offset and the surface height. The sim needs only robot.base_xy_m (the pan axis in the board
frame) and the surface height; a yaw of more than about 1 deg means: turn the board square to the arm.

Only numpy and yaml here, so both envs import it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = Path("configs/real_arm_board.yaml")
BASE_AXIS_X = 0.0388353 # The pan axis is this far in front of the base origin (so101.xml; a test checks it)
MAX_YAW_DEG = 1.5         # More than this: the sim (square board) is not a good model of the real setup

# Default marks: (name, x from the gate center (m, + = the robot's right), distance from the divider's face
# on the robot's side toward the robot (m)). Draw them on the play surface with a ruler.
DEFAULT_MARKS = (
    ("near_left", -0.06, 0.02),
    ("near_right", 0.06, 0.02),
    ("mid_center", 0.0, 0.06),
    ("far_left", -0.06, 0.08),      # 3 cm in front of the band line (11.15 cm from the divider face)
    ("far_right", 0.06, 0.08),
    ("far_center", 0.0, 0.13),      # Behind the band line, 3.15 cm from the end wall
)


def mark_xy(name: str, divider_thickness: float) -> np.ndarray:
    """Board (x, y) of a default mark."""
    for n, x, d in DEFAULT_MARKS:
        if n == name:
            return mark_to_board(x, d, divider_thickness)
    raise KeyError(f"No mark '{name}'. Marks: {[n for n, _, _ in DEFAULT_MARKS]}")


def mark_to_board(x_from_center: float, d_from_divider: float, divider_thickness: float) -> np.ndarray:
    """A ruler mark -> board frame (x, y) on the play surface."""
    return np.array([x_from_center, -(0.5 * divider_thickness + d_from_divider)])


def board_xy_to_robot_xy(xy_board, yaw: float = 0.0, offset=(0.0, 0.0)) -> np.ndarray:
    """Board (x, y) -> robot (x, y): square board turned by yaw about z, then shifted by offset."""
    xb, yb = np.asarray(xy_board, float).T
    sq = np.stack([yb, -xb], axis=-1)                     # Square board (sim convention)
    c, s = math.cos(yaw), math.sin(yaw)
    return sq @ np.array([[c, s], [-s, c]]) + np.asarray(offset, float)


@dataclass
class BoardFit:
    yaw: float                      # rad, board turn relative to the square sim board (+ = counterclockwise from above)
    offset: np.ndarray              # Robot (x, y) of the board origin
    surface_z: float                # Robot z of the play surface
    residuals_m: np.ndarray         # Per mark, distance between the fitted and the touched point (x, y)
    z_residuals_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def rms_mm(self) -> float:
        return float(1000 * np.sqrt(np.mean(self.residuals_m ** 2)))

    def pan_axis_in_board(self) -> np.ndarray:
        """robot.base_xy_m: the pan axis (robot (BASE_AXIS_X, 0)) in the board frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        sq = (np.array([BASE_AXIS_X, 0.0]) - self.offset) @ np.array([[c, s], [-s, c]]).T   # Undo the turn
        return np.array([-sq[1], sq[0]])                  # Square board: (x_b, y_b) = (-y, x)


def rotate_about_pan(robot_xyz, angle: float) -> np.ndarray:
    """Turn points about the pan axis (vertical, at robot (BASE_AXIS_X, 0)) by angle, counterclockwise from above.

    A sim pan change of +d turns the jaw tip by -d about this axis (radius and height unchanged), so a fitted
    board yaw is the same as a pan zero error: pan_change_for_zero_yaw = yaw.
    """
    p = np.array(robot_xyz, float)
    c, s = math.cos(angle), math.sin(angle)
    rel = p[..., :2] - [BASE_AXIS_X, 0.0]
    p[..., :2] = rel @ np.array([[c, s], [-s, c]]) + [BASE_AXIS_X, 0.0]
    return p


def fit_board(board_xy, robot_xyz) -> BoardFit:
    """Least-squares yaw and offset (2D Kabsch) and the mean surface height from >= 3 touched marks."""
    board_xy = np.asarray(board_xy, float)
    robot_xyz = np.asarray(robot_xyz, float)
    if len(board_xy) < 3:
        raise ValueError("Touch at least 3 marks.")
    a = board_xy_to_robot_xy(board_xy)                    # Square-board positions, before turn and shift
    b = robot_xyz[:, :2]
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, s], [-s, c]])
    offset = cb - ca @ rot
    res = np.linalg.norm(a @ rot + offset - b, axis=1)
    z = robot_xyz[:, 2]
    return BoardFit(yaw=yaw, offset=offset, surface_z=float(z.mean()), residuals_m=res, z_residuals_m=z - z.mean())


def save(fit: BoardFit, names, board_xy, robot_xyz, meta: dict, path=DEFAULT_PATH, readings=None):
    axis = fit.pan_axis_in_board()
    doc = {
        "meta": meta,
        "robot": {"base_xy_m": [round(float(v), 4) for v in axis]},
        "board": {"surface_z_m": round(fit.surface_z, 4)},
        "fit": {"yaw_deg": round(math.degrees(fit.yaw), 2), "offset_m": [round(float(v), 4) for v in fit.offset],
                "rms_mm": round(fit.rms_mm, 2)},
        "marks": {n: {"board_xy_m": [round(float(v), 4) for v in bxy], "tip_robot_m": [round(float(v), 4) for v in rx],
                      "error_mm": round(1000 * float(e), 1), "z_error_mm": round(1000 * float(ez), 1)}
                  for n, bxy, rx, e, ez in zip(names, board_xy, robot_xyz, fit.residuals_m, fit.z_residuals_m)},
    }
    if readings is not None:      # Raw LeRobot readings per mark (deg; gripper 0-100): touch_test --refit uses them
        for n, r in zip(names, readings):
            doc["marks"][n]["reading"] = {k: round(float(v), 3) for k, v in r.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Real board relative to the real arm. Written by: python -m src.slingpuck.deploy.touch_test\n"
                "# robot.base_xy_m replaces the placeholder in physics_params; record_trajectory.py reads this file.\n")
        yaml.safe_dump(doc, f, sort_keys=False)


def load(path=DEFAULT_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
