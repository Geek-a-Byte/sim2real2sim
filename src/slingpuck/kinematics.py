"""Goalkeeper kinematics shared by the sim and the robot interface.

Both the env and deploy/lerobot_interface.py map policy actions to pan angles
through this module, so the sim and the robot cannot disagree on the mapping.

Pan angle convention (board frame): pan = 0 puts the paddle center on the gate
line x = 0. Positive pan moves the paddle toward +x.
"""
from dataclasses import dataclass

import numpy as np

from src.slingpuck.physics.backend import PaddleState


@dataclass(frozen=True)
class GoalkeeperGeometry:
    base_xy: np.ndarray          # Pan axis position in the board frame
    arm_radius: float            # Pan axis to paddle center
    pan_max: float               # Pan range is [-pan_max, pan_max], rad
    paddle_half_width: float
    paddle_half_thickness: float
    pan_zero_offset: float       # shoulder_pan joint reading at pan = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "GoalkeeperGeometry":
        robot, gk, board = cfg["robot"], cfg["goalkeeper"], cfg["board"]
        base = np.array(robot["base_xy_m"], dtype=float)
        half_t = 0.5 * robot["paddle_thickness_m"]
        paddle_line_y = -(0.5 * board["divider_thickness_m"] + gk["paddle_standoff_m"] + half_t)
        radius = paddle_line_y - base[1]
        if radius <= 0:
            raise ValueError("robot base must be behind the paddle line (base y < paddle y)")
        geom = cls(base, radius, np.deg2rad(gk["pan_range_deg"]), 0.5 * robot["paddle_width_m"], half_t,
                   robot["pan_zero_offset_rad"])
        reach = abs(geom.paddle_center(geom.pan_max)[0]) + geom.paddle_half_width
        if reach > 0.5 * board["width_m"]:
            raise ValueError(f"paddle leaves the board at full pan: reach {reach:.4f} m")
        return geom

    def action_to_pan(self, action) -> float:
        """Policy action in [-1, 1] to board-frame pan angle."""
        return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0)) * self.pan_max

    def pan_to_action(self, pan: float) -> float:
        return float(np.clip(pan / self.pan_max, -1.0, 1.0))

    def pan_to_joint(self, pan: float) -> float:
        """Board-frame pan angle to the SO-101 shoulder_pan joint command."""
        return pan + self.pan_zero_offset

    def joint_to_pan(self, joint: float) -> float:
        return joint - self.pan_zero_offset

    def paddle_center(self, pan: float) -> np.ndarray:
        return self.base_xy + self.arm_radius * np.array([np.sin(pan), np.cos(pan)])

    def paddle_state(self, pan: float, pan_vel: float, restitution: float) -> PaddleState:
        # The paddle face stays perpendicular to the arm, so it turns by -pan.
        tangent = np.array([np.cos(pan), -np.sin(pan)])
        return PaddleState(
            center=self.paddle_center(pan),
            angle=-pan,
            vel=self.arm_radius * pan_vel * tangent,
            omega=-pan_vel,
            half_width=self.paddle_half_width,
            half_thickness=self.paddle_half_thickness,
            restitution=restitution,
        )
