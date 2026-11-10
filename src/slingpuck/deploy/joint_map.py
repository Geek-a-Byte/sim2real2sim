"""Joint map between the MuJoCo SO-101 and the real SO-101 (LeRobot, use_degrees=True).

The two angle systems differ per joint in unit, zero and direction:

    sim   radians, zero = the pose at qpos = 0 in so101.xml, + = the joint axis in so101.xml
    real  degrees, zero = the middle of the range swept in lerobot-calibrate, + = the motor direction

The map has one zero (degrees, a real reading) and one sign (+1 or -1) per arm joint:

    sim -> real (commands):  q_real_deg = zero_deg + sign * degrees(q_sim_rad)
    real -> sim (readings):  q_sim_rad  = sign * radians(q_real_deg - zero_deg)

The gripper (LeRobot RANGE_0_100) keeps two readings, closed and open, and the angle between them
(span_deg, from the motor steps in the LeRobot calibration). The sim jaws are closed at the lower
limit of the gripper joint (so101.xml, -10 deg) and open as the angle goes up:

    q_sim_grip_rad = SIM_GRIPPER_CLOSED_RAD + radians(span_deg) * (r - closed) / (open - closed)

The LeRobot zero moves when the arm is calibrated again, so the file keeps the SHA-256 of the
LeRobot calibration file. load() warns when it changed: then run deploy/calibrate_joints.py again.

Only numpy and yaml here: the project env (Python 3.14, MuJoCo) and the lerobot env (Python 3.12)
both import this file.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Same order as physics/mujoco_match.ARM_JOINTS (a test checks it). LeRobot uses the same names.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
DEFAULT_PATH = Path("configs/real_arm_joint_map.yaml")
MIN_SIGN_MOVE_DEG = 10.0     # A sign test move smaller than this is too close to hand shake and backlash
SIM_GRIPPER_CLOSED_RAD = -0.174533   # so101.xml gripper range: the jaw tips touch at the lower limit
SIM_GRIPPER_MAX_RAD = 1.745329       # so101.xml gripper range, upper limit
STS3215_MAX_RES = 4095
MIN_GRIPPER_RANGE_STEPS = 200        # A calibrated gripper range below this: the gripper did not move in lerobot-calibrate


def steps_to_deg(steps: float) -> float:
    return float(steps) * 360.0 / STS3215_MAX_RES


def file_sha256(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def estimate_sign(before_deg: float, after_deg: float, min_move_deg: float = MIN_SIGN_MOVE_DEG) -> int:
    """+1 if the real reading went up when the joint moved in the sim + direction, else -1."""
    move = after_deg - before_deg
    if abs(move) < min_move_deg:
        raise ValueError(f"The joint moved {np.radians(move):+.3f} rad. "
                         f"Move it at least {np.radians(min_move_deg):.2f} rad ({min_move_deg:.0f} deg).")
    return 1 if move > 0 else -1


def zero_from_reference(real_deg: float, sign: int, ref_sim_deg: float = 0.0) -> float:
    """The real reading at sim angle 0, from a real reading taken in a known sim pose (ref_sim_deg)."""
    return real_deg - sign * ref_sim_deg


@dataclass
class JointMap:
    zero_deg: dict = field(default_factory=lambda: {j: 0.0 for j in ARM_JOINTS})
    sign: dict = field(default_factory=lambda: {j: 1 for j in ARM_JOINTS})
    gripper_closed: float | None = None     # LeRobot 0-100 reading, jaws closed on nothing
    gripper_open: float | None = None       # LeRobot 0-100 reading, jaws fully open
    gripper_span_deg: float | None = None   # Motor angle from closed to open
    meta: dict = field(default_factory=dict)

    def real_to_sim(self, real_deg) -> np.ndarray:
        """LeRobot readings (dict 'joint' or 'joint.pos' -> deg, or a sequence in ARM_JOINTS order) -> sim rad."""
        r = self._as_array(real_deg)
        z = np.array([self.zero_deg[j] for j in ARM_JOINTS])
        s = np.array([self.sign[j] for j in ARM_JOINTS])
        return s * np.radians(r - z)

    def sim_to_real(self, q_sim_rad) -> dict:
        """Sim joint angles (rad, ARM_JOINTS order) -> LeRobot action dict {'joint.pos': deg}."""
        q = np.asarray(q_sim_rad, dtype=float)
        return {f"{j}.pos": float(self.zero_deg[j] + self.sign[j] * np.degrees(q[i]))
                for i, j in enumerate(ARM_JOINTS)}

    def _gripper_ready(self):
        if None in (self.gripper_closed, self.gripper_open, self.gripper_span_deg) \
                or abs(self.gripper_open - self.gripper_closed) < 1e-6:
            raise ValueError("No gripper map. Run calibrate_joints --steps gripper.")

    def gripper_to_sim(self, r: float) -> float:
        """LeRobot gripper reading (0-100) -> sim gripper angle (rad), clipped to the sim joint range."""
        self._gripper_ready()
        frac = (r - self.gripper_closed) / (self.gripper_open - self.gripper_closed)
        q = SIM_GRIPPER_CLOSED_RAD + np.radians(self.gripper_span_deg) * frac
        return float(np.clip(q, SIM_GRIPPER_CLOSED_RAD, SIM_GRIPPER_MAX_RAD))

    def gripper_to_real(self, q_rad: float) -> float:
        """Sim gripper angle (rad) -> LeRobot gripper command (0-100)."""
        self._gripper_ready()
        frac = (q_rad - SIM_GRIPPER_CLOSED_RAD) / np.radians(self.gripper_span_deg)
        return float(np.clip(self.gripper_closed + frac * (self.gripper_open - self.gripper_closed), 0.0, 100.0))

    @staticmethod
    def _as_array(real_deg) -> np.ndarray:
        if isinstance(real_deg, dict):
            return np.array([real_deg[f"{j}.pos"] if f"{j}.pos" in real_deg else real_deg[j] for j in ARM_JOINTS],
                            dtype=float)
        return np.asarray(real_deg, dtype=float)

    def save(self, path=DEFAULT_PATH):
        doc = {"meta": self.meta,
               "arm": {j: {"zero_deg": round(float(self.zero_deg[j]), 3), "sign": int(self.sign[j])}
                       for j in ARM_JOINTS},
               "gripper": {"closed": self.gripper_closed, "open": self.gripper_open,
                           "span_deg": None if self.gripper_span_deg is None else round(float(self.gripper_span_deg), 2)}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Real SO-101 joint map. Written by: python -m src.slingpuck.deploy.calibrate_joints\n"
                    "# sim -> real: q_real_deg = zero_deg + sign * degrees(q_sim_rad)\n"
                    "# Run the script again after each lerobot-calibrate (the LeRobot zero moves).\n")
            yaml.safe_dump(doc, f, sort_keys=False)

    @classmethod
    def load(cls, path=DEFAULT_PATH, check_calibration: bool = True) -> "JointMap":
        with open(path) as f:
            doc = yaml.safe_load(f)
        arm, grip, meta = doc["arm"], doc.get("gripper", {}), doc.get("meta", {})
        jm = cls(zero_deg={j: float(arm[j]["zero_deg"]) for j in ARM_JOINTS},
                 sign={j: int(arm[j]["sign"]) for j in ARM_JOINTS},
                 gripper_closed=grip.get("closed"), gripper_open=grip.get("open"),
                 gripper_span_deg=grip.get("span_deg"), meta=meta)
        if check_calibration and meta.get("lerobot_calibration_file"):
            now = file_sha256(meta["lerobot_calibration_file"])
            if now != meta.get("lerobot_calibration_sha256"):
                warnings.warn(f"The LeRobot calibration file {meta['lerobot_calibration_file']} changed after the "
                              f"joint map was made. Run calibrate_joints again.")
        return jm
