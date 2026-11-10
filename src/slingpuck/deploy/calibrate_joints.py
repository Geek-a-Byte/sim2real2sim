"""Measure the joint map between the MuJoCo SO-101 and the real SO-101 (steps 2-4 of the real-arm plan).

Run it in the lerobot env (Python 3.12), from the repo root, after lerobot-calibrate:

    conda activate lerobot
    python3 -m src.slingpuck.deploy.calibrate_joints --port /dev/tty.usbmodem5A7A0563651 --id group5

(python3, not python: ~/.zshrc aliases python to Homebrew python3.10, which has no LeRobot.)

In a second terminal, show the sim arm (project env, NOT the lerobot env; from the repo root, because
the path is relative; the arm holds qpos = 0 at the start):

    cd ~/Desktop/UTD/CS_6341/sim2real2sim
    python3 -m mujoco.viewer --mjcf=assets/mujoco/robotstudio_so101/scene.xml

The script never turns the motor torque on. It turns it off, so the arm falls when you let go:
keep a hand on the arm until each reading is done. Each move prompt shows the joint angle live.

Steps (--steps, default all, in this order):
  sign     per joint: drag its Control slider to + in the viewer, move the real joint the same way
  zero     put the real arm in the sim reference pose (default qpos = 0, else --ref-rad); 5 zeros
  gripper  readings with the jaws closed and fully open
  live     sim and real angles in rad, live (Ctrl-C to stop): compare with the viewer sliders

Writes configs/real_arm_joint_map.yaml (deploy/joint_map.py). A step that is not run keeps the
values of the existing file. Eye accuracy is a few degrees; the touch test refines the zeros.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import threading
import time
from pathlib import Path

import numpy as np

from src.slingpuck.deploy.joint_map import (ARM_JOINTS, DEFAULT_PATH, GRIPPER, MIN_GRIPPER_RANGE_STEPS,
                                            MIN_SIGN_MOVE_DEG, STS3215_MAX_RES, JointMap, estimate_sign,
                                            file_sha256, steps_to_deg, zero_from_reference)

STEPS = ("sign", "zero", "gripper", "live")
SAMPLES = 20           # Readings averaged per pose
SAMPLE_DT_S = 0.025
MAX_POSE_STD_DEG = 1.0  # More spread than this in one pose: the arm moved during the reading
NEAR_STOP_DEG = 15.0    # Start pose closer than this to a calibrated stop: the joint may not move in +

# What a + move of each sim joint does, from so101.xml: +0.3 rad at qpos = 0 moves the gripper body
# (robot frame, x forward = the way the gripper points at qpos = 0, z up):
#   pan (-0.01, -0.075, 0), lift (+0.025, 0, -0.071), elbow (-0.007, 0, -0.058), wrist_flex (-0.003, 0, -0.018).
SIM_PLUS = {
    "shoulder_pan": "the arm turns clockwise, seen from above",
    "shoulder_lift": "the upper arm tilts forward and down",
    "elbow_flex": "the forearm and gripper go down",
    "wrist_flex": "the gripper tips down",
    "wrist_roll": "the gripper turns about its own axis (look at the viewer for the direction)",
}


class Arm:
    """Read-only access to the real SO-101 through LeRobot. Torque off."""

    def __init__(self, port: str, robot_id: str):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            sys.exit(f"LeRobot is not importable with {sys.executable} ({e}).\n"
                     "Use the lerobot env's Python: ~/miniconda3/envs/lerobot/bin/python -m ...")
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
        if not self.robot.calibration_fpath.is_file():
            sys.exit(f"No LeRobot calibration at {self.robot.calibration_fpath}. Run lerobot-calibrate first.")
        self.calibration_file = self.robot.calibration_fpath
        self.robot.bus.connect()
        if not self.robot.bus.is_calibrated:
            self.robot.bus.disconnect(disable_torque=False)
            sys.exit("The motors do not match the LeRobot calibration file. Run lerobot-calibrate again.")
        steps = self.range_steps(GRIPPER)
        if steps < MIN_GRIPPER_RANGE_STEPS:
            self.robot.bus.disconnect(disable_torque=False)
            sys.exit(f"The LeRobot calibration has a gripper range of only {steps} motor steps "
                     f"({steps_to_deg(steps):.1f} deg): the gripper did not move during lerobot-calibrate,\n"
                     f"so every reading is 0 or 100. Run lerobot-calibrate again, and when it records the ranges,\n"
                     f"move EVERY joint, the gripper too (fully closed to fully open), through its full range.\n"
                     f"Then run this script again: a new calibration moves all the zeros.")

    def range_steps(self, joint: str) -> int:
        c = self.robot.calibration[joint]
        return int(c.range_max - c.range_min)

    def torque_off(self):
        input("\nThe motor torque turns OFF now, and the arm falls if nobody holds it.\n"
              "Hold the arm (or put it in its rest pose), then press Enter... ")
        self.robot.bus.disable_torque(num_retry=2)
        print("Torque is off. You can move the arm by hand.")

    def limits_deg(self, joint: str) -> tuple[float, float]:
        """The calibrated stops of an arm joint, in LeRobot degrees."""
        c = self.robot.calibration[joint]
        mid = (c.range_min + c.range_max) / 2
        return ((c.range_min - mid) * 360 / STS3215_MAX_RES, (c.range_max - mid) * 360 / STS3215_MAX_RES)

    def read(self) -> dict:
        """One reading: {'shoulder_pan': deg, ..., 'gripper': 0-100}."""
        return self.robot.bus.sync_read("Present_Position", num_retry=2)

    def read_mean(self, n: int = SAMPLES) -> tuple[dict, dict]:
        rows = []
        for _ in range(n):
            rows.append(self.read())
            time.sleep(SAMPLE_DT_S)
        keys = rows[0].keys()
        return ({k: float(np.mean([r[k] for r in rows])) for k in keys},
                {k: float(np.std([r[k] for r in rows])) for k in keys})

    def close(self):
        self.robot.bus.disconnect(disable_torque=True)


def rad(deg: float) -> str:
    return f"{np.radians(deg):+.3f} rad"


def fmt(joint: str, value: float) -> str:
    """A LeRobot reading for display: arm joints in rad, the gripper in its 0-100 units."""
    return f"{value:5.1f} /100" if joint == GRIPPER else rad(value)


def wait_enter(arm: Arm, joint: str | None = None, ref: float | None = None):
    """Wait for Enter. With a joint: show its angle (and the move from ref) live until then."""
    if joint is None:
        input()
        return
    stop = threading.Event()

    def show():
        while not stop.is_set():
            try:
                v = arm.read()[joint]
                moved = "" if ref is None else f"   moved {fmt(joint, v - ref)}"
                print(f"\r    live: {joint} {fmt(joint, v)}{moved}   ", end="", flush=True)
            except Exception:  # A failed serial read: skip one display update
                pass
            time.sleep(0.1)

    t = threading.Thread(target=show, daemon=True)
    t.start()
    try:
        input()
    finally:
        stop.set()
        t.join()
        print("\r\x1b[2K", end="", flush=True)    # Clear a live line printed after Enter


def steady_pose(arm: Arm, prompt: str, joint: str | None = None, ref: float | None = None) -> dict:
    """Ask for a pose, then average readings; ask again if the arm moved during the reading."""
    while True:
        print(prompt)
        print("    Keep your hand on the arm, hold it still, and press Enter with the other hand.")
        wait_enter(arm, joint, ref)
        mean, std = arm.read_mean()
        bad = [k for k in ARM_JOINTS if std[k] > MAX_POSE_STD_DEG]
        if not bad:
            return mean
        print(f"  The arm moved during the reading ({', '.join(bad)}). Try again.")


def step_sign(arm: Arm, jm: JointMap):
    print("\n=== Sign: one joint at a time (all angles in rad) ===\n"
          "For each joint, move the real joint the same way as the sim joint moves when its slider goes to +.\n"
          "In the viewer, open the Control panel (right side), drag the joint's slider to + and look.\n"
          "Set the slider back to 0 after each joint. The torque is off: a joint falls when you let go.")
    for j in ARM_JOINTS:
        lo, hi = arm.limits_deg(j)
        while True:
            print(f"\n[{j}] Sim +: {SIM_PLUS[j]}.")
            before = steady_pose(arm, f"  1. Start pose: put the real {j} where it can move 0.35-0.8 rad in the "
                                      f"sim + direction.", j)[j]
            if min(before - lo, hi - before) < NEAR_STOP_DEG:
                print(f"    Note: {j} is at {rad(before)}, near a stop ({rad(lo)} / {rad(hi)}). "
                      f"If it cannot move in +, start further from the stop.")
            after = steady_pose(arm, f"  2. Now move the real {j} 0.35-0.8 rad in the sim + direction. Keep the "
                                     f"other joints still. The live reading shows the move.", j, before)[j]
            try:
                jm.sign[j] = estimate_sign(before, after)
            except ValueError as e:
                print(f"    {e}\n    Possible causes: Enter came before the move, the joint fell back when you "
                      f"let go, or the joint is at a stop. Try again.")
                continue
            print(f"    {j}: reading {rad(before)} -> {rad(after)}, sign {jm.sign[j]:+d}")
            break


def step_zero(arm: Arm, jm: JointMap, ref_rad):
    ref = dict(zip(ARM_JOINTS, ref_rad))
    pose = ", ".join(f"{j} {ref[j]:+.2f}" for j in ARM_JOINTS)
    print(f"\n=== Zero: the sim reference pose (rad: {pose}) ===\n"
          "Set the viewer sliders to these values. Look at the sim arm from the side and from above.")
    mean = steady_pose(arm, "Put the real arm in the same pose as the sim arm. Support it (hand, box or books)\n"
                            "so that it does not fall while the script reads it.")
    print("  joint          real reading    sim ref     -> real reading at sim 0")
    for j in ARM_JOINTS:
        jm.zero_deg[j] = zero_from_reference(mean[j], jm.sign[j], np.degrees(ref[j]))
        print(f"  {j:13s} {rad(mean[j]):>12s} {ref[j]:+8.3f} rad -> {rad(jm.zero_deg[j])}")


def step_gripper(arm: Arm, jm: JointMap):
    print("\n=== Gripper ===")
    jm.gripper_closed = round(steady_pose(arm, "Close the jaws fully (no puck).", GRIPPER)[GRIPPER], 2)
    jm.gripper_open = round(steady_pose(arm, "Open the jaws fully.", GRIPPER)[GRIPPER], 2)
    jm.gripper_span_deg = steps_to_deg(arm.range_steps(GRIPPER) * abs(jm.gripper_open - jm.gripper_closed) / 100)
    print(f"  closed {jm.gripper_closed:.1f} /100, open {jm.gripper_open:.1f} /100, "
          f"{rad(jm.gripper_span_deg)} apart (sim: {jm.gripper_to_sim(jm.gripper_closed):+.3f} to "
          f"{jm.gripper_to_sim(jm.gripper_open):+.3f} rad)")


def live_table(jm: JointMap, r: dict) -> list[str]:
    """Three lines: names, the sim angle (rad, compare with the sliders), the real angle (rad, LeRobot)."""
    q = jm.real_to_sim(r)
    g_sim = f"{jm.gripper_to_sim(r[GRIPPER]):+13.3f}" if jm.gripper_span_deg else f"{'?':>13s}"
    names = "".join(f"{j:>14s}" for j in (*ARM_JOINTS, GRIPPER))
    sim = "".join(f"{v:+14.3f}" for v in q) + " " + g_sim
    real = "".join(f"{np.radians(r[j]):+14.3f}" for j in ARM_JOINTS) + f"{r[GRIPPER]:9.1f} /100"
    return [f"{'':10s}{names}", f"{'sim rad':10s}{sim}", f"{'real rad':10s}{real}"]


def step_live(arm: Arm, jm: JointMap):
    print("\n=== Live check (Ctrl-C to stop) ===\n"
          "Set a pose with the sliders, put the real arm in the same pose, and compare the 'sim rad' row\n"
          "with the slider values. 'real rad' is the LeRobot reading in rad (sim rad = sign x (real - zero)).\n")
    first = True
    try:
        while True:
            lines = live_table(jm, arm.read())
            if not first:
                print(f"\x1b[{len(lines)}F", end="")     # Back to the first line of the table
            print("\n".join(f"{line}\x1b[K" for line in lines), flush=True)
            first = False
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodem5A7A0563651 (lerobot-find-port)")
    p.add_argument("--id", default="group5", help="The --robot.id used in lerobot-calibrate")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    p.add_argument("--ref-rad", nargs=5, type=float, default=[0.0] * 5, metavar="RAD",
                   help="Sim reference pose for the zero step (slider values), in ARM_JOINTS order (default: 0)")
    p.add_argument("--out", default=str(DEFAULT_PATH))
    a = p.parse_args()

    out_exists = Path(a.out).is_file()
    jm = JointMap.load(a.out, check_calibration=False) if out_exists else JointMap()
    if not out_exists and ("zero" in a.steps or "live" in a.steps) and "sign" not in a.steps:
        sys.exit(f"{a.out} does not exist, so the signs are unknown. Run the sign step first.")

    arm = Arm(a.port, a.id)
    try:
        arm.torque_off()
        measured = [s for s in ("sign", "zero", "gripper") if s in a.steps]
        if "sign" in measured:
            step_sign(arm, jm)
        if "zero" in measured:
            step_zero(arm, jm, a.ref_rad)
        if "gripper" in measured:
            step_gripper(arm, jm)
        if measured:
            jm.meta.update({"robot_id": a.id, "port": a.port,
                            "date": datetime.datetime.now().isoformat(timespec="seconds"),
                            "steps": sorted(set(jm.meta.get("steps", [])) | set(measured)),
                            "ref_rad": [float(x) for x in a.ref_rad] if "zero" in measured
                            else jm.meta.get("ref_rad"),
                            "lerobot_calibration_file": str(arm.calibration_file),
                            "lerobot_calibration_sha256": file_sha256(arm.calibration_file)})
            jm.meta.pop("ref_deg", None)
            jm.save(a.out)
            print(f"\nSaved {a.out}")
        if "live" in a.steps:
            step_live(arm, jm)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
