"""Record the range of motion again for some joints only, and keep the rest of the LeRobot calibration.

Use it when lerobot-calibrate recorded too small a range for a joint (the joint was not moved to both ends).
The homing offsets do not change. The new range goes into the LeRobot calibration file and into the
motors' position limits. Run in the lerobot env:

    python3 -m src.slingpuck.deploy.fix_ranges --port /dev/tty.usbmodem5A7A0563651 --id group5 --joints shoulder_pan

A changed range moves the LeRobot zero of that joint (LeRobot 0 = the middle of the range), so run
calibrate_joints again afterwards. The old calibration file is kept as <id>.json.bak.
"""
from __future__ import annotations

import argparse
import shutil
import sys

ALL = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper")
WRAP_MARGIN_STEPS = 50       # A range this close to 0 or 4095: the joint crosses the encoder wrap
MIN_RANGE_STEPS = 1000       # About 88 deg; every SO-101 joint moves more than this


def check_range(joint: str, lo: int, hi: int) -> str | None:
    """None if the new range is usable, else the reason."""
    if lo < WRAP_MARGIN_STEPS or hi > 4095 - WRAP_MARGIN_STEPS:
        return (f"{joint}: the range {lo}-{hi} reaches the encoder wrap (0 / 4095). The middle pose of the "
                f"calibration was not near the joint's middle. Run the full lerobot-calibrate again.")
    if hi - lo < MIN_RANGE_STEPS:
        return f"{joint}: only {hi - lo} steps. Move the joint from one end stop to the other."
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="group5")
    p.add_argument("--joints", nargs="+", choices=ALL, required=True)
    a = p.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.id))
    if not robot.calibration_fpath.is_file():
        sys.exit(f"No calibration at {robot.calibration_fpath}. Run lerobot-calibrate first.")
    bus = robot.bus
    bus.connect()
    try:
        print("Present ranges (motor steps):")
        for j in a.joints:
            c = robot.calibration[j]
            print(f"  {j:14s} {c.range_min}-{c.range_max} ({c.range_max - c.range_min} steps)")
        input("\nThe torque turns OFF now. Hold the arm, then press Enter... ")
        bus.disable_torque()
        while True:
            print(f"\nMove {', '.join(a.joints)} slowly from one end stop to the other. Press Enter when done.")
            mins, maxes = bus.record_ranges_of_motion(list(a.joints))
            errors = [e for j in a.joints if (e := check_range(j, int(mins[j]), int(maxes[j])))]
            if not errors:
                break
            print("\n".join(errors))
            if any("wrap" in e for e in errors):
                sys.exit(1)
        backup = robot.calibration_fpath.with_suffix(".json.bak")
        shutil.copy(robot.calibration_fpath, backup)
        for j in a.joints:
            robot.calibration[j].range_min, robot.calibration[j].range_max = int(mins[j]), int(maxes[j])
            print(f"  {j:14s} new range {mins[j]}-{maxes[j]} ({maxes[j] - mins[j]} steps)")
        bus.write_calibration(robot.calibration)
        robot._save_calibration()
        print(f"Saved {robot.calibration_fpath} (old file: {backup}). Run calibrate_joints again.")
    finally:
        input("\nHold the arm (the torque stays off), then press Enter to disconnect... ")
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
