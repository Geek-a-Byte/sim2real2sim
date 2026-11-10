"""Vertical stop test: the arm moves its jaw tips straight down, 2 mm each second, until you press Enter at contact.

Where the jaws touch the board, the script records the sim's view of the jaw tips (forward kinematics of the
measured joints, through the joint map). The height of that point is the sim's view of the board surface; compare
it with the board file (touch_test) and with a ruler. Every second it prints the commanded and the measured tip
height: a difference shows a joint that sags under its load (as the elbow did in the first replays).

Run it in the lerobot env:

    python3 -m src.slingpuck.deploy.vertical_stop --port /dev/tty.usbmodem5A7A0563651

Per point:
  1. The torque is off. Close the jaws by hand. Hold the arm with the jaw tips 2-6 cm above the board, jaws
     pointing down (above a touch mark is best). The live line shows the tip height above the board file's surface.
  2. Press Enter. The torque turns on where the arm is, and the arm holds.
  3. Press Enter to start. The tips go down step by step. Press Enter at the first contact.
     The arm stops by itself: STOP_BELOW_MM under the expected surface, after MAX_TRAVEL_MM, or when a joint stays
     STALL_DEG behind its command for STALL_S.
  4. The arm goes back up to the start point and holds. Hold it before the torque turns off.
Writes results/real/vertical_stop_<time>.yaml.

Correct the board file's surface height with the median contact (no arm needed):

    python3 -m src.slingpuck.deploy.vertical_stop --apply results/real/vertical_stop_<time>.yaml
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.deploy import board_frame as bf
from src.slingpuck.deploy.joint_map import ARM_JOINTS, DEFAULT_PATH as MAP_PATH, GRIPPER, JointMap, file_sha256
from src.slingpuck.deploy.replay_trajectory import enter_pressed, read_loads

STEP_MM = 2.0
PERIOD_S = 1.0
STOP_BELOW_MM = 10.0         # Stop if the commanded tip goes this far below the expected surface
MAX_TRAVEL_MM = 80.0
STALL_DEG, STALL_S = 12.0, 1.0
START_MIN_MM, START_MAX_MM = 10.0, 100.0   # Allowed start heights above the expected surface
IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")  # Wrist roll and gripper stay as they are
DIR_WEIGHT = 0.02            # 1 rad of jaw direction change costs as much as 20 mm of tip error
OUT_DIR = Path("results/real")


class TipIK:
    """Jaw tip position (and jaw direction) as a function of the sim arm joints, with a small IK solver."""

    def __init__(self):
        from src.slingpuck.deploy.touch_test import TipModel
        self.model = TipModel()
        self.idx = [ARM_JOINTS.index(j) for j in IK_JOINTS]

    def tip_and_dir(self, q, q_grip) -> tuple[np.ndarray, np.ndarray]:
        tip = self.model.tip(q, q_grip)
        origin = self.model.d.xpos[self.model.m.body("gripper").id]
        v = tip - origin
        return tip, v / np.linalg.norm(v)

    def solve(self, q0, q_grip, target, direction, iters: int = 20) -> np.ndarray:
        """Joints near q0 that put the tip at target and keep the jaw direction (Gauss-Newton)."""
        q = np.array(q0, float)

        def res(qq):
            t, d = self.tip_and_dir(qq, q_grip)
            return np.concatenate([t - target, DIR_WEIGHT * (d - direction)])

        r = res(q)
        for _ in range(iters):
            jac = np.empty((len(r), len(self.idx)))
            for c, k in enumerate(self.idx):
                e = np.zeros_like(q)
                e[k] = 1e-6
                jac[:, c] = (res(q + e) - r) / 1e-6
            step = np.linalg.lstsq(jac, -r, rcond=None)[0]
            q[self.idx] += step
            r = res(q)
            if np.linalg.norm(step) < 1e-9:
                break
        return q


def descent_targets(start_tip, step_mm: float = STEP_MM, max_travel_mm: float = MAX_TRAVEL_MM):
    """Tip targets straight down from the start: start, start - step, ... (robot frame, m)."""
    n = int(max_travel_mm // step_mm)
    return [np.asarray(start_tip, float) - [0.0, 0.0, k * step_mm / 1000] for k in range(n + 1)]


def stop_reason(cmd_z: float, surface_z: float, travel_mm: float) -> str | None:
    if cmd_z < surface_z - STOP_BELOW_MM / 1000:
        return (f"the commanded tip is {STOP_BELOW_MM:.0f} mm below the expected surface and you did not press "
                f"Enter: the board is lower than expected, or the arm sags")
    if travel_mm > MAX_TRAVEL_MM:
        return f"{MAX_TRAVEL_MM:.0f} mm of travel"
    return None


def robot_to_board_xy(p, offset) -> np.ndarray:
    """Robot (x, y) -> board (x, y), board square to the arm (board_frame convention)."""
    return np.array([-(p[1] - offset[1]), p[0] - offset[0]])




def measure_point(arm, ik: TipIK, jm: JointMap, surface_z: float, offset, step_mm: float, period_s: float) -> dict:
    """One vertical stop. The arm holds at the start before and after."""
    from src.slingpuck.deploy.replay_trajectory import Stalled
    r0 = arm.read()
    q0 = jm.real_to_sim(r0)
    g_sim = jm.gripper_to_sim(r0[GRIPPER]) if jm.gripper_span_deg else -0.1745
    start_tip, direction = ik.tip_and_dir(q0, g_sim)
    input(f"Holding at tip height {1000 * (start_tip[2] - surface_z):+.1f} mm above the expected surface. "
          f"Press Enter to start the descent, then Enter again at the first contact... ")
    print(f"  {'step':>4s} {'cmd z':>8s} {'meas z':>8s} {'sag':>6s}  (mm above the expected surface)   "
          f"load % lift / elbow / wrist (overload protection at 80)")
    q_cmd, behind_since, rows, reason, touched, pressed = q0.copy(), {}, [], None, None, False
    for k, target in enumerate(descent_targets(start_tip, step_mm)):
        travel = k * step_mm
        reason = stop_reason(target[2], surface_z, travel)
        if reason:
            break
        q_cmd = ik.solve(q_cmd, g_sim, target, direction)
        cmd = jm.sim_to_real(q_cmd)
        arm.send({**cmd, f"{GRIPPER}.pos": r0[GRIPPER]})
        t_end = time.perf_counter() + period_s
        while True:                          # At least one Enter check and one stall check per step
            pressed = enter_pressed()
            meas = arm.read()
            now = time.perf_counter()
            for j in ARM_JOINTS:
                if abs(meas[j] - cmd[f"{j}.pos"]) > STALL_DEG:
                    if now - behind_since.setdefault(j, now) > STALL_S:
                        raise Stalled(f"{j} stayed more than {STALL_DEG:.0f} deg behind its command")
                else:
                    behind_since.pop(j, None)
            if pressed or now >= t_end:
                break
            time.sleep(0.05)
        meas = arm.read()
        tip_meas = ik.model.tip(jm.real_to_sim(meas), g_sim)
        loads = read_loads(arm)
        rows.append({"step": k, "cmd_tip_m": target.tolist(), "meas_tip_m": tip_meas.tolist(),
                     "reading": {j: float(meas[j]) for j in (*ARM_JOINTS, GRIPPER)}, "load_percent": loads})
        load_txt = "" if loads is None else " / ".join(f"{v:+5.1f}" for v in loads.values())
        print(f"  {k:4d} {1000 * (target[2] - surface_z):+8.1f} {1000 * (tip_meas[2] - surface_z):+8.1f} "
              f"{1000 * (tip_meas[2] - target[2]):+6.1f}   {load_txt}")
        if pressed:
            touched = rows[-1]
            break
    if touched is None and reason is None:
        reason = "the end of the steps"
    if touched is None:
        print(f"  Stopped: {reason}. No contact recorded.")
    print("  Going back up to the start point.")
    low = ik.model.tip(q_cmd, g_sim)[2]
    for z in np.append(np.arange(low + 0.005, start_tip[2], 0.005), start_tip[2]):
        q_cmd = ik.solve(q_cmd, g_sim, np.array([start_tip[0], start_tip[1], z]), direction)
        arm.send({**jm.sim_to_real(q_cmd), f"{GRIPPER}.pos": r0[GRIPPER]})
        time.sleep(0.3)
    result = {"start_tip_m": start_tip.tolist(), "start_board_xy_m": robot_to_board_xy(start_tip, offset).tolist(),
              "expected_surface_z_m": surface_z, "steps": rows, "stopped": reason}
    if touched is not None:
        z = touched["meas_tip_m"][2]
        result.update(contact_tip_m=touched["meas_tip_m"], contact_cmd_tip_m=touched["cmd_tip_m"],
                      contact_board_xy_m=robot_to_board_xy(touched["meas_tip_m"], offset).tolist(),
                      contact_minus_expected_mm=round(1000 * (z - surface_z), 1))
        print(f"  CONTACT: measured tip {1000 * (z - surface_z):+.1f} mm from the expected surface (commanded "
              f"{1000 * (touched['cmd_tip_m'][2] - surface_z):+.1f} mm); board xy "
              f"({1000 * result['contact_board_xy_m'][0]:+.0f}, {1000 * result['contact_board_xy_m'][1]:+.0f}) mm")
    return result


def live_until_enter(arm, ik: TipIK, jm: JointMap, surface_z: float, offset) -> np.ndarray:
    """Torque off: show the tip live until Enter; return the tip."""
    tip = None
    while not enter_pressed() or tip is None:
        try:
            r = arm.read()
            g = jm.gripper_to_sim(r[GRIPPER]) if jm.gripper_span_deg else -0.1745
            tip = ik.model.tip(jm.real_to_sim(r), g)
            bxy = robot_to_board_xy(tip, offset)
            print(f"\r    live: tip {1000 * (tip[2] - surface_z):+6.1f} mm above the surface, board xy "
                  f"({1000 * bxy[0]:+5.0f}, {1000 * bxy[1]:+5.0f}) mm   ", end="", flush=True)
        except Exception:
            pass
        time.sleep(0.1)
    print("\r\x1b[2K", end="")
    return tip


def surface_correction_mm(results: list) -> tuple[float, list]:
    """Median contact height above the expected surface (mm), and the contacts used."""
    c = [r["contact_minus_expected_mm"] for r in results if "contact_minus_expected_mm" in r]
    if len(c) < 2:
        raise ValueError("Fewer than 2 contacts: no correction.")
    return float(np.median(c)), c


def apply_correction(results_file, board_file=bf.DEFAULT_PATH) -> float:
    doc = yaml.safe_load(Path(results_file).read_text())
    corr, used = surface_correction_mm(doc["points"])
    board = bf.load(board_file)
    old = float(board["board"]["surface_z_m"])
    Path(board_file).with_suffix(".yaml.before_vertical_stop").write_text(Path(board_file).read_text())
    board["board"]["surface_z_m"] = round(old + corr / 1000, 4)
    board["meta"]["surface_from_vertical_stop"] = {"file": str(results_file), "contacts_mm": used,
                                                  "median_mm": round(corr, 2), "touch_test_surface_z_m": old}
    with open(board_file, "w") as f:
        f.write("# Real board relative to the real arm. Written by touch_test.py; surface height corrected by "
                "vertical_stop.py (torque on).\n")
        yaml.safe_dump(board, f, sort_keys=False)
    print(f"Contacts {used} mm -> median {corr:+.1f} mm. Surface z {1000 * old:.1f} -> "
          f"{1000 * board['board']['surface_z_m']:.1f} mm in {board_file} (old file: .before_vertical_stop).")
    print("Record the trajectories again (record_trajectory.py): they use the surface height.")
    return corr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--id", default="group5")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--board", default=str(bf.DEFAULT_PATH))
    p.add_argument("--step-mm", type=float, default=STEP_MM)
    p.add_argument("--period-s", type=float, default=PERIOD_S)
    p.add_argument("--apply", metavar="RESULTS_YAML", help="Correct the board surface height from saved results")
    a = p.parse_args()
    if a.apply:
        apply_correction(a.apply, a.board)
        return
    if not a.port:
        sys.exit("--port is needed (or --apply).")

    from src.slingpuck.deploy.replay_trajectory import RealArm, Stalled
    jm = JointMap.load(a.map, check_calibration=False)
    if file_sha256(jm.meta["lerobot_calibration_file"]) != jm.meta.get("lerobot_calibration_sha256"):
        sys.exit("The joint map belongs to an older LeRobot calibration. Run calibrate_joints first.")
    board = bf.load(a.board)
    surface_z, offset = float(board["board"]["surface_z_m"]), np.array(board["fit"]["offset_m"])
    ik = TipIK()
    arm = RealArm(a.port, a.id)
    results = []
    out = OUT_DIR / f"vertical_stop_{datetime.datetime.now():%Y%m%d_%H%M%S}.yaml"
    try:
        while True:
            input("\nThe torque turns OFF now (hold the arm), then press Enter... ")
            arm.bus.disable_torque(num_retry=2)
            print("Close the jaws by hand. Hold the tips 2-6 cm above the board, jaws pointing down. Press Enter "
                  "to turn the torque on there.")
            tip = live_until_enter(arm, ik, jm, surface_z, offset)
            h = 1000 * (tip[2] - surface_z)
            if not START_MIN_MM <= h <= START_MAX_MM:
                print(f"The tip is {h:+.0f} mm above the expected surface. Start between {START_MIN_MM:.0f} and "
                      f"{START_MAX_MM:.0f} mm.")
                continue
            arm.torque_on_here()
            try:
                results.append(measure_point(arm, ik, jm, surface_z, offset, a.step_mm, a.period_s))
            except Stalled as e:
                arm.hold_here()
                print(f"\nSTOPPED: {e}. Every joint holds where it is.")
                results.append({"stalled": str(e)})
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump({"map": a.map, "board": a.board, "points": results}, sort_keys=False))
            print(f"Saved {out}")
            if input("Press Enter for another point, or q + Enter to finish... ").strip().lower() == "q":
                break
    except KeyboardInterrupt:
        print("\nStopped. The arm holds its last target.")
    except ConnectionError as e:
        print(f"\nThe motors stopped answering: {e}. Turn the motor power off and on.")
        return
    arm.torque_off()


if __name__ == "__main__":
    main()
