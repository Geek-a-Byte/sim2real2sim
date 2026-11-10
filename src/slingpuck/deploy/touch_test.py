"""Step 3: find where the real board lies relative to the real arm, by touching marks with the jaw tips.

Before: run calibrate_joints (the joint map must belong to the current LeRobot calibration).
Draw 6 small marks (pen dots or tape crosses) on the play surface of the robot's half, measured with a ruler:

    mark          from the gate center        from the divider's face (toward the robot)
    near_left     6 cm to the robot's left    2 cm
    near_right    6 cm to the robot's right   2 cm
    mid_center    on the center line          6 cm
    far_left      6 cm to the robot's left    8 cm
    far_right     6 cm to the robot's right   8 cm
    far_center    on the center line          13 cm

(The robot's left = your left when you stand behind the robot and look along the board.)
Then, in the lerobot env:

    python3 -m src.slingpuck.deploy.touch_test --port /dev/tty.usbmodem5A7A0563651 --id group5

For each mark: close the jaws, put both jaw tips on the mark (jaws pointing down), hold, press Enter.
The script calculates the jaw tip position with the sim model (forward kinematics through the joint map),
then fits the board's position AND the zeros of pan, lift and elbow to the touches (a board yaw is
the same as a pan zero error, so the pan zero takes it and the sim board stays square). It shows the error per
mark before and after the fit, and a leave-one-out error (each mark predicted by a fit without it): that is the
honest accuracy. Aim for 3 mm or less. With your OK it writes the new zeros to the joint map (old map: .bak)
and the board to configs/real_arm_board.yaml.

Fit again from the saved readings, with no arm:  python3 -m src.slingpuck.deploy.touch_test --refit
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from src.slingpuck.deploy import board_frame as bf
from src.slingpuck.deploy.calibrate_joints import Arm
from src.slingpuck.deploy.joint_map import ARM_JOINTS, DEFAULT_PATH as MAP_PATH, GRIPPER, JointMap, file_sha256

SO101_XML = Path(__file__).resolve().parents[3] / "assets/mujoco/robotstudio_so101/so101.xml"
GOOD_RMS_MM = 3.0
# Fitted zeros. Wrist flex is not fitted by default: with the jaws down at every mark it trades off with lift and
# elbow, and on the first board the fit moved it 7.6 deg, to 2.8 deg from its stop (--fit-joints adds it).
FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
ALL_FIT_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")   # Wrist roll hardly moves the tip
PRIOR_DEG, PRIOR_M = 3.0, 0.001     # A zero change of 3 deg costs as much as 1 mm of touch error
MAX_ZERO_CHANGE_DEG = 6.0


class TipModel:
    """Forward kinematics of the jaw tips (midpoint of the two tip spheres, lowest point) in the robot frame."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(SO101_XML))
        self.d = mujoco.MjData(self.m)
        self.arm_q = [int(self.m.joint(j).qposadr[0]) for j in ARM_JOINTS]
        self.grip_q = int(self.m.joint(GRIPPER).qposadr[0])
        self.tips = [self.m.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
        self.r = float(self.m.geom_size[self.tips[0], 0])

    def tip(self, q_arm, q_grip: float) -> np.ndarray:
        self.d.qpos[self.arm_q] = q_arm
        self.d.qpos[self.grip_q] = q_grip
        self.mj.mj_kinematics(self.m, self.d)
        p = self.d.geom_xpos[self.tips].mean(axis=0)
        return p - [0.0, 0.0, self.r]


def tip_from_reading(model: TipModel, jm: JointMap, r: dict) -> np.ndarray:
    q_grip = jm.gripper_to_sim(r[GRIPPER]) if jm.gripper_span_deg else model.m.jnt_range[model.m.joint(GRIPPER).id][0]
    return model.tip(jm.real_to_sim(r), q_grip)


def touch(arm: Arm, model: TipModel, jm: JointMap, name: str, x: float, d: float):
    """One mark: (tip in the robot frame, mean reading) or None if skipped."""
    side = "on the center line" if abs(x) < 1e-9 else f"{100 * abs(x):.0f} cm to the robot's {'left' if x < 0 else 'right'}"
    print(f"\n[{name}] {side}, {100 * d:.0f} cm from the divider's face.")
    print("    Close the jaws, put both tips on the mark, hold still, press Enter (s + Enter: skip this mark).")
    stop = threading.Event()

    def show():
        while not stop.is_set():
            try:
                t = tip_from_reading(model, jm, arm.read())
                print(f"\r    live tip (robot frame): x {1000 * t[0]:+7.1f}  y {1000 * t[1]:+7.1f}  z {1000 * t[2]:+7.1f} mm   ",
                      end="", flush=True)
            except Exception:
                pass
            time.sleep(0.1)

    th = threading.Thread(target=show, daemon=True)
    th.start()
    try:
        answer = input()
    finally:
        stop.set()
        th.join()
        print("\r\x1b[2K", end="", flush=True)
    if answer.strip().lower() == "s":
        return None
    mean, std = arm.read_mean()
    if max(std[j] for j in ARM_JOINTS) > 1.0:
        print("    The arm moved during the reading. Touch this mark again.")
        return touch(arm, model, jm, name, x, d)
    t = tip_from_reading(model, jm, mean)
    print(f"    tip at x {1000 * t[0]:+.1f}, y {1000 * t[1]:+.1f}, z {1000 * t[2]:+.1f} mm (robot frame)")
    return t, mean


def _tips(model: TipModel, jm: JointMap, readings, dq, joints=FIT_JOINTS) -> np.ndarray:
    """Tip positions with sim joint changes dq (rad, in `joints` order) added to the mapped readings."""
    idx = [ARM_JOINTS.index(j) for j in joints]
    out = []
    for r in readings:
        q = jm.real_to_sim(r)
        q[idx] += dq
        g = jm.gripper_to_sim(r[GRIPPER]) if jm.gripper_span_deg else model.m.jnt_range[model.m.joint(GRIPPER).id][0]
        out.append(model.tip(q, g))
    return np.array(out)


def fit_zeros(model: TipModel, jm: JointMap, readings, board_xy, iters: int = 60, joints=FIT_JOINTS):
    """Levenberg-Marquardt over x = (dq for `joints`, board offset x y, surface z), the board square (yaw 0).
    `joints` must start with shoulder_pan (it takes the board yaw).

    Returns (dq rad, offset, surface_z, xy errors m, z errors m)."""
    board_xy = np.asarray(board_xy, float)
    assert joints[0] == "shoulder_pan"
    nj = len(joints)
    prior = math.radians(PRIOR_DEG)

    def residual(x):
        t = _tips(model, jm, readings, x[:nj], joints)
        pred = bf.board_xy_to_robot_xy(board_xy, 0.0, x[nj:nj + 2])
        return np.concatenate([(t[:, :2] - pred).ravel(), t[:, 2] - x[nj + 2], x[:nj] / prior * PRIOR_M])

    fit0 = bf.fit_board(board_xy, _tips(model, jm, readings, np.zeros(nj), joints))
    x = np.zeros(nj + 3)
    x[0] = fit0.yaw                                      # Pan change that removes the yaw (see rotate_about_pan)
    t1 = bf.rotate_about_pan(_tips(model, jm, readings, np.zeros(nj), joints), -fit0.yaw)
    f1 = bf.fit_board(board_xy, t1)
    x[nj:nj + 2], x[nj + 2] = f1.offset, f1.surface_z
    r = residual(x)
    lam = 1e-3
    for _ in range(iters):
        jac = np.empty((len(r), len(x)))
        for k in range(len(x)):
            e = np.zeros_like(x)
            e[k] = 1e-7
            jac[:, k] = (residual(x + e) - r) / 1e-7
        a = jac.T @ jac
        step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -jac.T @ r)
        r_new = residual(x + step)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
    t = _tips(model, jm, readings, x[:nj], joints)
    pred = bf.board_xy_to_robot_xy(board_xy, 0.0, x[nj:nj + 2])
    return x[:nj], x[nj:nj + 2], float(x[nj + 2]), np.linalg.norm(t[:, :2] - pred, axis=1), t[:, 2] - x[nj + 2]


def leave_one_out(model, jm, readings, board_xy, joints=FIT_JOINTS) -> np.ndarray:
    """xy error of each mark predicted by a fit of the other marks."""
    out = []
    board_xy = np.asarray(board_xy, float)
    for i in range(len(readings)):
        keep = [k for k in range(len(readings)) if k != i]
        dq, off, _, _, _ = fit_zeros(model, jm, [readings[k] for k in keep], board_xy[keep], iters=40, joints=joints)
        t = _tips(model, jm, [readings[i]], dq, joints)[0]
        out.append(float(np.linalg.norm(t[:2] - bf.board_xy_to_robot_xy(board_xy[i:i + 1], 0.0, off)[0])))
    return np.array(out)


def with_new_zeros(jm: JointMap, dq, joints=FIT_JOINTS) -> JointMap:
    """The joint map whose sim angles include dq: zero_new = zero - sign * degrees(dq)."""
    new = JointMap(zero_deg=dict(jm.zero_deg), sign=dict(jm.sign), gripper_closed=jm.gripper_closed,
                   gripper_open=jm.gripper_open, gripper_span_deg=jm.gripper_span_deg, meta=dict(jm.meta))
    for j, d in zip(joints, dq):
        new.zero_deg[j] = jm.zero_deg[j] - jm.sign[j] * math.degrees(d)
    return new


def report(names, fit0: bf.BoardFit, xy_err, z_err, loo, dq, jm: JointMap, new: JointMap, axis, surface_z,
           joints=FIT_JOINTS) -> None:
    print(f"\n=== Fit (board square to the arm; zeros fitted: {', '.join(joints)}) ===")
    print(f"  {'mark':12s} {'before':>8s} {'after':>8s} {'left out':>9s} {'z after':>8s}   (xy errors, mm)")
    for n, e0, e, l, ez in zip(names, fit0.residuals_m, xy_err, loo, z_err):
        print(f"  {n:12s} {1000 * e0:8.1f} {1000 * e:8.1f} {1000 * l:9.1f} {1000 * ez:+8.1f}")
    rms = lambda v: 1000 * float(np.sqrt(np.mean(np.asarray(v) ** 2)))
    print(f"  RMS: before {rms(fit0.residuals_m):.1f} mm (board yaw {math.degrees(fit0.yaw):+.2f} deg), after "
          f"{rms(xy_err):.1f} mm, left out {rms(loo):.1f} mm, z after {rms(z_err):.1f} mm")
    print("  Zero changes (LeRobot deg):")
    for j, d in zip(joints, dq):
        print(f"    {j:14s} {jm.zero_deg[j]:+8.2f} -> {new.zero_deg[j]:+8.2f}  ({-jm.sign[j] * math.degrees(d):+.2f})")
    print(f"  robot.base_xy_m (pan axis in the board frame): [{axis[0]:+.4f}, {axis[1]:+.4f}] m; "
          f"play surface {1000 * surface_z:+.1f} mm above the base origin")
    if rms(loo) > GOOD_RMS_MM:
        print(f"  WARNING: the left-out error is more than {GOOD_RMS_MM:.0f} mm. Check the marks with the ruler and "
              f"touch them again, with the tips exactly on the dots.")
    if max(abs(math.degrees(d)) for d in dq) > MAX_ZERO_CHANGE_DEG:
        print(f"  WARNING: a zero changes by more than {MAX_ZERO_CHANGE_DEG:.0f} deg. Check that joint in the mirror.")


def finish(model, jm, names, board_xy, readings, meta, map_path, out, ask=True, joints=FIT_JOINTS):
    fit0 = bf.fit_board(board_xy, _tips(model, jm, readings, np.zeros(len(joints)), joints))
    dq, off, sz, xy_err, z_err = fit_zeros(model, jm, readings, board_xy, joints=joints)
    loo = leave_one_out(model, jm, readings, board_xy, joints) if len(readings) >= 5 else np.full(len(readings), np.nan)
    new = with_new_zeros(jm, dq, joints)
    tips = _tips(model, new, readings, np.zeros(len(joints)), joints)
    fit = bf.fit_board(board_xy, tips)
    report(names, fit0, xy_err, z_err, loo, dq, jm, new, fit.pan_axis_in_board(), fit.surface_z, joints)
    if ask and input("\nWrite the new zeros to the joint map and the board file? [y/N] ").strip().lower() != "y":
        print("Nothing written.")
        return
    Path(map_path).with_suffix(".yaml.bak").write_text(Path(map_path).read_text())
    new.meta["touch_fit"] = {"date": meta["date"], "rms_mm": round(1000 * float(np.sqrt(np.mean(xy_err ** 2))), 2),
                             "left_out_rms_mm": round(1000 * float(np.sqrt(np.nanmean(loo ** 2))), 2)}
    new.save(map_path)
    bf.save(fit, names, board_xy, tips, {**meta, "zeros_fitted": True}, out, readings=readings)
    print(f"Saved {map_path} (old: {Path(map_path).with_suffix('.yaml.bak')}) and {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--id", default="group5")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--divider-mm", type=float, default=None,
                   help="Divider thickness (default: physics params, measured 7 mm)")
    p.add_argument("--out", default=str(bf.DEFAULT_PATH))
    p.add_argument("--refit", action="store_true", help="Fit again from the readings in --out (no arm)")
    p.add_argument("--fit-joints", nargs="+", choices=ALL_FIT_JOINTS, default=list(FIT_JOINTS),
                   help="Zeros to fit (first: shoulder_pan). Default: pan, lift, elbow")
    a = p.parse_args()

    from src.slingpuck.config import load_config
    cfg = load_config("latest")
    divider = cfg["board"]["divider_thickness_m"] if a.divider_mm is None else a.divider_mm / 1000
    if not Path(a.map).is_file():
        sys.exit(f"No joint map at {a.map}. Run calibrate_joints first.")
    jm = JointMap.load(a.map, check_calibration=False)
    model = TipModel()
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "robot_id": a.id,
            "divider_thickness_m": divider, "joint_map": a.map,
            "lerobot_calibration_sha256": jm.meta.get("lerobot_calibration_sha256")}

    if a.refit:
        doc = bf.load(a.out)
        names = [n for n, m in doc["marks"].items() if "reading" in m]
        if len(names) < 3:
            sys.exit(f"{a.out} has no saved readings (made before readings were saved). Touch the marks again.")
        board_xy = np.array([doc["marks"][n]["board_xy_m"] for n in names])
        readings = [doc["marks"][n]["reading"] for n in names]
        # The readings are LeRobot values: they do not depend on the map, so the fit starts from the present map
        finish(model, jm, names, board_xy, readings, {**doc["meta"], **meta}, a.map, a.out, joints=tuple(a.fit_joints))
        return

    if not a.port:
        sys.exit("--port is needed (or --refit).")
    arm = Arm(a.port, a.id)
    if file_sha256(arm.calibration_file) != jm.meta.get("lerobot_calibration_sha256"):
        arm.robot.bus.disconnect(disable_torque=False)
        sys.exit("The joint map belongs to an older LeRobot calibration. Run calibrate_joints first.")

    names, board_xy, readings = [], [], []
    try:
        arm.torque_off()
        print(f"\nDivider thickness: {1000 * divider:.1f} mm (board y of a mark = -(divider / 2 + distance)).")
        for name, x, d in bf.DEFAULT_MARKS:
            res = touch(arm, model, jm, name, x, d)
            if res is not None:
                names.append(name)
                board_xy.append(bf.mark_to_board(x, d, divider))
                readings.append(res[1])
    finally:
        input("\nHold the arm (the torque stays off), then press Enter to disconnect... ")
        arm.close()
    if len(names) < 3:
        sys.exit("Fewer than 3 marks: no fit.")
    finish(model, jm, names, np.array(board_xy), readings, meta, a.map, a.out, joints=tuple(a.fit_joints))


if __name__ == "__main__":
    main()
