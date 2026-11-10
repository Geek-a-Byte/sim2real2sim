"""Mirror: the MuJoCo SO-101 follows the real SO-101 live, through the joint map.

Move the real arm by hand (torque off) and compare its pose with the sim arm. Where they differ,
the joint map is wrong. Run it in the lerobot env with mjpython (macOS needs it for the live viewer):

    conda activate lerobot
    mjpython -m src.slingpuck.deploy.mirror --port /dev/tty.usbmodem5A7A0563651 --id group5

The viewer shows the sim angles and the real LeRobot readings, both in rad, at the top left.
The script never turns the motor torque on. It turns it off, so hold the arm.

Checks to do (in this order):
  1. Each joint alone: the sim joint must turn the same way and by the same amount.
  2. The zero pose (all sim angles 0): the real arm must look the same as the sim arm.
  3. The gripper: closed and fully open must look the same.
If a joint is off by a constant angle, run calibrate_joints --steps zero again.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.slingpuck.deploy.calibrate_joints import Arm
from src.slingpuck.deploy.joint_map import ARM_JOINTS, DEFAULT_PATH, GRIPPER, JointMap

SCENE = Path(__file__).resolve().parents[3] / "assets/mujoco/robotstudio_so101/scene.xml"
RATE_HZ = 30


def overlay(jm: JointMap, r: dict, q: np.ndarray, qg: float | None) -> tuple[str, str]:
    names = [*ARM_JOINTS, GRIPPER]
    sim = [f"{v:+7.3f}" for v in q] + ["      ?" if qg is None else f"{qg:+7.3f}"]
    real = [f"{np.radians(r[j]):+7.3f}" for j in ARM_JOINTS] + [f"{r[GRIPPER]:5.1f} /100"]
    return ("\n".join(["joint", *names]),
            "\n".join(["sim rad   real rad", *(f"{s}   {x}" for s, x in zip(sim, real))]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="group5")
    p.add_argument("--map", default=str(DEFAULT_PATH))
    a = p.parse_args()

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        sys.exit(f"MuJoCo is not importable with {sys.executable} ({e}). Use the lerobot env: pip install mujoco")
    if not Path(a.map).is_file():
        sys.exit(f"No joint map at {a.map}. Run calibrate_joints first.")
    jm = JointMap.load(a.map)
    if jm.gripper_span_deg is None:
        print("The joint map has no gripper span: the sim gripper stays closed. Run calibrate_joints --steps gripper.")

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    arm_q = [int(m.joint(j).qposadr[0]) for j in ARM_JOINTS]
    grip_q = int(m.joint(GRIPPER).qposadr[0])

    arm = Arm(a.port, a.id)
    try:
        arm.torque_off()
        with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
            print("Mirror running. Close the viewer window or press Ctrl-C to stop.")
            while v.is_running():
                t0 = time.perf_counter()
                try:
                    r = arm.read()
                except Exception as e:      # A failed serial read: keep the last pose
                    print(f"read failed: {e}")
                    continue
                q = jm.real_to_sim(r)
                qg = jm.gripper_to_sim(r[GRIPPER]) if jm.gripper_span_deg else None
                with v.lock():
                    d.qpos[arm_q] = q
                    d.qpos[grip_q] = m.jnt_range[m.joint(GRIPPER).id][0] if qg is None else qg
                    d.qvel[:] = 0
                    mujoco.mj_forward(m, d)
                    left, right = overlay(jm, r, q, qg)
                    v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
                v.sync()
                time.sleep(max(0.0, 1.0 / RATE_HZ - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


if __name__ == "__main__":
    main()
