"""Step 4b: play a recorded sim trajectory on the real SO-101, open loop, with safety checks.

Run it in the lerobot env. Without --execute it only checks the trajectory (no motion):

    python3 -m src.slingpuck.deploy.replay_trajectory results/real/traj_<time> --port /dev/tty.usbmodem5A7A0563651
    python3 -m src.slingpuck.deploy.replay_trajectory results/real/traj_<time> --port ... --execute --speed 0.25

Order of the first tests: a --hover-mm 30 trajectory at --speed 0.25, then 0.5, then a trajectory on the board
(hover 0) at 0.25 --until grasp with a puck on the mark, then the full shot.

What it does with --execute:
  0. Turns the torque off. You hold the arm by hand near the first frame (jaws down, tips above the first
     point); a live line shows how far each joint and the tip are from it. The start must be within
     START_MAX_JOINT_DEG per joint and START_MAX_TIP_MM. (A joint-space move from the rest pose went through a
     stretched-out pose that the elbow could not hold.)
  1. Turns the torque on at the present pose (the goal positions are set to the present positions first).
  2. Moves slowly from the present pose to the first frame. Waits for Enter.
  3. Plays the frames at --speed x the sim speed (--rate Hz). Each arm command may change at most --max-step-deg
     from the LAST COMMAND. (Not LeRobot's max_relative_target: it limits the goal relative to the PRESENT
     position, so it limits the motor's force too. On the first run it stalled the elbow under gravity.)
  4. Holds the last frame (no move back to the rest pose: the same stretched-out path).
  5. Asks you to hold the arm, then turns the torque off.
The log has the loads of lift, elbow and wrist flex; the summary warns above LOAD_WARN_PERCENT.
Stall guard: if an arm joint stays more than STALL_DEG behind its command for STALL_S (a blocked or overloaded
joint), the script stops: every joint holds where it is, and the script asks before the torque goes off.
Ctrl-C at any time: the arm stops and holds its last target; then the script asks before the torque goes off.
Writes replay_<time>.npz in the trajectory folder: time, commanded and measured LeRobot angles.
"""
from __future__ import annotations

import argparse
import datetime
import json
import select
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.slingpuck.deploy.joint_map import (ARM_JOINTS, DEFAULT_PATH as MAP_PATH, GRIPPER, STS3215_MAX_RES, JointMap,
                                            file_sha256)

LIMIT_MARGIN_DEG = 3.0       # Commands must stay this far inside the calibrated stops
MAX_SPEED_DEG_S = 180.0      # Peak joint speed allowed at the chosen playback speed (STS3215: about 270 deg/s)
SPEED_WINDOW_S = 0.1         # Speeds are measured over this much play time. Single-frame jumps (the sim script
                             # starts some moves a few mm from where the last one ended) are reported apart
APPROACH_DEG_S = 20.0        # Speed of the moves from and back to the start pose
HOLD_S = 0.5
START_MAX_JOINT_DEG = 20.0   # Hand-held start: every arm joint this close to the first frame ...
START_MAX_TIP_MM = 40.0      # ... and the jaw tips this close to the first frame's tips
STALL_DEG = 12.0             # A joint this far behind its command ...
STALL_S = 0.4                # ... for this long: stop


def load_traj(folder: Path) -> dict:
    z = np.load(Path(folder) / "traj.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def cut_until(traj: dict, move: str | None) -> dict:
    """Keep the frames up to the end of the named move."""
    if move is None:
        return traj
    hits = np.nonzero(traj["move"] == move)[0]
    if not len(hits):
        raise ValueError(f"No move '{move}' in the trajectory. Moves: {list(dict.fromkeys(traj['move'].tolist()))}")
    end = hits[-1] + 1
    return {k: v[:end] for k, v in traj.items()}


def to_real(traj: dict, jm: JointMap) -> tuple[np.ndarray, np.ndarray | None]:
    """Sim frames -> LeRobot arm angles (n, 5) in deg, and gripper commands (n,) in 0-100 (None: no gripper map)."""
    arm = np.array([[jm.sim_to_real(q)[f"{j}.pos"] for j in ARM_JOINTS] for q in traj["q"]])
    grip = np.array([jm.gripper_to_real(g) for g in traj["grip"]]) if jm.gripper_span_deg else None
    return arm, grip


def limits_from_calibration(path) -> dict:
    """Calibrated stops of the arm joints in LeRobot degrees (LeRobot 0 = the middle of the range)."""
    cal = json.loads(Path(path).read_text())
    out = {}
    for j in ARM_JOINTS:
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]
        half = (hi - lo) / 2 * 360 / STS3215_MAX_RES
        out[j] = (-half, half)
    return out


def check(arm_deg: np.ndarray, t_play: np.ndarray, limits: dict) -> tuple[list[str], dict]:
    """Problems (empty = safe to play) and per-joint stats."""
    problems, stats = [], {}
    w = max(1, int(round(SPEED_WINDOW_S / np.median(np.diff(t_play))))) if len(t_play) > 1 else 1
    for i, j in enumerate(ARM_JOINTS):
        lo, hi = limits[j]
        a = arm_deg[:, i]
        if len(a) > w:
            speed = float(np.max(np.abs(a[w:] - a[:-w]) / (t_play[w:] - t_play[:-w])))
            jump = float(np.max(np.abs(np.diff(a))))
        else:
            speed = jump = 0.0
        stats[j] = {"min": float(a.min()), "max": float(a.max()), "lo": lo, "hi": hi, "peak_deg_s": speed,
                    "max_jump_deg": jump}
        if a.min() < lo + LIMIT_MARGIN_DEG or a.max() > hi - LIMIT_MARGIN_DEG:
            problems.append(f"{j}: commands {a.min():+.1f} to {a.max():+.1f} deg, calibrated stops {lo:+.1f} / {hi:+.1f} "
                            f"deg (margin {LIMIT_MARGIN_DEG:.0f}). Check the joint map, the calibration and the board position.")
        if speed > MAX_SPEED_DEG_S:
            problems.append(f"{j}: peak speed {speed:.0f} deg/s at this playback speed (limit {MAX_SPEED_DEG_S:.0f}). "
                            f"Use a lower --speed.")
    return problems, stats


def sample(t_src: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of frame values at time t (clamped to the ends)."""
    if values.ndim == 1:
        return np.interp(t, t_src, values)
    return np.array([np.interp(t, t_src, values[:, i]) for i in range(values.shape[1])])


def action(arm_deg: np.ndarray, grip: float | None) -> dict:
    a = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, arm_deg)}
    if grip is not None:
        a[f"{GRIPPER}.pos"] = float(grip)
    return a


def print_report(meta_path: Path, traj: dict, arm_deg, grip, t_play, problems, stats):
    if meta_path.is_file():
        meta = yaml.safe_load(meta_path.read_text())
        print(f"Trajectory: sim outcome {meta.get('sim_outcome')}, hover {1000 * meta.get('hover_m', 0):.0f} mm, "
              f"board file {meta.get('board_file')}")
    print(f"Frames: {len(t_play)}, play time {t_play[-1]:.1f} s, moves: "
          f"{' -> '.join(dict.fromkeys(traj['move'].tolist()))}")
    print(f"  {'joint':14s} {'commands (deg)':>20s} {'stops (deg)':>18s} {'peak deg/s':>11s} {'max jump':>9s}")
    for j in ARM_JOINTS:
        s = stats[j]
        print(f"  {j:14s} {s['min']:+8.1f} to {s['max']:+7.1f} {s['lo']:+8.1f} / {s['hi']:+6.1f} {s['peak_deg_s']:10.0f} "
              f"{s['max_jump_deg']:7.1f} deg")
    print(f"  (peak speed over {SPEED_WINDOW_S:.1f} s windows; a jump is one frame, and the --max-step-deg limit spreads it)")
    print(f"  gripper: " + ("no gripper map, the gripper is not commanded" if grip is None
                            else f"{grip.min():.1f} to {grip.max():.1f} /100"))
    print("\n".join(["Problems:", *problems]) if problems else "Checks passed.")


class Stalled(Exception):
    pass


LOAD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
LOAD_WARN_PERCENT = 70.0     # The motors' overload protection starts at 80 % (Overload_Torque)


def load_percent(value: int) -> float:
    """Present_Load in % of the motor's maximum. LeRobot already decodes the sign (bit 10): value is signed 0.1 %."""
    return int(value) / 10.0


def enter_pressed() -> bool:
    """True if a line (Enter) is waiting on stdin; reads it. Never blocks (macOS / Linux terminals)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def start_check(reading: dict, first_deg, tip, first_tip) -> tuple[bool, str, float, float]:
    """Is the hand-held pose close enough to the first frame? (ok, worst joint, its difference deg, tip mm)."""
    diffs = {j: abs(reading[j] - first_deg[i]) for i, j in enumerate(ARM_JOINTS)}
    worst = max(diffs, key=diffs.get)
    tip_mm = 1000 * float(np.linalg.norm(np.asarray(tip) - np.asarray(first_tip)))
    return diffs[worst] <= START_MAX_JOINT_DEG and tip_mm <= START_MAX_TIP_MM, worst, diffs[worst], tip_mm


def read_loads(arm) -> dict | None:
    """Load of lift, elbow and wrist flex in % (None if the arm cannot say)."""
    bus = getattr(arm, "bus", None)
    if bus is None:
        return None
    try:
        raw = bus.sync_read("Present_Load", list(LOAD_JOINTS), normalize=False, num_retry=1)
        return {k: load_percent(v) for k, v in raw.items()}
    except Exception:
        return None


class RealArm:
    """The SO-101 with torque on, through LeRobot (no max_relative_target: see the module docstring)."""

    def __init__(self, port: str, robot_id: str):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
        self.bus = self.robot.bus
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            sys.exit("The motors do not match the LeRobot calibration file.")

    def torque_on_here(self):
        present = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        self.bus.sync_write("Goal_Position", present, normalize=False)   # No jump when the torque comes on
        self.robot.configure()                                           # Gains, gripper current limit, torque on

    def read(self) -> dict:
        return self.bus.sync_read("Present_Position", num_retry=2)

    def send(self, a: dict) -> dict:
        return self.robot.send_action(a)

    def hold_here(self):
        """Goal = present position for every motor: a joint that pushes against something stops pushing."""
        present = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        self.bus.sync_write("Goal_Position", present, normalize=False)

    def torque_off(self):
        input("\nHold the arm: the torque turns OFF when you press Enter... ")
        try:
            self.bus.disconnect(disable_torque=True)
        except Exception as e:      # The motors do not answer (no power): nothing to turn off
            print(f"The motors did not answer ({e}). Turn the motor power off and on before the next run.")


class Runner:
    """One command stream: a step limit relative to the last command, a stall guard, and a log."""

    def __init__(self, arm, keys, rate: float, max_step_deg: float):
        self.arm, self.keys, self.rate, self.max_step = arm, list(keys), rate, max_step_deg
        self.last = None
        self.behind_since = {}
        self.t0 = time.perf_counter()
        self.log = {"t": [], "phase": [], "cmd": [], "meas": [], "load": []}

    def command(self, target: dict, phase: str) -> dict:
        target = dict(target)
        if self.last is not None:
            for j in ARM_JOINTS:
                target[j] = float(np.clip(target[j], self.last[j] - self.max_step, self.last[j] + self.max_step))
        self.arm.send({f"{k}.pos": float(target[k]) for k in self.keys})
        meas = self.arm.read()
        now = time.perf_counter() - self.t0
        self.log["t"].append(now)
        self.log["phase"].append(phase)
        self.log["cmd"].append([target[k] for k in self.keys])
        self.log["meas"].append([meas[j] for j in (*ARM_JOINTS, GRIPPER)])
        loads = read_loads(self.arm)
        self.log["load"].append([np.nan] * len(LOAD_JOINTS) if loads is None else [loads[j] for j in LOAD_JOINTS])
        self.last = target
        for j in ARM_JOINTS:
            if abs(meas[j] - target[j]) > STALL_DEG:
                since = self.behind_since.setdefault(j, now)
                if now - since > STALL_S:
                    raise Stalled(f"{j} stayed more than {STALL_DEG:.0f} deg behind its command for {STALL_S} s "
                                  f"(command {target[j]:+.1f}, present {meas[j]:+.1f} deg) during '{phase}'")
            else:
                self.behind_since.pop(j, None)
        return meas

    def arrays(self) -> dict:
        return {"t": np.array(self.log["t"]), "phase": np.array(self.log["phase"]),
                "cmd": np.array(self.log["cmd"]), "meas": np.array(self.log["meas"]),
                "load": np.array(self.log["load"], dtype=float).reshape(-1, len(LOAD_JOINTS))}


def move_between(run: Runner, start: dict, end: dict, phase: str) -> None:
    """Straight line in joint space from start to end at about APPROACH_DEG_S."""
    span = max(abs(end[k] - start[k]) for k in run.keys if k != GRIPPER)
    n = max(2, int(run.rate * max(1.0, span / APPROACH_DEG_S)))
    for s in np.linspace(0.0, 1.0, n):
        t0 = time.perf_counter()
        run.command({k: start[k] + s * (end[k] - start[k]) for k in run.keys}, phase)
        time.sleep(max(0.0, 1.0 / run.rate - (time.perf_counter() - t0)))


def play(run: Runner, traj_t, arm_deg, grip, speed: float) -> None:
    t_end = traj_t[-1] / speed
    start = time.perf_counter()
    while True:
        now = time.perf_counter() - start
        if now > t_end:
            break
        target = dict(zip(ARM_JOINTS, sample(traj_t, arm_deg, now * speed)))
        if grip is not None:
            target[GRIPPER] = float(sample(traj_t, grip, now * speed))
        run.command(target, "play")
        time.sleep(max(0.0, 1.0 / run.rate - (time.perf_counter() - start - now)))


def summary(log: dict) -> None:
    for phase in dict.fromkeys(log["phase"].tolist()):
        rows = log["phase"] == phase
        err = log["meas"][rows, :5] - log["cmd"][rows, :5]
        print(f"Tracking error during '{phase}' (measured - commanded, deg; {rows.sum()} commands):")
        for i, j in enumerate(ARM_JOINTS):
            print(f"  {j:14s} RMS {np.sqrt(np.mean(err[:, i] ** 2)):5.1f}   max {np.max(np.abs(err[:, i])):5.1f}")
        load = log["load"][rows]
        if len(load) and not np.isnan(load).all():
            peak = np.nanmax(np.abs(load), axis=0)
            print("  peak load %: " + ", ".join(f"{j} {v:.0f}" for j, v in zip(LOAD_JOINTS, peak)))
            if peak.max() > LOAD_WARN_PERCENT:
                print(f"  WARNING: a load above {LOAD_WARN_PERCENT:.0f} % (the overload protection starts at 80 %).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traj", help="results/real/traj_<time>")
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="group5")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--speed", type=float, default=0.25, help="Playback speed relative to the sim (default 0.25)")
    p.add_argument("--rate", type=float, default=50.0, help="Command rate, Hz")
    p.add_argument("--until", default=None, help="Stop after this move (e.g. grasp, pull)")
    p.add_argument("--max-step-deg", type=float, default=4.0,
                   help="Largest change of an arm command from the last command (default 4 deg: 200 deg/s at 50 Hz)")
    p.add_argument("--execute", action="store_true", help="Move the arm. Without it: checks only")
    a = p.parse_args()

    folder = Path(a.traj)
    jm = JointMap.load(a.map, check_calibration=False)
    cal_file = jm.meta.get("lerobot_calibration_file")
    if not cal_file or file_sha256(cal_file) != jm.meta.get("lerobot_calibration_sha256"):
        sys.exit("The joint map belongs to an older LeRobot calibration. Run calibrate_joints first.")
    traj = cut_until(load_traj(folder), a.until)
    arm_deg, grip = to_real(traj, jm)
    t_play = traj["t"] / a.speed
    problems, stats = check(arm_deg, t_play, limits_from_calibration(cal_file))
    print_report(folder / "meta.yaml", traj, arm_deg, grip, t_play, problems, stats)
    if problems:
        sys.exit("Not played.")
    if not a.execute:
        print("Dry run: add --execute to move the arm.")
        return

    keys = [*ARM_JOINTS, *([GRIPPER] if grip is not None else [])]
    from src.slingpuck.deploy.touch_test import TipModel
    tips = TipModel()
    g_first = float(traj["grip"][0])
    first_tip = tips.tip(traj["q"][0], g_first)
    arm = RealArm(a.port, a.id)
    run = Runner(arm, keys, a.rate, a.max_step_deg)
    connection_lost = False
    try:
        input("\nThe torque turns OFF now: hold the arm, then press Enter... ")
        arm.bus.disable_torque(num_retry=2)
        meta = yaml.safe_load((folder / "meta.yaml").read_text()) if (folder / "meta.yaml").is_file() else {}
        above = "" if "surface_z_m" not in meta else f", {1000 * (first_tip[2] - meta['surface_z_m']):.0f} mm above the board"
        print(f"Hold the arm near the first frame (the guard pose: jaw tips above the gate center{above}), jaws down.\n"
              f"Get within {START_MAX_JOINT_DEG:.0f} deg on every joint and {START_MAX_TIP_MM:.0f} mm at the tips "
              f"(the live line says OK), then press Enter: the torque turns on there.")
        while True:
            r = arm.read()
            ok, worst, diff, tip_mm = start_check(r, arm_deg[0], tips.tip(jm.real_to_sim(r), g_first), first_tip)
            print(f"\r    live: tip {tip_mm:5.0f} mm from the first frame; worst joint {worst} {diff:5.1f} deg "
                  f"{'OK' if ok else '  '}   ", end="", flush=True)
            if enter_pressed():
                print()
                if ok:
                    break
                print("    Not close enough yet. Keep holding, move closer, and press Enter again.")
            time.sleep(0.1)
        arm.torque_on_here()
        here = arm.read()
        first = dict(zip(keys, [*arm_deg[0], *([grip[0]] if grip is not None else [])]))
        move_between(run, here, first, "approach")
        input(f"At the first frame. Press Enter to play {t_play[-1]:.1f} s ({a.speed} x sim speed)... ")
        play(run, traj["t"], arm_deg, grip, a.speed)
        time.sleep(HOLD_S)
        print("Done. The arm holds the last frame.")
    except Stalled as e:
        print(f"\nSTOPPED: {e}.")
        print("Every joint now holds where it is. Look for something that blocks the arm, or a joint that is too weak.")
        try:
            arm.hold_here()
        except Exception:
            pass
    except ConnectionError as e:
        connection_lost = True
        print(f"\nThe motors stopped answering: {e}\nUsually the motor power dropped (a high load can pull the supply "
              f"voltage down) or the power was cut. Turn the motor power off and on, then check the arm.")
    except KeyboardInterrupt:
        print("\nStopped. The arm holds its last target.")
    finally:
        log = run.arrays()
        if len(log["t"]):
            out = folder / f"replay_{datetime.datetime.now():%Y%m%d_%H%M%S}.npz"
            np.savez(out, **log, keys=np.array(keys), meas_joints=np.array([*ARM_JOINTS, GRIPPER]), speed=a.speed)
            summary(log)
            print(f"Saved {out}")
        if connection_lost:
            print("The torque is probably off already (the motors lost power). Hold the arm.")
        else:
            arm.torque_off()


if __name__ == "__main__":
    main()
