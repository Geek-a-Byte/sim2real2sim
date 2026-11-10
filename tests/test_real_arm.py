"""Steps 3-4 for the real arm: board fit, range checks, trajectory record and replay checks (no hardware)."""
import json
import math

import numpy as np
import pytest

from src.slingpuck.deploy import board_frame as bf
from src.slingpuck.deploy import replay_trajectory as rp
from src.slingpuck.deploy.fix_ranges import check_range
from src.slingpuck.deploy.joint_map import ARM_JOINTS, JointMap


def touches(yaw_deg, offset, surface_z, noise_m=0.0, seed=0):
    rng = np.random.default_rng(seed)
    board = np.array([bf.mark_to_board(x, d, 0.01) for _, x, d in bf.DEFAULT_MARKS])
    xy = bf.board_xy_to_robot_xy(board, math.radians(yaw_deg), offset) + rng.normal(0, noise_m, board.shape)
    z = surface_z + rng.normal(0, noise_m, len(board))
    return board, np.column_stack([xy, z])


def test_board_fit_recovers_yaw_offset_and_surface():
    board, robot = touches(2.0, (0.25, 0.01), 0.011)
    fit = bf.fit_board(board, robot)
    assert math.degrees(fit.yaw) == pytest.approx(2.0, abs=1e-9)
    assert fit.offset == pytest.approx([0.25, 0.01], abs=1e-12)
    assert fit.surface_z == pytest.approx(0.011) and fit.rms_mm < 1e-6


def test_board_fit_with_noise_and_too_few_marks():
    board, robot = touches(-1.0, (0.24, -0.005), 0.01, noise_m=0.001)
    fit = bf.fit_board(board, robot)
    assert abs(math.degrees(fit.yaw) + 1.0) < 1.0 and 0.3 < fit.rms_mm < 2.0
    with pytest.raises(ValueError):
        bf.fit_board(board[:2], robot[:2])


def test_pan_axis_matches_the_sim_board_placement():
    """With no yaw, the fitted base_xy_m puts the sim board where the touches were (so101_slingpuck.board_pose)."""
    so = pytest.importorskip("src.slingpuck.so101_slingpuck")
    assert bf.BASE_AXIS_X == pytest.approx(so.BASE_AXIS_X, abs=1e-4)
    board, robot = touches(0.0, (0.26, 0.012), 0.01)
    axis = bf.BoardFit(0.0, np.array([0.26, 0.012]), 0.01, np.zeros(6)).pan_axis_in_board()
    cfg = {"robot": {"base_xy_m": list(axis)}, "board": {"outer_height_m": 0.025, "wall_height_m": 0.015}}
    for bxy, r in zip(board, robot):
        assert so.board_to_robot(cfg, bxy)[:2] == pytest.approx(r[:2], abs=2e-4)


def test_board_file_round_trip(tmp_path):
    board, robot = touches(0.5, (0.25, 0.0), 0.01)
    fit = bf.fit_board(board, robot)
    bf.save(fit, [n for n, _, _ in bf.DEFAULT_MARKS], board, robot, {"robot_id": "t"}, tmp_path / "b.yaml")
    doc = bf.load(tmp_path / "b.yaml")
    assert doc["robot"]["base_xy_m"] == pytest.approx(list(fit.pan_axis_in_board()), abs=1e-4)
    assert doc["fit"]["yaw_deg"] == pytest.approx(0.5, abs=0.01)


def test_marks_are_on_the_robot_half_and_left_is_negative_x():
    for name, x, d in bf.DEFAULT_MARKS:
        bx, by = bf.mark_to_board(x, d, 0.01)
        assert by < -0.005
        assert ("left" in name) == (bx < 0) or "center" in name


def test_range_check():
    assert check_range("shoulder_pan", 1000, 3100) is None
    assert "Move the joint" in check_range("shoulder_pan", 1563, 2050)        # The second calibration
    assert "wrap" in check_range("shoulder_pan", 10, 2000)


def calibration_file(tmp_path, pan_steps=2372):
    cal = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 2048 - s // 2, "range_max": 2048 + s // 2}
           for i, (j, s) in enumerate(zip(ARM_JOINTS, (pan_steps, 2500, 2290, 2390, 4095)))}
    path = tmp_path / "group5.json"
    path.write_text(json.dumps(cal))
    return path


def traj(n=101, dt=0.01, amp=0.3):
    t = np.arange(n) * dt
    q = np.zeros((n, 5))
    q[:, 1] = amp * np.sin(np.pi * t / t[-1])
    move = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    return {"t": t, "q": q, "grip": np.full(n, -0.17), "move": move}


def test_limits_from_calibration(tmp_path):
    lim = rp.limits_from_calibration(calibration_file(tmp_path, pan_steps=487))
    assert lim["shoulder_pan"] == pytest.approx((-21.4, 21.4), abs=0.1)
    assert lim["wrist_roll"][1] == pytest.approx(180.0, abs=0.1)


def test_replay_checks_limits_and_speed(tmp_path):
    jm = JointMap()
    lim = rp.limits_from_calibration(calibration_file(tmp_path))
    tr = traj()
    arm, grip = rp.to_real(tr, jm)
    assert grip is None                                                     # No gripper map: no gripper command
    assert arm[:, 1].max() == pytest.approx(math.degrees(0.3), abs=0.01)
    assert rp.check(arm, tr["t"] / 0.25, lim)[0] == []
    ramp = traj(n=201)
    ramp["q"][:, 1] = np.linspace(0.0, 1.0, 201)                            # 1 rad in 2 s: 28.6 deg/s at 1x
    ramp_deg = rp.to_real(ramp, jm)[0]
    _, stats = rp.check(ramp_deg, ramp["t"], lim)
    assert stats["shoulder_lift"]["peak_deg_s"] == pytest.approx(28.6, abs=0.1)
    assert rp.check(ramp_deg, ramp["t"] / 6.0, lim)[0] == []                 # 172 deg/s
    assert any("peak speed" in p for p in rp.check(ramp_deg, ramp["t"] / 7.0, lim)[0])   # 200 deg/s
    far = traj(amp=2.0)                                                     # 115 deg: past the lift stop
    assert any("stops" in p for p in rp.check(rp.to_real(far, jm)[0], far["t"] / 0.1, lim)[0])


def test_single_frame_jump_is_not_a_speed_problem(tmp_path):
    lim = rp.limits_from_calibration(calibration_file(tmp_path))
    tr = traj(n=401, amp=0.0)
    tr["q"][200:, 2] = math.radians(3.0)                                    # A 3 deg step, as at a move change
    arm, _ = rp.to_real(tr, JointMap())
    probs, stats = rp.check(arm, tr["t"] / 1.0, lim)
    assert probs == [] and stats["elbow_flex"]["max_jump_deg"] == pytest.approx(3.0)


def test_cut_until_and_sample():
    tr = traj()
    cut = rp.cut_until(tr, "a")
    assert len(cut["t"]) == 50 and set(cut["move"]) == {"a"}
    with pytest.raises(ValueError):
        rp.cut_until(tr, "nope")
    v = np.column_stack([tr["t"], 2 * tr["t"]])
    assert rp.sample(tr["t"], v, 0.555) == pytest.approx([0.555, 1.11])
    assert rp.sample(tr["t"], tr["t"], 99.0) == pytest.approx(tr["t"][-1])  # Clamped at the end


def test_action_has_the_gripper_only_with_a_map():
    assert "gripper.pos" not in rp.action(np.zeros(5), None)
    assert rp.action(np.zeros(5), 40.0)["gripper.pos"] == 40.0


def test_record_setup_uses_the_board_file_and_hover(tmp_path):
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy.record_trajectory import setup_config
    board, robot = touches(0.0, (0.26, 0.0), 0.012)
    fit = bf.fit_board(board, robot)
    bf.save(fit, [n for n, _, _ in bf.DEFAULT_MARKS], board, robot, {}, tmp_path / "b.yaml")
    cfg, notes = setup_config(tmp_path / "b.yaml", 0.03, None)
    surface = cfg["board"]["outer_height_m"] - cfg["board"]["wall_height_m"]
    assert surface == pytest.approx(0.012 + 0.03, abs=1e-4)
    assert cfg["robot"]["base_xy_m"] == pytest.approx(list(fit.pan_axis_in_board()), abs=1e-4)
    assert notes["board_file"] == str(tmp_path / "b.yaml")


# Sim joint angles of the 6 marks (IK of the jaws pointing down, from the S3 scene)
MARK_Q = np.array([[-0.32, 0.17, 0.23, 1.17, -1.85], [0.28, 0.15, 0.26, 1.16, -1.24], [-0.03, -0.24, 0.72, 1.09, -1.55],
                   [-0.52, -0.45, 0.91, 1.11, -2.04], [0.45, -0.49, 0.94, 1.12, -1.07], [-0.06, -1.0, 1.24, 1.34, -1.58]])


def synthetic_touches(true_map, offset=(0.25, -0.02)):
    """Readings of the 6 poses with the true map, and board marks exactly under the true tips (board square)."""
    from src.slingpuck.deploy.touch_test import TipModel
    model = TipModel()
    tips = np.array([model.tip(q, -0.17) for q in MARK_Q])
    sq = tips[:, :2] - offset                                  # Square board: robot xy = (y_b, -x_b) + offset
    board_xy = np.column_stack([-sq[:, 1], sq[:, 0]])
    readings = [{**{k.removesuffix(".pos"): v for k, v in true_map.sim_to_real(q).items()}, "gripper": 2.0}
                for q in MARK_Q]
    return model, board_xy, readings, tips


def test_touch_fit_finds_zero_errors_and_the_board():
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy.touch_test import ALL_FIT_JOINTS as FIT_JOINTS, fit_zeros, leave_one_out, with_new_zeros
    true = JointMap()                                          # Zeros 0, signs +1
    model, board_xy, readings, tips = synthetic_touches(true)
    wrong = JointMap(zero_deg={**true.zero_deg, "shoulder_pan": -6.0, "shoulder_lift": 2.0, "elbow_flex": -1.5,
                               "wrist_flex": 1.0})
    from src.slingpuck.deploy.touch_test import _tips
    before = bf.fit_board(board_xy, _tips(model, wrong, readings, np.zeros(4), FIT_JOINTS))
    # Lift, elbow and wrist zero errors of 1-2 deg mostly shift all points (the board offset takes that) and
    # change the height: small xy error, a height spread, and the pan error shows as a board yaw
    assert abs(math.degrees(before.yaw)) > 5.0 and np.ptp(before.z_residuals_m) > 0.003
    dq, off, sz, xy_err, z_err = fit_zeros(model, wrong, readings, board_xy, joints=FIT_JOINTS)
    fixed = with_new_zeros(wrong, dq, FIT_JOINTS)
    assert fixed.zero_deg["shoulder_pan"] == pytest.approx(0.0, abs=0.3)      # The yaw goes into the pan zero
    for j in FIT_JOINTS[1:]:
        # With the jaws down at every mark, lift, elbow and wrist flex trade off: the fit finds a combination
        # that puts the tips right, not each true zero
        assert fixed.zero_deg[j] == pytest.approx(0.0, abs=1.0), j
    assert 1000 * np.sqrt(np.mean(xy_err ** 2)) < 0.5 and np.max(np.abs(z_err)) < 0.001
    assert off == pytest.approx([0.25, -0.02], abs=0.002) and sz == pytest.approx(tips[:, 2].mean(), abs=0.002)
    assert np.max(leave_one_out(model, wrong, readings, board_xy, FIT_JOINTS)) < 0.001


def test_pan_change_turns_the_tip_about_the_pan_axis():
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy.touch_test import TipModel
    model = TipModel()
    q = MARK_Q[2].copy()
    t0 = model.tip(q, -0.17)
    q[0] += math.radians(4.0)
    assert model.tip(q, -0.17) == pytest.approx(bf.rotate_about_pan(t0, -math.radians(4.0)), abs=1e-6)


class FakeServoArm:
    """Joints follow the goal at most `speed` deg per command; `blocked` joints do not move past a value."""

    def __init__(self, start, speed=10.0, blocked=None):
        self.pos = dict(start)
        self.speed, self.blocked = speed, blocked or {}
        self.sent = []

    def send(self, a):
        goal = {k.removesuffix(".pos"): v for k, v in a.items()}
        self.sent.append(goal)
        for k, g in goal.items():
            step = float(np.clip(g - self.pos[k], -self.speed, self.speed))
            self.pos[k] += step
            if k in self.blocked:
                self.pos[k] = max(self.pos[k], self.blocked[k])
        return a

    def read(self):
        return dict(self.pos)


START = {"shoulder_pan": 0.0, "shoulder_lift": -90.0, "elbow_flex": 95.0, "wrist_flex": 0.0, "wrist_roll": 0.0,
         "gripper": 5.0}


def test_runner_limits_steps_from_the_last_command_not_from_the_present():
    """The first run: the goal clamped to the present position stalled the elbow under gravity."""
    arm = FakeServoArm(START, speed=0.0)                       # A joint that does not move at all
    run = rp.Runner(arm, ARM_JOINTS, rate=50, max_step_deg=4.0)
    run.command({**START}, "approach")
    run.command({**START, "elbow_flex": 60.0}, "approach")     # 35 deg away: limited to 4 from the last COMMAND
    assert arm.sent[-1]["elbow_flex"] == pytest.approx(91.0)
    run.command({**START, "elbow_flex": 60.0}, "approach")
    assert arm.sent[-1]["elbow_flex"] == pytest.approx(87.0)   # Keeps going although the joint did not move


def test_runner_stops_a_stalled_joint(monkeypatch):
    clock = iter(np.arange(0.0, 10.0, 0.02))
    monkeypatch.setattr(rp.time, "perf_counter", lambda: next(clock))
    arm = FakeServoArm(START, speed=5.0, blocked={"elbow_flex": 80.0})
    run = rp.Runner(arm, ARM_JOINTS, rate=50, max_step_deg=4.0)
    with pytest.raises(rp.Stalled, match="elbow_flex"):
        rp.move_between(run, START, {**START, "elbow_flex": 7.0}, "approach")
    log = run.arrays()
    assert (log["phase"] == "approach").all() and log["meas"][-1, 2] == pytest.approx(80.0)


def test_runner_passes_a_joint_that_follows(monkeypatch):
    clock = iter(np.arange(0.0, 20.0, 0.02))
    monkeypatch.setattr(rp.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    arm = FakeServoArm(START, speed=5.0)
    run = rp.Runner(arm, ARM_JOINTS, rate=50, max_step_deg=4.0)
    end = {**START, "shoulder_lift": 8.0, "elbow_flex": 7.0}
    rp.move_between(run, START, end, "approach")
    assert arm.pos["elbow_flex"] == pytest.approx(7.0) and arm.pos["shoulder_lift"] == pytest.approx(8.0)


def test_vertical_stop_ik_moves_the_tip_straight_down():
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy.vertical_stop import TipIK
    ik = TipIK()
    q0 = MARK_Q[2]
    tip0, dir0 = ik.tip_and_dir(q0, -0.17)
    q = q0.copy()
    for k in range(1, 6):                                      # 5 steps of 2 mm
        q = ik.solve(q, -0.17, tip0 - [0, 0, 0.002 * k], dir0)
    tip, d = ik.tip_and_dir(q, -0.17)
    assert tip == pytest.approx(tip0 - [0, 0, 0.010], abs=2e-4)
    assert np.degrees(np.arccos(np.clip(d @ dir0, -1, 1))) < 1.0
    assert q[4] == q0[4]                                       # Wrist roll does not move


def test_vertical_stop_rules_and_frames():
    from src.slingpuck.deploy import vertical_stop as vs
    t = vs.descent_targets([0.2, 0.0, 0.05], 2.0, 10.0)
    assert len(t) == 6 and t[-1][2] == pytest.approx(0.04)
    assert vs.stop_reason(0.012, 0.012, 20) is None
    assert "below the expected surface" in vs.stop_reason(0.0015, 0.012, 20)
    assert "travel" in vs.stop_reason(0.05, 0.012, 81)
    bxy = np.array([[0.03, -0.06]])
    r = bf.board_xy_to_robot_xy(bxy, 0.0, (0.27, 0.002))[0]
    assert vs.robot_to_board_xy(r, (0.27, 0.002)) == pytest.approx(bxy[0])
    assert rp.load_percent(800) == 80.0 and rp.load_percent(-32) == -3.2   # LeRobot gives the signed value


def test_vertical_stop_records_the_contact(monkeypatch):
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy import vertical_stop as vs
    jm = JointMap(gripper_closed=2.0, gripper_open=98.0, gripper_span_deg=130.0)
    start = {**{k.removesuffix(".pos"): v for k, v in jm.sim_to_real(MARK_Q[2]).items()}, "gripper": 2.0}
    arm = FakeServoArm(start, speed=100.0)                     # Follows every command at once
    ik = vs.TipIK()
    tip0 = ik.model.tip(MARK_Q[2], jm.gripper_to_sim(2.0))
    surface = tip0[2] - 0.006                                  # The board is 6 mm under the start
    presses = iter([False] * 3 + [True])                       # Enter at the 4th step (6 mm down)
    monkeypatch.setattr(vs, "enter_pressed", lambda: next(presses))
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(vs.time, "sleep", lambda s: None)
    res = vs.measure_point(arm, ik, jm, surface, (0.27, 0.0), 2.0, 0.0)
    assert res["contact_minus_expected_mm"] == pytest.approx(0.0, abs=0.3)
    assert len(res["steps"]) == 4 and res["stopped"] is None
    back = ik.model.tip(jm.real_to_sim(arm.read()), jm.gripper_to_sim(2.0))
    assert back == pytest.approx(tip0, abs=5e-4)               # Back at the start


def test_vertical_stop_correction_uses_the_median(tmp_path):
    from src.slingpuck.deploy import vertical_stop as vs
    board, robot = touches(0.0, (0.26, 0.0), 0.0118)
    fit = bf.fit_board(board, robot)
    bf.save(fit, [n for n, _, _ in bf.DEFAULT_MARKS], board, robot, {}, tmp_path / "b.yaml")
    res = {"points": [{"contact_minus_expected_mm": v} for v in (7.2, 13.0, 7.8, 7.0)] + [{"stalled": "x"}]}
    (tmp_path / "r.yaml").write_text(__import__("yaml").safe_dump(res))
    assert vs.apply_correction(tmp_path / "r.yaml", tmp_path / "b.yaml") == pytest.approx(7.5)
    assert bf.load(tmp_path / "b.yaml")["board"]["surface_z_m"] == pytest.approx(0.0118 + 0.0075, abs=1e-4)
    with pytest.raises(ValueError):
        vs.surface_correction_mm([{"contact_minus_expected_mm": 1.0}])


def test_replay_start_check():
    first = np.array([0.0, 8.0, 7.0, 80.0, 98.0])
    near = dict(zip(ARM_JOINTS, first + [5, -10, 12, 3, -15]))
    ok, worst, diff, tip_mm = rp.start_check(near, first, [0.27, 0.0, 0.05], [0.27, 0.01, 0.06])
    assert ok and worst == "wrist_roll" and diff == pytest.approx(15) and tip_mm == pytest.approx(14.1, abs=0.1)
    far = dict(zip(ARM_JOINTS, first + [0, -95, 90, 0, 0]))          # The rest pose: refused
    assert not rp.start_check(far, first, [0.27, 0.0, 0.05], [0.27, 0.0, 0.05])[0]
    assert not rp.start_check(near, first, [0.27, 0.0, 0.10], [0.27, 0.0, 0.05])[0]   # Tips 5 cm away


def test_runner_log_has_loads_or_nan():
    arm = FakeServoArm(START, speed=100.0)
    run = rp.Runner(arm, ARM_JOINTS, rate=50, max_step_deg=4.0)
    run.command(dict(START), "play")
    log = run.arrays()
    assert log["load"].shape == (1, 3) and np.isnan(log["load"]).all()   # The fake arm has no load register
