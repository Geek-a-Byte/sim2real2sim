import numpy as np
import pytest

from src.slingpuck.deploy import calibrate_joints as cj
from src.slingpuck.deploy.joint_map import ARM_JOINTS, GRIPPER, JointMap, estimate_sign, zero_from_reference

ZERO = {"shoulder_pan": 3.0, "shoulder_lift": -88.0, "elbow_flex": 91.5, "wrist_flex": -2.0, "wrist_roll": 47.0}
SIGN = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": -1, "wrist_flex": 1, "wrist_roll": -1}


def true_map():
    return JointMap(zero_deg=dict(ZERO), sign=dict(SIGN))


def test_sim_to_real_and_back_is_the_identity():
    jm = true_map()
    q = np.random.default_rng(0).uniform(-1.5, 1.5, len(ARM_JOINTS))
    action = jm.sim_to_real(q)
    assert set(action) == {f"{j}.pos" for j in ARM_JOINTS}
    assert np.allclose(jm.real_to_sim(action), q)
    assert np.allclose(jm.sim_to_real(np.zeros(5))["elbow_flex.pos"], ZERO["elbow_flex"])


def test_real_to_sim_accepts_names_with_and_without_pos():
    jm = true_map()
    r = {j: ZERO[j] + 10.0 for j in ARM_JOINTS}
    assert np.allclose(jm.real_to_sim(r), jm.real_to_sim({f"{j}.pos": v for j, v in r.items()}))
    assert np.allclose(np.degrees(jm.real_to_sim(r)), [10.0 * SIGN[j] for j in ARM_JOINTS])


def test_sign_needs_a_large_move():
    assert estimate_sign(-5.0, 25.0) == 1
    assert estimate_sign(30.0, 0.0) == -1
    with pytest.raises(ValueError):
        estimate_sign(0.0, 4.0)


def test_zero_from_a_non_zero_reference_pose():
    # Sim pose elbow = +30 deg, sign -1: the real reading is zero - 30
    assert zero_from_reference(ZERO["elbow_flex"] - 30.0, -1, 30.0) == pytest.approx(ZERO["elbow_flex"])


def test_save_load_and_calibration_change_warning(tmp_path):
    calib = tmp_path / "group5.json"
    calib.write_text("{}")
    jm = true_map()
    jm.gripper_closed, jm.gripper_open = 1.5, 98.0
    jm.meta = {"lerobot_calibration_file": str(calib), "lerobot_calibration_sha256": cj.file_sha256(calib)}
    jm.save(tmp_path / "map.yaml")
    back = JointMap.load(tmp_path / "map.yaml")
    assert back.zero_deg == pytest.approx(ZERO) and back.sign == SIGN and back.gripper_open == 98.0
    calib.write_text('{"changed": 1}')
    with pytest.warns(UserWarning, match="calibrate_joints"):
        JointMap.load(tmp_path / "map.yaml")


def test_arm_joint_order_matches_the_sim():
    mm = pytest.importorskip("src.slingpuck.physics.mujoco_match")
    assert ARM_JOINTS == mm.ARM_JOINTS


class FakeArm:
    """A real arm with the true map above. The 'person' sets the pose the script asks for."""

    def __init__(self):
        self.real = {j: ZERO[j] for j in ARM_JOINTS} | {GRIPPER: 50.0}

    def set_sim_deg(self, j, deg):
        self.real[j] = ZERO[j] + SIGN[j] * deg

    def limits_deg(self, j):
        return (-110.0, 110.0)

    def range_steps(self, j):
        return 1250                   # About 110 deg of motor travel

    def read(self):
        return dict(self.real)

    def read_mean(self, n=cj.SAMPLES):
        return dict(self.real), {k: 0.1 for k in self.real}


def test_sign_and_zero_steps_recover_the_true_map(monkeypatch):
    arm = FakeArm()
    script = []                       # One action per input() call, in order
    for j in ARM_JOINTS:
        script += [lambda j=j: arm.set_sim_deg(j, 0.0), lambda j=j: arm.set_sim_deg(j, 30.0)]
    ref = [10.0, -20.0, 30.0, 0.0, 5.0]
    script.append(lambda: [arm.set_sim_deg(j, d) for j, d in zip(ARM_JOINTS, ref)])
    script += [lambda: arm.real.update({GRIPPER: 2.0}), lambda: arm.real.update({GRIPPER: 97.0})]
    actions = iter(script)
    monkeypatch.setattr("builtins.input", lambda *_: next(actions)())

    jm = JointMap()                   # Starts with zeros and signs +1
    cj.step_sign(arm, jm)
    cj.step_zero(arm, jm, np.radians(ref))      # The zero step takes the slider values (rad)
    cj.step_gripper(arm, jm)
    assert jm.sign == SIGN
    assert jm.zero_deg == pytest.approx(ZERO)
    assert (jm.gripper_closed, jm.gripper_open) == (2.0, 97.0)
    assert jm.gripper_span_deg == pytest.approx(1250 * 0.95 * 360 / 4095)


def test_a_pose_that_moves_during_the_reading_is_asked_again(monkeypatch):
    arm = FakeArm()
    stds = iter([{**{j: 3.0 for j in ARM_JOINTS}, GRIPPER: 0.0}, {**{j: 0.1 for j in ARM_JOINTS}, GRIPPER: 0.0}])
    monkeypatch.setattr(arm, "read_mean", lambda n=0: (dict(arm.real), next(stds)))
    asked = []
    monkeypatch.setattr("builtins.input", lambda *_: asked.append(1))
    cj.steady_pose(arm, "pose")
    assert len(asked) == 2


def test_a_joint_that_falls_back_before_enter_is_asked_again(monkeypatch, capsys):
    """The first real run: the person let go of shoulder_lift to press Enter, and it fell back to its stop."""
    arm = FakeArm()
    j = "shoulder_lift"
    script = [lambda: arm.set_sim_deg(j, 0.0), lambda: arm.set_sim_deg(j, 0.0),      # Fell back: no move
              lambda: arm.set_sim_deg(j, 0.0), lambda: arm.set_sim_deg(j, 30.0)]     # Held in the new pose
    actions = iter(script)
    monkeypatch.setattr("builtins.input", lambda *_: next(actions)())
    monkeypatch.setattr(cj, "ARM_JOINTS", (j,))
    jm = JointMap()
    cj.step_sign(arm, jm)
    assert jm.sign[j] == SIGN[j]
    assert "fell back" in capsys.readouterr().out


def test_live_reading_shows_the_move(monkeypatch, capsys):
    arm = FakeArm()
    arm.set_sim_deg("shoulder_pan", 30.0)

    def slow_enter(*_):
        import time
        time.sleep(0.25)              # Time for the live thread to print
    monkeypatch.setattr("builtins.input", slow_enter)
    cj.wait_enter(arm, "shoulder_pan", ZERO["shoulder_pan"])
    assert "moved +0.524 rad" in capsys.readouterr().out    # 30 deg


def gripper_map():
    return JointMap(gripper_closed=2.0, gripper_open=96.0, gripper_span_deg=100.0)


def test_gripper_closed_and_open_map_to_the_sim_jaws():
    from src.slingpuck.deploy.joint_map import SIM_GRIPPER_CLOSED_RAD, SIM_GRIPPER_MAX_RAD
    jm = gripper_map()
    assert jm.gripper_to_sim(2.0) == pytest.approx(SIM_GRIPPER_CLOSED_RAD)
    assert np.degrees(jm.gripper_to_sim(96.0) - SIM_GRIPPER_CLOSED_RAD) == pytest.approx(100.0)
    assert jm.gripper_to_sim(100.0) <= SIM_GRIPPER_MAX_RAD and jm.gripper_to_sim(-5.0) == SIM_GRIPPER_CLOSED_RAD
    for r in (2.0, 30.0, 70.0, 96.0):
        assert jm.gripper_to_real(jm.gripper_to_sim(r)) == pytest.approx(r)
    with pytest.raises(ValueError):
        JointMap().gripper_to_sim(50.0)


def test_sim_gripper_constants_match_the_model():
    mujoco = pytest.importorskip("mujoco")
    from src.slingpuck.deploy.joint_map import SIM_GRIPPER_CLOSED_RAD, SIM_GRIPPER_MAX_RAD
    m = mujoco.MjModel.from_xml_path("assets/mujoco/robotstudio_so101/so101.xml")
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[m.joint("gripper").id]
    assert (lo, hi) == pytest.approx((SIM_GRIPPER_CLOSED_RAD, SIM_GRIPPER_MAX_RAD), abs=1e-5)

    def gap(q):
        d.qpos[m.joint("gripper").qposadr[0]] = q
        mujoco.mj_kinematics(m, d)
        return np.linalg.norm(d.geom_xpos[m.geom("fixed_jaw_sph_tip1").id] - d.geom_xpos[m.geom("moving_jaw_sph_tip1").id])
    assert gap(lo) < gap(0.5) < gap(hi)          # The jaws are closed at the lower limit and open as q goes up


def test_mirror_overlay_shows_every_joint():
    pytest.importorskip("mujoco")
    from src.slingpuck.deploy.mirror import overlay
    jm = gripper_map()
    r = {**{j: 10.0 for j in ARM_JOINTS}, GRIPPER: 50.0}
    left, right = overlay(jm, r, np.zeros(5), jm.gripper_to_sim(50.0))
    assert len(left.splitlines()) == len(right.splitlines()) == 7
    assert "gripper" in left and "50.0 /100" in right


def test_live_table_shows_sim_and_real_in_rad():
    jm = JointMap(zero_deg=dict(ZERO), sign=dict(SIGN), gripper_closed=2.0, gripper_open=96.0, gripper_span_deg=100.0)
    sim_rad = [0.5, -1.0, 0.25, 0.0, 1.5]
    r = {k.removesuffix(".pos"): v for k, v in jm.sim_to_real(sim_rad).items()} | {GRIPPER: 2.0}
    names, sim, real = cj.live_table(jm, r)
    assert names.split() == [*ARM_JOINTS, GRIPPER]
    assert [float(x) for x in sim.split()[2:7]] == pytest.approx(sim_rad, abs=1e-3)
    assert float(sim.split()[7]) == pytest.approx(-0.175, abs=1e-3)               # Closed jaws
    assert [float(x) for x in real.split()[2:7]] == pytest.approx(np.radians([r[j] for j in ARM_JOINTS]), abs=1e-3)
