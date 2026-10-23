import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.so101_slingpuck import BASE_AXIS_X, board_to_robot, build  # noqa: E402


@pytest.fixture(scope="module")
def scene(cfg):
    m = build(cfg).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def test_board_lies_in_front_of_the_robot(cfg):
    hl = 0.5 * cfg["board"]["outer_length_m"]
    assert BASE_AXIS_X < board_to_robot(cfg, (0, -hl))[0]
    assert abs(board_to_robot(cfg, (0, 0))[1]) < 1e-9  # Centered left to right


def test_turning_axis_matches_config(cfg, scene):
    m, d = scene
    axis = d.xanchor[m.joint("shoulder_pan").id]
    np.testing.assert_allclose(axis[:2], board_to_robot(cfg, cfg["robot"]["base_xy_m"])[:2], atol=1e-4)


def test_board_and_robot_stand_on_the_same_table(cfg, scene):
    m, d = scene
    g = m.geom("surface").id
    assert d.geom_xpos[g][2] - m.geom_size[g][2] == pytest.approx(0.0, abs=1e-4)
    assert "lb_wall_back" not in {m.geom(i).name for i in range(m.ngeom)}


def test_flicked_puck_slows_with_coulomb_friction(cfg, scene):
    m, _ = scene
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    j = m.joint("puck_a0_y")
    d.qvel[j.dofadr[0]] = 0.5
    n = int(round(0.1 / m.opt.timestep))
    mujoco.mj_step(m, d, nstep=n)
    decel = (0.5 - d.qvel[j.dofadr[0]]) / (n * m.opt.timestep)
    assert decel == pytest.approx(cfg["puck"]["friction_kinetic"] * 9.81, rel=0.03)


def test_wall_bounce_keeps_the_configured_restitution(cfg, scene):
    m, _ = scene
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    a = m.joint("puck_a4_x").dofadr[0]   # Outermost puck in the row, heading for the side wall
    d.qvel[a] = 1.5
    v_in = None
    for _ in range(int(0.2 / m.opt.timestep)):
        before = d.qvel[a]
        mujoco.mj_step(m, d)
        if v_in is None and d.ncon:
            v_in = before
        if v_in is not None and d.ncon == 0 and d.qvel[a] < 0:
            break
    assert -d.qvel[a] / v_in == pytest.approx(cfg["board"]["restitution"], abs=0.03)


def test_opponent_camera_faces_the_robot(scene):
    m, d = scene
    cam = m.camera("opponent").id
    forward = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    assert forward[0] < -0.5  # Looks back toward the robot (-x)
