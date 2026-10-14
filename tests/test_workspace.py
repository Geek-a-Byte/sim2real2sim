import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck import workspace as ws  # noqa: E402
from src.slingpuck.so101_slingpuck import board_to_robot, build  # noqa: E402


@pytest.fixture(scope="module")
def scene(cfg):
    m = build(cfg).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def test_board_fits_between_the_side_walls_in_front_of_the_robot(cfg):
    hl = 0.5 * cfg["board"]["outer_length_m"]
    near, far = board_to_robot(cfg, (0, -hl))[0], board_to_robot(cfg, (0, hl))[0]
    assert ws.BASE_AXIS_X < near and far < ws.FAR_X - 0.15  # Room for the opponent's hands at the open end
    assert abs(board_to_robot(cfg, (0, 0))[1]) < 1e-9  # Centered left to right
    assert 0.5 * cfg["board"]["outer_width_m"] < ws.MAT_Y


def test_turning_axis_matches_config(cfg, scene):
    m, d = scene
    axis = d.xanchor[m.joint("shoulder_pan").id]
    np.testing.assert_allclose(axis[:2], board_to_robot(cfg, cfg["robot"]["base_xy_m"])[:2], atol=1e-4)


def test_play_surface_sits_on_the_mat(cfg, scene):
    m, d = scene
    g = m.geom("surface").id
    bottom = d.geom_xpos[g][2] - m.geom_size[g][2]
    assert bottom == pytest.approx(ws.MAT_TOP, abs=1e-4)
    assert ws.play_surface_z(cfg) == pytest.approx(-cfg["robot"]["base_z_m"], abs=1e-4)


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


def test_far_end_is_open_for_the_opponent(scene):
    m, _ = scene
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert "lb_wall_back" not in names and {"lb_wall_left", "lb_wall_right"} <= names
    left = m.geom("lb_wall_left").id
    assert m.geom_pos[left][0] + m.geom_size[left][0] == pytest.approx(ws.FAR_X)


def test_c920_looks_at_the_mat(scene):
    m, d = scene
    cam = m.camera("c920").id
    forward = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    to_mat = np.array([(ws.MAT_X[0] + ws.MAT_X[1]) / 2, 0, ws.MAT_TOP]) - d.cam_xpos[cam]
    assert math.degrees(math.acos(forward @ to_mat / np.linalg.norm(to_mat))) < 1.0
