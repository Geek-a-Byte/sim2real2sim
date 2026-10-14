import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.envs.sling_env import launch_and_fly  # noqa: E402
from src.slingpuck.physics.mujoco_sling import MujocoSlingSim, ShotPlan, grasp_geometry  # noqa: E402


@pytest.fixture(scope="module")
def sim(cfg):
    return MujocoSlingSim(cfg)


def test_grasp_opening_fits_the_puck(cfg):
    g = grasp_geometry(2.0 * cfg["puck"]["radius_m"])
    assert g.q_grasp < g.q_open


@pytest.mark.parametrize("xy", [(0.0, 0.0), (0.08, 0.0), (-0.08, -0.03), (0.06, -0.03)])
def test_ik_reaches_band_and_pull_positions_gripper_down(sim, xy):
    """xy is relative to the band center: (x, pull back in -y)."""
    target = np.array([xy[0], sim.band.band_y + xy[1], sim.z_grasp])
    q = sim.solve_ik(target)
    d = sim._ik_data
    d.qpos[sim._arm_qadr] = q
    mujoco.mj_kinematics(sim.model, d)
    np.testing.assert_allclose(d.site_xpos[sim._site], target, atol=1e-4)
    R = d.site_xmat[sim._site].reshape(3, 3)
    assert R[2, 0] < -0.999  # Approach axis points down
    assert abs(R[1, 2]) < 0.01  # Jaws close along x


def test_corner_contact_normal_is_radial(sim, cfg):
    """Puck hits the right divider corner off center: it must bounce back, not slide through."""
    sim.apply_params(cfg)
    m, d = sim.model, sim.data
    mujoco.mj_resetData(m, d)
    r = cfg["puck"]["radius_m"]
    x = 0.5 * cfg["board"]["gate_width_m"] - 0.6 * r  # Corner 0.6 r to the side of the center: ~37 deg normal
    d.qpos[sim._puck_qadr] = [x, -0.03, 0.0]
    d.qvel[sim._puck_vadr] = [0.0, 1.67, 0.0]
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert d.qvel[sim._puck_vadr[1]] < 0.0
    assert d.qpos[sim._puck_qadr[1]] < 0.0


@pytest.mark.parametrize("x0,pull_x,depth", [(0.03, 0.0, 0.024), (-0.05, 0.01, 0.028)])
def test_shot_matches_2d_model(sim, cfg, x0, pull_x, depth):
    sim.apply_params(cfg)
    pull = np.array([pull_x, sim.band.band_y - depth])
    res = sim.run(ShotPlan(x0, pull_x, pull))
    ref = launch_and_fly(cfg, sim.band, res.release_pos, None)
    # The arm holds the puck at the target under band load.
    assert np.linalg.norm(res.release_pos - pull) < 1e-3
    assert res.max_band_force_n > 1.0
    assert res.launched
    np.testing.assert_allclose(np.linalg.norm(res.exit_vel), np.linalg.norm(ref.exit_vel), rtol=0.02)
    assert res.success == ref.success
    assert res.miss_m == pytest.approx(ref.miss_m, abs=1e-3)


def test_puck_follows_the_grasp_while_held(sim, cfg):
    sim.apply_params(cfg)
    errors = []

    def on_frame(s):
        if s.held:
            errors.append(np.linalg.norm(s.site_pos()[:2] - s.puck_pos))

    sim.run(ShotPlan(0.0, 0.0, np.array([0.0, sim.band.band_y - 0.02])), on_frame, frame_dt=0.02)
    assert errors and max(errors) < 1e-6
