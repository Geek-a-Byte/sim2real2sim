import copy

import numpy as np
import pytest
from scipy.integrate import quad

from src.slingpuck.physics.arm_deflection import pull_direction, solve_pull
from src.slingpuck.physics.band_model import BandModel, BandParams


def band(exponent=1.0, loss=0.15, transfer=0.7, k=200.0, natural=0.15):
    return BandModel(BandParams(0.1854, natural, k, exponent, loss, transfer), band_y=-0.1)


@pytest.mark.parametrize("exponent", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("loss", [0.0, 0.15, 0.4])
def test_hysteresis_loop_closes_and_loses_the_set_fraction(exponent, loss):
    b = band(exponent, loss)
    s0, s1 = b.s_rest, b.s_rest + 0.01
    # Closed loop: unloading meets loading at both ends.
    assert b.tension_unload(s0, s1) == pytest.approx(b.tension_load(s0))
    assert b.tension_unload(s1, s1) == pytest.approx(b.tension_load(s1))
    # Unloading is never above loading.
    s = np.linspace(s0, s1, 50)
    assert np.all(b.tension_unload(s, s1) <= b.tension_load(s) + 1e-12)
    w_load = quad(lambda x: float(b.tension_load(x)), s0, s1)[0]
    w_unload = quad(lambda x: float(b.tension_unload(x, s1)), s0, s1)[0]
    assert w_load == pytest.approx(b.work_load(s0, s1), rel=1e-6)
    assert 1.0 - w_unload / w_load == pytest.approx(loss, abs=1e-6)


def test_tension_monotonic_and_pretension():
    b = band(exponent=1.5)
    s = np.linspace(0, 0.06, 100)
    assert np.all(np.diff(b.tension_load(s)) >= 0)
    assert b.s_rest == pytest.approx(0.1854 - 0.15)
    assert b.tension_load(b.s_rest) > 0  # Pre-stretched band


def test_force_points_forward_and_is_zero_on_the_band_line():
    b = band()
    np.testing.assert_allclose(b.force_load((0.0, b.band_y)), [0.0, 0.0], atol=1e-12)
    f = b.force_load((0.0, b.band_y - 0.02))
    assert f[1] > 0 and abs(f[0]) < 1e-12


def test_release_energy_balance():
    b = band(loss=0.2, transfer=1.0)
    pos = (0.0, b.band_y - 0.025)
    s_max = b.elongation(pos)
    r = b.release(pos, mass=0.027, mu=0.0, dt=2e-5)
    expected = (1 - 0.2) * b.work_load(b.s_rest, s_max)
    assert r.band_work_j == pytest.approx(expected, rel=0.01)
    assert 0.5 * 0.027 * float(r.exit_vel @ r.exit_vel) == pytest.approx(expected, rel=0.01)
    assert r.exit_pos[1] == pytest.approx(b.band_y)


def test_energy_transfer_scales_exit_speed():
    pos = (0.0, -0.125)
    v1 = np.linalg.norm(band(transfer=1.0).release(pos, 0.027, 0.2).exit_vel)
    v2 = np.linalg.norm(band(transfer=0.5).release(pos, 0.027, 0.2).exit_vel)
    assert v2 / v1 == pytest.approx(np.sqrt(0.5))


def test_angled_pull_launches_almost_straight():
    """Documents the main Phase 2 finding: the pull angle barely steers the puck."""
    b = band()
    straight = b.release((0.0, b.band_y - 0.025), 0.027, 0.2).exit_vel
    p = np.array([0.0, b.band_y]) + 0.025 * pull_direction(np.deg2rad(30))
    angled = b.release(p, 0.027, 0.2).exit_vel
    assert abs(np.degrees(np.arctan2(straight[0], straight[1]))) < 1e-6
    assert abs(np.degrees(np.arctan2(angled[0], angled[1]))) < 3.0


def test_deflection_matches_compliance_times_force():
    b = band()
    c = 0.002
    r = solve_pull(b, (0.0, b.band_y), 0.0, 0.03, c)
    resisting = -float(r.force @ pull_direction(0.0))
    assert r.pull_m + c * resisting == pytest.approx(0.03, abs=1e-7)
    assert r.pull_m < 0.03
    assert solve_pull(b, (0.0, b.band_y), 0.0, 0.03, 0.0).pull_m == pytest.approx(0.03)


def test_config_band(cfg):
    b = BandModel.from_config(cfg)
    assert b.band_y == pytest.approx(-0.5 * cfg["board"]["length_m"] + cfg["band"]["band_offset_from_end_wall_m"])
    bad = copy.deepcopy(cfg)
    bad["band"]["energy_transfer"] = 0.0
    with pytest.raises(ValueError):
        BandModel.from_config(bad)


def test_tiny_pull_does_not_launch():
    b = band()
    # Band force at 0.2 mm depth is below sliding friction (mu * m * g ~ 0.05 N).
    r = b.release((0.0, b.band_y - 0.0002), mass=0.027, mu=0.2)
    assert not r.launched and np.all(r.exit_vel == 0)
    assert b.release((0.0, b.band_y - 0.02), mass=0.027, mu=0.2).launched
