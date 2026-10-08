import copy

import numpy as np
import pytest

from src.slingpuck.kinematics import GoalkeeperGeometry


def test_action_mapping_range_and_inverse(cfg):
    g = GoalkeeperGeometry.from_config(cfg)
    assert g.pan_max == pytest.approx(np.deg2rad(cfg["goalkeeper"]["pan_range_deg"]))
    assert g.action_to_pan([1.0]) == pytest.approx(g.pan_max)
    assert g.action_to_pan(np.array([-3.0])) == pytest.approx(-g.pan_max)  # Clipped
    for a in np.linspace(-1, 1, 9):
        assert g.pan_to_action(g.action_to_pan([a])) == pytest.approx(a)
    assert g.joint_to_pan(g.pan_to_joint(0.1)) == pytest.approx(0.1)


def test_paddle_geometry(cfg):
    g = GoalkeeperGeometry.from_config(cfg)
    face_y = g.paddle_center(0.0)[1] + g.paddle_half_thickness
    expected = -(0.5 * cfg["board"]["divider_thickness_m"] + cfg["goalkeeper"]["paddle_standoff_m"])
    assert face_y == pytest.approx(expected)
    s = g.paddle_state(0.2, 1.5, 0.5)
    # Center velocity is tangent to the arc, magnitude R * pan_vel.
    radial = s.center - g.base_xy
    assert abs(radial @ s.vel) < 1e-12
    assert np.linalg.norm(s.vel) == pytest.approx(g.arm_radius * 1.5)
    assert s.angle == pytest.approx(-0.2)


def test_rejects_paddle_outside_board(cfg):
    bad = copy.deepcopy(cfg)
    bad["goalkeeper"]["pan_range_deg"] = 40.0
    with pytest.raises(ValueError, match="leaves the board"):
        GoalkeeperGeometry.from_config(bad)


def test_robot_stub_uses_shared_mapping():
    import inspect

    from src.slingpuck.deploy import lerobot_interface
    src = inspect.getsource(lerobot_interface.SO101Controller.write_action)
    assert "self.geom.pan_to_joint(self.geom.action_to_pan(" in src
