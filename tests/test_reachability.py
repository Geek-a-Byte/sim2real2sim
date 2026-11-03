import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.slingpuck.general.common import ObjectShape  # noqa: E402
from src.slingpuck.general.reachability import (SEED_POSES, ReachabilityChecker, SO101Kinematics,  # noqa: E402
                                                 approach_dir, tangential_dir)


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


def test_ik_recovers_poses_from_forward_kinematics(kin):
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(20):
        q = rng.uniform(0.6 * kin.lo, 0.6 * kin.hi)
        p, R = kin.fk(q)
        solved += any(kin.ik(p, R[:, 0], R[:, 2], q0=q0)[0] for q0 in SEED_POSES)
    assert solved >= 18


def test_ik_stays_inside_joint_limits(kin):
    ok, q, *_ = kin.ik((0.6, 0.0, 0.02), approach_dir((0.6, 0, 0), 0), None)
    assert not ok and np.all(q >= kin.lo - 1e-9) and np.all(q <= kin.hi + 1e-9)


def test_gripper_opening_is_monotonic(kin):
    gaps = [kin.jaw_gap(q) for q in np.linspace(*kin.gripper_range, 8)]
    assert np.all(np.diff(gaps) > 0) and kin.max_opening == pytest.approx(gaps[-1])
    assert kin.jaw_gap(kin.gripper_for_opening(0.04)) == pytest.approx(0.04, abs=1e-4)


def test_approach_directions_are_in_the_radial_plane():
    p = (0.2, 0.2, 0.05)
    for tilt in (0, 30, 60, 90):
        a = approach_dir(p, tilt)
        assert np.linalg.norm(a) == pytest.approx(1.0)
        assert a @ tangential_dir(p) == pytest.approx(0.0, abs=1e-12)   # 5-DoF: no sideways component
    assert approach_dir(p, 0) == pytest.approx([0, 0, -1])


def test_checker_accepts_near_goals_and_names_the_reason_for_others(gcfg, shapes):
    ch = ReachabilityChecker(gcfg)
    cube = shapes["cube"]
    assert ch.check(cube, (0.25, 0.0, 0.015), 0.3).ok
    assert ch.check(cube, (0.6, 0.0, 0.015)).reason == "ik"           # Too far
    assert ch.check(cube, (-0.1, 0.0, 0.015)).reason == "ik"          # Behind the base
    wide = ObjectShape("wide", "box", (0.08, 0.08, 0.02), 0.1)
    assert ch.check(wide, (0.25, 0.0, 0.02)).reason == "gripper_width"


def test_top_down_grasp_closes_across_the_narrow_side(gcfg, shapes):
    ch = ReachabilityChecker(gcfg)
    bar = shapes["bar"]                       # 7 cm long, 2.4 cm wide
    r = ch.check(bar, (0.25, 0.0, 0.012), 0.0)
    assert r.ok and r.tilt_deg == 0
    assert abs(r.jaw @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)


def test_grasp_pose_has_no_collision(gcfg, shapes):
    ch = ReachabilityChecker(gcfg)
    r = ch.check(shapes["cube"], (0.25, 0.05, 0.015), 0.0)
    assert r.ok and not ch.kin.in_collision(r.q, ch.kin.gripper_for_opening(0.04))


def test_map_agrees_with_the_exact_check(gcfg, shapes, reach_map):
    exact, fast = ReachabilityChecker(gcfg), ReachabilityChecker(gcfg, reach_map)
    rng = np.random.default_rng(1)
    lo, hi = np.array(gcfg["reach"]["workspace_min_m"]), np.array(gcfg["reach"]["workspace_max_m"])
    agree = 0
    pts = rng.uniform(lo, hi, (30, 3))
    for p in pts:
        agree += exact.check(shapes["cube"], p).ok == fast.check(shapes["cube"], p, exact=False).ok
    assert agree >= 22        # Coarse 8 cm map in the test; the real map uses 2 cm voxels
    assert not fast.check(shapes["cube"], (0.8, 0.0, 0.02), exact=False).ok


def test_map_save_and_load(tmp_path, reach_map, gcfg):
    from src.slingpuck.general.reachability import ReachabilityMap

    reach_map.save(tmp_path / "m.npz")
    loaded = ReachabilityMap.load(tmp_path / "m.npz")
    assert np.array_equal(loaded.grid, reach_map.grid) and loaded.matches(gcfg)
