from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from src.slingpuck.physics.mujoco_sling import HOME_XYZ, MujocoSlingSim  # noqa: E402
from src.slingpuck.sysid.fit import _until_gap  # noqa: E402
from src.slingpuck.twin.calibration import BoardCalibration, board_corners  # noqa: E402
from src.slingpuck.twin.detector import PuckDetector  # noqa: E402
from src.slingpuck.twin.scene import TwinScene, project  # noqa: E402
from src.slingpuck.twin.tracker import MultiPuckTracker  # noqa: E402

W, H = 640, 480


def puck(i, x, y, hidden=False):
    return SimpleNamespace(id=i, pos=np.array([x, y]), vel=np.zeros(2), hidden=hidden)


@pytest.fixture(scope="module")
def rig(cfg):
    scene = TwinScene(cfg, 6)
    sim = MujocoSlingSim(cfg)
    scene.set_arm(np.r_[sim.solve_ik(HOME_XYZ), sim.grasp.q_open])
    corners = project(scene.model, scene.data, "overhead", np.c_[board_corners(cfg), np.zeros(4)], W, H)
    calib = BoardCalibration.from_corners(cfg, corners, 0.6, (0.0, 0.0))
    return SimpleNamespace(scene=scene, sim=sim, calib=calib, renderer=mujoco.Renderer(scene.model, H, W),
                           detector=PuckDetector(cfg, calib))


def render(rig):
    rig.renderer.update_scene(rig.scene.data, camera="overhead")
    return rig.renderer.render()[:, :, ::-1].copy()


def test_calibration_round_trip(rig, cfg, tmp_path):
    xy = np.array([[0.05, -0.1], [-0.08, 0.12]])
    for h in (0.0, cfg["puck"]["thickness_m"]):
        np.testing.assert_allclose(rig.calib.pixel_to_board(rig.calib.board_to_pixel(xy, h), h), xy, atol=1e-9)
    rig.calib.save(tmp_path / "c.yaml")
    np.testing.assert_allclose(BoardCalibration.load(tmp_path / "c.yaml").H_px_to_board, rig.calib.H_px_to_board)


def test_detector_finds_pucks_within_a_millimetre(rig):
    truth = np.array([[0.03, rig.sim.band.band_y], [0.06, 0.05], [-0.06, 0.12], [0.07, -0.04]])
    rig.scene.set_pucks([puck(i, *p) for i, p in enumerate(truth)])
    dets = rig.detector.detect(render(rig))
    assert len(dets) == len(truth)
    for p in truth:
        assert min(np.linalg.norm(d.xy - p) for d in dets) < 1.5e-3


def test_touching_pucks_are_split(rig):
    truth = np.array([[0.0, 0.13], [0.026, 0.13]])
    rig.scene.set_pucks([puck(i, *p) for i, p in enumerate(truth)])
    dets = rig.detector.detect(render(rig))
    assert len(dets) == 2
    for p in truth:
        assert min(np.linalg.norm(d.xy - p) for d in dets) < 5e-3


def test_arm_mask_removes_gripper_detections(rig, cfg):
    held = np.array([0.03, rig.sim.band.band_y - 0.02])
    q = np.r_[rig.sim.solve_ik([*held, 0.5 * cfg["puck"]["thickness_m"]]), rig.sim.grasp.q_grasp]
    rig.scene.set_arm(q)
    rig.scene.set_pucks([puck(0, *held)])
    img = render(rig)
    assert rig.detector.detect(img, rig.scene.arm_mask(rig.calib, H, W)) == []
    np.testing.assert_allclose(rig.scene.grasp_xy(), held, atol=1e-3)  # Held position from the joints


def test_tracker_keeps_identities_through_hiding_and_crossing(cfg):
    trk = MultiPuckTracker(cfg, n_pucks=2)
    dt = 1 / 60
    ids_a = set()
    for k in range(60):
        t = k * dt
        a = np.array([-0.05 + 0.1 * t, 0.0])          # Moves right
        b = np.array([0.05 - 0.1 * t, 0.002])         # Moves left, crosses a
        dets = [a] if 20 <= k < 35 else [a, b]        # b hidden for 0.25 s
        if k == 40:
            dets.append(np.array([0.0, 0.1]))         # One false detection
        states = trk.update(t, dets)
        if k > 5:
            ids_a.add(min(states, key=lambda s: np.linalg.norm(s.pos - a)).id)
    assert len(trk.tracks) == 2 and len(ids_a) == 1


def test_hidden_puck_reappears_far_away_with_fixed_count(cfg):
    trk = MultiPuckTracker(cfg, n_pucks=2)
    for k in range(10):
        trk.update(k / 60, [np.array([0.0, -0.1]), np.array([0.05, 0.1])])
    first = {round(s.pos[1], 2): s.id for s in trk.states()}
    for k in range(10, 40):
        trk.update(k / 60, [np.array([0.05, 0.1])])  # The first puck is picked up
    states = trk.update(40 / 60, [np.array([-0.04, -0.12]), np.array([0.05, 0.1])])  # Put back elsewhere
    assert len(trk.tracks) == 2
    back = min(states, key=lambda s: np.linalg.norm(s.pos - [-0.04, -0.12]))
    assert back.id == first[-0.1]


def test_sysid_segment_stops_at_a_hidden_gap():
    t = np.array([0.0, 1, 2, 3, 4, 9, 10]) / 60
    assert _until_gap(list(range(7)), t) == [0, 1, 2, 3, 4]


def test_synthetic_session_end_to_end(tmp_path):
    from src.slingpuck.twin.replay import run
    from src.slingpuck.twin.synthetic import generate
    session = generate(tmp_path / "s", n_gate=1, n_divider=1, seed=3)
    summary = run(session, quiet=True)
    assert summary["tracks"] == 5
    shots = pd.read_csv(session / "robot_shots.csv")
    by_id = {s["shot_id"]: s for s in summary["shots"]}
    for _, s in shots.iterrows():
        m = by_id[s.shot_id]
        assert m["puck_id"] is not None and m["n_rest"] > 10 and m["n_pulled"] > 10 and m["n_flight"] > 10
        assert m["success"] == (abs(s.cmd_slide_m) < 0.01)  # Straight shot goes through, divider shot does not
    df = pd.read_csv(session / "twin" / "shots.csv")
    assert set(df.phase) == {"rest", "pulled", "flight"}


def test_detector_tells_black_from_white(rig):
    rig.scene.set_arm(np.r_[rig.sim.solve_ik(HOME_XYZ), rig.sim.grasp.q_open])  # Arm away from the board
    truth = [(0.03, rig.sim.band.band_y, "black"), (-0.06, -0.04, "black"), (0.06, 0.05, "white"),
             (-0.06, 0.12, "white"), (0.0, 0.0, "white"), (0.07, -0.09, "white")]
    rig.scene.set_pucks([SimpleNamespace(id=i, pos=np.array(p[:2]), hidden=False, colour=p[2])
                         for i, p in enumerate(truth)])
    dets = rig.detector.detect(render(rig))
    assert len(dets) == len(truth)
    for x, y, colour in truth:
        d = min(dets, key=lambda d: np.linalg.norm(d.xy - [x, y]))
        assert d.colour == colour and np.linalg.norm(d.xy - [x, y]) < 2e-3


def test_tracker_never_swaps_colours(cfg):
    trk = MultiPuckTracker(cfg, n_pucks=2)
    for k in range(60):
        t = k / 60
        black = np.array([-0.05 + 0.1 * t, 0.0])
        white = np.array([0.05 - 0.1 * t, 0.0])   # Same line: the two pucks meet head-on at t = 0.5 s
        states = trk.update(t, [black, white], ["black", "white"])
        if k > 5:
            by_colour = {s.colour: s for s in states}
            assert np.linalg.norm(by_colour["black"].pos - black) < 0.01
