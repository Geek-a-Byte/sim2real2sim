"""Overhead camera (sensing/): board homography, puck detection on synthetic camera frames, tracker, snapshot."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.slingpuck.sensing.board_view import BoardView, apply_h, corner_board_xy, fit_homography  # noqa: E402
from src.slingpuck.sensing.puck_detector import Detection, PuckDetector  # noqa: E402
from src.slingpuck.sensing.puck_tracker import PuckTracker  # noqa: E402

WOOD = (160, 195, 220)       # BGR, light yellow wood
BLACK = (28, 28, 28)
WHITE = (245, 245, 245)
CANVAS_PX_PER_MM = 4.0

# Board (x, y) in m and colour of the pucks in the synthetic scene. Touching pair at the robot end.
PUCKS = [((0.000, -0.060), "black"), ((-0.060, -0.120), "black"), ((-0.0346, -0.120), "black"),
         ((0.070, -0.030), "white"), ((0.020, 0.080), "white"), ((-0.050, 0.120), "black")]


def render_canvas(cfg, pucks=PUCKS, radius_scale=1.0):
    """Top-down image of the play area at CANVAS_PX_PER_MM (top = opponent end), with printed lines and a graphic."""
    w, l = cfg["board"]["width_m"], cfg["board"]["length_m"]
    s = 1000 * CANVAS_PX_PER_MM
    img = np.zeros((int(round(l * s)), int(round(w * s)), 3), np.uint8)
    img[:] = WOOD

    def px(xy):
        return int(round((xy[0] + w / 2) * s)), int(round((l / 2 - xy[1]) * s))

    for x in (-0.03, 0.03):                                   # Printed lines along the board
        cv2.line(img, px((x, -l / 2)), px((x, l / 2)), BLACK, 6)
    cv2.circle(img, px((0.06, 0.03)), int(0.012 * s), BLACK, 8)   # A printed ring: not a solid puck
    cv2.line(img, px((0.045, 0.03)), px((0.075, 0.03)), BLACK, 8)
    g = np.array([-0.06, 0.03])                                   # A printed "glove": dark disk, light notches at the rim
    cv2.circle(img, px(g), int(0.0115 * s), BLACK, -1)
    for a in (0.3, 2.0, 4.2):
        cv2.circle(img, px(g + 0.0115 * np.array([np.cos(a), np.sin(a)])), int(0.0035 * s), WOOD, -1)
    rng = np.random.default_rng(0)
    for xy, colour in pucks:
        r = int(round(radius_scale * cfg["puck"]["radius_m"] * s))
        cv2.circle(img, px(xy), r, BLACK if colour == "black" else WHITE, -1, cv2.LINE_AA)
        if colour == "white":
            cv2.circle(img, px(xy), r, (200, 200, 200), 3, cv2.LINE_AA)    # Shaded edge
    img = np.clip(img.astype(int) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


def to_camera(canvas, corners_cam, size=(960, 720)):
    """Warp the top-down canvas into a camera frame whose play-area corners land at corners_cam (CORNER_ORDER)."""
    h, w = canvas.shape[:2]
    canvas_corners = np.array([[0, h], [w, h], [w, 0], [0, 0]], float)    # CORNER_ORDER in canvas pixels
    H = cv2.getPerspectiveTransform(canvas_corners.astype(np.float32), np.asarray(corners_cam, np.float32))
    frame = np.full((size[1], size[0], 3), 90, np.uint8)          # Table around the board
    return cv2.warpPerspective(canvas, H, size, dst=frame, borderMode=cv2.BORDER_TRANSPARENT)


# Camera corner pixels: a tilted view with the robot end at the bottom, and a view turned so the robot is at the left.
TILTED = [[260, 640], [700, 630], [640, 90], [320, 100]]
TURNED = [[120, 160], [120, 560], [820, 600], [820, 120]]


def detector(cfg, corners, size=(960, 720), **kw):
    view = BoardView(corners, cfg["board"]["width_m"], cfg["board"]["length_m"], image_size=size, **kw)
    return view, PuckDetector(view, cfg["puck"]["radius_m"], 0.0)


def match(dets, truth):
    """For each true puck: distance (m) to the nearest detection and its colour."""
    out = []
    for xy, colour in truth:
        d = min(dets, key=lambda e: np.hypot(*(e.xy - xy)))
        out.append((float(np.hypot(*(d.xy - xy))), d.colour == colour))
    return out


def test_homography_matches_opencv_and_round_trips(cfg):
    view, _ = detector(cfg, TILTED)
    board = corner_board_xy(cfg["board"]["width_m"], cfg["board"]["length_m"])
    ref = cv2.getPerspectiveTransform(np.float32(TILTED), np.float32(board))
    assert view.H_img_to_board == pytest.approx(ref / ref[2, 2], rel=1e-6, abs=1e-9)
    pts = np.random.default_rng(1).uniform([300, 120], [650, 600], (20, 2))
    assert view.board_to_img(view.img_to_board(pts)) == pytest.approx(pts, abs=1e-6)
    assert view.img_to_board(np.array(TILTED, float)) == pytest.approx(board, abs=1e-9)
    # More than 4 pairs: least squares gives the exact map back
    H = fit_homography(TILTED, board)
    extra = apply_h(H, pts)
    assert fit_homography(np.vstack([TILTED, pts]), np.vstack([board, extra])) == pytest.approx(H, rel=1e-6, abs=1e-9)


def test_rectified_view_puts_the_robot_end_at_the_bottom(cfg):
    view, _ = detector(cfg, TURNED)
    l = cfg["board"]["length_m"]
    robot_end = view.board_to_rect([0.0, -l / 2])
    opponent_end = view.board_to_rect([0.0, l / 2])
    right = view.board_to_rect([0.05, 0.0])
    assert robot_end[1] > opponent_end[1]                 # Image v grows downward
    assert right[0] > view.board_to_rect([0.0, 0.0])[0]   # Robot's right is image right
    assert view.rect_size[0] < view.rect_size[1]          # Tall: the board length is vertical


@pytest.mark.parametrize("corners", [TILTED, TURNED], ids=["tilted", "turned"])
def test_detects_every_puck_within_2mm_with_its_colour(cfg, corners):
    frame = to_camera(render_canvas(cfg), corners)
    view, det = detector(cfg, corners)
    dets = det.detect(frame)
    assert len(dets) == len(PUCKS), [(d.colour, np.round(1000 * d.xy)) for d in dets]   # No ring, no lines
    for err, colour_ok in match(dets, PUCKS):
        assert err < 0.002 and colour_ok
    assert sum(d.side == "robot" for d in dets) == sum(xy[1] < 0 for xy, _ in PUCKS)


def test_no_pucks_no_detections(cfg):
    frame = to_camera(render_canvas(cfg, pucks=[]), TILTED)
    _, det = detector(cfg, TILTED)
    assert det.detect(frame) == []


def test_parallax_correction(cfg):
    view, _ = detector(cfg, TILTED, camera_height_m=0.6, camera_xy_m=(0.01, -0.02))
    assert view.correct_parallax([0.01, -0.02], 0.0102) == pytest.approx([0.01, -0.02])     # Under the lens
    seen = np.array([0.11, -0.12])
    true = view.correct_parallax(seen, 0.0102)
    # The ray from the lens through the true top-face center hits the surface at the seen point
    lens = np.array([0.01, -0.02, 0.6])
    top = np.array([*true, 0.0102])
    hit = lens + (top - lens) * (lens[2] / (lens[2] - top[2]))
    assert hit[:2] == pytest.approx(seen, abs=1e-12)
    no_cam, _ = detector(cfg, TILTED)
    assert no_cam.correct_parallax(seen, 0.0102) == pytest.approx(seen)


def test_bad_settings_and_corners_rejected(cfg):
    view, _ = detector(cfg, TILTED)
    with pytest.raises(KeyError):
        PuckDetector(view, 0.0127, params={"hough_parm2": 20})
    with pytest.raises(ValueError):
        BoardView([[0, 0], [1, 1], [2, 2], [3, 3]], 0.2, 0.33)


def test_calibration_file_round_trip(cfg, tmp_path):
    view, _ = detector(cfg, TILTED, camera_height_m=0.55, camera_xy_m=(0.0, 0.0))
    path = view.save(tmp_path / "cam.yaml", detect={"hough_param2": 22}, source="test")
    back = BoardView.from_config(cfg, path)
    assert back.H_img_to_board == pytest.approx(view.H_img_to_board, rel=1e-4, abs=1e-6)
    assert back.camera_height_m == 0.55 and back.image_size == (960, 720)
    assert PuckDetector.from_config(cfg, back).p["hough_param2"] == 22


def det(xy, colour="black"):
    return Detection(np.asarray(xy, float), colour, 0.0127, np.zeros(2), 0.0, 0.0)


def test_tracker_follows_a_fast_puck_and_keeps_ids():
    tr = PuckTracker(meas_noise_std_m=0.001, max_speed_m_s=3.5)
    rng = np.random.default_rng(0)
    v = np.array([0.4, -2.5])                    # An incoming shot, 2.5 m/s toward the robot
    rest = np.array([-0.05, -0.12])
    for k in range(15):
        t = k / 60.0
        p = np.array([0.0, 0.14]) + v * t
        tr.update(t, [det(p + rng.normal(0, 0.001, 2), "white"), det(rest + rng.normal(0, 0.001, 2))])
    states = tr.estimate()
    assert len(states) == 2 and len(tr.tracks) == 2               # One track per puck the whole time
    moving = max(states, key=lambda s: s.speed)
    still = min(states, key=lambda s: s.speed)
    assert moving.vel == pytest.approx(v, abs=0.15) and moving.colour == "white"
    # 1 mm camera noise at 60 fps: a resting puck shows about 0.1 m/s, far below a shot
    assert still.speed < 0.2 and still.xy == pytest.approx(rest, abs=0.002) and still.side == "robot"
    ahead = tr.estimate(tr.t + 0.05)                               # Latency compensation
    assert max(ahead, key=lambda s: s.speed).xy == pytest.approx(moving.xy + 0.05 * moving.vel, abs=1e-9)


def test_tracker_drops_lost_pucks_and_needs_two_hits():
    tr = PuckTracker(max_missed_s=0.1)
    tr.update(0.0, [det([0.0, -0.1])])
    assert tr.estimate() == [] and len(tr.estimate(confirmed=False)) == 1
    tr.update(0.03, [det([0.0, -0.1])])
    assert len(tr.estimate()) == 1
    tr.update(0.2, [])
    assert tr.tracks == []


def test_snapshot_rejects_a_moving_puck(cfg, tmp_path, monkeypatch):
    from src.slingpuck.sensing import overhead_camera as oc

    view, _ = detector(cfg, TILTED)
    calib = view.save(tmp_path / "cam.yaml")
    frames = [to_camera(render_canvas(cfg, [((0.0, -0.06), "black"), ((-0.08 + 0.016 * k, 0.10), "white")]), TILTED)
              for k in range(10)]                                   # The white puck slides 16 mm per frame

    class Frames:
        def __init__(self, *_a, **_k):
            self.k, self.cap, self.is_file = 0, None, True

        def read(self):
            self.k += 1
            return self.k <= len(frames), frames[min(self.k, len(frames)) - 1], self.k / 30

        def release(self):
            pass

    monkeypatch.setattr(oc, "FrameSource", Frames)
    pucks = oc.locate_pucks("fake", frames=10, calib=calib, cfg=cfg)
    assert [p["colour"] for p in pucks] == ["black"]
    assert pucks[0]["xy"] == pytest.approx([0.0, -0.06], abs=0.002)


def test_snapshot_from_an_image_file(cfg, tmp_path):
    from src.slingpuck.deploy.record_trajectory import choose_puck, guard_xy
    from src.slingpuck.sensing.overhead_camera import locate_pucks, robot_half

    frame = to_camera(render_canvas(cfg), TILTED)
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), frame)
    view, _ = detector(cfg, TILTED)
    calib = view.save(tmp_path / "cam.yaml")
    pucks = locate_pucks(str(img), frames=5, calib=calib, cfg=cfg)
    assert len(pucks) == len(PUCKS) and all(p["seen"] == 1.0 for p in pucks)
    mine = robot_half(pucks, cfg)
    assert len(mine) == sum(xy[1] < 0 for xy, _ in PUCKS)
    xy = np.array([p["xy"] for p in mine])
    i, shot, how = choose_puck(cfg, xy, "nearest")
    assert how == "nearest" and shot is None
    assert xy[i] == pytest.approx(PUCKS[0][0], abs=0.002)            # (0, -6 cm) is nearest to the guard pose
    assert np.linalg.norm(xy[i] - guard_xy(cfg)) == pytest.approx(np.min(np.linalg.norm(xy - guard_xy(cfg), axis=1)))
    assert choose_puck(cfg, xy, "2")[0] == 2
    with pytest.raises(IndexError):
        choose_puck(cfg, xy, str(len(xy)))


def test_greedy_pick_uses_the_shot_selector(greedy_cfg, shot_model):
    from src.slingpuck.deploy.record_trajectory import choose_puck, guard_xy
    from src.slingpuck.primitives.sling import ShotSelector

    xy = np.array([[0.0, -0.06], [0.06, -0.12], [-0.09, -0.04]])
    i, shot, how = choose_puck(greedy_cfg, xy, "greedy", model=shot_model)
    ref = ShotSelector(greedy_cfg, shot_model).choose(xy, guard_xy(greedy_cfg))
    assert i == ref.index and shot == pytest.approx(ref.shot) and how.startswith("greedy")


def test_greedy_pick_checks_the_model_against_the_unmoved_board(greedy_cfg, shot_model, monkeypatch):
    """setup_config() moves the sim board to the real surface height. The shot model must still load."""
    import copy

    from src.slingpuck.deploy import record_trajectory as rt
    from src.slingpuck.shot import model as sm

    seen = {}

    def fake_load_for(cfg):
        seen["outer_height_m"] = cfg["board"]["outer_height_m"]
        return shot_model

    monkeypatch.setattr(sm, "load_for", fake_load_for)
    moved = copy.deepcopy(greedy_cfg)
    moved["board"]["outer_height_m"] += 0.03                      # As setup_config() with a surface height and hover
    i, shot, how = rt.choose_puck(moved, np.array([[0.0, -0.06], [0.05, -0.1]]), "greedy")
    assert how.startswith("greedy") and shot is not None
    assert seen["outer_height_m"] == pytest.approx(greedy_cfg["board"]["outer_height_m"])
