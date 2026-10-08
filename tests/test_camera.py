import numpy as np
import pytest

from slingpuck.sensing.camera_model import CameraModel

DT = 0.001


def cam_cfg(**kw):
    base = {"fps": 60, "latency_s": 0.05, "noise_std_m": 0.0, "dropout_prob": 0.0}
    base.update(kw)
    return base


def run(cam, duration, signal=lambda t: np.array([t, -t])):
    """Return a list of (t_received, frame)."""
    out = []
    for k in range(1, int(round(duration / DT)) + 1):
        t = k * DT
        out += [(t, f) for f in cam.observe(t, signal(t))]
    return out


def test_frame_rate():
    cam = CameraModel(cam_cfg(latency_s=0.0), rng=np.random.default_rng(0))
    frames = run(cam, 1.0)
    assert len(frames) == 61  # Captures at t = 0, 1/60, ..., 1.0 (both ends)
    gaps = np.diff([f.t_capture for _, f in frames])
    np.testing.assert_allclose(gaps, 1 / 60, atol=DT)


@pytest.mark.parametrize("latency", [0.0, 0.02, 0.05, 0.1])
def test_latency_shifts_signal(latency):
    cam = CameraModel(cam_cfg(latency_s=latency), rng=np.random.default_rng(0))
    frames = run(cam, 1.0)
    for t_recv, f in frames:
        # The value is the true signal at capture time, not at receive time.
        np.testing.assert_allclose(f.xy, [f.t_capture, -f.t_capture])
        # It arrives at the first step at or after capture + latency.
        assert latency - 1e-9 <= t_recv - f.t_capture < latency + DT


def test_each_frame_delivered_once():
    cam = CameraModel(cam_cfg(), rng=np.random.default_rng(0))
    frames = run(cam, 1.0)
    captures = [f.t_capture for _, f in frames]
    assert len(captures) == len(set(captures))


def test_dropout_rate():
    cam = CameraModel(cam_cfg(dropout_prob=0.2), rng=np.random.default_rng(1))
    frames = run(cam, 100.0, signal=lambda t: np.zeros(2))
    assert cam.n_captured == 6001
    assert cam.n_dropped / cam.n_captured == pytest.approx(0.2, abs=0.02)
    assert len(frames) + cam.n_dropped <= cam.n_captured


def test_noise_std():
    cam = CameraModel(cam_cfg(noise_std_m=0.003), rng=np.random.default_rng(2))
    frames = run(cam, 50.0, signal=lambda t: np.zeros(2))
    xy = np.array([f.xy for _, f in frames])
    assert xy.std() == pytest.approx(0.003, rel=0.05)
    assert np.abs(xy.mean()) < 3e-4


def test_seeded_reproducible():
    a = run(CameraModel(cam_cfg(noise_std_m=0.01, dropout_prob=0.1), rng=np.random.default_rng(7)), 1.0)
    b = run(CameraModel(cam_cfg(noise_std_m=0.01, dropout_prob=0.1), rng=np.random.default_rng(7)), 1.0)
    assert len(a) == len(b)
    for (_, fa), (_, fb) in zip(a, b):
        np.testing.assert_array_equal(fa.xy, fb.xy)
