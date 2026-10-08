import numpy as np

from slingpuck.sensing.camera_model import CameraModel, Frame
from slingpuck.sensing.kalman_tracker import KalmanTracker

DT = 0.001


def track_constant_velocity(latency, noise, dropout, duration=1.0, seed=0):
    p0, v = np.array([-0.05, 0.15]), np.array([0.1, -0.3])
    cam = CameraModel({"fps": 60, "latency_s": latency, "noise_std_m": noise, "dropout_prob": dropout},
                      rng=np.random.default_rng(seed))
    trk = KalmanTracker(noise, accel_noise_std_m_s2=1.0)
    last_meas = None
    for k in range(1, int(round(duration / DT)) + 1):
        t = k * DT
        for f in cam.observe(t, p0 + v * t):
            trk.update(f)
            last_meas = f.xy
    truth = p0 + v * t
    return trk, t, truth, v, last_meas


def test_not_initialized_before_first_frame():
    trk = KalmanTracker(0.003, 5.0)
    assert trk.estimate(0.0) is None
    trk.update(Frame(0.0, 0.05, np.array([0.1, 0.2])))
    pos, vel, _ = trk.estimate(0.0)
    np.testing.assert_allclose(pos, [0.1, 0.2])
    np.testing.assert_allclose(vel, [0.0, 0.0])


def test_converges_on_constant_velocity():
    trk, t, truth, v, _ = track_constant_velocity(latency=0.0, noise=0.003, dropout=0.0)
    pos, vel, _ = trk.estimate(t)
    # Bounds are ~3 sigma of the steady-state filter error for accel_noise_std = 1.
    assert np.linalg.norm(pos - truth) < 0.006
    assert np.linalg.norm(vel - v) < 0.1


def test_latency_compensation_beats_raw_measurement():
    latency = 0.05
    trk, t, truth, v, last_meas = track_constant_velocity(latency=latency, noise=0.003, dropout=0.0)
    pos, _, _ = trk.estimate(t)
    raw_error = np.linalg.norm(last_meas - truth)
    est_error = np.linalg.norm(pos - truth)
    assert raw_error > 0.8 * latency * np.linalg.norm(v)  # Raw frame lags by ~latency
    assert est_error < 0.25 * raw_error


def test_survives_dropouts():
    trk, t, truth, v, _ = track_constant_velocity(latency=0.05, noise=0.003, dropout=0.3, seed=3)
    pos, vel, _ = trk.estimate(t)
    assert np.linalg.norm(pos - truth) < 0.012
    assert np.linalg.norm(vel - v) < 0.12


def test_covariance_stays_symmetric_psd():
    trk, t, *_ = track_constant_velocity(latency=0.05, noise=0.003, dropout=0.1)
    _, _, P = trk.estimate(t)
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0)
