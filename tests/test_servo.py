import numpy as np
import pytest

from slingpuck.physics.servo_model import ServoModel

DT = 0.01


def servo_cfg(**kw):
    base = {"max_rate_deg_s": 1e6, "lag_tau_s": 0.0, "deadband_deg": 0.0, "command_latency_s": 0.0}
    base.update(kw)
    return base


@pytest.mark.parametrize("latency_s", [0.0, 0.01, 0.03, 0.05])
def test_latency_shifts_command_by_exact_steps(latency_s):
    servo = ServoModel(servo_cfg(command_latency_s=latency_s), DT)
    n = int(round(latency_s / DT))
    trace = [float(servo.step(1.0)) for _ in range(n + 3)]
    # A command sent at step 1 must first move the servo at step n + 1.
    assert trace[:n] == [0.0] * n
    assert trace[n] == pytest.approx(1.0)


def test_latency_shift_on_arbitrary_signal():
    servo = ServoModel(servo_cfg(command_latency_s=0.04), DT)
    cmd = np.sin(np.arange(50) * 0.3)
    out = np.array([float(servo.step(c)) for c in cmd])
    np.testing.assert_allclose(out[4:], cmd[:-4])
    np.testing.assert_allclose(out[:4], 0.0)


def test_rate_limit():
    servo = ServoModel(servo_cfg(max_rate_deg_s=300.0), DT)
    step = np.deg2rad(300.0) * DT
    for k in range(1, 6):
        assert float(servo.step(10.0)) == pytest.approx(k * step)
    assert float(servo.vel) == pytest.approx(np.deg2rad(300.0))


def test_first_order_lag_reaches_63_percent_at_tau():
    tau = 0.1
    servo = ServoModel(servo_cfg(lag_tau_s=tau), 0.001)
    for _ in range(100):
        pos = float(servo.step(1.0))
    assert pos == pytest.approx(1.0 - np.exp(-1.0), abs=1e-6)


def test_deadband_ignores_small_errors():
    servo = ServoModel(servo_cfg(deadband_deg=1.0), DT)
    for _ in range(10):
        servo.step(np.deg2rad(0.9))
    assert float(servo.pos) == 0.0
    servo.step(np.deg2rad(1.1))
    assert float(servo.pos) == pytest.approx(np.deg2rad(1.1))


def test_vector_joints_independent():
    servo = ServoModel(servo_cfg(max_rate_deg_s=300.0), DT, init_pos=np.zeros(3))
    pos = servo.step(np.array([0.001, -10.0, 10.0]))
    max_step = np.deg2rad(300.0) * DT
    np.testing.assert_allclose(pos, [0.001, -max_step, max_step])


def test_config_servo_loads(cfg):
    servo = ServoModel(cfg["servo"], 0.001)
    assert servo.latency_steps == 30
    assert servo.max_rate == pytest.approx(np.deg2rad(300.0))
