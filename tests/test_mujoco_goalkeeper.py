import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv  # noqa: E402
from src.slingpuck.envs.mujoco_goalkeeper_env import MujocoGoalkeeperEnv  # noqa: E402
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper  # noqa: E402
from src.slingpuck.physics.mujoco_goalkeeper import MujocoGoalkeeperSim  # noqa: E402
from src.slingpuck.policies.scripted import CenterBlocker  # noqa: E402

DT = 0.001


@pytest.fixture(scope="module")
def sim(cfg):
    return MujocoGoalkeeperSim(cfg)


@pytest.mark.parametrize("pan", [0.0, 0.2, -0.3])
def test_paddle_on_the_2d_arc(sim, pan):
    sim.reset_arm(pan)
    np.testing.assert_allclose(sim.paddle_center()[:2], sim.layout.geom.paddle_center(pan), atol=2e-4)
    assert sim.pose.settled_error_m < 2e-4


def test_pan_sign_matches_2d_convention(sim):
    sim.reset_arm(0.0)
    sim.set_pan_target(0.2)
    for _ in range(300):
        sim.step(DT)
    assert sim.pan == pytest.approx(0.2, abs=0.01)
    assert sim.paddle_center()[0] > 0.03  # Positive pan moves the paddle toward +x


def test_friction_matches_coulomb(sim, cfg):
    mu = cfg["puck"]["friction_kinetic"]
    sim.reset_arm(0.0)
    sim.reset((-0.07, 0.10), (0.6, 0.0))
    for _ in range(600):
        sim.step(DT)
    assert np.all(sim.vel == 0.0)
    stop_x = -0.07 + 0.6**2 / (2 * mu * 9.81)
    assert sim.pos[0, 0] == pytest.approx(stop_x, abs=1e-3)


def bounce(sim, pos, vel, axis):
    sim.reset(pos, vel)
    v_in = vel[axis]
    for _ in range(250):
        sim.step(DT)
        v = sim.data.qvel[sim._puck_vadr[axis]]
        if np.sign(v) != np.sign(v_in):
            for _ in range(12):
                sim.step(DT)
            return -sim.data.qvel[sim._puck_vadr[axis]] / v_in
    raise AssertionError("no bounce")


def test_wall_restitution_calibrated(sim, cfg):
    e, mu = cfg["board"]["restitution"], cfg["puck"]["friction_kinetic"]
    speed = 2.0
    measured = np.mean([bounce(sim, (0.05 + d, 0.10), (speed, 0.0), 0) for d in np.linspace(0, 0.003, 5)])
    expected = e - 2 * mu * 9.81 * 0.016 / speed  # Friction during contact and the 12 ms after
    assert measured == pytest.approx(expected, abs=0.02)


def test_paddle_restitution_calibrated(sim, cfg):
    sim.reset_arm(0.0)
    measured = np.mean([bounce(sim, (0.0, 0.06 - d), (0.0, -2.0), 1) for d in np.linspace(0, 0.003, 5)])
    assert measured == pytest.approx(cfg["board"]["paddle_restitution"], abs=0.06)


def test_gate_crossing_and_divider(sim):
    sim.reset_arm(0.35)  # Paddle out of the way
    sim.reset((0.0, 0.08), (0.0, -1.5))
    crossings = []
    for _ in range(150):
        crossings += sim.step(DT).crossings
    assert [c.direction for c in crossings] == [-1]
    sim.reset((0.05, 0.08), (0.0, -1.5))  # Hits the divider
    crossings = []
    for _ in range(150):
        crossings += sim.step(DT).crossings
    assert crossings == [] and sim.pos[0, 1] > 0.0


def test_check_env_and_same_spaces_as_2d(cfg):
    env = MujocoGoalkeeperEnv(cfg)
    check_env(env, skip_render_check=True)
    check_env(PrivilegedObsWrapper(MujocoGoalkeeperEnv(cfg)), skip_render_check=True)
    ref = GoalkeeperEnv(cfg)
    assert env.observation_space == ref.observation_space
    assert env.action_space == ref.action_space
    assert env.privileged_space == ref.privileged_space


def test_center_saves_slow_shot(cfg):
    env = MujocoGoalkeeperEnv(cfg)
    obs, info = env.reset(seed=1, options={"speed": 1.0, "start_pan": 0.0, "randomize": False})
    while True:
        obs, reward, terminated, truncated, info = env.step(CenterBlocker().predict(obs)[0])
        if terminated or truncated:
            break
    assert info["outcome"] == "save"


def test_randomization_updates_model_in_place(cfg):
    env = MujocoGoalkeeperEnv(cfg)
    m = env.mj.model
    gates = set()
    for s in range(4):
        env.reset(seed=s)
        gid = m.geom("divider_right").id
        gates.add(round(float(m.geom_pos[gid][0] - m.geom_size[gid][0]) * 2, 6))  # Gate width
        assert env.mj._mu == env.cfg["puck"]["friction_kinetic"]
    assert len(gates) == 4
