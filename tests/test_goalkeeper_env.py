import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from slingpuck.envs.wrappers import PrivilegedObsWrapper
from slingpuck.policies.scripted import CenterBlocker, HoldStart


def rollout(env, policy, **reset_kw):
    obs, info = env.reset(**reset_kw)
    total, observations = 0.0, [obs]
    while True:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        observations.append(obs)
        if terminated or truncated:
            return info, total, observations


@pytest.fixture
def no_prelaunch(cfg):
    c = copy.deepcopy(cfg)
    c["goalkeeper"]["prelaunch_s"] = [0.0, 0.0]
    return c


def test_check_env_box_and_dict(cfg):
    check_env(GoalkeeperEnv(cfg), skip_render_check=True)
    check_env(PrivilegedObsWrapper(GoalkeeperEnv(cfg)), skip_render_check=True)


def test_seeded_episodes_repeat(cfg):
    a = rollout(GoalkeeperEnv(cfg), CenterBlocker(), seed=5)
    b = rollout(GoalkeeperEnv(cfg), CenterBlocker(), seed=5)
    np.testing.assert_array_equal(np.array(a[2]), np.array(b[2]))
    assert a[0]["outcome"] == b[0]["outcome"]


def test_threats_only_samples_threats(cfg):
    env = GoalkeeperEnv(cfg)
    assert all(env.reset(seed=s)[1]["threat"] for s in range(30))


def test_center_saves_slow_shot(no_prelaunch):
    env = GoalkeeperEnv(no_prelaunch)
    info, total, _ = rollout(env, CenterBlocker(), seed=1, options={"speed": 1.0, "start_pan": 0.0})
    assert info["outcome"] == "save"
    assert total == pytest.approx(1.0)


def test_far_paddle_concedes_fast_shot(no_prelaunch):
    env = GoalkeeperEnv(no_prelaunch)
    env.reset(seed=0)
    info, total, _ = rollout(env, HoldStart(), seed=2,
                             options={"speed": 3.0, "start_pan": env.geom.pan_max, "randomize": False})
    assert info["outcome"] == "goal"
    assert total == pytest.approx(-1.0)


def test_smoothness_penalty(cfg):
    env = GoalkeeperEnv(cfg)
    env.reset(seed=0, options={"start_pan": 0.0})
    _, reward, terminated, _, _ = env.step(np.array([1.0]))
    assert not terminated
    assert reward == pytest.approx(-cfg["goalkeeper"]["smoothness_weight"] * 1.0)


def test_policy_obs_is_tracked_not_ground_truth(cfg):
    c = copy.deepcopy(cfg)
    c["goalkeeper"]["paddle_locked_until_launch"] = False  # Start before any camera frame
    env = GoalkeeperEnv(c)
    obs, info = env.reset(seed=3)
    names = GoalkeeperEnv.POLICY_OBS_NAMES
    # Before the first camera frame the tracker has no estimate: zeros and valid = 0.
    assert obs[names.index("trk_valid")] == 0.0
    assert np.all(obs[:4] == 0.0)
    assert info["privileged"][1] != 0.0  # The true puck y is known only to the critic
    steps = 0
    while obs[names.index("trk_valid")] == 0.0:
        obs, _, _, _, info = env.step(np.zeros(1))
        steps += 1
    # The first frame cannot arrive before the camera latency has passed.
    assert steps * env.control_dt >= env.cfg["camera"]["latency_s"] - 1e-9
    hl = 0.5 * cfg["board"]["length_m"]
    np.testing.assert_allclose(info["privileged"][:2] * hl, env.sim.pos[0], atol=1e-6)
    assert not np.allclose(obs[:2], info["privileged"][:2])  # Camera noise and latency


def test_domain_randomization_changes_physics_per_episode(cfg):
    env = GoalkeeperEnv(cfg)
    frictions = set()
    for s in range(5):
        env.reset(seed=s)
        frictions.add(env.cfg["puck"]["friction_kinetic"])
    assert len(frictions) == 5
    env.reset(seed=0, options={"randomize": False})
    assert env.cfg["puck"]["friction_kinetic"] == cfg["puck"]["friction_kinetic"]


def test_locked_paddle_holds_until_launch(cfg):
    env = GoalkeeperEnv(cfg)
    pan0 = 0.7 * env.geom.pan_max
    obs, info = env.reset(seed=4, options={"start_pan": pan0, "release_delay": 0.0})
    names = GoalkeeperEnv.POLICY_OBS_NAMES
    assert env.launched
    assert env.t == pytest.approx(env.shot.prelaunch_s, abs=env.physics_dt)
    assert float(env.servo.pos) == pytest.approx(pan0)  # Did not move while locked
    assert obs[names.index("trk_valid")] == 1.0          # Tracker is warm at release
    assert obs[names.index("prev_action")] == pytest.approx(0.7)


def test_release_delay(cfg):
    env = GoalkeeperEnv(cfg)
    env.reset(seed=4, options={"release_delay": 0.05})
    assert env.t == pytest.approx(env.shot.prelaunch_s + 0.05, abs=env.physics_dt)
    assert env.release_delay == 0.05


def test_long_release_delay_ends_at_first_step(cfg):
    env = GoalkeeperEnv(cfg)
    env.reset(seed=4, options={"release_delay": 1.0, "start_pan": env.geom.pan_max, "speed": 3.0})
    _, reward, terminated, _, info = env.step(np.zeros(1))
    assert terminated and info["outcome"] == "goal"
