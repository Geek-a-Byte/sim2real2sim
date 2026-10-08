import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.slingpuck.envs.sling_env import SlingEnv, launch_and_fly, realize_pull
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper
from src.slingpuck.physics.band_model import BandModel
from src.slingpuck.policies.sling_scripted import SlideCenter, SlideCenterCorrect


def run(env, policy, **kw):
    obs, info = env.reset(**kw)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.predict(obs)[0])
        steps += 1
        if terminated or truncated:
            return info, reward, steps


def test_check_env(cfg):
    check_env(SlingEnv(cfg), skip_render_check=True)
    check_env(PrivilegedObsWrapper(SlingEnv(cfg)), skip_render_check=True)


def test_two_steps_with_correction_one_without(cfg):
    assert run(SlingEnv(cfg), SlideCenter(), seed=0)[2] == 2
    c = copy.deepcopy(cfg)
    c["sling"]["correction_step"] = False
    assert run(SlingEnv(c), SlideCenter(), seed=0)[2] == 1


def test_seeded_repeat(cfg):
    a = run(SlingEnv(cfg), SlideCenterCorrect(cfg), seed=7)
    b = run(SlingEnv(cfg), SlideCenterCorrect(cfg), seed=7)
    assert a[0]["miss_m"] == b[0]["miss_m"] and a[1] == b[1]


def test_noiseless_center_shot_scores_and_off_center_misses(cfg):
    band = BandModel.from_config(cfg)
    pull = realize_pull(cfg, band, 0.0, 0.0, 0.025, rng=None)
    hit = launch_and_fly(cfg, band, pull.pos, rng=None)
    assert hit.success and hit.miss_m < 1e-6
    # Off center, the asymmetric band V pushes the puck back toward the band center a little.
    pull = realize_pull(cfg, band, 0.03, 0.0, 0.025, rng=None)
    miss = launch_and_fly(cfg, band, pull.pos, rng=None)
    assert not miss.success and 0.015 < miss.miss_m < 0.03
    assert miss.exit_vel[0] < 0.0


def test_reward_success_and_shaping(cfg):
    env = SlingEnv(cfg)
    info, reward, _ = run(env, SlideCenter(), seed=3, options={"randomize": False})
    shaping = cfg["sling"]["shaping_weight"] * np.exp(-(info["miss_m"] / cfg["sling"]["shaping_scale_m"]) ** 2)
    assert reward == pytest.approx(float(info["success"]) + shaping)


def test_observation_band_depth_after_pull(cfg):
    env = SlingEnv(cfg)
    obs, _ = env.reset(seed=1, options={"randomize": False})
    names = SlingEnv.POLICY_OBS_NAMES
    assert obs[names.index("stage")] == 0 and obs[names.index("band_depth")] == 0
    obs, *_ = env.step(np.array([0.0, 0.0, 1.0]))
    depth = obs[names.index("band_depth")] * cfg["band"]["max_pull_m"]
    assert obs[names.index("stage")] == 1
    # The camera sees the realized pull (shorter than commanded, by the deflection).
    assert depth == pytest.approx(env.pull.pull_m, abs=0.003)
    assert env.pull.pull_m < cfg["band"]["max_pull_m"]


def test_scripted_baselines_score(cfg):
    env = SlingEnv(cfg)
    rates = {}
    for pol in (SlideCenter(), SlideCenterCorrect(cfg)):
        rates[pol.name] = np.mean([run(env, pol, seed=s)[0]["success"] for s in range(100)])
    assert rates["center"] > 0.8 and rates["center_correct"] > 0.8
