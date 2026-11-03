import numpy as np
import pytest
import torch as th

mujoco = pytest.importorskip("mujoco")

from stable_baselines3 import PPO  # noqa: E402

from src.slingpuck.general.goal_env import GoalEnv  # noqa: E402
from src.slingpuck.general.policy import PairedPointEncoder, PairedPointPolicy  # noqa: E402


def test_encoder_ignores_point_order_and_masked_points():
    enc = PairedPointEncoder()
    pts = th.randn(2, 16, 7)
    pts[..., 6] = 1.0
    pts[1, 10:, 6] = 0.0
    out = enc(pts)
    perm = pts[:, th.randperm(16)]
    assert th.allclose(enc(perm), out, atol=1e-5)
    changed = pts.clone()
    changed[1, 10:, :6] = 99.0                     # Masked points: their values do not matter
    assert th.allclose(enc(changed)[1], out[1], atol=1e-5)


def test_actor_ignores_privileged_critic_uses_it(gcfg, reach_map, tmp_path):
    env = GoalEnv(gcfg, reach_map=reach_map)
    model = PPO(PairedPointPolicy, env, n_steps=64, batch_size=32, n_epochs=1, seed=0)
    model.learn(64)
    p = model.policy
    obs, _ = env.reset(seed=1)
    a = {k: th.as_tensor(v[None]) for k, v in obs.items()}
    b = {**a, "privileged": a["privileged"] + 1.0}
    with th.no_grad():
        assert th.allclose(p.get_distribution(a).distribution.mean, p.get_distribution(b).distribution.mean)
        assert not th.allclose(p.predict_values(a), p.predict_values(b))
    model.save(tmp_path / "m")
    loaded = PPO.load(tmp_path / "m", custom_objects={"policy_class": PairedPointPolicy})
    np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
