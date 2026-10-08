import numpy as np
import torch as th
from stable_baselines3 import PPO

from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from src.slingpuck.envs.wrappers import PrivilegedObsWrapper
from src.slingpuck.policies.asymmetric import AsymmetricActorCriticPolicy


def test_actor_ignores_privileged_critic_uses_it(cfg, tmp_path):
    env = PrivilegedObsWrapper(GoalkeeperEnv(cfg))
    model = PPO(AsymmetricActorCriticPolicy, env, n_steps=64, batch_size=32, n_epochs=1, seed=0)
    model.learn(64)
    p = model.policy
    assert p.pi_features_extractor.features_dim == env.observation_space["policy"].shape[0]
    assert p.vf_features_extractor.features_dim == (env.observation_space["policy"].shape[0]
                                                     + env.observation_space["privileged"].shape[0])
    obs, _ = env.reset(seed=1)
    a = {k: th.as_tensor(v[None]) for k, v in obs.items()}
    b = {"policy": a["policy"], "privileged": a["privileged"] + 1.0}
    with th.no_grad():
        assert th.equal(p.get_distribution(a).distribution.mean, p.get_distribution(b).distribution.mean)
        assert not th.equal(p.predict_values(a), p.predict_values(b))

    model.save(tmp_path / "m")
    loaded = PPO.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
