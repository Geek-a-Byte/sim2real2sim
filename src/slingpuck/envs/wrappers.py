"""Env wrappers."""
import gymnasium as gym
from gymnasium import spaces


class PrivilegedObsWrapper(gym.ObservationWrapper):
    """Dict observation {"policy": actor obs, "privileged": critic-only obs}.

    The wrapped env must define `privileged_space` and put the privileged vector
    in info["privileged"] at reset and step. Use with
    policies.asymmetric.AsymmetricActorCriticPolicy, whose actor reads only "policy".
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "policy": env.observation_space,
            "privileged": env.unwrapped.privileged_space,
        })
        self._privileged = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._privileged = info["privileged"]
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._privileged = info["privileged"]
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        return {"policy": obs, "privileged": self._privileged}
