"""Paired Point Encoding policy for SB3 PPO (Dex4D), with an asymmetric critic.

Each object point arrives as [current xyz, target xyz, valid]. A shared MLP encodes every point, and
masked mean + max pooling makes the result independent of the point order and of masked (unseen) points.
The actor reads points + proprio (what the robot can sense). The critic also reads the privileged sim
state, which is used only in training (privileged teacher).
"""
import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.slingpuck.policies.asymmetric import AsymmetricMlpExtractor

ACTOR_KEYS = ("points", "proprio")
CRITIC_KEYS = ("points", "proprio", "privileged")


class PairedPointEncoder(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 64, out: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out), nn.ReLU())
        self.out_dim = 2 * out

    def forward(self, points: th.Tensor) -> th.Tensor:
        valid = points[..., -1:]                       # (B, N, 1)
        f = self.mlp(points)                           # (B, N, out)
        n = valid.sum(1).clamp(min=1.0)
        mean = (f * valid).sum(1) / n
        mx = (f * valid - 1e6 * (1 - valid)).max(1).values
        return th.cat([mean, mx], dim=1)


class PairedPointExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, keys=ACTOR_KEYS):
        enc = PairedPointEncoder(observation_space["points"].shape[-1])
        flat = [k for k in keys if k != "points"]
        dim = enc.out_dim + sum(int(observation_space[k].shape[0]) for k in flat)
        super().__init__(observation_space, features_dim=dim)
        self.encoder, self.flat_keys = enc, flat

    def forward(self, obs) -> th.Tensor:
        return th.cat([self.encoder(obs["points"]), *(obs[k] for k in self.flat_keys)], dim=1)


class PairedPointPolicy(ActorCriticPolicy):
    """Actor: PairedPointExtractor(points, proprio). Critic: its own extractor with the privileged state."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict) or set(CRITIC_KEYS) - set(observation_space.spaces):
            raise ValueError(f"PairedPointPolicy needs a Dict observation with keys {CRITIC_KEYS}")
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = PairedPointExtractor
        kwargs.pop("features_extractor_kwargs", None)
        self._extractors_built = 0
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        # ActorCriticPolicy.__init__ builds the actor extractor first, then the critic extractor.
        keys = ACTOR_KEYS if self._extractors_built == 0 else CRITIC_KEYS
        self._extractors_built += 1
        return PairedPointExtractor(self.observation_space, keys)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(self.pi_features_extractor.features_dim,
                                                    self.vf_features_extractor.features_dim,
                                                    self.net_arch, self.activation_fn, self.device)
