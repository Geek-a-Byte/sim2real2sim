"""Asymmetric actor-critic for SB3 PPO.

The actor reads only obs["policy"] (tracked, deployable inputs). The critic
reads obs["policy"] and obs["privileged"] (sim ground truth and randomized
params). The critic is used only in training, so the deployed actor never
needs privileged inputs. Use with envs.wrappers.PrivilegedObsWrapper.
"""
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor

ACTOR_KEYS = ("policy",)
CRITIC_KEYS = ("policy", "privileged")


class KeySelectExtractor(BaseFeaturesExtractor):
    """Concatenate the selected keys of a Dict observation."""

    def __init__(self, observation_space: gym.spaces.Dict, keys=ACTOR_KEYS):
        dim = sum(int(observation_space[k].shape[0]) for k in keys)
        super().__init__(observation_space, features_dim=dim)
        self.keys = tuple(keys)

    def forward(self, observations) -> th.Tensor:
        return th.cat([observations[k] for k in self.keys], dim=1)


class AsymmetricMlpExtractor(nn.Module):
    """MlpExtractor with different input sizes for the actor and the critic."""

    def __init__(self, pi_dim, vf_dim, net_arch, activation_fn, device):
        super().__init__()
        pi = MlpExtractor(pi_dim, net_arch, activation_fn, device)
        vf = MlpExtractor(vf_dim, net_arch, activation_fn, device)
        self.policy_net, self.latent_dim_pi = pi.policy_net, pi.latent_dim_pi
        self.value_net, self.latent_dim_vf = vf.value_net, vf.latent_dim_vf

    def forward(self, features):
        pi_features, vf_features = features
        return self.forward_actor(pi_features), self.forward_critic(vf_features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(observation_space, gym.spaces.Dict) or set(CRITIC_KEYS) - set(observation_space.spaces):
            raise ValueError(f"AsymmetricActorCriticPolicy needs a Dict observation with keys {CRITIC_KEYS}")
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = KeySelectExtractor
        kwargs.pop("features_extractor_kwargs", None)
        self._extractors_built = 0
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        # ActorCriticPolicy.__init__ builds the actor extractor first, then the critic extractor.
        keys = ACTOR_KEYS if self._extractors_built == 0 else CRITIC_KEYS
        self._extractors_built += 1
        return KeySelectExtractor(self.observation_space, keys)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.pi_features_extractor.features_dim,
            self.vf_features_extractor.features_dim,
            self.net_arch,
            self.activation_fn,
            self.device,
        )
