"""Scripted goalkeeper baselines. Same predict() signature as SB3 models.

- CenterBlocker: always command pan = 0 (paddle centered on the gate). The gate
  is barely wider than the puck, so this is a strong reference policy.
- HoldStart: keep the start pan (never move). Lower bound.
"""
import numpy as np

from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv

_PREV_ACTION = GoalkeeperEnv.POLICY_OBS_NAMES.index("prev_action")


class CenterBlocker:
    name = "center"

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs)
        return np.zeros(obs.shape[:-1] + (1,), dtype=np.float32), state


class HoldStart:
    name = "hold"

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs)
        return obs[..., _PREV_ACTION:_PREV_ACTION + 1].astype(np.float32), state
