"""Scripted Phase 2 baselines. Same predict() signature as SB3 models.

The band launches the puck almost straight forward, so a good shot is: slide the
puck in front of the gate (x = 0), pull straight back, release.

- SlideCenter: step 1 slides to x = 0 and pulls straight back; step 2 no correction.
- SlideCenterCorrect: as SlideCenter, then uses the tracked pulled puck to shift
  it laterally back to x = 0 before release (uses the camera, like the policy).
"""
import numpy as np

from src.slingpuck.envs.sling_env import SlingEnv

_X = SlingEnv.POLICY_OBS_NAMES.index("trk_x")
_STAGE = SlingEnv.POLICY_OBS_NAMES.index("stage")


class SlideCenter:
    name = "center"

    def __init__(self, pull_frac: float = 0.8):
        self.pull_action = 2.0 * pull_frac - 1.0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs, dtype=float)
        action = np.zeros(obs.shape[:-1] + (3,), dtype=np.float32)
        aim = obs[..., _STAGE] < 0.5
        action[..., 2] = np.where(aim, self.pull_action, 0.0)
        return action, state


class SlideCenterCorrect(SlideCenter):
    name = "center_correct"

    def __init__(self, config: dict, pull_frac: float = 0.8):
        super().__init__(pull_frac)
        self.half_length = 0.5 * config["board"]["length_m"]
        self.max_correction = config["sling"]["correction_slide_m"]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        action, state = super().predict(obs, state, episode_start, deterministic)
        obs = np.asarray(obs, dtype=float)
        correct = obs[..., _STAGE] >= 0.5
        x = obs[..., _X] * self.half_length
        action[..., 0] = np.where(correct, np.clip(-x / self.max_correction, -1.0, 1.0), action[..., 0])
        return action, state
