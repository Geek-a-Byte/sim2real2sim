"""Scripted Phase 3 policies (high-level selectors). Same predict() signature as SB3 models.

Actions: 0 BLOCK, 1 SLING, 2 HOLD (see envs/match_env.py).

- AlwaysBlock:   never slings (cannot win; a defensive reference).
- GreedySling:   slings whenever it can and never returns to the gate.
- ReloadRule:    the plan's baseline: "sling only when the opponent is reloading".
                 Reloading is detected from the hand: it is away from the band line.
                 Otherwise go back to (or stay at) the gate.
- TellRule:      like ReloadRule, but also slings while the opponent aims if the hand
                 x shows a likely miss (|hand x| above a threshold). Uses the tell.
- OracleReload:  ReloadRule with the opponent's true phase (privileged input, needs a
                 PrivilegedObsWrapper env). An upper reference for the rule.
"""
import numpy as np

from src.slingpuck.envs.match_env import BLOCK, HOLD, SLING, MatchEnv
from src.slingpuck.opponent.scripted_opponent import PHASES

N = MatchEnv.POLICY_OBS_NAMES.index
P = MatchEnv.PRIVILEGED_OBS_NAMES.index


class _Base:
    name = "base"

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if isinstance(obs, dict):
            return np.array(self.act(obs["policy"], obs["privileged"])), state
        return np.array(self.act(obs, None)), state

    @staticmethod
    def arm(obs, s):
        return obs[N(f"arm_{s}")] > 0.5


class AlwaysBlock(_Base):
    name = "always_block"

    def act(self, obs, priv):
        return BLOCK


class GreedySling(_Base):
    name = "greedy_sling"

    def act(self, obs, priv):
        return SLING


class ReloadRule(_Base):
    """Sling only while the opponent reloads (hand away from its band line)."""
    name = "reload_rule"

    def __init__(self, config: dict, margin_m: float = 0.01):
        hl = 0.5 * config["board"]["length_m"]
        self.half_length = hl
        self.band_y = hl - config["band"]["band_offset_from_end_wall_m"]
        self.margin = margin_m

    def reloading(self, obs, priv):
        return obs[N("hand_y")] * self.half_length < self.band_y - self.margin

    def safe(self, obs, priv):
        return self.reloading(obs, priv)

    def act(self, obs, priv):
        incoming = obs[N("puck_valid")] > 0.5
        if self.safe(obs, priv) and not incoming:
            return SLING  # Ignored by the env when it cannot sling (busy or no puck)
        return BLOCK


class TellRule(ReloadRule):
    """Also sling during the opponent's aim when the hand x shows a likely miss."""
    name = "tell_rule"

    def __init__(self, config: dict, miss_threshold_m: float = 0.006, margin_m: float = 0.01):
        super().__init__(config, margin_m)
        self.miss_threshold = miss_threshold_m

    def safe(self, obs, priv):
        hand_x = obs[N("hand_x")] * self.half_length
        return self.reloading(obs, priv) or abs(hand_x) > self.miss_threshold


class OracleReload(ReloadRule):
    """Uses the opponent's true phase (privileged)."""
    name = "oracle_reload"

    def reloading(self, obs, priv):
        return priv[P("opp_reload")] > 0.5 or priv[P("opp_idle")] > 0.5


__all__ = ["AlwaysBlock", "GreedySling", "ReloadRule", "TellRule", "OracleReload", "BLOCK", "SLING", "HOLD",
           "PHASES"]
