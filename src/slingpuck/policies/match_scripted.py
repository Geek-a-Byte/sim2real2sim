"""Scripted match policies (high-level selectors). Same predict() signature as SB3 models.

Actions: 0 BLOCK, 1 SLING, 2 WAIT (see envs/match_env.py).

- AlwaysBlock:   never slings (cannot win; a defensive reference).
- GreedySling:   slings whenever it can and never returns to the gate.
- ReloadRule:    the plan's baseline: "sling only when the opponent is reloading".
                 Reloading is detected from the hand: it is away from the band line.
                 Otherwise go back to (or stay at) the gate.
- TellRule:      like ReloadRule, but also slings while the opponent aims if the hand
                 x shows a likely miss (|hand x| above a threshold). Uses the tell.
- OracleReload:  ReloadRule with the opponent's true phase (privileged input, needs a
                 PrivilegedObsWrapper env). An upper reference for the rule.
- GreedyRule:    layer 3 of the adaptation plan (race of rates). Sling when the best shot clears more
                 pucks than blocking saves over the same time (see the class).
"""
import numpy as np

from src.slingpuck.envs.match_env import BLOCK, SHOT_TIME_SCALE, SLING, WAIT, MatchEnv
from src.slingpuck.opponent.scripted_opponent import PHASES, threat_half_width
from src.slingpuck.primitives.block import BlockPrimitive

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


class GreedyRule(ReloadRule):
    """Greedy game decision (layer 3): take the action with the most expected pucks per second.

    SLING clears P(through) pucks in T_shot seconds (the robot's own best shot, from the observation).
    During those T_shot seconds the gate is open, so BLOCK is worth the threats it would save in the same
    window: p_threat * p_block for each opponent shot that arrives in it. Sling when
        P(through) / T_shot  >  expected saved threats / T_shot.
    Both values are taken over the same window. With the time until the next shot as T_block (the plan's
    first form), blocking an aimed shot always looks better and the rule becomes reload_rule.

    The opponent model (threat probability, cycle time, aim time) is the robot's estimate of this opponent
    from env.yaml `opponent`; with measured profiles it comes from the team member's video.
    """
    name = "greedy_rule"

    def __init__(self, config: dict, margin_m: float = 0.01):
        super().__init__(config, margin_m)
        o, m = config["opponent"], config["match"]
        self.threat_prob = o["threat_prob"]
        hold = 0.5 * sum(o["hold_time_range_s"])
        self.cycle_s = o["reload_time_mu_s"] + o["place_time_s"] + o["pull_time_s"] + hold
        flight = (self.band_y - 0.5 * config["board"]["divider_thickness_m"]) / np.mean(o["shot_speed_range_m_s"])
        self.aim_to_gate_s = 0.5 * o["pull_time_s"] + hold + flight     # Aiming now -> the puck at the gate
        self.reload_to_gate_s = o["reload_time_mu_s"] + o["place_time_s"] + self.aim_to_gate_s
        self.half_threat = threat_half_width(config)
        self.block = BlockPrimitive.from_config(config)
        self.back_s = m["move_band_to_gate_s"]

    def saved_threats(self, obs, window: float) -> float:
        """Expected threats that a blocking arm saves in the next `window` seconds."""
        hl = self.half_length
        if obs[N("gate_covered")] > 0.5:                            # When the arm got back to the gate
            ready = -np.inf
        else:
            ready = 0.0 if self.arm(obs, "gate") else self.back_s
        if obs[N("puck_valid")] > 0.5:                              # A tracked opponent puck in flight
            x, y, vy = obs[N("puck_x")] * hl, obs[N("puck_y")] * hl, obs[N("puck_vy")] * 3.0
            first_p = float(abs(x) <= self.half_threat + 0.002)
            first_t = max(y, 0.0) / max(-vy, 0.1)
        elif self.reloading(obs, None):
            first_p, first_t = self.threat_prob, self.reload_to_gate_s
        else:                                                       # Aiming: the next shot comes soon
            first_p, first_t = self.threat_prob, self.aim_to_gate_s
        saved = 0.0
        t = first_t
        p = first_p
        while t <= window:
            saved += p * self.block.p_save(t - ready)
            t, p = t + self.cycle_s, self.threat_prob
        return saved

    def act(self, obs, priv):
        p, t_shot = obs[N("shot_p")], obs[N("shot_time")] * SHOT_TIME_SCALE
        if obs[N("agent_pucks")] <= 0 or t_shot <= 0:
            return BLOCK
        return SLING if p > self.saved_threats(obs, t_shot) else BLOCK


__all__ = ["AlwaysBlock", "GreedySling", "ReloadRule", "TellRule", "OracleReload", "GreedyRule", "BLOCK", "SLING",
           "WAIT", "PHASES"]
