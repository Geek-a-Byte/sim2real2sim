"""Phase 3: hierarchical match against a scripted opponent (event-level sim).

The high-level policy picks one of three frozen primitives every decision_dt_s:
  0 BLOCK  go to (or stay at) the gate; the Phase 1 block skill defends it
  1 SLING  shoot the next puck (Phase 2 skill); the arm leaves the gate
  2 HOLD   keep the arm where it is (at the band: stay ready for another sling)

Arm states: GATE -> (SLING) TO_BAND -> SLINGING -> AT_BAND -> (BLOCK) TO_GATE -> GATE.
SLING from AT_BAND goes straight to SLINGING. During TO_BAND, SLINGING, AT_BAND
and TO_GATE the gate is undefended: this is the explicit recovery time.

Fidelity. Millions of decisions are needed, so nothing is simulated at 1 kHz:
- Opponent shots fly straight with Coulomb friction (analytic arrival time).
- A threat that reaches the gate is saved with the measured probability of the
  frozen block primitive (primitives/block.py): p_gate(speed) if the paddle was
  settled at the gate when the shot launched, p_edge(speed, delay) if the arm was
  just back from the band (delay = how late it arrived after the launch), and 0
  if the arm was away from the gate.
- An agent sling succeeds with the measured Phase 2 probability.

Puck bookkeeping: each side starts with pucks_per_side. A puck in flight belongs
to no side until it resolves. Rewards and the win rule come from `scoring`
(placeholder: first side with no pucks wins).

Observation (what the robot can sense): the tracked incoming opponent puck (after
the camera latency), the opponent hand (position, velocity), the arm state, puck
counts and time. The loaded opponent puck is hidden under the hand. The critic
also gets the opponent's internal state (privileged).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.slingpuck.opponent.scripted_opponent import PHASES, Opponent, Shot
from src.slingpuck.physics.band_model import GRAVITY
from src.slingpuck.primitives.block import BlockTable, load_or_build_block_table
from src.slingpuck.primitives.sling import SlingPrimitive

BLOCK, SLING, HOLD = 0, 1, 2
ACTION_NAMES = ("block", "sling", "hold")
ARM_STATES = ("gate", "to_band", "slinging", "at_band", "to_gate")
OBS_CLIP = 5.0


class MatchEnv(gym.Env):
    metadata = {"render_modes": []}
    POLICY_OBS_NAMES = ("puck_valid", "puck_x", "puck_y", "puck_vx", "puck_vy",
                        "hand_x", "hand_y", "hand_vx", "hand_vy",
                        *(f"arm_{s}" for s in ARM_STATES), "arm_time_left", "paddle_settled",
                        "agent_pucks", "opp_pucks", "own_in_flight", "time_left")
    PRIVILEGED_OBS_NAMES = (*(f"opp_{p}" for p in PHASES), "opp_time_to_release", "next_threat", "next_x",
                            "next_speed", "incoming_threat", "incoming_time_to_gate")

    def __init__(self, config: dict, block_table: BlockTable | None = None):
        super().__init__()
        self.cfg = config
        self.m = config["match"]
        self.scoring = config["scoring"]
        self.block = block_table if block_table is not None else load_or_build_block_table(config)
        self.sling = SlingPrimitive.from_config(config)
        hl = 0.5 * config["board"]["length_m"]
        self.half_length = hl
        r = config["puck"]["radius_m"]
        self.opp_band_y = hl - config["band"]["band_offset_from_end_wall_m"]
        self.gate_face_y = 0.5 * config["board"]["divider_thickness_m"] + r  # Opponent side of the divider
        self.decel = config["puck"]["friction_kinetic"] * GRAVITY
        self.n_sub = max(1, int(round(self.m["decision_dt_s"] / self.m["sim_dt_s"])))
        self.max_steps = int(np.ceil(self.m["max_time_s"] / self.m["decision_dt_s"]))
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.POLICY_OBS_NAMES),), dtype=np.float32)
        self.privileged_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(len(self.PRIVILEGED_OBS_NAMES),),
                                           dtype=np.float32)

    # ------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        """options: tell_strength (override for sweeps)."""
        super().reset(seed=seed)
        options = options or {}
        cfg = self.cfg
        if "tell_strength" in options:
            cfg = {**cfg, "opponent": {**cfg["opponent"], "tell_strength": float(options["tell_strength"])}}
        self.opponent = Opponent(cfg, self.np_random)
        n = self.m["pucks_per_side"]
        self.agent_side, self.opp_side = n, n
        self.t = 0.0
        self.steps = 0
        self.arm = "gate"
        self.arm_until = 0.0              # End time of the current arm motion
        self.gate_ready_since = -np.inf   # Time the arm got back near the gate (None = away)
        self.incoming: list[dict] = []    # Opponent pucks in flight
        self.own: dict | None = None      # Agent puck in flight
        self.stats = {"sent": 0, "received": 0, "slings": 0, "sling_hits": 0, "threats": 0, "saves": 0,
                      "conceded_while_away": 0}
        self.opponent.reset(0.0, self.opp_side > 0)
        return self._obs(), self._info(None)

    # -------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        reward = self._apply_action(action)
        outcome = None
        for _ in range(self.n_sub):
            self.t += self.m["sim_dt_s"]
            reward += self._advance()
            outcome = self._match_over()
            if outcome is not None:
                break
        self.steps += 1
        terminated = outcome is not None
        truncated = not terminated and self.steps >= self.max_steps
        if outcome == "win":
            reward += self.scoring["reward_win"]
        elif outcome == "loss":
            reward += self.scoring["reward_loss"]
        if truncated:
            outcome = "time"
        return self._obs(), float(reward), terminated, truncated, self._info(outcome)

    def can_sling(self) -> bool:
        return self.arm in ("gate", "at_band") and self.agent_side > 0 and self.own is None

    def _apply_action(self, action) -> float:
        if action == SLING and self.can_sling():
            self.stats["slings"] += 1
            self.gate_ready_since = None
            if self.arm == "gate":
                self.arm, self.arm_until = "to_band", self.t + self.sling.move_gate_to_band_s
            else:
                self.arm, self.arm_until = "slinging", self.t + self.sling.at_band_s
        elif action == BLOCK and self.arm == "at_band":
            self.arm, self.arm_until = "to_gate", self.t + self.m["move_band_to_gate_s"]
        return 0.0

    def _advance(self) -> float:
        reward = 0.0
        t = self.t
        # Arm state machine
        if self.arm == "to_band" and t >= self.arm_until:
            self.arm, self.arm_until = "slinging", self.arm_until + self.sling.at_band_s
        if self.arm == "slinging" and t >= self.arm_until:
            self.agent_side -= 1
            self.own = {"t_resolve": self.arm_until + self.sling.flight_s,
                        "hit": bool(self.np_random.random() < self.sling.success_prob)}
            self.arm = "at_band"
        if self.arm == "to_gate" and t >= self.arm_until:
            self.arm, self.gate_ready_since = "gate", self.arm_until
        # Own puck
        if self.own is not None and t >= self.own["t_resolve"]:
            if self.own["hit"]:
                self.opp_side += 1
                self.stats["sent"] += 1
                self.stats["sling_hits"] += 1
                reward += self.scoring["reward_puck_sent"]
            else:
                self.agent_side += 1
            self.own = None
        # Opponent: shots launched now leave the opponent side
        for shot in self.opponent.advance(t, self.opp_side > 0):
            self.opp_side -= 1
            self.incoming.append(self._flight(shot))
        if self.opponent.next_shot is None and self.opp_side > 0:
            self.opponent.advance(t, True)
        # Opponent pucks reaching the gate
        for f in [f for f in self.incoming if t >= f["t_gate"]]:
            self.incoming.remove(f)
            reward += self._resolve(f)
        return reward

    def _flight(self, shot: Shot) -> dict:
        dist = self.opp_band_y - self.gate_face_y
        v, a = shot.speed, self.decel
        disc = v * v - 2 * a * dist
        reaches = disc > 0
        t_gate = shot.t_launch + ((v - np.sqrt(disc)) / a if reaches else v / a)
        return {"shot": shot, "t_gate": t_gate, "reaches": reaches}

    def _resolve(self, f) -> float:
        shot = f["shot"]
        if not (shot.threat and f["reaches"]):
            self.opp_side += 1  # Hit the divider (or fell short) and stays on the opponent side
            return 0.0
        self.stats["threats"] += 1
        if self.gate_ready_since is None:
            p = 0.0
        elif self.gate_ready_since <= shot.t_launch - self.m["settle_after_return_s"]:
            p = self.block.p_save(shot.speed, at_gate=True)
        else:
            p = self.block.p_save(shot.speed, at_gate=False, delay=max(0.0, self.gate_ready_since - shot.t_launch))
        if self.np_random.random() < p:
            self.stats["saves"] += 1
            self.opp_side += 1
            return 0.0
        if self.gate_ready_since is None:
            self.stats["conceded_while_away"] += 1
        self.agent_side += 1
        self.stats["received"] += 1
        return self.scoring["reward_puck_received"]

    def _match_over(self):
        if self.scoring["win_rule"] != "empty_side":
            return None
        if self.agent_side == 0 and self.own is None and not any(f["shot"].threat for f in self.incoming):
            return "win"
        if self.opp_side == 0 and not self.incoming and self.own is None:
            return "loss"
        return None

    # ------------------------------------------------------- observation
    def _incoming_state(self):
        """Earliest-arriving opponent puck in flight that the camera has seen, true [x, y, vx, vy]."""
        latency = self.cfg["camera"]["latency_s"]
        seen = [f for f in self.incoming if self.t - f["shot"].t_launch >= latency]
        if not seen:
            return None, None
        f = min(seen, key=lambda f: f["t_gate"])
        shot = f["shot"]
        tau = min(self.t - shot.t_launch, shot.speed / self.decel)
        speed = shot.speed - self.decel * tau
        y = self.opp_band_y - (shot.speed * tau - 0.5 * self.decel * tau * tau)
        return np.array([shot.x_launch, y, 0.0, -speed]), f

    def _obs(self):
        hl = self.half_length
        n = self.m["pucks_per_side"]
        puck, _ = self._incoming_state()
        if puck is None:
            puck_part = [0.0] * 5
        else:
            pos = puck[:2] + self.np_random.normal(0.0, self.m["puck_track_noise_m"], 2)
            vel = puck[2:] + self.np_random.normal(0.0, self.m["puck_track_vel_noise_m_s"], 2)
            puck_part = [1.0, pos[0] / hl, pos[1] / hl, vel[0] / 3.0, vel[1] / 3.0]
        if self.m["observe_hand"]:
            h = self.opponent.hand_observed(self.t)
            hand_part = [h[0] / hl, h[1] / hl, h[2], h[3]]
        else:
            hand_part = [0.0] * 4
        arm = [float(self.arm == s) for s in ARM_STATES]
        settled = self.gate_ready_since is not None and self.t - self.gate_ready_since >= self.m["settle_after_return_s"]
        obs = np.array([*puck_part, *hand_part, *arm, max(self.arm_until - self.t, 0.0), float(settled),
                        self.agent_side / n, self.opp_side / n, float(self.own is not None),
                        1.0 - self.t / self.m["max_time_s"]])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def privileged_obs(self):
        hl = self.half_length
        phase = self.opponent.phase(self.t)
        nxt = self.opponent.next_shot
        ttr = self.opponent.time_to_release(self.t)
        _, f = self._incoming_state()
        true_in = min(self.incoming, key=lambda f: f["t_gate"]) if self.incoming else None
        obs = np.array([*(float(phase == p) for p in PHASES), min(ttr, 3.0) / 3.0,
                        float(nxt.threat) if nxt else 0.0, (nxt.x_launch / hl) if nxt else 0.0,
                        (nxt.speed / 3.0) if nxt else 0.0,
                        float(true_in["shot"].threat) if true_in else 0.0,
                        min(true_in["t_gate"] - self.t, 3.0) / 3.0 if true_in else 1.0])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "agent_side": self.agent_side, "opp_side": self.opp_side,
                "arm": self.arm, "opp_phase": self.opponent.phase(self.t), "can_sling": self.can_sling(),
                "privileged": self.privileged_obs(), **({"stats": dict(self.stats)} if outcome else {})}
