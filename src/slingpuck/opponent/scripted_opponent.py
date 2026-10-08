"""Scripted human opponent for Phase 3.

Cycle: reload -> place -> pull -> hold -> release, then the next cycle if the
opponent still has a puck on its side.
  reload: the hand moves to a puck on the opponent half and picks it up.
  place:  the hand moves the puck to the opponent band line.
  pull:   the hand pulls the puck back (pull ~ shot speed).
  hold:   a short random pause.
  release: the puck launches straight toward the agent (-y) from x_launch.

The hand covers the loaded puck from the overhead camera, so the camera sees
only the hand. Shots fly straight (Phase 2 finding), so x_launch decides whether
a shot is a threat (goes through the gate). The tell: during place, pull and hold
the hand x is   tell * x_launch + (1 - tell) * x_decoy,   with x_decoy drawn from
the same distribution as x_launch. tell_strength 0 = the hand says nothing about
the threat; 1 = it shows the launch x exactly.
"""
from dataclasses import dataclass

import numpy as np

THREAT_HALF_WIDTH_M = 0.003   # Launch |x| below this goes through the gate (Phase 2: about +-3 mm)
MISS_MIN_M = 0.006            # Non-threat shots launch at least this far from the center
LAUNCH_X_MAX_M = 0.06
PICK_X_MAX_M = 0.07
PICK_Y_RANGE_M = (0.02, 0.09)  # Where loose pucks lie on the opponent half (in front of the band)
HOME = np.array([0.0, 0.16])   # Hand rest position when the opponent has no puck
PHASES = ("reload", "place", "pull", "hold", "idle")


@dataclass(frozen=True)
class Shot:
    t_launch: float
    x_launch: float
    speed: float
    threat: bool


@dataclass(frozen=True)
class _Cycle:
    times: np.ndarray        # Phase boundaries: reload start, place start, pull start, hold start, release
    waypoints: np.ndarray    # Hand position at each boundary (times[i] -> waypoints[i])
    shot: Shot


class Opponent:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg["opponent"]
        self.rng = rng
        self.band_y = 0.5 * cfg["board"]["length_m"] - cfg["band"]["band_offset_from_end_wall_m"]
        self.cycle: _Cycle | None = None
        self.idle_since = 0.0
        self.idle_pos = HOME.copy()

    # ------------------------------------------------------------ sampling
    def _launch_x(self, threat: bool) -> float:
        if threat:
            return float(self.rng.uniform(-THREAT_HALF_WIDTH_M, THREAT_HALF_WIDTH_M))
        return float(self.rng.choice([-1, 1]) * self.rng.uniform(MISS_MIN_M, LAUNCH_X_MAX_M))

    def _start_cycle(self, t0: float, start_pos: np.ndarray):
        c = self.cfg
        threat = bool(self.rng.random() < c["threat_prob"])
        x_launch = self._launch_x(threat)
        x_decoy = self._launch_x(bool(self.rng.random() < c["threat_prob"]))
        x_hand = c["tell_strength"] * x_launch + (1.0 - c["tell_strength"]) * x_decoy
        lo, hi = c["shot_speed_range_m_s"]
        speed = float(self.rng.uniform(lo, hi))
        pull = c["max_pull_m"] * (0.3 + 0.7 * (speed - lo) / max(hi - lo, 1e-9))
        reload = max(0.3, float(self.rng.normal(c["reload_time_mu_s"], c["reload_time_std_s"])))
        hold = float(self.rng.uniform(*c["hold_time_range_s"]))
        times = t0 + np.cumsum([0.0, reload, c["place_time_s"], c["pull_time_s"], hold])
        pick = np.array([self.rng.uniform(-PICK_X_MAX_M, PICK_X_MAX_M), self.rng.uniform(*PICK_Y_RANGE_M)])
        on_band = np.array([x_hand, self.band_y])
        pulled = np.array([x_hand, self.band_y + pull])
        # Reload: move to the pick point in the first half, then wait there.
        waypoints = np.array([start_pos, pick, on_band, pulled, pulled])
        self.cycle = _Cycle(times, waypoints, Shot(float(times[-1]), x_launch, speed, threat))
        self._reload_mid = (t0 + 0.5 * reload)

    # ---------------------------------------------------------------- API
    def reset(self, t0: float, has_puck: bool):
        self.cycle = None
        self.idle_pos = HOME.copy()
        self.idle_since = t0
        if has_puck:
            self._start_cycle(t0, HOME.copy())

    def advance(self, t: float, has_puck: bool) -> list[Shot]:
        """Advance to time t. Returns the shots released up to t."""
        shots = []
        while self.cycle is not None and t >= self.cycle.shot.t_launch:
            shot = self.cycle.shot
            shots.append(shot)
            end_pos = self.cycle.waypoints[-1]
            has_puck_after = has_puck and len(shots) == 1  # The env updates counts after each shot
            self.cycle = None
            self.idle_pos, self.idle_since = end_pos, shot.t_launch
            if has_puck_after:
                self._start_cycle(shot.t_launch, end_pos)
        if self.cycle is None and has_puck and not shots:
            self._start_cycle(t, self.idle_pos)
        return shots

    def phase(self, t: float) -> str:
        if self.cycle is None:
            return "idle"
        k = int(np.searchsorted(self.cycle.times, t, side="right")) - 1
        return PHASES[min(max(k, 0), 3)]

    def hand_true(self, t: float) -> np.ndarray:
        """True hand [x, y, vx, vy]."""
        if self.cycle is None:
            return np.array([*self.idle_pos, 0.0, 0.0])
        times, wp = self.cycle.times, self.cycle.waypoints
        if t <= times[1]:  # Reload: travel to the pick point during the first half, then wait
            t_mid = self._reload_mid
            if t <= t_mid:
                f = (t - times[0]) / max(t_mid - times[0], 1e-9)
                v = (wp[1] - wp[0]) / max(t_mid - times[0], 1e-9)
                return np.array([*(wp[0] + f * (wp[1] - wp[0])), *v])
            return np.array([*wp[1], 0.0, 0.0])
        k = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        dt = max(times[k + 1] - times[k], 1e-9)
        f = np.clip((t - times[k]) / dt, 0.0, 1.0)
        v = (wp[k + 1] - wp[k]) / dt
        return np.array([*(wp[k] + f * (wp[k + 1] - wp[k])), *v])

    def hand_observed(self, t: float) -> np.ndarray:
        h = self.hand_true(t)
        noise = self.cfg["hand_noise_m"]
        h[:2] += self.rng.normal(0.0, noise, 2)
        h[2:] += self.rng.normal(0.0, noise / 0.05, 2)  # Velocity from noisy positions over ~50 ms
        return h

    def time_to_release(self, t: float) -> float:
        return np.inf if self.cycle is None else max(self.cycle.shot.t_launch - t, 0.0)

    @property
    def next_shot(self) -> Shot | None:
        return None if self.cycle is None else self.cycle.shot
