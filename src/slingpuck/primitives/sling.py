"""Sling primitive: which puck and which shot, how long it takes, and its success probability.

Two models (shot.selection in env.yaml):

fixed    the old black box: move_gate_to_band_s (if the arm starts at the gate), then fetch_time_s plus
         sling_core_s, then the puck flies sling_flight_s and goes through with sling_success_prob.
greedy   layer 2 of the adaptation plan. For every puck on the agent half and every candidate shot
         (launch x, aim, pull) on a grid, the shot model gives P(through), and the time is
             T = pick_time + (|arm -> puck| + |puck -> seat|) / arm_speed + seat_release + pull / draw_speed,
         plus free_puck_s if another puck leaves no room for the open jaws (the arm first pushes the puck out).
         Choose the puck and shot with the highest (P + ucb_beta * std) / T. Pucks jammed against a side
         wall go last.
random   a random puck (the "random puck order" baseline), with the most likely shot.

All times are placeholders until they are measured on the robot.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlingPrimitive:
    move_gate_to_band_s: float
    fetch_s: float
    core_s: float
    flight_s: float
    success_prob: float

    @classmethod
    def from_config(cls, cfg: dict) -> "SlingPrimitive":
        m = cfg["match"]
        return cls(m["move_gate_to_band_s"], m["fetch_time_s"], m["sling_core_s"], m["sling_flight_s"],
                   m["sling_success_prob"])

    @property
    def at_band_s(self) -> float:
        """Time from the arm at the band to the release: fetch the puck, then the sling core."""
        return self.fetch_s + self.core_s


@dataclass
class ShotChoice:
    index: int                # Index of the puck in the list that was given
    shot: np.ndarray          # (launch x, aim, pull)
    p: float                  # P(through), the shot model's mean
    score_p: float            # P used for the choice (mean + ucb_beta * std)
    time_s: float             # From the arm's position to the release
    seat: np.ndarray          # Where the puck sits on the band (board frame)


class ShotSelector:
    def __init__(self, cfg: dict, model):
        s, b = cfg["shot"], cfg["board"]
        self.s = s
        self.mode = s["selection"]
        if self.mode not in ("greedy", "random"):
            raise ValueError(f"ShotSelector needs shot.selection greedy or random, not {self.mode!r}")
        self.band_y = -(0.5 * b["length_m"] - cfg["band"]["band_offset_from_end_wall_m"])
        self.wall_x = 0.5 * b["width_m"] - cfg["puck"]["radius_m"]
        axes = [np.linspace(*s[k], n) for k, n in zip(("launch_x_range_m", "aim_range_rad", "pull_range_m"),
                                                      s["grid"])]
        self.shots = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        self.p, std = model.predict(self.shots)
        self.score_p = self.p + s["ucb_beta"] * std
        self.draw_s = self.shots[:, 2] / s["draw_speed_m_s"]
        self.seats = np.stack([self.shots[:, 0], np.full(len(self.shots), self.band_y)], 1)

    def times(self, puck_xy, from_xy) -> np.ndarray:
        """T for one puck and every candidate shot."""
        s = self.s
        carry = np.linalg.norm(np.asarray(puck_xy) - from_xy) + np.linalg.norm(self.seats - puck_xy, axis=1)
        return s["pick_time_s"] + carry / s["arm_speed_m_s"] + s["seat_release_s"] + self.draw_s

    def free_times(self, pucks_xy) -> np.ndarray:
        """Extra time per puck: free_puck_s if a neighbour leaves no room for the open jaws."""
        d = np.abs(pucks_xy[:, None] - pucks_xy[None])
        near = (d[..., 0] < self.s["jaw_room_x_m"]) & (d[..., 1] < self.s["jaw_room_y_m"])
        np.fill_diagonal(near, False)
        return near.any(1) * self.s["free_puck_s"]

    def jammed(self, puck_xy) -> bool:
        return abs(puck_xy[0]) > self.wall_x - self.s["jam_margin_m"]

    def choose(self, pucks_xy, from_xy, rng=None) -> ShotChoice | None:
        if len(pucks_xy) == 0:
            return None
        pucks_xy, from_xy = np.asarray(pucks_xy, float), np.asarray(from_xy, float)
        free = [i for i in range(len(pucks_xy)) if not self.jammed(pucks_xy[i])] or list(range(len(pucks_xy)))
        extra = self.free_times(pucks_xy)
        if self.mode == "random" and rng is not None:
            i = int(free[int(rng.integers(len(free)))])
            k = int(np.argmax(self.p))
            t = self.times(pucks_xy[i], from_xy)[k] + extra[i]
        else:
            T = np.array([self.times(pucks_xy[i], from_xy) + extra[i] for i in free])    # (n_free, n_shots)
            v = self.score_p[None] / T
            a, k = np.unravel_index(int(np.argmax(v)), v.shape)
            i, t = int(free[a]), float(T[a, k])
        return ShotChoice(i, self.shots[k].copy(), float(self.p[k]), float(self.score_p[k]), float(t),
                          self.seats[k].copy())
