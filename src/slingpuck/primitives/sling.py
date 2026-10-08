"""Frozen sling primitive for Phase 3: timing and success probability.

The Phase 2 skill (slide, pull, look, correct, release) is used as a black box:
it takes move_gate_to_band_s (if the arm starts at the gate), then fetch_time_s
(bring the next puck from the agent half to the band) plus sling_core_s,
then the puck flies sling_flight_s to the gate and goes through with
probability sling_success_prob (measured by eval/hole_rate.py).
"""
from dataclasses import dataclass


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
