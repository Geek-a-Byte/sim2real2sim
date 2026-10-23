"""Block primitive: the probability that the robot saves a threat at the gate.

The robot guards the gate with its closed gripper (physics/mujoco_match.py). The
match runs millions of decisions, so it does not simulate the block. A threat is
saved if the gripper covers the gate before the puck gets there:

  time_to_cover = (time the puck reaches the gate) - (time the arm got back to the gate)
  p_save = save_prob * Phi((time_to_cover - cover_time_s) / cover_time_std_s)

Phi is the standard normal CDF, so cover_time_std_s is the spread of the real
cover time (0 = a hard cut). An arm that is away from the gate saves nothing
(the env handles that case). All three values are placeholders until they are
measured on the robot.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockPrimitive:
    save_prob: float         # Save probability when the gripper covers the gate in time
    cover_time_s: float      # From the arm back at the gate to the gate covered
    cover_time_std_s: float

    @classmethod
    def from_config(cls, cfg: dict) -> "BlockPrimitive":
        m = cfg["match"]
        return cls(m["block_save_prob"], m["cover_time_s"], m["cover_time_std_s"])

    def p_save(self, time_to_cover_s: float) -> float:
        margin = time_to_cover_s - self.cover_time_s
        if self.cover_time_std_s <= 0.0:
            return self.save_prob if margin >= 0.0 else 0.0
        return self.save_prob * 0.5 * (1.0 + math.erf(margin / (self.cover_time_std_s * math.sqrt(2.0))))
