"""Physics backend interface for puck simulation.

Envs talk only to this interface, so the fast NumPy sim (Fast2DPuckSim) can be
swapped for a PyBullet (or later Isaac) backend without env changes.

Board frame: origin at the board center, x across the width, y along the length.
The agent's half is y < 0, the opponent's half is y > 0, and the center divider
with the gate is at y = 0.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GateCrossing:
    puck: int       # Puck index
    direction: int  # +1: agent half -> opponent half, -1: opponent half -> agent half
    x: float        # Crossing x position


@dataclass(frozen=True)
class PaddleState:
    """Kinematic box moved by the arm. The sim does not push back on it."""
    center: np.ndarray       # (2,) m
    angle: float             # Rotation of the long axis from +x, rad
    vel: np.ndarray          # (2,) center velocity, m/s
    omega: float             # Angular velocity, rad/s (counter-clockwise positive)
    half_width: float        # Along the long axis, m
    half_thickness: float    # Along the normal, m
    restitution: float


@dataclass(frozen=True)
class StepEvents:
    crossings: list[GateCrossing] = field(default_factory=list)
    paddle_contacts: list[int] = field(default_factory=list)  # Puck indices that touched the paddle


class PuckPhysicsBackend(ABC):
    n_pucks: int

    @abstractmethod
    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Set puck positions and velocities, shape (n_pucks, 2)."""

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of (pos, vel), shape (n_pucks, 2)."""

    @abstractmethod
    def set_paddle(self, paddle: PaddleState | None) -> None:
        """Set the paddle pose for the next step(s). None removes the paddle."""

    @abstractmethod
    def step(self, dt: float) -> StepEvents:
        """Advance the sim by dt and return the events in this step."""

    @abstractmethod
    def kinetic_energy(self) -> float:
        """Total translational kinetic energy of all pucks, in J."""
