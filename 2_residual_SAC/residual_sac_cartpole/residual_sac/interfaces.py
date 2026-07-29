from abc import ABC, abstractmethod

import numpy as np


class Plant(ABC):
    """Interface that the completed nonlinear plant must implement."""

    @abstractmethod
    def reset(self, initial_state: np.ndarray, pole_mass_scale: float) -> np.ndarray:
        """Set state and mass, then return a copy of the resulting state."""

    @abstractmethod
    def step(self, force: float) -> tuple[np.ndarray, bool]:
        """Advance one control period and return (next_state, track_violation)."""


class PhysicsController(ABC):
    """Interface for the energy-shaping plus hysteretic-LQR controller."""

    def reset(self) -> None:
        """Reset controller mode/hysteresis state at the start of a rollout."""

    @abstractmethod
    def action(self, state: np.ndarray) -> float:
        """Return the scalar physics-controller force for the current state."""

