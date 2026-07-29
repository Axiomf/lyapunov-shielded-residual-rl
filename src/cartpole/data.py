from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class State:
    """The only state format passed through the project."""

    x: float
    theta: float
    x_dot: float
    theta_dot: float

    def as_array(self) -> FloatArray:
        return np.array(
            [self.x, self.theta, self.x_dot, self.theta_dot],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: FloatArray) -> "State":
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (4,):
            raise ValueError("State array must have shape (4,).")
        return cls(
            x=float(values[0]),
            theta=float(values[1]),
            x_dot=float(values[2]),
            theta_dot=float(values[3]),
        )


@dataclass(frozen=True)
class ControlDecision:
    """One controller output and small diagnostics for later logging."""

    force: float
    physics_force: float
    residual_force: float
    controller_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    state: State
    decision: ControlDecision
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class Rollout:
    mu: float
    initial_state: State
    transitions: tuple[Transition, ...]
    track_violation: bool

    @property
    def total_reward(self) -> float:
        return sum(item.reward for item in self.transitions)

