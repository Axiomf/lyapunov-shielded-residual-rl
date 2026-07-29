from typing import Protocol

import numpy as np

from .data import FloatArray


class Actor(Protocol):
    """Small boundary between this project and any SAC library.

    Input:
        observation: normalized full state, shape (4,), without mu.
        deterministic: False during data collection; True for evaluation.

    Output:
        One scalar in [-1, 1].
    """

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        ...


class ZeroActor:
    """Runnable placeholder. It produces no residual correction."""

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        del observation, deterministic
        return 0.0


class RandomActor:
    """Simple smoke-test actor; this is not SAC."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        del observation
        if deterministic:
            return 0.0
        return float(self._rng.uniform(-1.0, 1.0))

