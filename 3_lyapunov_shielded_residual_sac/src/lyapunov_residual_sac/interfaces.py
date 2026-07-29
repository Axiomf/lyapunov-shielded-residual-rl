"""Small interfaces implemented by the other two project parts."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class PhysicsController(Protocol):
    """Energy-shaping / LQR controller supplied by the physics project."""

    def action(self, state: FloatArray) -> float:
        """Return the current physics-controller force."""

    def lqr_action(self, state: FloatArray) -> float:
        """Return nominal LQR force for the shield's fallback."""


class ResidualPolicy(Protocol):
    """Frozen SAC actor supplied by the residual-SAC project."""

    def action(
        self,
        normalized_state: FloatArray,
        deterministic: bool = True,
    ) -> float:
        """Return a dimensionless residual action, nominally in [-1, 1]."""


class StateNormalizer(Protocol):
    """The same observation normalization used during SAC training."""

    def __call__(self, state: FloatArray) -> FloatArray:
        """Return the normalized full state."""

