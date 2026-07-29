"""Configuration objects with small, explicit validation checks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShieldConfig:
    """Parameters of the local one-step Lyapunov shield."""

    rho: float
    alpha: float
    u_max: float = 10.0
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.u_max <= 0.0:
            raise ValueError("u_max must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class ControllerConfig:
    """Residual scaling and actuator bound."""

    beta: float = 3.0
    u_max: float = 10.0

    def __post_init__(self) -> None:
        if self.u_max <= 0.0:
            raise ValueError("u_max must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.beta > 0.3 * self.u_max + 1e-12:
            raise ValueError("beta must not exceed 0.3 * u_max")


@dataclass(frozen=True)
class LQRModel:
    """Nominal discrete local model used by the shield.

    The model is z_next = A @ z + B * u, where z is the local upright state
    error and u is the total force sent to the plant.
    """

    A: FloatArray
    B: FloatArray
    P: FloatArray
    equilibrium_state: FloatArray

    def __post_init__(self) -> None:
        a = np.asarray(self.A, dtype=float)
        b = np.asarray(self.B, dtype=float).reshape(-1)
        p = np.asarray(self.P, dtype=float)
        equilibrium = np.asarray(self.equilibrium_state, dtype=float).reshape(-1)

        n = equilibrium.size
        if a.shape != (n, n):
            raise ValueError(f"A must have shape {(n, n)}, got {a.shape}")
        if b.shape != (n,):
            raise ValueError(f"B must have shape {(n,)} or {(n, 1)}")
        if p.shape != (n, n):
            raise ValueError(f"P must have shape {(n, n)}, got {p.shape}")
        if not np.allclose(p, p.T, atol=1e-10):
            raise ValueError("P must be symmetric")
        if np.min(np.linalg.eigvalsh(p)) <= 0.0:
            raise ValueError("P must be positive definite")

        # Store normalized arrays even when callers supplied lists or B as n x 1.
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "P", p)
        object.__setattr__(self, "equilibrium_state", equilibrium)

