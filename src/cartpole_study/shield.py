"""Analytical Lyapunov projection for one bounded force input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .config import ControllerConfig, ShieldConfig
from .plant import CartPolePlant, wrap_angle


@dataclass(frozen=True)
class ShieldResult:
    force: float
    active: bool
    changed: bool
    infeasible: bool


class LyapunovShield:
    """Project a candidate force onto a local Lyapunov half-space.

    The plant is input-affine and has one input, so the derivative condition is
    a scalar interval constraint; a QP dependency is unnecessary.
    """

    def __init__(
        self,
        lyapunov_matrix: np.ndarray,
        controller_config: ControllerConfig,
        shield_config: ShieldConfig,
    ):
        self.p = np.asarray(lyapunov_matrix, dtype=np.float64)
        self.controller_config = controller_config
        self.config = shield_config

    def project(
        self,
        actual_plant: CartPolePlant,
        state: ArrayLike,
        nominal_force: float,
        candidate_force: float,
    ) -> ShieldResult:
        state = np.asarray(state, dtype=np.float64).copy()
        state[2] = wrap_angle(float(state[2]))
        if abs(state[2]) > self.config.active_angle:
            return ShieldResult(candidate_force, False, False, False)

        force_limit = actual_plant.config.force_limit
        residual_limit = self.controller_config.residual_force_limit
        lower = max(-force_limit, nominal_force - residual_limit)
        upper = min(force_limit, nominal_force + residual_limit)

        gradient = 2.0 * self.p @ state
        f_zero = actual_plant.derivative(state, 0.0)
        f_one = actual_plant.derivative(state, 1.0)
        offset = float(gradient @ f_zero)
        coefficient = float(gradient @ (f_one - f_zero))
        bound = -self.config.alpha * float(state @ state)

        infeasible = lower > upper
        if abs(coefficient) <= self.config.tolerance:
            infeasible = infeasible or offset > bound + self.config.tolerance
        elif coefficient > 0.0:
            upper = min(upper, (bound - offset) / coefficient)
            infeasible = infeasible or lower > upper
        else:
            lower = max(lower, (bound - offset) / coefficient)
            infeasible = infeasible or lower > upper

        if infeasible:
            fallback = float(np.clip(nominal_force, -force_limit, force_limit))
            return ShieldResult(
                force=fallback,
                active=True,
                changed=not np.isclose(fallback, candidate_force),
                infeasible=True,
            )

        projected = float(np.clip(candidate_force, lower, upper))
        return ShieldResult(
            force=projected,
            active=True,
            changed=not np.isclose(projected, candidate_force),
            infeasible=False,
        )
