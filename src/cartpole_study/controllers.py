"""Nominal controller and deterministic policy adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_continuous_are

from .config import ControllerConfig
from .plant import CartPolePlant, wrap_angle
from .shield import LyapunovShield, ShieldResult


class NominalController:
    """Energy shaping away from upright and LQR close to upright."""

    def __init__(self, nominal_plant: CartPolePlant, config: ControllerConfig):
        self.plant = nominal_plant
        self.config = config
        a, b = nominal_plant.upright_linear_model()
        q = np.diag(np.asarray(config.lqr_q, dtype=np.float64))
        r = np.array([[config.lqr_r]], dtype=np.float64)
        self.lyapunov_matrix = solve_continuous_are(a, b, q, r)
        self.gain = np.linalg.solve(r, b.T @ self.lyapunov_matrix)

    def action(self, state: ArrayLike) -> float:
        state = np.asarray(state, dtype=np.float64).copy()
        state[2] = wrap_angle(float(state[2]))
        if abs(state[2]) <= self.config.switch_angle:
            force = -(self.gain @ state).item()
        else:
            energy_error = self.plant.pole_energy(state) - (
                self.plant.pole_mass
                * self.plant.config.gravity
                * self.plant.config.pole_length
            )
            force = (
                self.config.energy_gain
                * energy_error
                * state[3]
                * np.cos(state[2])
                - self.config.cart_position_gain * state[0]
                - self.config.cart_velocity_gain * state[1]
            )
        return float(
            np.clip(force, -self.plant.config.force_limit, self.plant.config.force_limit)
        )

    def lyapunov_value(self, state: ArrayLike) -> float:
        local_state = np.asarray(state, dtype=np.float64).copy()
        local_state[2] = wrap_angle(float(local_state[2]))
        return float(local_state @ self.lyapunov_matrix @ local_state)

    def lyapunov_derivative(
        self, actual_plant: CartPolePlant, state: ArrayLike, force: float
    ) -> float:
        local_state = np.asarray(state, dtype=np.float64).copy()
        local_state[2] = wrap_angle(float(local_state[2]))
        gradient = 2.0 * self.lyapunov_matrix @ local_state
        return float(gradient @ actual_plant.derivative(local_state, force))


@dataclass(frozen=True)
class PolicyResult:
    force: float
    nominal_force: float
    residual_force: float
    shield: ShieldResult | None = None


class ClosedLoopPolicy:
    """Convert a normalized SAC residual into the applied physical force."""

    def __init__(
        self,
        name: str,
        nominal: NominalController,
        residual_limit: float,
        model: Any | None = None,
        shield: LyapunovShield | None = None,
    ):
        if name not in {"nominal", "residual", "shielded"}:
            raise ValueError(f"Unknown controller: {name}")
        if name != "nominal" and model is None:
            raise ValueError(f"{name} requires a trained SAC model")
        self.name = name
        self.nominal = nominal
        self.residual_limit = residual_limit
        self.model = model
        self.shield = shield

    def action(self, actual_plant: CartPolePlant, state: ArrayLike) -> PolicyResult:
        nominal_force = self.nominal.action(state)
        residual_force = 0.0
        if self.model is not None:
            normalized, _ = self.model.predict(
                np.asarray(state, dtype=np.float32), deterministic=True
            )
            residual_force = self.residual_limit * float(np.asarray(normalized).item())

        candidate = float(
            np.clip(
                nominal_force + residual_force,
                -actual_plant.config.force_limit,
                actual_plant.config.force_limit,
            )
        )
        shield_result = None
        if self.shield is not None:
            shield_result = self.shield.project(
                actual_plant=actual_plant,
                state=state,
                nominal_force=nominal_force,
                candidate_force=candidate,
            )
            candidate = shield_result.force
            residual_force = candidate - nominal_force

        return PolicyResult(
            force=candidate,
            nominal_force=nominal_force,
            residual_force=residual_force,
            shield=shield_result,
        )
