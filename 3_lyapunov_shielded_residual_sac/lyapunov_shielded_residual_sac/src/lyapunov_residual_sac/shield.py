"""Pure mathematics and a small wrapper for the scalar Lyapunov shield."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import LQRModel, ShieldConfig


FloatArray = NDArray[np.float64]
ActionInterval = tuple[float, float]


def quadratic_value(z: FloatArray, p: FloatArray) -> float:
    """Return V(z) = z.T P z."""

    return float(z @ p @ z)


def nominal_next_state(
    z: FloatArray,
    action: float,
    a: FloatArray,
    b: FloatArray,
) -> FloatArray:
    """Apply the discrete nominal local model."""

    return a @ z + b * float(action)


def lyapunov_constraint_coefficients(
    z: FloatArray,
    model: LQRModel,
    alpha: float,
) -> tuple[float, float, float]:
    """Return qa, qb, qc for g(u) = qa*u^2 + qb*u + qc.

    The constraint g(u) <= 0 is exactly
    V(Az + Bu) - V(z) <= -alpha * V(z).
    """

    az = model.A @ z
    qa = float(model.B @ model.P @ model.B)
    qb = float(2.0 * model.B @ model.P @ az)
    qc = float(az @ model.P @ az - (1.0 - alpha) * quadratic_value(z, model.P))
    return qa, qb, qc


def feasible_interval(
    qa: float,
    qb: float,
    qc: float,
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
) -> Optional[ActionInterval]:
    """Solve qa*u^2 + qb*u + qc <= 0 inside [lower, upper].

    For a positive-definite P, qa = B.T P B is non-negative, so the feasible
    set is an interval. Linear and action-independent edge cases are handled
    explicitly.
    """

    if lower > upper:
        raise ValueError("lower action bound must not exceed upper bound")
    if qa < -tolerance:
        raise ValueError("expected a convex scalar constraint")

    if abs(qa) <= tolerance:
        if abs(qb) <= tolerance:
            return (lower, upper) if qc <= tolerance else None

        boundary = -qc / qb
        if qb > 0.0:
            interval = (lower, min(upper, boundary))
        else:
            interval = (max(lower, boundary), upper)
        return interval if interval[0] <= interval[1] + tolerance else None

    discriminant = qb * qb - 4.0 * qa * qc
    if discriminant < -tolerance:
        return None

    # Treat a tiny negative value as roundoff at a tangent point.
    root_term = np.sqrt(max(0.0, discriminant))
    first_root = (-qb - root_term) / (2.0 * qa)
    second_root = (-qb + root_term) / (2.0 * qa)

    interval = (max(lower, first_root), min(upper, second_root))
    return interval if interval[0] <= interval[1] + tolerance else None


def project_to_interval(action: float, interval: ActionInterval) -> float:
    """Return the nearest scalar point in a closed interval."""

    return float(np.clip(action, interval[0], interval[1]))


def upright_state_error(
    state: FloatArray,
    equilibrium_state: FloatArray,
    angle_index: int = 1,
) -> FloatArray:
    """Return local state error, wrapping the pole angle to [-pi, pi)."""

    z = np.asarray(state, dtype=float).reshape(-1) - equilibrium_state
    # The research model is four-dimensional and uses angle_index=1. Allow
    # lower-dimensional synthetic models in unit tests without special cases.
    if 0 <= angle_index < z.size:
        z[angle_index] = (z[angle_index] + np.pi) % (2.0 * np.pi) - np.pi
    return z


def realized_lyapunov_change(
    state: FloatArray,
    next_state: FloatArray,
    model: LQRModel,
) -> float:
    """Compute empirical Delta V from two states returned by the real plant.

    Unlike ``ShieldResult.nominal_delta_value``, this function does not use
    A or B to predict the next state. Call it after the plant transition.
    """

    z = upright_state_error(
        np.asarray(state, dtype=float).reshape(-1),
        model.equilibrium_state,
    )
    next_z = upright_state_error(
        np.asarray(next_state, dtype=float).reshape(-1),
        model.equilibrium_state,
    )
    if z.shape != model.equilibrium_state.shape:
        raise ValueError("state shape does not match the nominal model")
    if next_z.shape != model.equilibrium_state.shape:
        raise ValueError("next_state shape does not match the nominal model")
    return quadratic_value(next_z, model.P) - quadratic_value(z, model.P)


@dataclass(frozen=True)
class ShieldResult:
    """Output and diagnostics from one shield evaluation."""

    action: float
    proposed_action: float
    value: float
    nominal_delta_value: float
    inside_region: bool
    projected: bool
    feasible: bool
    used_lqr_fallback: bool
    constraint_satisfied: bool
    feasible_interval: Optional[ActionInterval]


class LyapunovShield:
    """Project a scalar action under a nominal local Lyapunov condition."""

    def __init__(self, model: LQRModel, config: ShieldConfig) -> None:
        self.model = model
        self.config = config

    def state_error(self, state: FloatArray) -> FloatArray:
        """Convert the full state to wrapped local upright coordinates."""

        state_array = np.asarray(state, dtype=float).reshape(-1)
        if state_array.shape != self.model.equilibrium_state.shape:
            raise ValueError(
                "state shape does not match the nominal model equilibrium"
            )
        return upright_state_error(state_array, self.model.equilibrium_state)

    def apply(
        self,
        state: FloatArray,
        proposed_action: float,
        lqr_fallback_action: float,
    ) -> ShieldResult:
        """Apply the local shield, or pass the action through outside V <= rho."""

        z = self.state_error(state)
        proposed = float(
            np.clip(
                proposed_action,
                -self.config.u_max,
                self.config.u_max,
            )
        )
        value = quadratic_value(z, self.model.P)
        inside = value <= self.config.rho + self.config.tolerance

        interval: Optional[ActionInterval] = None
        feasible = True
        projected = False
        used_fallback = False
        final_action = proposed

        if inside:
            coefficients = lyapunov_constraint_coefficients(
                z,
                self.model,
                self.config.alpha,
            )
            interval = feasible_interval(
                *coefficients,
                lower=-self.config.u_max,
                upper=self.config.u_max,
                tolerance=self.config.tolerance,
            )

            feasible = interval is not None
            if interval is None:
                final_action = float(
                    np.clip(
                        lqr_fallback_action,
                        -self.config.u_max,
                        self.config.u_max,
                    )
                )
                used_fallback = True
            else:
                final_action = project_to_interval(proposed, interval)
                projected = abs(final_action - proposed) > self.config.tolerance

        next_z = nominal_next_state(
            z,
            final_action,
            self.model.A,
            self.model.B,
        )
        delta_value = quadratic_value(next_z, self.model.P) - value
        constraint_satisfied = (
            delta_value
            <= -self.config.alpha * value + self.config.tolerance
        )

        return ShieldResult(
            action=final_action,
            proposed_action=proposed,
            value=value,
            nominal_delta_value=delta_value,
            inside_region=inside,
            projected=projected,
            feasible=feasible,
            used_lqr_fallback=used_fallback,
            constraint_satisfied=constraint_satisfied,
            feasible_interval=interval,
        )
