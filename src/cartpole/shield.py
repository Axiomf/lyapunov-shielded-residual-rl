from dataclasses import dataclass

import numpy as np

from .config import PlantConfig, ShieldConfig
from .control_math import LQRData, nominal_lqr_force
from .data import State
from .plant import step_rk4


@dataclass(frozen=True)
class ShieldResult:
    force: float
    active: bool
    projected: bool
    infeasible: bool
    v_before: float
    nominal_delta_v: float


def lyapunov_value(state: State, p: np.ndarray) -> float:
    z = state.as_array()
    return float(z @ p @ z)


def project_with_lyapunov_shield(
    state: State,
    proposed_force: float,
    nominal_plant: PlantConfig,
    lqr: LQRData,
    config: ShieldConfig,
) -> ShieldResult:
    """Approximate the nearest admissible scalar action with a force grid.

    The feasibility check is based on the nominal nonlinear one-step map:

        V(z_next) - V(z) <= -alpha * V(z)

    Outside V <= rho, the proposed bounded force is returned unchanged.
    """

    proposed_force = float(
        np.clip(proposed_force, -nominal_plant.u_max, nominal_plant.u_max)
    )
    v_before = lyapunov_value(state, lqr.p)

    if v_before > config.rho:
        return ShieldResult(
            force=proposed_force,
            active=False,
            projected=False,
            infeasible=False,
            v_before=v_before,
            nominal_delta_v=float("nan"),
        )

    grid = np.linspace(
        -nominal_plant.u_max,
        nominal_plant.u_max,
        config.grid_size,
    )
    candidates = np.unique(np.append(grid, proposed_force))
    feasible: list[tuple[float, float]] = []

    for force in candidates:
        next_state = step_rk4(state, float(force), nominal_plant)
        delta_v = lyapunov_value(next_state, lqr.p) - v_before
        threshold = -config.alpha * v_before + config.feasibility_tolerance
        if delta_v <= threshold:
            feasible.append((float(force), float(delta_v)))

    if feasible:
        selected_force, selected_delta_v = min(
            feasible,
            key=lambda item: abs(item[0] - proposed_force),
        )
        return ShieldResult(
            force=selected_force,
            active=True,
            projected=not np.isclose(selected_force, proposed_force),
            infeasible=False,
            v_before=v_before,
            nominal_delta_v=selected_delta_v,
        )

    fallback = nominal_lqr_force(state, lqr, nominal_plant)
    fallback_next = step_rk4(state, fallback, nominal_plant)
    fallback_delta_v = lyapunov_value(fallback_next, lqr.p) - v_before
    return ShieldResult(
        force=fallback,
        active=True,
        projected=not np.isclose(fallback, proposed_force),
        infeasible=True,
        v_before=v_before,
        nominal_delta_v=fallback_delta_v,
    )

