from dataclasses import dataclass

import numpy as np

from .config import PlantConfig, ShieldConfig
from .data import State

from .plant import step_rk4
from .control_math import LQRData, nominal_lqr_force



@dataclass(frozen=True)
class ShieldResult:
    """Applied force and shield diagnostics for one control period.

    The fields separate three events that are useful in the experiments:

    - ``active`` means the state was inside the nominal ellipsoid
      ``V(z) <= rho``, so the shield tested the one-step decrease condition.
    - ``projected`` means the returned force differs from the bounded proposed
      force. An active shield need not project an already feasible proposal.
    - ``infeasible`` means no force in the *finite sampled candidate set*
      satisfied the nominal decrease test, so nominal LQR was used as fallback.
      It does not prove that the continuous force interval is infeasible.

    ``v_before`` is the value of the nominal quadratic Lyapunov candidate at the
    current state. ``nominal_delta_v`` is its one-step change under the nominal
    RK4 model and returned force; it is ``NaN`` when the shield is inactive and
    therefore does not evaluate a one-step constraint.

    The dataclass is frozen so logged controller decisions cannot be modified
    after a rollout.
    """

    force: float
    active: bool
    projected: bool
    infeasible: bool
    v_before: float
    nominal_delta_v: float


def lyapunov_value(state: State, P: np.ndarray) -> float:
    """Evaluate the nominal quadratic Lyapunov candidate ``V(z) = z.T P z``.

    Here ``z = s - s_star`` is the displacement from the upright equilibrium.
    This project uses ``s_star = [0, 0, 0, 0]``, so the canonical state array is
    already ``z``. The matrix ``P`` is the discrete-time LQR Riccati matrix.

    This function only evaluates a scalar candidate function. Whether ``V``
    decreases must be checked against a specified closed-loop one-step map.
    """

    z = state.as_array()
    return float(z @ P @ z)


def project_with_lyapunov_shield(
    state: State,
    proposed_force: float,
    nominal_plant: PlantConfig,
    lqr: LQRData,
    config: ShieldConfig,
) -> ShieldResult:
    """Filter a proposed force with a local nominal Lyapunov condition.

    This function implements the project's shielded-residual controller near
    the upright equilibrium. In dynamical-systems terms, it is a sampled-data,
    control-Lyapunov-like action filter. For the nominal nonlinear one-step map
    ``F_h^0`` produced by zero-order hold and RK4, it approximately solves

    ``minimize_u  |u - u_proposed|``

    subject to

    ``V(F_h^0(z, u)) - V(z) <= -alpha * V(z)``

    and ``|u| <= u_max``. The absolute value is the scalar Euclidean distance,
    so the selected force changes the proposed physics-plus-residual action as
    little as possible among the tested candidates.

    The procedure is:

    1. Clip the proposal to the actuator limits.
    2. Return it unchanged when ``V(z) > rho``.
    3. Inside ``V(z) <= rho``, test a deterministic force grid plus the exact
       proposed force using the nominal sampled-data model.
    4. Return the feasible candidate nearest to the proposal. If the sampled
       set has no feasible candidate, return the bounded nominal LQR force.

    The condition is model based: it uses the controller's nominal pole mass,
    not the mass-mismatched evaluation plant. Also, the scalar force interval is
    sampled rather than solved continuously, and the fallback is not asserted
    to satisfy the decrease inequality. Therefore ``V <= rho`` is a chosen
    shield-activation ellipsoid, not a certified region of attraction. Realized
    Lyapunov change on the true plant, local Jacobian spectra, and empirical
    basin measurements remain experimental evidence rather than global safety
    or stability guarantees.

    Args:
        state: Current cart-pole state in canonical project order.
        proposed_force: Physics-plus-residual force before shield projection.
        nominal_plant: Controller model and sampled-data integration settings.
        lqr: Nominal LQR matrices, including the Riccati matrix ``P``.
        config: Shield ellipsoid, decrease rate, grid, and tolerance settings.

    Returns:
        The applied force together with activation, projection, feasibility,
        and nominal Lyapunov diagnostics.
    """

    # Enforce the physical actuator interval before measuring projection distance.
    proposed_force = float(
        np.clip(proposed_force, -nominal_plant.u_max, nominal_plant.u_max)
    )
    v_before = lyapunov_value(state, lqr.P)

    # The shield is intentionally local; outside its ellipsoid it has no authority.
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
    # Including the exact proposal avoids unnecessary grid quantization when the
    # unmodified proposal already satisfies the nominal decrease condition.
    candidates = np.unique(np.append(grid, proposed_force))
    feasible: list[tuple[float, float]] = []
    threshold = -config.alpha * v_before + config.feasibility_tolerance

    for force in candidates:
        # Use the same nominal zero-order-held RK4 map as the controller model.
        next_state = step_rk4(state, float(force), nominal_plant)
        delta_v = lyapunov_value(next_state, lqr.P) - v_before
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

    # This is an operational fallback, not a claim that the decrease constraint
    # is feasible. Its nominal Delta V is still recorded for later analysis.
    fallback = nominal_lqr_force(state, lqr, nominal_plant)
    fallback_next = step_rk4(state, fallback, nominal_plant)
    fallback_delta_v = lyapunov_value(fallback_next, lqr.P) - v_before
    return ShieldResult(
        force=fallback,
        active=True,
        projected=not np.isclose(fallback, proposed_force),
        infeasible=True,
        v_before=v_before,
        nominal_delta_v=fallback_delta_v,
    )

