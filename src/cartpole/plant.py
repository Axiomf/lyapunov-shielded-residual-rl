"""Nonlinear cart-pole plant and sampled-data transition utilities.

Theoretical role in this project
--------------------------------
The continuous plant is the nonlinear ordinary differential equation

    s_dot = f(s, u; p),

where ``s = [x, theta, x_dot, theta_dot]`` and ``p`` contains the physical
parameters. Passing the actual plant configuration models the rollout plant;
passing the nominal configuration gives the model used by the controller or
shield. In particular, pole-mass mismatch enters through the value of
``PlantConfig.pole_mass``.

The controller is digital, so :func:`step_rk4` turns the ODE into the
sampled-data map

    s_(k+1) = F_h(s_k, u_k; p),

using a zero-order-held input and RK4 integration. This one-step map is the
object used in closed-loop simulations, finite-difference Jacobians, empirical
basin tests, and nominal one-step Lyapunov-change calculations.

This module enforces the actuator bound and detects the cart-position limit.
Those operations are simulation constraints and checks; they are not, by
themselves, formal safety or region-of-attraction guarantees.
"""

import math

import numpy as np

from .config import ObservationConfig, PlantConfig
from .data import FloatArray, State


def wrap_angle(angle: float) -> float:
    """Return the representative of ``angle`` in ``[-pi, pi)``.

    Angles that differ by an integer multiple of ``2*pi`` represent the same
    pole orientation. The half-open interval gives that equivalence class one
    consistent numerical representative; in particular, ``pi`` maps to
    ``-pi``.
    """

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip_force(force: float, plant: PlantConfig) -> float:
    """Project a scalar force onto the actuator interval.

    This implements ``u = clip(u_proposed, -u_max, u_max)`` for all three
    controllers. For a scalar, clipping is also the Euclidean projection onto
    the closed admissible interval ``[-u_max, u_max]``.
    """

    return float(np.clip(force, -plant.u_max, plant.u_max))


def state_derivative(
    state_array: FloatArray,
    force: float,
    plant: PlantConfig,
) -> FloatArray:
    """Evaluate the continuous nonlinear dynamics ``s_dot = f(s, u; p)``.

    Args:
        state_array: State ``[x, theta, x_dot, theta_dot]`` with shape ``(4,)``.
        force: Horizontal cart force ``u`` in newtons. This function does not
            clip it; clipping occurs at the sampled transition boundary.
        plant: Actual or nominal physical parameters ``p``.

    Returns:
        The derivative ``[x_dot, theta_dot, x_ddot, theta_ddot]`` with shape
        ``(4,)``.

    The coordinate ``theta = 0`` is upright, so it is an unstable open-loop
    equilibrium orientation. The pole is represented by a point mass ``m_p``
    at distance ``l`` from the pivot. Eliminating the coupled accelerations
    gives

        D = m_c + m_p sin(theta)^2,
        x_ddot = (u - m_p g sin(theta) cos(theta)
                  + m_p l theta_dot^2 sin(theta)) / D,
        theta_ddot = (g sin(theta) - cos(theta) x_ddot) / l.

    Consequently, every state ``[x, 0, 0, 0]`` with zero force is an
    equilibrium of the unconstrained open-loop equations. The closed-loop
    controller selects the desired cart location and may create an empirical
    equilibrium bias when a frozen residual action is nonzero near upright.
    """

    # Position x does not appear explicitly because the ideal track is
    # translationally invariant; x matters separately through its track limit.
    _, theta, x_dot, theta_dot = state_array
    m_c = plant.cart_mass
    m_p = plant.pole_mass
    length = plant.pole_length
    gravity = plant.gravity

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # This denominator results from solving the two coupled mechanical
    # equations for x_ddot and theta_ddot.
    denominator = m_c + m_p * sin_theta**2

    x_ddot = (
        force
        - m_p * gravity * sin_theta * cos_theta
        + m_p * length * theta_dot**2 * sin_theta
    ) / denominator
    theta_ddot = (
        gravity * sin_theta / length
        - cos_theta * x_ddot / length
    )

    return np.array(
        [x_dot, theta_dot, x_ddot, theta_ddot],
        dtype=np.float64,
    )


def step_rk4(state: State, force: float, plant: PlantConfig) -> State:
    """Advance the plant by one sampled-data control period.

    The proposed force is clipped once and then held constant for the complete
    period ``control_dt`` (zero-order hold). The period is divided into
    ``rk4_substeps`` equal intervals, and classical fourth-order Runge--Kutta
    approximates the nonlinear flow on each interval.

    Mathematically, this function approximates the discrete transition map
    ``F_h``. Analyses that use one-step differences, such as
    ``Delta V = V(s_(k+1)) - V(s_k)``, must use this same map and the appropriate
    nominal or actual plant configuration to remain consistent.
    """

    # Saturate at the plant interface, then use exactly the same force in every
    # RK4 stage to model the zero-order hold.
    force = clip_force(force, plant)
    y = state.as_array()
    h = plant.control_dt / plant.rk4_substeps

    for _ in range(plant.rk4_substeps):
        k1 = state_derivative(y, force, plant)
        k2 = state_derivative(y + 0.5 * h * k1, force, plant)
        k3 = state_derivative(y + 0.5 * h * k2, force, plant)
        k4 = state_derivative(y + h * k3, force, plant)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return State.from_array(y)


def normalize_state(state: State, config: ObservationConfig) -> FloatArray:
    """Construct the normalized observation supplied to the SAC actor.

    The state order remains ``[x, theta, x_dot, theta_dot]``. The angle is
    first represented in ``[-pi, pi)``, each component is divided by its
    matching scale, and the result is clipped componentwise to ``[-1, 1]``.

    The pole-mass multiplier ``mu`` is intentionally absent. Thus a frozen
    policy cannot condition directly on ``mu``; any robustness to mass mismatch
    must arise from the observed state evolution learned under domain
    randomization. Wrapping affects only the actor observation, not the
    integrated physical state.
    """

    values = state.as_array()
    values[1] = wrap_angle(values[1])
    scales = np.asarray(config.scales, dtype=np.float64)
    return np.clip(values / scales, -1.0, 1.0)


def has_track_violation(state: State, plant: PlantConfig) -> bool:
    """Return whether the cart lies strictly outside the permitted track.

    The boundary ``abs(x) == x_limit`` is still admissible; a violation occurs
    only when ``abs(x) > x_limit``. This predicate reports the event used in
    evaluation and does not modify the state or claim forward invariance.
    """

    return abs(state.x) > plant.x_limit

