import math

import numpy as np

from .config import ObservationConfig, PlantConfig
from .data import FloatArray, State


def wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip_force(force: float, plant: PlantConfig) -> float:
    return float(np.clip(force, -plant.u_max, plant.u_max))


def state_derivative(
    state_array: FloatArray,
    force: float,
    plant: PlantConfig,
) -> FloatArray:
    """Continuous nonlinear cart-pole dynamics.

    Input:
        state_array: [x, theta, x_dot, theta_dot], shape (4,).
        force: scalar horizontal force in newtons.
        plant: actual or nominal parameters.

    Output:
        [x_dot, theta_dot, x_ddot, theta_ddot], shape (4,).

    The pole is treated as a point mass at distance `pole_length`.
    theta = 0 is the unstable upright position.
    """

    _, theta, x_dot, theta_dot = state_array
    m_c = plant.cart_mass
    m_p = plant.pole_mass
    length = plant.pole_length
    gravity = plant.gravity

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
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
    """Advance one control period with a zero-order-held force."""

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
    """Return the actor observation; mass multiplier is intentionally absent."""

    values = state.as_array()
    values[1] = wrap_angle(values[1])
    scales = np.asarray(config.scales, dtype=np.float64)
    return np.clip(values / scales, -1.0, 1.0)


def has_track_violation(state: State, plant: PlantConfig) -> bool:
    return abs(state.x) > plant.x_limit

