from dataclasses import dataclass
import math

import numpy as np

from .config import CartPoleParams, SwingUpConfig
from .math_utils import clip, wrap_angle


@dataclass(frozen=True)
class EnergyOutput:
    force: float
    desired_cart_acceleration: float
    energy: float
    desired_energy: float
    energy_error: float


def pendulum_energy(state, params):
    """Pole energy, with its maximum at the upright equilibrium."""

    theta = float(state[1])
    theta_dot = float(state[3])
    m = params.pole_mass
    length = params.pole_com_length
    inertia_at_pivot = params.pole_inertia + m * length * length
    kinetic = 0.5 * inertia_at_pivot * theta_dot * theta_dot
    potential = m * params.gravity * length * math.cos(theta)
    return kinetic + potential


def upright_energy(params):
    return params.pole_mass * params.gravity * params.pole_com_length


def desired_cart_acceleration(state, params, config):
    """Acceleration command that pumps pole energy and recenters the cart."""

    x, theta, x_dot, theta_dot = np.asarray(state, dtype=float)
    energy = pendulum_energy(state, params)
    desired = upright_energy(params)
    error = energy - desired

    energy_term = (
        config.energy_gain * error * theta_dot * math.cos(theta)
    )
    centering_term = (
        -config.cart_position_gain * x - config.cart_velocity_gain * x_dot
    )
    acceleration = energy_term + centering_term

    # The symmetric energy law cannot leave the exact downward rest state.
    angle_from_down = wrap_angle(theta - math.pi)
    if (
        abs(angle_from_down) < config.kick_angle
        and abs(theta_dot) < config.kick_speed
        and abs(energy_term) < 1e-12
    ):
        acceleration += config.kick_acceleration

    return clip(
        acceleration,
        -config.max_cart_acceleration,
        config.max_cart_acceleration,
    )


def force_for_cart_acceleration(state, acceleration, params):
    """Partial feedback linearization: choose force for x_ddot=acceleration."""

    _, theta, x_dot, theta_dot = np.asarray(state, dtype=float)
    m = params.pole_mass
    length = params.pole_com_length
    inertia_at_pivot = params.pole_inertia + m * length * length

    theta_acceleration = (
        m * params.gravity * length * math.sin(theta)
        - params.pole_damping * theta_dot
        - m * length * math.cos(theta) * acceleration
    ) / inertia_at_pivot

    force = (
        (params.cart_mass + m) * acceleration
        + params.cart_damping * x_dot
        + m * length * math.cos(theta) * theta_acceleration
        - m * length * theta_dot * theta_dot * math.sin(theta)
    )
    return force


def energy_shaping_control(state, params, config=None):
    if config is None:
        config = SwingUpConfig()

    acceleration = desired_cart_acceleration(state, params, config)
    force = force_for_cart_acceleration(state, acceleration, params)
    energy = pendulum_energy(state, params)
    desired = upright_energy(params)

    return EnergyOutput(
        force=float(force),
        desired_cart_acceleration=float(acceleration),
        energy=float(energy),
        desired_energy=float(desired),
        energy_error=float(energy - desired),
    )
