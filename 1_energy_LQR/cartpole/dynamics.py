import math

import numpy as np

from .config import CartPoleParams


def mass_matrix(theta, params):
    """Return D(q) in D(q) q_ddot = rhs, q = [x, theta]."""

    m = params.pole_mass
    length = params.pole_com_length
    total_pole_inertia = params.pole_inertia + m * length * length

    return np.array(
        [
            [params.cart_mass + m, m * length * math.cos(theta)],
            [m * length * math.cos(theta), total_pole_inertia],
        ],
        dtype=float,
    )


def accelerations(state, force, params):
    """Compute [x_ddot, theta_ddot] from the nonlinear equations."""

    _, theta, x_dot, theta_dot = np.asarray(state, dtype=float)
    m = params.pole_mass
    length = params.pole_com_length

    rhs = np.array(
        [
            force
            - params.cart_damping * x_dot
            + m * length * theta_dot * theta_dot * math.sin(theta),
            m * params.gravity * length * math.sin(theta)
            - params.pole_damping * theta_dot,
        ],
        dtype=float,
    )
    return np.linalg.solve(mass_matrix(theta, params), rhs)


def derivative(state, force, params):
    """Continuous-time vector field for s = [x, theta, x_dot, theta_dot]."""

    state = np.asarray(state, dtype=float)
    x_ddot, theta_ddot = accelerations(state, force, params)
    return np.array([state[2], state[3], x_ddot, theta_ddot], dtype=float)


def rk4_step(state, force, dt, params):
    """One fourth-order Runge--Kutta step with force held constant."""

    state = np.asarray(state, dtype=float)
    k1 = derivative(state, force, params)
    k2 = derivative(state + 0.5 * dt * k1, force, params)
    k3 = derivative(state + 0.5 * dt * k2, force, params)
    k4 = derivative(state + dt * k3, force, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step_zoh(state, force, params):
    """Advance one control period using RK4 and zero-order-held force."""

    substeps = max(1, math.ceil(params.control_period / params.integration_step))
    dt = params.control_period / substeps
    next_state = np.asarray(state, dtype=float).copy()
    for _ in range(substeps):
        next_state = rk4_step(next_state, force, dt, params)
    return next_state
