"""Nonlinear point-mass cart-pole model.

State order is ``[cart_position, cart_velocity, pole_angle, pole_angular_rate]``.
The upright equilibrium is at angle zero; the hanging configuration is at pi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import PlantConfig

State = NDArray[np.float64]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class CartPolePlant:
    """Continuous dynamics and fixed-step RK4 integration."""

    def __init__(self, config: PlantConfig, pole_mass: float | None = None):
        self.config = config
        self.pole_mass = (
            config.nominal_pole_mass if pole_mass is None else pole_mass
        )

    def derivative(self, state: ArrayLike, force: float) -> State:
        x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
        del x
        m = self.pole_mass
        cart_mass = self.config.cart_mass
        length = self.config.pole_length
        gravity = self.config.gravity
        force = float(np.clip(force, -self.config.force_limit, self.config.force_limit))

        sine = np.sin(theta)
        cosine = np.cos(theta)
        denominator = cart_mass + m - m * cosine**2
        x_ddot = (
            force + m * length * sine * theta_dot**2 - m * gravity * sine * cosine
        ) / denominator
        theta_ddot = (gravity * sine - cosine * x_ddot) / length
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)

    def step(self, state: ArrayLike, force: float) -> State:
        """Advance one configured sample using RK4."""
        state = np.asarray(state, dtype=np.float64)
        dt = self.config.dt
        k1 = self.derivative(state, force)
        k2 = self.derivative(state + 0.5 * dt * k1, force)
        k3 = self.derivative(state + 0.5 * dt * k2, force)
        k4 = self.derivative(state + dt * k3, force)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[2] = wrap_angle(next_state[2])
        return next_state

    def pole_energy(self, state: ArrayLike) -> float:
        """Pole energy with its maximum at the upright equilibrium."""
        theta = float(np.asarray(state)[2])
        theta_dot = float(np.asarray(state)[3])
        m = self.pole_mass
        length = self.config.pole_length
        return float(
            0.5 * m * length**2 * theta_dot**2
            + m * self.config.gravity * length * np.cos(theta)
        )

    def upright_linear_model(self) -> tuple[State, NDArray[np.float64]]:
        """Continuous-time linearization at the upright equilibrium."""
        cart_mass = self.config.cart_mass
        m = self.pole_mass
        length = self.config.pole_length
        gravity = self.config.gravity
        a = np.zeros((4, 4), dtype=np.float64)
        b = np.zeros((4, 1), dtype=np.float64)
        a[0, 1] = 1.0
        a[1, 2] = -(m * gravity) / cart_mass
        a[2, 3] = 1.0
        a[3, 2] = ((cart_mass + m) * gravity) / (cart_mass * length)
        b[1, 0] = 1.0 / cart_mass
        b[3, 0] = -1.0 / (cart_mass * length)
        return a, b
