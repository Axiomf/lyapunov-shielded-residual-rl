"""
Runnable toy backend for software smoke tests.

This is not the cart-pole model and must not be used for research results.
"""

import numpy as np

from residual_sac.interfaces import PhysicsController, Plant


class DummyPlant(Plant):
    def __init__(self, control_period: float = 0.02) -> None:
        self.control_period = control_period
        self.state = np.zeros(4, dtype=np.float64)
        self.mass_scale = 1.0

    def reset(self, initial_state: np.ndarray, pole_mass_scale: float) -> np.ndarray:
        self.state = np.asarray(initial_state, dtype=np.float64).copy()
        self.mass_scale = float(pole_mass_scale)
        return self.state.copy()

    def step(self, force: float) -> tuple[np.ndarray, bool]:
        # Toy nonlinear dynamics with RK4, included only so the project runs.
        def derivative(state: np.ndarray) -> np.ndarray:
            x, theta, x_dot, theta_dot = state
            return np.array(
                [
                    x_dot,
                    theta_dot,
                    force / self.mass_scale - 0.15 * x_dot,
                    9.0 * np.sin(theta)
                    + 1.1 * force / self.mass_scale
                    - 0.2 * theta_dot,
                ],
                dtype=np.float64,
            )

        dt = self.control_period
        k1 = derivative(self.state)
        k2 = derivative(self.state + 0.5 * dt * k1)
        k3 = derivative(self.state + 0.5 * dt * k2)
        k4 = derivative(self.state + dt * k3)
        self.state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        track_violation = bool(abs(self.state[0]) > 2.4)
        return self.state.copy(), track_violation


class DummyPhysicsController(PhysicsController):
    def __init__(self, u_max: float = 10.0) -> None:
        self.u_max = u_max

    def action(self, state: np.ndarray) -> float:
        x, theta, x_dot, theta_dot = state
        force = -0.8 * x - 0.7 * x_dot - 7.0 * theta - 2.0 * theta_dot
        return float(np.clip(force, -self.u_max, self.u_max))

