from collections.abc import Callable
from math import pi

import numpy as np

from residual_sac.config import EnvironmentConfig
from residual_sac.interfaces import PhysicsController, Plant


RewardFunction = Callable[[np.ndarray, float, np.ndarray, bool], float]


def wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""
    return float((angle + pi) % (2.0 * pi) - pi)


def normalize_state(
    state: np.ndarray,
    state_scale: tuple[float, float, float, float],
    clip_value: float,
) -> np.ndarray:
    """Normalize the full state without exposing the mass multiplier."""
    normalized_state = np.asarray(state, dtype=np.float32).copy()
    if normalized_state.shape != (4,):
        raise ValueError("state must have shape (4,)")

    normalized_state[1] = wrap_angle(float(normalized_state[1]))
    normalized_state /= np.asarray(state_scale, dtype=np.float32)
    return np.clip(normalized_state, -clip_value, clip_value).astype(np.float32)


def placeholder_reward(
    state: np.ndarray,
    force: float,
    next_state: np.ndarray,
    track_violation: bool,
) -> float:
    """Temporary reward; replace it with the shared experiment reward."""
    del state
    x, theta, x_dot, theta_dot = np.asarray(next_state, dtype=np.float64)
    theta = wrap_angle(float(theta))

    cost = (
        2.0 * theta**2
        + 0.1 * theta_dot**2
        + 0.1 * x**2
        + 0.01 * x_dot**2
        + 0.001 * force**2
    )
    if track_violation:
        cost += 100.0
    return float(1.0 - cost)


class ResidualCartPoleEnvironment:
    """
    Adds the residual to the physics action and owns rollout bookkeeping.

    This class is intentionally independent of Gymnasium.
    """

    observation_size = 4
    action_size = 1

    def __init__(
        self,
        plant: Plant,
        physics_controller: PhysicsController,
        config: EnvironmentConfig,
        reward_function: RewardFunction = placeholder_reward,
        seed: int = 0,
    ) -> None:
        self.plant = plant
        self.physics_controller = physics_controller
        self.config = config
        self.reward_function = reward_function
        self.random = np.random.default_rng(seed)

        self.state = np.zeros(4, dtype=np.float64)
        self.mass_scale = 1.0
        self.step_count = 0
        self.has_reset = False

    def reset(
        self,
        initial_state: np.ndarray | None = None,
        pole_mass_scale: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if initial_state is None:
            initial_state = np.zeros(4, dtype=np.float64)
        initial_state = np.asarray(initial_state, dtype=np.float64)
        if initial_state.shape != (4,):
            raise ValueError("initial_state must have shape (4,)")

        if pole_mass_scale is None:
            pole_mass_scale = self.random.uniform(
                self.config.train_mu_min,
                self.config.train_mu_max,
            )
        if pole_mass_scale <= 0.0:
            raise ValueError("pole_mass_scale must be positive")

        self.mass_scale = float(pole_mass_scale)
        self.physics_controller.reset()
        self.state = np.asarray(
            self.plant.reset(initial_state.copy(), self.mass_scale),
            dtype=np.float64,
        )
        if self.state.shape != (4,):
            raise ValueError("plant.reset must return an array with shape (4,)")

        self.step_count = 0
        self.has_reset = True
        observation = self._observation()
        return observation, {"mass_scale": self.mass_scale}

    def step(
        self,
        residual_action: float | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | bool]]:
        if not self.has_reset:
            raise RuntimeError("reset must be called before step")

        action_array = np.asarray(residual_action, dtype=np.float64).reshape(-1)
        if action_array.size != 1:
            raise ValueError("residual_action must contain exactly one value")

        normalized_residual = float(np.clip(action_array[0], -1.0, 1.0))
        physics_force = float(self.physics_controller.action(self.state.copy()))
        residual_force = self.config.beta * normalized_residual
        proposed_force = physics_force + residual_force
        applied_force = float(
            np.clip(proposed_force, -self.config.u_max, self.config.u_max)
        )

        previous_state = self.state.copy()
        next_state, track_violation = self.plant.step(applied_force)
        self.state = np.asarray(next_state, dtype=np.float64)
        if self.state.shape != (4,):
            raise ValueError("plant.step must return a state with shape (4,)")

        self.step_count += 1
        track_violation = bool(track_violation)
        terminated = track_violation and self.config.terminate_on_track_violation
        truncated = self.step_count >= self.config.max_steps
        reward = self.reward_function(
            previous_state,
            applied_force,
            self.state.copy(),
            track_violation,
        )

        info: dict[str, float | bool] = {
            "mass_scale": self.mass_scale,
            "physics_force": physics_force,
            "normalized_residual": normalized_residual,
            "residual_force": residual_force,
            "proposed_force": proposed_force,
            "applied_force": applied_force,
            "force_was_clipped": not np.isclose(proposed_force, applied_force),
            "track_violation": track_violation,
        }
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        return normalize_state(
            self.state,
            self.config.state_scale,
            self.config.observation_clip,
        )

