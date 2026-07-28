"""Gymnasium environment whose action is a bounded residual force."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ExperimentConfig
from .controllers import NominalController
from .plant import CartPolePlant, wrap_angle
from .shield import LyapunovShield


class ResidualCartPoleEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train SAC on normalized residual actions in [-1, 1]."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        shielded: bool,
        fixed_mass: float | None = None,
    ):
        super().__init__()
        self.config = config
        self.shielded = shielded
        self.fixed_mass = fixed_mass
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        high = np.array(
            [
                2.0 * config.plant.track_limit,
                20.0,
                np.pi,
                30.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        nominal_plant = CartPolePlant(config.plant, config.plant.nominal_pole_mass)
        self.nominal = NominalController(nominal_plant, config.controller)
        self.shield = (
            LyapunovShield(
                self.nominal.lyapunov_matrix,
                config.controller,
                config.shield,
            )
            if shielded
            else None
        )
        self.plant = nominal_plant
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if self.fixed_mass is None:
            low, high = self.config.training.pole_mass_range
            pole_mass = float(self.np_random.uniform(low, high))
        else:
            pole_mass = self.fixed_mass
        self.plant = CartPolePlant(self.config.plant, pole_mass)

        noise = np.asarray(self.config.training.start_noise, dtype=np.float64)
        self.state = self.np_random.uniform(-noise, noise)
        self.state[2] = wrap_angle(np.pi + self.state[2])
        self.steps = 0
        return self._observation(), {"pole_mass": pole_mass}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_residual = float(np.clip(np.asarray(action).item(), -1.0, 1.0))
        nominal_force = self.nominal.action(self.state)
        candidate = nominal_force + (
            self.config.controller.residual_force_limit * normalized_residual
        )
        candidate = float(
            np.clip(
                candidate,
                -self.config.plant.force_limit,
                self.config.plant.force_limit,
            )
        )

        shield_result = None
        force = candidate
        if self.shield is not None:
            shield_result = self.shield.project(
                self.plant, self.state, nominal_force, candidate
            )
            force = shield_result.force

        self.state = self.plant.step(self.state, force)
        self.steps += 1
        terminated = abs(self.state[0]) > self.config.plant.track_limit
        truncated = self.steps >= self.config.training.episode_steps

        theta = wrap_angle(float(self.state[2]))
        reward = (
            np.cos(theta)
            - 0.10 * self.state[0] ** 2
            - 0.01 * self.state[1] ** 2
            - 0.005 * self.state[3] ** 2
            - 0.001 * force**2
        )
        if terminated:
            reward -= 10.0

        info = {
            "pole_mass": self.plant.pole_mass,
            "nominal_force": nominal_force,
            "residual_force": force - nominal_force,
            "applied_force": force,
            "shield_active": bool(shield_result and shield_result.active),
            "shield_changed": bool(shield_result and shield_result.changed),
            "shield_infeasible": bool(shield_result and shield_result.infeasible),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        observation = self.state.copy()
        observation[2] = wrap_angle(float(observation[2]))
        return observation.astype(np.float32)
