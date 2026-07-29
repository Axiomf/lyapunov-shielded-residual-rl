from enum import Enum
import math
from typing import Protocol

import numpy as np

from .actor import Actor
from .config import ExperimentConfig
from .control_math import LQRData, build_lqr, nominal_lqr_force
from .data import ControlDecision, State
from .plant import normalize_state, wrap_angle
from .shield import project_with_lyapunov_shield


class Controller(Protocol):
    def reset(self) -> None:
        ...

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        ...


class ControllerKind(str, Enum):
    PHYSICS = "physics"
    RESIDUAL_SAC = "residual_sac"
    SHIELDED_RESIDUAL_SAC = "shielded_residual_sac"


class PhysicsController:
    """Stateful swing-up/LQR controller with hysteresis."""

    def __init__(self, config: ExperimentConfig, lqr: LQRData) -> None:
        self._config = config
        self._lqr = lqr
        self._mode = "swing_up"

    def reset(self) -> None:
        self._mode = "swing_up"

    def _update_mode(self, state: State) -> None:
        theta = abs(wrap_angle(state.theta))
        theta_dot = abs(state.theta_dot)
        switch = self._config.physics

        if self._mode == "swing_up":
            if theta <= switch.enter_theta and theta_dot <= switch.enter_theta_dot:
                self._mode = "lqr"
        else:
            if theta >= switch.exit_theta or theta_dot >= switch.exit_theta_dot:
                self._mode = "swing_up"

    def _energy_shaping_force(self, state: State) -> float:
        """Small readable starter law; replace/tune for the actual study."""

        plant = self._config.plant
        gains = self._config.physics
        theta = wrap_angle(state.theta)

        energy = (
            0.5 * plant.pole_mass * plant.pole_length**2 * state.theta_dot**2
            + plant.pole_mass * plant.gravity * plant.pole_length * math.cos(theta)
        )
        desired_energy = plant.pole_mass * plant.gravity * plant.pole_length
        energy_error = energy - desired_energy

        force = (
            gains.energy_gain * energy_error * state.theta_dot * math.cos(theta)
            - gains.cart_position_gain * state.x
            - gains.cart_velocity_gain * state.x_dot
        )

        # Exact downward rest is a symmetry; give the starter a deterministic kick.
        near_downward_rest = (
            abs(abs(theta) - math.pi) < 0.10
            and abs(state.theta_dot) < 0.05
        )
        if near_downward_rest:
            force += gains.kick_force

        return float(np.clip(force, -plant.u_max, plant.u_max))

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        del deterministic
        self._update_mode(state)

        if self._mode == "lqr":
            force = nominal_lqr_force(state, self._lqr, self._config.plant)
        else:
            force = self._energy_shaping_force(state)

        return ControlDecision(
            force=force,
            physics_force=force,
            residual_force=0.0,
            controller_name=ControllerKind.PHYSICS.value,
            diagnostics={"physics_mode": self._mode},
        )


class ResidualController:
    def __init__(
        self,
        physics: PhysicsController,
        actor: Actor,
        config: ExperimentConfig,
    ) -> None:
        self._physics = physics
        self._actor = actor
        self._config = config

    def reset(self) -> None:
        self._physics.reset()

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        physics = self._physics.act(state, deterministic)
        observation = normalize_state(state, self._config.observation)
        actor_action = float(
            np.clip(self._actor.act(observation, deterministic), -1.0, 1.0)
        )
        residual_force = self._config.residual.beta * actor_action
        total_force = float(
            np.clip(
                physics.force + residual_force,
                -self._config.plant.u_max,
                self._config.plant.u_max,
            )
        )

        diagnostics = dict(physics.diagnostics)
        diagnostics["actor_action"] = actor_action
        return ControlDecision(
            force=total_force,
            physics_force=physics.force,
            residual_force=residual_force,
            controller_name=ControllerKind.RESIDUAL_SAC.value,
            diagnostics=diagnostics,
        )


class ShieldedResidualController:
    def __init__(
        self,
        residual: ResidualController,
        config: ExperimentConfig,
        lqr: LQRData,
    ) -> None:
        self._residual = residual
        self._config = config
        self._lqr = lqr

    def reset(self) -> None:
        self._residual.reset()

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        proposed = self._residual.act(state, deterministic)
        shield = project_with_lyapunov_shield(
            state=state,
            proposed_force=proposed.force,
            nominal_plant=self._config.plant,
            lqr=self._lqr,
            config=self._config.shield,
        )

        diagnostics = dict(proposed.diagnostics)
        diagnostics.update(
            {
                "shield_active": shield.active,
                "shield_projected": shield.projected,
                "shield_infeasible": shield.infeasible,
                "v_before": shield.v_before,
                "nominal_delta_v": shield.nominal_delta_v,
            }
        )
        return ControlDecision(
            force=shield.force,
            physics_force=proposed.physics_force,
            residual_force=proposed.residual_force,
            controller_name=ControllerKind.SHIELDED_RESIDUAL_SAC.value,
            diagnostics=diagnostics,
        )


def build_controller(
    kind: ControllerKind,
    config: ExperimentConfig,
    actor: Actor,
) -> Controller:
    """Single factory so all three controllers share nominal objects."""

    lqr = build_lqr(config.plant, config.lqr)
    physics = PhysicsController(config, lqr)

    if kind == ControllerKind.PHYSICS:
        return physics

    residual = ResidualController(physics, actor, config)
    if kind == ControllerKind.RESIDUAL_SAC:
        return residual
    if kind == ControllerKind.SHIELDED_RESIDUAL_SAC:
        return ShieldedResidualController(residual, config, lqr)

    raise ValueError(f"Unknown controller kind: {kind}")

