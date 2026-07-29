"""Composition of the physics controller, frozen SAC actor, and shield."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ControllerConfig
from .interfaces import PhysicsController, ResidualPolicy, StateNormalizer
from .shield import LyapunovShield, ShieldResult


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControllerOutput:
    """Final action and all values needed for experiment-side logging."""

    action: float
    physics_action: float
    residual_action_raw: float
    residual_action_clipped: float
    unshielded_action: float
    shield: ShieldResult


class ShieldedResidualController:
    """Third controller: bounded residual SAC with a local Lyapunov shield."""

    def __init__(
        self,
        physics: PhysicsController,
        residual_policy: ResidualPolicy,
        normalizer: StateNormalizer,
        shield: LyapunovShield,
        config: ControllerConfig,
    ) -> None:
        if abs(config.u_max - shield.config.u_max) > 1e-12:
            raise ValueError("controller and shield must use the same u_max")

        self.physics = physics
        self.residual_policy = residual_policy
        self.normalizer = normalizer
        self.shield = shield
        self.config = config

    def act(
        self,
        state: FloatArray,
        deterministic: bool = True,
    ) -> ControllerOutput:
        """Return one zero-order-held force for the current full state."""

        state_array = np.asarray(state, dtype=float).reshape(-1)
        normalized_state = np.asarray(
            self.normalizer(state_array.copy()),
            dtype=float,
        ).reshape(-1)

        physics_action = float(self.physics.action(state_array.copy()))
        residual_raw = float(
            self.residual_policy.action(
                normalized_state,
                deterministic=deterministic,
            )
        )
        residual_clipped = float(np.clip(residual_raw, -1.0, 1.0))

        unshielded_action = float(
            np.clip(
                physics_action + self.config.beta * residual_clipped,
                -self.config.u_max,
                self.config.u_max,
            )
        )

        lqr_fallback = float(self.physics.lqr_action(state_array.copy()))
        shield_result = self.shield.apply(
            state=state_array,
            proposed_action=unshielded_action,
            lqr_fallback_action=lqr_fallback,
        )

        return ControllerOutput(
            action=shield_result.action,
            physics_action=physics_action,
            residual_action_raw=residual_raw,
            residual_action_clipped=residual_clipped,
            unshielded_action=unshielded_action,
            shield=shield_result,
        )

