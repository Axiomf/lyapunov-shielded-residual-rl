from dataclasses import dataclass

import numpy as np

from .config import CartPoleParams, LQRConfig, SwingUpConfig, SwitchConfig
from .energy import energy_shaping_control
from .lqr import build_discrete_lqr
from .math_utils import clip, wrap_angle


@dataclass(frozen=True)
class ControlOutput:
    force: float
    unsaturated_force: float
    mode: str
    switched: bool


def regulator_state(state):
    """Return the local error state used by the upright LQR."""

    error = np.asarray(state, dtype=float).copy()
    error[1] = wrap_angle(error[1])
    return error


def inside_thresholds(state, angle, angle_speed, position, cart_speed):
    x, theta, x_dot, theta_dot = regulator_state(state)
    return (
        abs(theta) <= angle
        and abs(theta_dot) <= angle_speed
        and abs(x) <= position
        and abs(x_dot) <= cart_speed
    )


class PhysicsController:
    """Energy swing-up plus discrete LQR with hysteretic mode switching."""

    def __init__(
        self,
        nominal_params,
        lqr_config=None,
        swingup_config=None,
        switch_config=None,
    ):
        if not isinstance(nominal_params, CartPoleParams):
            raise TypeError("nominal_params must be CartPoleParams")

        self.params = nominal_params
        self.swingup_config = swingup_config or SwingUpConfig()
        self.switch_config = switch_config or SwitchConfig()
        self.lqr = build_discrete_lqr(
            nominal_params, lqr_config or LQRConfig()
        )
        self.mode = "swingup"

    def reset(self):
        """Reset hybrid memory before every independent rollout."""

        self.mode = "swingup"

    def _should_enter_balance(self, state):
        c = self.switch_config
        return inside_thresholds(
            state,
            c.enter_angle,
            c.enter_angle_speed,
            c.enter_position,
            c.enter_cart_speed,
        )

    def _should_leave_balance(self, state):
        c = self.switch_config
        return not inside_thresholds(
            state,
            c.exit_angle,
            c.exit_angle_speed,
            c.exit_position,
            c.exit_cart_speed,
        )

    def action(self, state):
        previous_mode = self.mode

        if self.mode == "swingup" and self._should_enter_balance(state):
            self.mode = "balance"
        elif self.mode == "balance" and self._should_leave_balance(state):
            self.mode = "swingup"

        if self.mode == "balance":
            error = regulator_state(state)
            raw_force = -(self.lqr.gain @ error).item()
        else:
            raw_force = energy_shaping_control(
                state, self.params, self.swingup_config
            ).force

        force = clip(
            raw_force, -self.params.max_force, self.params.max_force
        )
        return ControlOutput(
            force=float(force),
            unsaturated_force=float(raw_force),
            mode=self.mode,
            switched=(self.mode != previous_mode),
        )
