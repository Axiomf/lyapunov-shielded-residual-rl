import math

import numpy as np

from cartpole import CartPoleParams, PhysicsController, SwingUpConfig
from cartpole.dynamics import accelerations
from cartpole.energy import (
    desired_cart_acceleration,
    force_for_cart_acceleration,
    pendulum_energy,
)


def test_energy_term_drives_energy_error_toward_zero():
    params = CartPoleParams(cart_damping=0.0, pole_damping=0.0)
    config = SwingUpConfig(
        energy_gain=1.2,
        cart_position_gain=0.0,
        cart_velocity_gain=0.0,
        kick_acceleration=0.0,
    )
    state = np.array([0.0, 2.4, 0.0, 1.3])
    acceleration = desired_cart_acceleration(state, params, config)

    energy_error = pendulum_energy(state, params) - (
        params.pole_mass * params.gravity * params.pole_com_length
    )
    energy_rate = (
        -params.pole_mass
        * params.pole_com_length
        * acceleration
        * state[3]
        * math.cos(state[1])
    )
    assert energy_error * energy_rate < 0.0


def test_hysteresis_prevents_chattering_between_thresholds():
    controller = PhysicsController(CartPoleParams())

    near_upright = np.array([0.0, 0.05, 0.0, 0.1])
    first = controller.action(near_upright)
    assert first.mode == "balance"
    assert first.switched

    between_thresholds = np.array([0.0, 0.27, 0.0, 1.5])
    second = controller.action(between_thresholds)
    assert second.mode == "balance"
    assert not second.switched

    outside_exit_region = np.array([0.0, 0.51, 0.0, 0.0])
    third = controller.action(outside_exit_region)
    assert third.mode == "swingup"
    assert third.switched


def test_force_is_saturated():
    params = CartPoleParams(max_force=2.0)
    controller = PhysicsController(params)
    output = controller.action(np.array([0.0, 0.15, 0.0, 1.0]))
    assert abs(output.force) <= params.max_force


def test_partial_feedback_linearization_delivers_requested_acceleration():
    params = CartPoleParams()
    state = np.array([0.2, 1.1, -0.3, 1.7])
    requested_acceleration = -2.4
    force = force_for_cart_acceleration(
        state, requested_acceleration, params
    )
    actual_acceleration = accelerations(state, force, params)[0]
    assert abs(actual_acceleration - requested_acceleration) < 1e-12
