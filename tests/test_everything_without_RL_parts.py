"""Tests for the project parts that do not require SAC or RL training.

This file checks the shared data format, nonlinear plant, sampled-data LQR,
physics controller, residual-controller plumbing with simple test actors,
Lyapunov shield, reward, and rollout synchronization. It intentionally does
not import the Gymnasium environment, training module, or Stable-Baselines3.
"""

import math
import unittest
from dataclasses import replace

import numpy as np

from cartpole.actor import RandomActor, ZeroActor
from cartpole.config import ExperimentConfig, ResidualConfig, ShieldConfig
from cartpole.control_math import (
    build_lqr,
    finite_difference_discrete_model,
    nominal_lqr_force,
    one_step_array,
)
from cartpole.controllers import (
    ControllerKind,
    PhysicsController,
    ResidualController,
    ShieldedResidualController,
    build_controller,
)
from cartpole.data import ControlDecision, Rollout, State, Transition
from cartpole.plant import (
    clip_force,
    has_track_violation,
    normalize_state,
    state_derivative,
    step_rk4,
    wrap_angle,
)
from cartpole.shield import lyapunov_value, project_with_lyapunov_shield
from cartpole.simulation import run_rollout, starter_reward


class ConstantActor:
    """Small non-learning actor used to test residual-controller plumbing."""

    def __init__(self, action: float) -> None:
        self.action = action
        self.observations: list[np.ndarray] = []
        self.deterministic_flags: list[bool] = []

    def act(self, observation: np.ndarray, deterministic: bool) -> float:
        self.observations.append(observation.copy())
        self.deterministic_flags.append(deterministic)
        return self.action


class ConstantController:
    """Simple controller used to test rollout logic independently."""

    def __init__(self, force: float) -> None:
        self.force = force
        self.reset_count = 0
        self.deterministic_flags: list[bool] = []

    def reset(self) -> None:
        self.reset_count += 1

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        del state
        self.deterministic_flags.append(deterministic)
        return ControlDecision(
            force=self.force,
            physics_force=self.force,
            residual_force=0.0,
            controller_name="constant_test_controller",
        )


class ConfigurationAndDataTests(unittest.TestCase):
    def test_default_settings_match_the_project_scope(self) -> None:
        config = ExperimentConfig()

        self.assertEqual(config.plant.control_dt, 0.02)
        self.assertEqual(config.plant.u_max, 10.0)
        self.assertEqual(config.plant.x_limit, 2.4)
        self.assertLessEqual(config.residual.beta, 0.3 * config.plant.u_max)
        self.assertEqual(config.training.seeds, (0, 1, 2))

    def test_mass_multiplier_changes_only_the_plant_pole_mass(self) -> None:
        nominal = ExperimentConfig().plant
        actual = nominal.with_mass_multiplier(1.3)
        expected = replace(nominal, pole_mass=1.3 * nominal.pole_mass)

        self.assertEqual(actual, expected)
        self.assertEqual(nominal.pole_mass, 0.1)

    def test_invalid_non_rl_settings_are_rejected(self) -> None:
        config = ExperimentConfig()

        with self.subTest("residual authority"), self.assertRaises(ValueError):
            ExperimentConfig(residual=ResidualConfig(beta=3.01))

        with self.subTest("RK4 substeps"), self.assertRaises(ValueError):
            ExperimentConfig(plant=replace(config.plant, rk4_substeps=0))

        with self.subTest("shield grid"), self.assertRaises(ValueError):
            ExperimentConfig(shield=replace(config.shield, grid_size=2))

    def test_state_array_round_trip_preserves_order_and_dtype(self) -> None:
        state = State(x=1.0, theta=2.0, x_dot=3.0, theta_dot=4.0)
        values = state.as_array()

        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(values.dtype, np.float64)
        self.assertEqual(State.from_array(values), state)

        values[0] = 99.0
        self.assertEqual(state.x, 1.0)

    def test_state_rejects_arrays_with_the_wrong_shape(self) -> None:
        invalid_arrays = [np.zeros(3), np.zeros((4, 1)), np.zeros((1, 4))]

        for values in invalid_arrays:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                State.from_array(values)

    def test_rollout_total_reward_uses_synchronized_transitions(self) -> None:
        state = State(0.0, 0.0, 0.0, 0.0)
        decision = ControlDecision(0.0, 0.0, 0.0, "test")
        transitions = (
            Transition(state, decision, 1.5, state, False),
            Transition(state, decision, -0.25, state, False),
        )
        rollout = Rollout(1.0, state, transitions, track_violation=False)

        self.assertEqual(rollout.total_reward, 1.25)


class PlantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()
        self.plant = self.config.plant

    def test_angle_wrapping_uses_one_consistent_interval(self) -> None:
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(2.0 * math.pi + 0.2), 0.2)

    def test_force_clipping_respects_the_shared_actuator_limit(self) -> None:
        self.assertEqual(clip_force(4.0, self.plant), 4.0)
        self.assertEqual(clip_force(100.0, self.plant), self.plant.u_max)
        self.assertEqual(clip_force(-100.0, self.plant), -self.plant.u_max)

    def test_upright_is_an_open_loop_equilibrium_at_zero_force(self) -> None:
        state = State(x=0.7, theta=0.0, x_dot=0.0, theta_dot=0.0)

        derivative = state_derivative(state.as_array(), 0.0, self.plant)
        next_state = step_rk4(state, 0.0, self.plant)

        np.testing.assert_allclose(derivative, np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(next_state.as_array(), state.as_array(), atol=1e-12)

    def test_state_derivative_satisfies_the_coupled_dynamics(self) -> None:
        plant = self.plant.with_mass_multiplier(1.3)
        state = State(0.7, 0.43, -0.8, 1.2)
        force = 2.3

        derivative = state_derivative(state.as_array(), force, plant)
        x_ddot = derivative[2]
        theta_ddot = derivative[3]
        sin_theta = math.sin(state.theta)
        cos_theta = math.cos(state.theta)

        cart_equation_force = (
            (plant.cart_mass + plant.pole_mass) * x_ddot
            + plant.pole_mass * plant.pole_length * cos_theta * theta_ddot
            - plant.pole_mass
            * plant.pole_length
            * sin_theta
            * state.theta_dot**2
        )
        pole_equation = plant.pole_length * theta_ddot + cos_theta * x_ddot

        np.testing.assert_allclose(
            derivative[:2],
            [state.x_dot, state.theta_dot],
            atol=1e-12,
        )
        self.assertAlmostEqual(cart_equation_force, force, places=12)
        self.assertAlmostEqual(pole_equation, plant.gravity * sin_theta, places=12)

    def test_rk4_clips_once_at_the_plant_interface(self) -> None:
        state = State(0.0, 0.2, 0.0, 0.0)

        saturated = step_rk4(state, 100.0, self.plant)
        limited = step_rk4(state, self.plant.u_max, self.plant)

        np.testing.assert_allclose(saturated.as_array(), limited.as_array(), atol=1e-12)

    def test_mass_mismatch_changes_the_nonlinear_transition(self) -> None:
        state = State(0.1, 0.6, -0.2, 1.0)
        light_plant = self.plant.with_mass_multiplier(0.6)
        heavy_plant = self.plant.with_mass_multiplier(1.4)

        light_next = step_rk4(state, 2.0, light_plant)
        heavy_next = step_rk4(state, 2.0, heavy_plant)

        self.assertFalse(
            np.allclose(light_next.as_array(), heavy_next.as_array(), atol=1e-10)
        )

    def test_state_normalization_preserves_order_wraps_and_clips(self) -> None:
        state = State(1.2, 3.0 * math.pi, -2.5, 20.0)

        observation = normalize_state(state, self.config.observation)

        np.testing.assert_allclose(observation, [0.5, -1.0, -0.5, 1.0])
        self.assertEqual(observation.shape, (4,))
        self.assertEqual(observation.dtype, np.float64)

    def test_track_boundary_is_allowed_but_crossing_is_a_violation(self) -> None:
        self.assertFalse(has_track_violation(State(2.4, 0.0, 0.0, 0.0), self.plant))
        self.assertTrue(
            has_track_violation(State(2.400001, 0.0, 0.0, 0.0), self.plant)
        )


class NominalControlMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExperimentConfig()
        cls.lqr = build_lqr(cls.config.plant, cls.config.lqr)

    def test_array_adapter_matches_the_project_rk4_map(self) -> None:
        state = State(0.1, -0.2, 0.3, -0.4)
        force = 1.2

        adapter_result = one_step_array(state.as_array(), force, self.config.plant)
        direct_result = step_rk4(state, force, self.config.plant).as_array()

        np.testing.assert_allclose(adapter_result, direct_result, atol=1e-12)

    def test_finite_difference_model_has_the_expected_shapes(self) -> None:
        A, B = finite_difference_discrete_model(
            self.config.plant,
            self.config.lqr.finite_difference_epsilon,
        )

        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(B.shape, (4, 1))
        self.assertTrue(np.all(np.isfinite(A)))
        self.assertTrue(np.all(np.isfinite(B)))

    def test_lqr_data_is_positive_definite_and_locally_stabilizing(self) -> None:
        lqr = self.lqr

        np.testing.assert_allclose(lqr.P, lqr.P.T, atol=1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(lqr.P) > 0.0))

        closed_loop = lqr.A - lqr.B @ lqr.K
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(closed_loop))))
        self.assertLess(spectral_radius, 1.0)

    def test_nominal_lqr_force_is_zero_at_origin_and_clipped(self) -> None:
        zero_force = nominal_lqr_force(
            State(0.0, 0.0, 0.0, 0.0),
            self.lqr,
            self.config.plant,
        )
        clipped_force = nominal_lqr_force(
            State(1_000_000.0, 0.0, 0.0, 0.0),
            self.lqr,
            self.config.plant,
        )

        self.assertAlmostEqual(zero_force, 0.0)
        self.assertAlmostEqual(abs(clipped_force), self.config.plant.u_max)


class NonLearningActorTests(unittest.TestCase):
    def test_zero_actor_always_returns_zero(self) -> None:
        actor = ZeroActor()
        observation = np.array([0.2, -0.3, 0.4, -0.5])

        self.assertEqual(actor.act(observation, deterministic=True), 0.0)
        self.assertEqual(actor.act(observation, deterministic=False), 0.0)

    def test_random_actor_is_seeded_bounded_and_deterministic_when_requested(self) -> None:
        observation = np.zeros(4)
        first = RandomActor(seed=9)
        second = RandomActor(seed=9)

        self.assertEqual(first.act(observation, deterministic=True), 0.0)
        first_samples = [first.act(observation, deterministic=False) for _ in range(4)]
        second_samples = [second.act(observation, deterministic=False) for _ in range(4)]

        np.testing.assert_allclose(first_samples, second_samples)
        self.assertTrue(np.all(np.asarray(first_samples) >= -1.0))
        self.assertTrue(np.all(np.asarray(first_samples) <= 1.0))


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExperimentConfig()
        cls.lqr = build_lqr(cls.config.plant, cls.config.lqr)

    def test_physics_controller_uses_hysteresis_and_reset(self) -> None:
        controller = PhysicsController(self.config, self.lqr)

        entered = controller.act(State(0.0, 0.1, 0.0, 0.1))
        stayed = controller.act(State(0.0, 0.25, 0.0, 1.0))
        exited = controller.act(State(0.0, 0.36, 0.0, 0.0))
        controller.reset()
        after_reset = controller.act(State(0.0, 0.25, 0.0, 1.0))

        self.assertEqual(entered.diagnostics["physics_mode"], "lqr")
        self.assertEqual(stayed.diagnostics["physics_mode"], "lqr")
        self.assertEqual(exited.diagnostics["physics_mode"], "swing_up")
        self.assertEqual(after_reset.diagnostics["physics_mode"], "swing_up")

    def test_physics_controller_kicks_from_downward_rest(self) -> None:
        controller = PhysicsController(self.config, self.lqr)

        decision = controller.act(State(0.0, math.pi, 0.0, 0.0))

        self.assertEqual(decision.diagnostics["physics_mode"], "swing_up")
        self.assertAlmostEqual(decision.force, self.config.physics.kick_force)
        self.assertEqual(decision.residual_force, 0.0)

    def test_residual_controller_scales_action_and_passes_normalized_state(self) -> None:
        actor = ConstantActor(action=0.5)
        physics = PhysicsController(self.config, self.lqr)
        controller = ResidualController(physics, actor, self.config)
        state = State(0.0, 0.0, 0.0, 0.0)

        decision = controller.act(state, deterministic=False)

        self.assertAlmostEqual(decision.physics_force, 0.0)
        self.assertAlmostEqual(decision.residual_force, 0.5 * self.config.residual.beta)
        self.assertAlmostEqual(decision.force, decision.residual_force)
        np.testing.assert_allclose(actor.observations[0], np.zeros(4))
        self.assertEqual(actor.deterministic_flags, [False])

    def test_residual_controller_clips_actor_and_total_force(self) -> None:
        actor = ConstantActor(action=5.0)
        physics = PhysicsController(self.config, self.lqr)
        controller = ResidualController(physics, actor, self.config)
        state = State(-100.0, math.pi, 0.0, 0.0)

        decision = controller.act(state)

        self.assertAlmostEqual(decision.physics_force, self.config.plant.u_max)
        self.assertAlmostEqual(decision.residual_force, self.config.residual.beta)
        self.assertAlmostEqual(decision.force, self.config.plant.u_max)
        self.assertAlmostEqual(decision.diagnostics["actor_action"], 1.0)

    def test_shielded_controller_projects_a_nonzero_origin_action(self) -> None:
        actor = ConstantActor(action=1.0)
        physics = PhysicsController(self.config, self.lqr)
        residual = ResidualController(physics, actor, self.config)
        controller = ShieldedResidualController(residual, self.config, self.lqr)

        decision = controller.act(State(0.0, 0.0, 0.0, 0.0))

        self.assertAlmostEqual(decision.residual_force, self.config.residual.beta)
        self.assertAlmostEqual(decision.force, 0.0)
        self.assertTrue(decision.diagnostics["shield_active"])
        self.assertTrue(decision.diagnostics["shield_projected"])
        self.assertFalse(decision.diagnostics["shield_infeasible"])

    def test_controller_factory_builds_exactly_the_three_comparators(self) -> None:
        expected_names = {
            ControllerKind.PHYSICS.value,
            ControllerKind.RESIDUAL_SAC.value,
            ControllerKind.SHIELDED_RESIDUAL_SAC.value,
        }
        actual_names = set()

        for kind in ControllerKind:
            controller = build_controller(kind, self.config, ZeroActor())
            decision = controller.act(State(0.0, 0.0, 0.0, 0.0))
            actual_names.add(decision.controller_name)
            self.assertIsInstance(decision, ControlDecision)
            self.assertLessEqual(abs(decision.force), self.config.plant.u_max)

        self.assertEqual(actual_names, expected_names)
        self.assertEqual(len(ControllerKind), 3)


class ShieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExperimentConfig()
        cls.lqr = build_lqr(cls.config.plant, cls.config.lqr)

    def test_lyapunov_value_is_zero_at_origin_positive_and_even(self) -> None:
        origin = State(0.0, 0.0, 0.0, 0.0)
        state = State(0.01, -0.02, 0.03, -0.04)
        negative_state = State(-0.01, 0.02, -0.03, 0.04)

        self.assertAlmostEqual(lyapunov_value(origin, self.lqr.P), 0.0)
        self.assertGreater(lyapunov_value(state, self.lqr.P), 0.0)
        self.assertAlmostEqual(
            lyapunov_value(state, self.lqr.P),
            lyapunov_value(negative_state, self.lqr.P),
        )

    def test_inactive_shield_returns_the_bounded_proposal_unchanged(self) -> None:
        result = project_with_lyapunov_shield(
            state=State(10.0, 0.0, 0.0, 0.0),
            proposed_force=100.0,
            nominal_plant=self.config.plant,
            lqr=self.lqr,
            config=self.config.shield,
        )

        self.assertFalse(result.active)
        self.assertFalse(result.projected)
        self.assertFalse(result.infeasible)
        self.assertEqual(result.force, self.config.plant.u_max)
        self.assertTrue(math.isnan(result.nominal_delta_v))

    def test_active_shield_keeps_an_exact_feasible_proposal(self) -> None:
        result = project_with_lyapunov_shield(
            state=State(0.0, 0.0, 0.0, 0.0),
            proposed_force=0.0,
            nominal_plant=self.config.plant,
            lqr=self.lqr,
            config=self.config.shield,
        )

        self.assertTrue(result.active)
        self.assertFalse(result.projected)
        self.assertFalse(result.infeasible)
        self.assertAlmostEqual(result.force, 0.0)
        self.assertLessEqual(
            result.nominal_delta_v,
            -self.config.shield.alpha * result.v_before
            + self.config.shield.feasibility_tolerance,
        )

    def test_active_shield_projects_an_infeasible_origin_proposal(self) -> None:
        result = project_with_lyapunov_shield(
            state=State(0.0, 0.0, 0.0, 0.0),
            proposed_force=3.0,
            nominal_plant=self.config.plant,
            lqr=self.lqr,
            config=self.config.shield,
        )

        self.assertTrue(result.active)
        self.assertTrue(result.projected)
        self.assertFalse(result.infeasible)
        self.assertAlmostEqual(result.force, 0.0)

    def test_sampled_infeasibility_uses_nominal_lqr_fallback(self) -> None:
        state = State(0.0, 0.01, 0.0, 0.0)
        impossible_decrease = ShieldConfig(rho=1_000_000.0, alpha=2.0, grid_size=41)

        result = project_with_lyapunov_shield(
            state=state,
            proposed_force=3.0,
            nominal_plant=self.config.plant,
            lqr=self.lqr,
            config=impossible_decrease,
        )
        expected_fallback = nominal_lqr_force(state, self.lqr, self.config.plant)

        self.assertTrue(result.active)
        self.assertTrue(result.infeasible)
        self.assertAlmostEqual(result.force, expected_fallback)
        self.assertTrue(math.isfinite(result.nominal_delta_v))


class RewardAndRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()

    def short_config(self, steps: int) -> ExperimentConfig:
        return replace(
            self.config,
            rollout=replace(
                self.config.rollout,
                horizon_seconds=steps * self.config.plant.control_dt,
            ),
        )

    def test_starter_reward_matches_the_documented_stage_cost(self) -> None:
        next_state = State(1.0, 2.0 * math.pi + 0.2, 2.0, -3.0)
        force = 4.0
        expected_cost = (
            0.2**2
            + 0.1 * 1.0**2
            + 0.01 * 2.0**2
            + 0.02 * (-3.0) ** 2
            + 0.001 * force**2
        )

        reward = starter_reward(State(9.0, 9.0, 9.0, 9.0), force, next_state)

        self.assertAlmostEqual(reward, -expected_cost)

    def test_starter_reward_adds_track_failure_cost_at_the_boundary(self) -> None:
        next_state = State(2.4, 0.0, 0.0, 0.0)
        expected_cost = 0.1 * 2.4**2 + 100.0

        reward = starter_reward(State(0.0, 0.0, 0.0, 0.0), 0.0, next_state)

        self.assertAlmostEqual(reward, -expected_cost)

    def test_rollout_has_aligned_transitions_for_the_full_horizon(self) -> None:
        controller = ConstantController(force=0.0)
        config = self.short_config(steps=3)
        initial_state = State(0.0, 0.0, 0.0, 0.0)

        rollout = run_rollout(controller, initial_state, mu=1.2, config=config)

        self.assertEqual(controller.reset_count, 1)
        self.assertEqual(controller.deterministic_flags, [True, True, True])
        self.assertEqual(rollout.mu, 1.2)
        self.assertEqual(rollout.initial_state, initial_state)
        self.assertEqual(len(rollout.transitions), 3)
        self.assertFalse(rollout.track_violation)
        self.assertEqual(rollout.total_reward, 0.0)

        for index, transition in enumerate(rollout.transitions):
            self.assertFalse(transition.terminated)
            if index > 0:
                self.assertEqual(
                    transition.state,
                    rollout.transitions[index - 1].next_state,
                )

    def test_rollout_uses_one_fixed_mismatched_mass(self) -> None:
        config = self.short_config(steps=2)
        initial_state = State(0.0, 0.4, 0.0, 0.0)

        light = run_rollout(
            ConstantController(force=1.0),
            initial_state,
            mu=0.6,
            config=config,
        )
        heavy = run_rollout(
            ConstantController(force=1.0),
            initial_state,
            mu=1.4,
            config=config,
        )

        self.assertEqual(light.mu, 0.6)
        self.assertEqual(heavy.mu, 1.4)
        self.assertFalse(
            np.allclose(
                light.transitions[-1].next_state.as_array(),
                heavy.transitions[-1].next_state.as_array(),
                atol=1e-10,
            )
        )
        self.assertTrue(
            all(item.decision.force == 1.0 for item in light.transitions)
        )
        self.assertTrue(
            all(item.decision.force == 1.0 for item in heavy.transitions)
        )

    def test_rollout_keeps_the_first_track_violation_then_stops(self) -> None:
        controller = ConstantController(force=0.0)
        initial_state = State(2.39, 0.0, 1.0, 0.0)

        rollout = run_rollout(
            controller,
            initial_state,
            mu=1.0,
            config=self.short_config(steps=5),
        )

        self.assertTrue(rollout.track_violation)
        self.assertEqual(len(rollout.transitions), 1)
        self.assertTrue(rollout.transitions[0].terminated)
        self.assertGreater(
            abs(rollout.transitions[0].next_state.x),
            self.config.plant.x_limit,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
