import math
import unittest
from dataclasses import replace

import numpy as np

from cartpole.config import ExperimentConfig
from cartpole.control_math import build_lqr
from cartpole.controllers import PhysicsController
from cartpole.data import State
from cartpole.evaluation import (
    SuccessCriteria,
    analyze_local_closed_loop,
    evaluate_rollout,
    make_latin_hypercube_states,
    make_theta_theta_dot_slice,
)
from cartpole.simulation import run_rollout


class PhysicsBaselineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = ExperimentConfig()
        self.config = replace(
            config,
            rollout=replace(config.rollout, horizon_seconds=0.10),
        )
        self.lqr = build_lqr(self.config.plant, self.config.lqr)

    def controller_factory(self) -> PhysicsController:
        return PhysicsController(self.config, self.lqr)

    def test_latin_hypercube_states_are_seeded_and_bounded(self) -> None:
        bounds = (
            (-2.4, 2.4),
            (-math.pi, math.pi),
            (-5.0, 5.0),
            (-10.0, 10.0),
        )
        first = make_latin_hypercube_states(20, bounds, seed=7)
        repeated = make_latin_hypercube_states(20, bounds, seed=7)
        independent = make_latin_hypercube_states(20, bounds, seed=8)

        self.assertEqual(first.shape, (20, 4))
        np.testing.assert_allclose(first, repeated)
        self.assertFalse(np.array_equal(first, independent))
        lower = np.asarray([item[0] for item in bounds])
        upper = np.asarray([item[1] for item in bounds])
        self.assertTrue(np.all(first >= lower))
        self.assertTrue(np.all(first <= upper))

    def test_theta_slice_uses_canonical_state_order(self) -> None:
        theta, theta_dot, states = make_theta_theta_dot_slice(
            -math.pi,
            math.pi,
            5,
            -4.0,
            4.0,
            7,
        )

        self.assertEqual(theta.shape, (5,))
        self.assertEqual(theta_dot.shape, (7,))
        self.assertEqual(states.shape, (5, 7, 4))
        np.testing.assert_allclose(states[:, :, 0], 0.0)
        np.testing.assert_allclose(states[:, :, 2], 0.0)
        np.testing.assert_allclose(states[:, 0, 1], theta)
        np.testing.assert_allclose(states[0, :, 3], theta_dot)

    def test_upright_fixed_point_and_local_jacobian(self) -> None:
        analysis = analyze_local_closed_loop(
            controller_factory=self.controller_factory,
            mu=1.0,
            config=self.config,
        )

        self.assertTrue(analysis.root_converged)
        self.assertLess(analysis.upright_residual_norm, 1e-12)
        self.assertLess(analysis.equilibrium_bias_norm, 1e-10)
        self.assertEqual(analysis.jacobian.shape, (4, 4))
        self.assertEqual(analysis.eigenvalues.shape, (4,))
        self.assertTrue(np.all(np.isfinite(analysis.jacobian)))
        self.assertLess(analysis.spectral_radius, 1.0)

    def test_origin_rollout_satisfies_success_and_metric_conventions(self) -> None:
        rollout = run_rollout(
            controller=self.controller_factory(),
            initial_state=State(0.0, 0.0, 0.0, 0.0),
            mu=1.2,
            config=self.config,
            deterministic=True,
        )
        metrics = evaluate_rollout(
            rollout=rollout,
            config=self.config,
            lqr=self.lqr,
            criteria=SuccessCriteria(final_window_seconds=0.04),
        )

        self.assertTrue(metrics.completed_horizon)
        self.assertTrue(metrics.success)
        self.assertFalse(metrics.track_violation)
        self.assertEqual(metrics.settling_time_seconds, 0.0)
        self.assertEqual(metrics.controller_name, "physics")
        self.assertEqual(metrics.lqr_fraction, 1.0)
        self.assertEqual(metrics.swing_up_fraction, 0.0)
        self.assertEqual(metrics.rms_force, 0.0)
        self.assertEqual(metrics.local_lyapunov_step_count, len(rollout.transitions))
        self.assertEqual(metrics.realized_v_condition_fraction, 1.0)
        self.assertFalse(metrics.shield_applicable)
        self.assertIsNone(metrics.shield_activation_fraction)


if __name__ == "__main__":
    unittest.main()
