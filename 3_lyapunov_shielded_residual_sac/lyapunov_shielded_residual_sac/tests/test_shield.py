import unittest

import numpy as np

from lyapunov_residual_sac import LQRModel, LyapunovShield, ShieldConfig
from lyapunov_residual_sac.shield import (
    feasible_interval,
    realized_lyapunov_change,
    upright_state_error,
)


def one_dimensional_shield(
    *,
    rho: float = 100.0,
    alpha: float = 0.2,
    u_max: float = 2.0,
    b: float = 1.0,
) -> LyapunovShield:
    model = LQRModel(
        A=np.array([[1.0]]),
        B=np.array([b]),
        P=np.array([[1.0]]),
        equilibrium_state=np.array([0.0]),
    )
    return LyapunovShield(
        model,
        ShieldConfig(rho=rho, alpha=alpha, u_max=u_max),
    )


class FeasibleIntervalTests(unittest.TestCase):
    def test_quadratic_interval(self) -> None:
        interval = feasible_interval(1.0, 0.0, -1.0, -2.0, 2.0)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], -1.0)
        self.assertAlmostEqual(interval[1], 1.0)

    def test_linear_interval(self) -> None:
        interval = feasible_interval(0.0, 2.0, -1.0, -2.0, 2.0)
        self.assertEqual(interval, (-2.0, 0.5))

    def test_infeasible_constant_constraint(self) -> None:
        self.assertIsNone(
            feasible_interval(0.0, 0.0, 1.0, -2.0, 2.0)
        )


class LyapunovShieldTests(unittest.TestCase):
    def test_inside_region_projects_to_nearest_feasible_action(self) -> None:
        shield = one_dimensional_shield()
        result = shield.apply(
            state=np.array([0.5]),
            proposed_action=1.0,
            lqr_fallback_action=-0.5,
        )

        expected_upper = np.sqrt(0.8 * 0.5**2) - 0.5
        self.assertTrue(result.inside_region)
        self.assertTrue(result.feasible)
        self.assertTrue(result.projected)
        self.assertFalse(result.used_lqr_fallback)
        self.assertAlmostEqual(result.action, expected_upper)
        self.assertTrue(result.constraint_satisfied)

    def test_outside_region_passes_bounded_action_through(self) -> None:
        shield = one_dimensional_shield(rho=0.1, u_max=2.0)
        result = shield.apply(
            state=np.array([1.0]),
            proposed_action=5.0,
            lqr_fallback_action=-1.0,
        )

        self.assertFalse(result.inside_region)
        self.assertEqual(result.action, 2.0)
        self.assertFalse(result.projected)
        self.assertFalse(result.used_lqr_fallback)

    def test_infeasible_constraint_uses_clipped_lqr_fallback(self) -> None:
        shield = one_dimensional_shield(
            rho=200.0,
            alpha=0.5,
            u_max=1.0,
            b=0.1,
        )
        result = shield.apply(
            state=np.array([10.0]),
            proposed_action=0.5,
            lqr_fallback_action=-4.0,
        )

        self.assertTrue(result.inside_region)
        self.assertFalse(result.feasible)
        self.assertTrue(result.used_lqr_fallback)
        self.assertEqual(result.action, -1.0)
        self.assertFalse(result.constraint_satisfied)

    def test_angle_error_wraps_around_upright(self) -> None:
        state = np.array([0.0, 2.0 * np.pi - 0.1, 0.0, 0.0])
        error = upright_state_error(state, np.zeros(4))
        self.assertAlmostEqual(error[1], -0.1)

    def test_realized_change_uses_observed_next_state(self) -> None:
        model = LQRModel(
            A=np.eye(4),
            B=np.zeros(4),
            P=np.eye(4),
            equilibrium_state=np.zeros(4),
        )
        state = np.array([0.0, 0.2, 0.0, 0.0])
        next_state = np.array([0.0, 0.1, 0.0, 0.0])
        self.assertAlmostEqual(
            realized_lyapunov_change(state, next_state, model),
            -0.03,
        )


if __name__ == "__main__":
    unittest.main()
