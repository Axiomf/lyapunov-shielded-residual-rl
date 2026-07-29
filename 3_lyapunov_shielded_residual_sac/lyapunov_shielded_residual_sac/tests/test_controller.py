import unittest

import numpy as np

from lyapunov_residual_sac import (
    ControllerConfig,
    LQRModel,
    LyapunovShield,
    ShieldConfig,
    ShieldStats,
    ShieldedResidualController,
)


class ConstantPhysics:
    def action(self, state: np.ndarray) -> float:
        del state
        return 1.0

    def lqr_action(self, state: np.ndarray) -> float:
        return float(-state[0])


class ConstantResidual:
    def __init__(self, value: float) -> None:
        self.value = value
        self.last_deterministic = None

    def action(
        self,
        normalized_state: np.ndarray,
        deterministic: bool = True,
    ) -> float:
        del normalized_state
        self.last_deterministic = deterministic
        return self.value


def identity(state: np.ndarray) -> np.ndarray:
    return state


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        model = LQRModel(
            A=np.array([[1.0]]),
            B=np.array([1.0]),
            P=np.array([[1.0]]),
            equilibrium_state=np.array([0.0]),
        )
        self.shield = LyapunovShield(
            model,
            ShieldConfig(rho=0.1, alpha=0.2, u_max=10.0),
        )

    def test_composes_and_clips_residual(self) -> None:
        residual = ConstantResidual(2.5)
        controller = ShieldedResidualController(
            physics=ConstantPhysics(),
            residual_policy=residual,
            normalizer=identity,
            shield=self.shield,
            config=ControllerConfig(beta=3.0, u_max=10.0),
        )

        # V=1 is outside rho, so the proposed action passes through.
        output = controller.act(np.array([1.0]), deterministic=True)

        self.assertEqual(output.residual_action_raw, 2.5)
        self.assertEqual(output.residual_action_clipped, 1.0)
        self.assertEqual(output.unshielded_action, 4.0)
        self.assertEqual(output.action, 4.0)
        self.assertTrue(residual.last_deterministic)

    def test_stats_count_inside_projection(self) -> None:
        residual = ConstantResidual(1.0)
        controller = ShieldedResidualController(
            physics=ConstantPhysics(),
            residual_policy=residual,
            normalizer=identity,
            shield=self.shield,
            config=ControllerConfig(beta=3.0, u_max=10.0),
        )
        output = controller.act(np.array([0.1]))
        stats = ShieldStats()
        stats.update(output)

        self.assertEqual(stats.steps, 1)
        self.assertEqual(stats.inside_region_steps, 1)
        self.assertEqual(stats.projected_steps, 1)

    def test_rejects_residual_scale_above_project_limit(self) -> None:
        with self.assertRaises(ValueError):
            ControllerConfig(beta=3.01, u_max=10.0)


if __name__ == "__main__":
    unittest.main()

