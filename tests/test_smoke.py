import unittest

import numpy as np

from cartpole.actor import ZeroActor
from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind, build_controller
from cartpole.data import State
from cartpole.plant import normalize_state, state_derivative, step_rk4
from cartpole.simulation import run_rollout


class StarterSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()

    def test_upright_is_an_equilibrium_at_zero_force(self) -> None:
        state = State(0.0, 0.0, 0.0, 0.0)
        derivative = state_derivative(
            state.as_array(),
            0.0,
            self.config.plant,
        )
        next_state = step_rk4(state, 0.0, self.config.plant)
        np.testing.assert_allclose(derivative, np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(next_state.as_array(), np.zeros(4), atol=1e-12)

    def test_actor_observation_has_one_shared_shape_and_range(self) -> None:
        observation = normalize_state(
            State(5.0, 7.0, -20.0, 50.0),
            self.config.observation,
        )
        self.assertEqual(observation.shape, (4,))
        self.assertTrue(np.all(observation >= -1.0))
        self.assertTrue(np.all(observation <= 1.0))

    def test_all_three_controllers_use_same_decision_type(self) -> None:
        actor = ZeroActor()
        state = State(0.0, 0.05, 0.0, 0.0)
        names = []

        for kind in ControllerKind:
            controller = build_controller(kind, self.config, actor)
            decision = controller.act(state)
            names.append(decision.controller_name)
            self.assertLessEqual(abs(decision.force), self.config.plant.u_max)

        self.assertEqual(len(set(names)), 3)

    def test_rollout_keeps_requested_mu(self) -> None:
        controller = build_controller(
            ControllerKind.PHYSICS,
            self.config,
            ZeroActor(),
        )
        rollout = run_rollout(
            controller,
            State(0.0, 0.1, 0.0, 0.0),
            mu=1.17,
            config=self.config,
        )
        self.assertEqual(rollout.mu, 1.17)
        self.assertGreater(len(rollout.transitions), 0)


if __name__ == "__main__":
    unittest.main()

