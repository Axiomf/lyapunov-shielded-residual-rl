import math
import unittest
from dataclasses import replace

import numpy as np

from cartpole.config import ExperimentConfig
from cartpole.data import State
from cartpole.environment import ResidualCartPoleEnvironment


class ResidualCartPoleEnvironmentTests(unittest.TestCase):
    def test_reset_is_seeded_and_observation_contains_only_state(self) -> None:
        def random_initial_state(rng: np.random.Generator) -> State:
            return State(
                x=float(rng.uniform(-0.1, 0.1)),
                theta=math.pi,
                x_dot=0.0,
                theta_dot=0.0,
            )

        env = ResidualCartPoleEnvironment(
            initial_state_sampler=random_initial_state,
        )
        first_observation, first_info = env.reset(seed=7)
        second_observation, second_info = env.reset(seed=7)

        self.assertEqual(first_observation.shape, (4,))
        np.testing.assert_allclose(first_observation, second_observation)
        self.assertEqual(first_info["mu"], second_info["mu"])
        self.assertNotIn("mu", first_observation.dtype.names or ())

    def test_mass_stays_fixed_during_an_episode(self) -> None:
        env = ResidualCartPoleEnvironment()
        _, reset_info = env.reset(seed=11)
        episode_mu = reset_info["mu"]
        episode_pole_mass = reset_info["pole_mass"]

        for _ in range(5):
            observation, _, terminated, truncated, info = env.step([0.0])
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(info["mu"], episode_mu)
            self.assertEqual(info["pole_mass"], episode_pole_mass)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

    def test_step_clips_and_scales_actor_action(self) -> None:
        env = ResidualCartPoleEnvironment()
        env.reset(seed=3)

        _, reward, terminated, truncated, info = env.step(np.array([5.0]))

        self.assertEqual(info["actor_action"], 1.0)
        self.assertEqual(info["physics_force"], 1.0)
        self.assertEqual(info["residual_force"], 3.0)
        self.assertEqual(info["proposed_force"], 4.0)
        self.assertEqual(info["applied_force"], 4.0)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_shield_filters_the_residual_near_upright(self) -> None:
        def upright_state(rng: np.random.Generator) -> State:
            del rng
            return State(0.0, 0.0, 0.0, 0.0)

        env = ResidualCartPoleEnvironment(
            shielded=True,
            initial_state_sampler=upright_state,
        )
        env.reset(seed=5)

        _, _, terminated, truncated, info = env.step(1.0)

        self.assertTrue(info["shield_active"])
        self.assertTrue(info["shield_projected"])
        self.assertFalse(info["shield_infeasible"])
        self.assertAlmostEqual(info["proposed_force"], 3.0)
        self.assertAlmostEqual(info["applied_force"], 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_horizon_sets_truncated_not_terminated(self) -> None:
        config = ExperimentConfig()
        config = replace(
            config,
            rollout=replace(
                config.rollout,
                horizon_seconds=config.plant.control_dt,
            ),
        )
        env = ResidualCartPoleEnvironment(config=config)
        env.reset(seed=13)

        _, _, terminated, truncated, _ = env.step(0.0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(0.0)

    def test_invalid_action_shape_is_rejected(self) -> None:
        env = ResidualCartPoleEnvironment()
        env.reset(seed=17)

        with self.assertRaises(ValueError):
            env.step(np.zeros(2))


if __name__ == "__main__":
    unittest.main()
