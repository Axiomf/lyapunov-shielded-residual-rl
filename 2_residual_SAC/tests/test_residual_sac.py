import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.dummy_backend import DummyPhysicsController, DummyPlant
from residual_sac.config import EnvironmentConfig, SACConfig
from residual_sac.environment import (
    ResidualCartPoleEnvironment,
    normalize_state,
)

try:
    import torch

    from residual_sac.replay_buffer import ReplayBuffer
    from residual_sac.sac import SACAgent
except ImportError:
    torch = None


class EnvironmentTests(unittest.TestCase):
    def test_beta_limit_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentConfig(u_max=10.0, beta=3.01)

    def test_normalization_wraps_angle(self) -> None:
        state = np.array([0.0, 3.0 * np.pi, 0.0, 0.0])
        observation = normalize_state(
            state,
            (2.4, np.pi, 5.0, 10.0),
            5.0,
        )
        self.assertAlmostEqual(float(observation[1]), -1.0, places=6)

    def test_residual_is_scaled_and_force_is_clipped(self) -> None:
        config = EnvironmentConfig(beta=3.0, max_steps=2)
        environment = ResidualCartPoleEnvironment(
            DummyPlant(config.control_period),
            DummyPhysicsController(config.u_max),
            config,
        )
        _, reset_info = environment.reset(
            np.array([0.0, -2.0, 0.0, 0.0]),
            pole_mass_scale=0.9,
        )
        _, _, _, _, info = environment.step(np.array([1.5]))

        self.assertAlmostEqual(reset_info["mass_scale"], 0.9)
        self.assertAlmostEqual(info["normalized_residual"], 1.0)
        self.assertAlmostEqual(info["residual_force"], 3.0)
        self.assertLessEqual(abs(info["applied_force"]), 10.0)
        self.assertAlmostEqual(info["mass_scale"], 0.9)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SACTests(unittest.TestCase):
    def test_action_range_update_and_checkpoint(self) -> None:
        sac_config = SACConfig(hidden_sizes=(16, 16), seed=4)
        agent = SACAgent(4, 1, sac_config)
        observation = np.zeros(4, dtype=np.float32)

        deterministic_action_1 = agent.act(observation, deterministic=True)
        deterministic_action_2 = agent.act(observation, deterministic=True)
        self.assertTrue(np.all(deterministic_action_1 <= 1.0))
        self.assertTrue(np.all(deterministic_action_1 >= -1.0))
        np.testing.assert_allclose(
            deterministic_action_1,
            deterministic_action_2,
        )

        replay = ReplayBuffer(64, 4, 1, seed=2)
        random = np.random.default_rng(2)
        for _ in range(32):
            current = random.normal(size=4).astype(np.float32)
            action = random.uniform(-1.0, 1.0, size=1).astype(np.float32)
            following = random.normal(size=4).astype(np.float32)
            replay.add(current, action, 0.5, following, False)

        losses = agent.update(replay.sample(16, agent.device))
        self.assertTrue(np.isfinite(losses["actor_loss"]))
        self.assertTrue(np.isfinite(losses["critic_loss"]))
        self.assertGreater(losses["alpha"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent.pt"
            agent.save(checkpoint)
            loaded_agent = SACAgent.from_checkpoint(checkpoint)
            loaded_action = loaded_agent.act(observation, deterministic=True)
            np.testing.assert_allclose(
                agent.act(observation, deterministic=True),
                loaded_action,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()

