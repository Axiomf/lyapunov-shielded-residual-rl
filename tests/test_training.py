import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind
from cartpole.training import TrainedSACActor, train_sac


class SACTrainingTests(unittest.TestCase):
    def test_small_training_run_saves_artifacts_and_returns_actor(self) -> None:
        with TemporaryDirectory() as output_directory:
            config = ExperimentConfig()
            config = replace(
                config,
                rollout=replace(
                    config.rollout,
                    horizon_seconds=4 * config.plant.control_dt,
                ),
                training=replace(
                    config.training,
                    total_timesteps=12,
                    learning_starts=4,
                    buffer_size=64,
                    batch_size=4,
                    hidden_sizes=(8, 8),
                    output_directory=output_directory,
                ),
            )

            result = train_sac(ControllerKind.RESIDUAL_SAC, config, seed=7)

            self.assertEqual(
                set(result.metrics),
                {"episode_return", "episode_length"},
            )
            self.assertTrue(all(np.isfinite(list(result.metrics.values()))))
            self.assertEqual(result.extra["seed"], 7)

            checkpoint_path = Path(result.extra["checkpoint_path"])
            self.assertTrue(checkpoint_path.is_file())
            run_directory = checkpoint_path.parent
            self.assertTrue((run_directory / "config.json").is_file())
            self.assertTrue((run_directory / "metrics.json").is_file())

            with (run_directory / "config.json").open(encoding="utf-8") as file:
                saved_config = json.load(file)
            self.assertEqual(saved_config["seed"], 7)
            self.assertEqual(
                saved_config["controller_kind"],
                ControllerKind.RESIDUAL_SAC.value,
            )
            self.assertEqual(
                saved_config["experiment_config"]["training"]["total_timesteps"],
                12,
            )

            with (run_directory / "metrics.json").open(encoding="utf-8") as file:
                saved_metrics = json.load(file)
            self.assertEqual(saved_metrics["metrics"], result.metrics)
            self.assertTrue(all(saved_metrics["interface_checks"].values()))
            self.assertEqual(saved_metrics["replay_entries"], 12)

            for deterministic in (False, True):
                action = result.actor.act(np.zeros(4), deterministic=deterministic)
                self.assertIsInstance(action, float)
                self.assertGreaterEqual(action, -1.0)
                self.assertLessEqual(action, 1.0)

            loaded_actor = TrainedSACActor.load(checkpoint_path)
            loaded_action = loaded_actor.act(np.zeros(4), deterministic=True)
            self.assertIsInstance(loaded_action, float)
            self.assertGreaterEqual(loaded_action, -1.0)
            self.assertLessEqual(loaded_action, 1.0)

    def test_small_shielded_training_run_uses_same_interface(self) -> None:
        with TemporaryDirectory() as output_directory:
            config = ExperimentConfig()
            config = replace(
                config,
                rollout=replace(
                    config.rollout,
                    horizon_seconds=2 * config.plant.control_dt,
                ),
                training=replace(
                    config.training,
                    total_timesteps=4,
                    learning_starts=4,
                    buffer_size=16,
                    batch_size=4,
                    hidden_sizes=(8, 8),
                    output_directory=output_directory,
                ),
            )

            result = train_sac(
                ControllerKind.SHIELDED_RESIDUAL_SAC,
                config,
                seed=3,
            )

            checkpoint_path = Path(result.extra["checkpoint_path"])
            self.assertTrue(checkpoint_path.is_file())
            with (checkpoint_path.parent / "metrics.json").open(
                encoding="utf-8"
            ) as file:
                saved_metrics = json.load(file)
            self.assertTrue(all(saved_metrics["interface_checks"].values()))

    def test_physics_controller_is_not_trained(self) -> None:
        with self.assertRaises(ValueError):
            train_sac(ControllerKind.PHYSICS, ExperimentConfig(), seed=0)


if __name__ == "__main__":
    unittest.main()
