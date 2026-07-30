"""Small end-to-end test for SAC training, saving, loading, and inference.

This is an interface smoke test. The short training run is not evidence of policy
convergence, robustness, local stability, safety, or an empirical basin.
"""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind, build_controller
from cartpole.data import Rollout, State
from cartpole.simulation import run_rollout
from cartpole.training import TrainingResult, TrainedSACActor, train_sac


ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "artifacts" / "test_training_and_inference"
)
TRAINING_SEED = 11
INFERENCE_MASS_MULTIPLIER = 1.1


def make_test_config() -> ExperimentConfig:
    """Return a small synchronized configuration for an end-to-end smoke test."""

    config = ExperimentConfig()
    return replace(
        config,
        rollout=replace(
            config.rollout,
            horizon_seconds=10 * config.plant.control_dt,
        ),
        training=replace(
            config.training,
            total_timesteps=32,
            learning_starts=8,
            buffer_size=128,
            batch_size=8,
            hidden_sizes=(16, 16),
            output_directory=str(ARTIFACT_DIRECTORY),
        ),
    )


def train_and_store_test_model(config: ExperimentConfig) -> TrainingResult:
    """Train one residual SAC smoke model and store its project artifacts."""

    return train_sac(
        controller_kind=ControllerKind.RESIDUAL_SAC,
        config=config,
        seed=TRAINING_SEED,
    )


def load_and_test_model(
    checkpoint_path: Path,
    config: ExperimentConfig,
) -> tuple[TrainedSACActor, Rollout]:
    """Load the stored actor and run one deterministic sampled-data rollout."""

    actor = TrainedSACActor.load(checkpoint_path, device=config.training.device)
    controller = build_controller(
        kind=ControllerKind.RESIDUAL_SAC,
        config=config,
        actor=actor,
    )
    initial_state = State(x=0.0, theta=0.05, x_dot=0.0, theta_dot=0.0)
    rollout = run_rollout(
        controller=controller,
        initial_state=initial_state,
        mu=INFERENCE_MASS_MULTIPLIER,
        config=config,
        deterministic=True,
    )
    return actor, rollout


class TrainingAndInferenceTests(unittest.TestCase):
    def test_train_store_load_and_run_deterministic_inference(self) -> None:
        config = make_test_config()

        training_result = train_and_store_test_model(config)
        checkpoint_path = Path(training_result.extra["checkpoint_path"])

        self.assertTrue(checkpoint_path.is_file())
        self.assertTrue((checkpoint_path.parent / "config.json").is_file())
        self.assertTrue((checkpoint_path.parent / "metrics.json").is_file())

        test_observation = np.array([0.0, 0.05, 0.0, 0.0], dtype=np.float64)
        action_before_loading = training_result.actor.act(
            test_observation,
            deterministic=True,
        )

        loaded_actor, rollout = load_and_test_model(checkpoint_path, config)
        action_after_loading = loaded_actor.act(
            test_observation,
            deterministic=True,
        )

        self.assertAlmostEqual(action_before_loading, action_after_loading, places=7)
        self.assertGreaterEqual(action_after_loading, -1.0)
        self.assertLessEqual(action_after_loading, 1.0)

        self.assertEqual(rollout.mu, INFERENCE_MASS_MULTIPLIER)
        self.assertGreater(len(rollout.transitions), 0)

        for transition in rollout.transitions:
            decision = transition.decision
            self.assertEqual(
                decision.controller_name,
                ControllerKind.RESIDUAL_SAC.value,
            )
            self.assertLessEqual(abs(decision.force), config.plant.u_max)
            self.assertLessEqual(abs(decision.residual_force), config.residual.beta)
            self.assertTrue(np.all(np.isfinite(transition.next_state.as_array())))


if __name__ == "__main__":
    unittest.main()
