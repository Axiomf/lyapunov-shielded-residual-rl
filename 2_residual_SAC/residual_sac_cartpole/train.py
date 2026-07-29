import argparse

from examples.dummy_backend import DummyPhysicsController, DummyPlant
from residual_sac.config import EnvironmentConfig, SACConfig, TrainingConfig
from residual_sac.environment import ResidualCartPoleEnvironment
from residual_sac.replay_buffer import ReplayBuffer
from residual_sac.sac import SACAgent
from residual_sac.training import EpisodeSummary, train


def print_progress(summary: EpisodeSummary) -> None:
    print(
        f"episode={summary.episode:03d} "
        f"steps={summary.steps:04d} "
        f"mu={summary.mass_scale:.3f} "
        f"return={summary.return_value:.1f} "
        f"track_violation={summary.track_violation}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual SAC")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/sac_checkpoint.pt",
    )
    arguments = parser.parse_args()

    environment_config = EnvironmentConfig(
        beta=arguments.beta,
        max_steps=arguments.max_steps,
    )
    training_config = TrainingConfig(
        episodes=arguments.episodes,
        seed=arguments.seed,
    )
    sac_config = SACConfig(seed=arguments.seed)

    # Replace these two dummy objects with examples.project_adapters instances.
    plant = DummyPlant(environment_config.control_period)
    physics_controller = DummyPhysicsController(environment_config.u_max)
    environment = ResidualCartPoleEnvironment(
        plant,
        physics_controller,
        environment_config,
        seed=arguments.seed,
    )

    agent = SACAgent(
        environment.observation_size,
        environment.action_size,
        sac_config,
    )
    replay_buffer = ReplayBuffer(
        training_config.replay_capacity,
        environment.observation_size,
        environment.action_size,
        seed=arguments.seed,
    )

    train(
        environment,
        agent,
        replay_buffer,
        training_config,
        progress_callback=print_progress,
    )
    agent.save(arguments.checkpoint)
    print(f"saved checkpoint: {arguments.checkpoint}")


if __name__ == "__main__":
    main()

