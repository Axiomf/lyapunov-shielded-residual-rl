from collections.abc import Callable
from dataclasses import dataclass
from math import pi

import numpy as np

from residual_sac.config import TrainingConfig
from residual_sac.environment import ResidualCartPoleEnvironment
from residual_sac.replay_buffer import ReplayBuffer
from residual_sac.sac import SACAgent


InitialStateSampler = Callable[[np.random.Generator], np.ndarray]
ProgressCallback = Callable[["EpisodeSummary"], None]


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    total_steps: int
    mass_scale: float
    return_value: float
    track_violation: bool
    final_actor_loss: float | None
    final_critic_loss: float | None


def default_initial_state(random: np.random.Generator) -> np.ndarray:
    """Placeholder distribution; replace with the shared training distribution."""
    return np.array(
        [
            random.uniform(-0.25, 0.25),
            random.uniform(-pi, pi),
            random.uniform(-0.25, 0.25),
            random.uniform(-0.5, 0.5),
        ],
        dtype=np.float64,
    )


def train(
    environment: ResidualCartPoleEnvironment,
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
    initial_state_sampler: InitialStateSampler = default_initial_state,
    progress_callback: ProgressCallback | None = None,
) -> list[EpisodeSummary]:
    random = np.random.default_rng(config.seed)
    summaries: list[EpisodeSummary] = []
    total_steps = 0

    for episode in range(config.episodes):
        # No mass is passed: environment samples one mu and holds it for the rollout.
        observation, reset_info = environment.reset(initial_state_sampler(random))
        episode_return = 0.0
        track_violation = False
        last_losses: dict[str, float] | None = None

        for episode_step in range(1, environment.config.max_steps + 1):
            if total_steps < config.random_steps:
                action = random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            else:
                action = agent.act(observation, deterministic=False)

            (
                next_observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = environment.step(action)

            # Store only true termination. Time-limit truncation still bootstraps.
            replay_buffer.add(
                observation,
                action,
                reward,
                next_observation,
                terminated,
            )
            observation = next_observation
            episode_return += reward
            total_steps += 1
            track_violation = track_violation or bool(
                step_info["track_violation"]
            )

            can_update = (
                total_steps >= config.update_after
                and len(replay_buffer) >= config.batch_size
            )
            if can_update:
                for _ in range(config.updates_per_step):
                    batch = replay_buffer.sample(config.batch_size, agent.device)
                    last_losses = agent.update(batch)

            if terminated or truncated:
                break

        summary = EpisodeSummary(
            episode=episode,
            steps=episode_step,
            total_steps=total_steps,
            mass_scale=float(reset_info["mass_scale"]),
            return_value=float(episode_return),
            track_violation=track_violation,
            final_actor_loss=(
                last_losses["actor_loss"] if last_losses is not None else None
            ),
            final_critic_loss=(
                last_losses["critic_loss"] if last_losses is not None else None
            ),
        )
        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(summary)

    return summaries

