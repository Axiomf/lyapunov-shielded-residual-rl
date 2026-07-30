"""Train bounded residual SAC policies with episode-wise mass randomization.

The Stable-Baselines3 objects stay inside this module. The rest of the project
continues to use the small :class:`Actor` protocol and the shared cart-pole data
objects.
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import stable_baselines3
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from .actor import Actor
from .config import ExperimentConfig
from .controllers import ControllerKind
from .data import State
from .plant import normalize_state


@dataclass(frozen=True)
class TrainingResult:
    """Project-level result returned by one seeded SAC training run."""

    actor: Actor
    metrics: dict[str, float]
    extra: dict[str, Any]


class TrainedSACActor:
    """Expose a trained Stable-Baselines3 SAC model through ``Actor``."""

    def __init__(self, model: SAC) -> None:
        self._model = model

    def act(self, observation: np.ndarray, deterministic: bool) -> float:
        """Return one finite scalar residual action in ``[-1, 1]``."""

        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (4,):
            raise ValueError("SAC observations must have shape (4,).")
        if not np.all(np.isfinite(values)):
            raise ValueError("SAC observations must be finite.")

        action, _ = self._model.predict(values, deterministic=deterministic)
        action_values = np.asarray(action, dtype=np.float64)
        if action_values.shape != (1,):
            raise RuntimeError("The trained SAC actor did not return one action.")

        scalar_action = float(action_values[0])
        if not math.isfinite(scalar_action):
            raise RuntimeError("The trained SAC actor returned a non-finite action.")
        return float(np.clip(scalar_action, -1.0, 1.0))

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> "TrainedSACActor":
        """Load an actor checkpoint saved by :func:`train_sac`."""

        model = SAC.load(str(checkpoint_path), device=device)
        return cls(model)


class _ResidualSACEnvironment(gym.Env):
    """Small Gymnasium adapter around ``ResidualCartPoleEnvironment``.

    Stable-Baselines3 uses a one-element array for its continuous action space.
    This adapter validates that array and passes a Python scalar into the project
    environment. It also checks the synchronized observation and fixed-mass
    episode contracts while training data are collected.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, environment: Any, config: ExperimentConfig) -> None:
        super().__init__()
        self._environment = environment
        self._config = config
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self._episode_mu: float | None = None
        self._episode_pole_mass: float | None = None
        self.checked_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and sample one fixed plant mass for the new episode."""

        super().reset(seed=seed)
        del options
        observation, info = self._environment.reset(seed=seed)
        self._episode_mu = float(info["mu"])
        self._episode_pole_mass = float(info["pole_mass"])
        return self._validated_observation(observation, info), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Validate the SAC action and advance the project environment."""

        action_values = np.asarray(action, dtype=np.float64)
        if action_values.shape != (1,):
            raise ValueError("SAC must provide exactly one scalar action.")

        scalar_action = float(action_values[0])
        if not math.isfinite(scalar_action):
            raise ValueError("SAC actions must be finite.")
        if scalar_action < -1.0 or scalar_action > 1.0:
            raise ValueError("SAC actions must remain in [-1, 1].")

        observation, reward, terminated, truncated, info = self._environment.step(
            scalar_action
        )
        self._validate_step_info(info, scalar_action)
        self.checked_steps += 1

        return (
            self._validated_observation(observation, info),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _validated_observation(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Confirm that the observation is exactly the normalized four-state."""

        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (4,):
            raise RuntimeError("The actor observation must have shape (4,).")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("The actor observation must contain finite values.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise RuntimeError("The normalized observation must remain in [-1, 1].")

        state = info.get("state")
        if not isinstance(state, State):
            raise TypeError("Environment info must preserve the physical State.")
        expected = normalize_state(state, self._config.observation)
        if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "The SAC observation must contain only the normalized full state."
            )
        return values

    def _validate_step_info(
        self,
        info: dict[str, Any],
        requested_action: float,
    ) -> None:
        """Confirm scalar action handling and fixed mass within an episode."""

        actor_action = info.get("actor_action")
        if not isinstance(actor_action, float):
            raise TypeError("The project environment must store a scalar action.")
        if not -1.0 <= actor_action <= 1.0:
            raise RuntimeError("The stored actor action left [-1, 1].")
        if not math.isclose(actor_action, requested_action, abs_tol=1e-7):
            raise RuntimeError("The project environment changed an in-range action.")

        if self._episode_mu is None or self._episode_pole_mass is None:
            raise RuntimeError("The environment was stepped before reset.")
        if float(info["mu"]) != self._episode_mu:
            raise RuntimeError("The mass multiplier changed within an episode.")
        if float(info["pole_mass"]) != self._episode_pole_mass:
            raise RuntimeError("The plant pole mass changed within an episode.")


class _EpisodeMetricsCallback(BaseCallback):
    """Collect episode returns and lengths without changing environment data."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_return = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=np.bool_)
        if rewards.shape != (1,) or dones.shape != (1,):
            raise RuntimeError("Training expects exactly one environment.")
        if not np.all(np.isfinite(rewards)):
            raise RuntimeError("The environment produced a non-finite reward.")

        self._current_return += float(rewards[0])
        self._current_length += 1
        if bool(dones[0]):
            self.episode_returns.append(self._current_return)
            self.episode_lengths.append(self._current_length)
            self._current_return = 0.0
            self._current_length = 0
        return True

    def scalar_metrics(self) -> dict[str, float]:
        """Return completed-episode means or the smoke-run prefix metrics."""

        if self.episode_returns:
            episode_return = float(np.mean(self.episode_returns))
            episode_length = float(np.mean(self.episode_lengths))
        else:
            episode_return = float(self._current_return)
            episode_length = float(self._current_length)
        return {
            "episode_return": episode_return,
            "episode_length": episode_length,
        }

    @property
    def current_prefix(self) -> dict[str, float]:
        """Return the unfinished final trajectory prefix for saved diagnostics."""

        return {
            "return": float(self._current_return),
            "length": float(self._current_length),
        }


def sample_rollout_mass(
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> float:
    """Draw one pole-mass multiplier for a complete training rollout."""

    return float(rng.uniform(config.training.mu_min, config.training.mu_max))


def _seed_everything(seed: int, using_cuda: bool) -> None:
    """Seed Python, NumPy, PyTorch through SB3, and SB3 helper generators."""

    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed, using_cuda=using_cuda)


def _assert_finite_replay_buffer(model: SAC) -> int:
    """Fail if any populated replay entry is non-finite or has a wrong shape."""

    replay_buffer = model.replay_buffer
    if replay_buffer is None:
        raise RuntimeError("SAC did not create a replay buffer.")

    used_entries = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
    if used_entries < 1:
        raise RuntimeError("SAC did not add any transitions to replay.")

    for name in (
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "timeouts",
    ):
        stored_values = getattr(replay_buffer, name, None)
        if stored_values is None:
            continue
        populated_values = np.asarray(stored_values[:used_entries])
        if not np.all(np.isfinite(populated_values)):
            raise RuntimeError(f"Replay buffer field {name!r} contains non-finite values.")

    observations = np.asarray(replay_buffer.observations[:used_entries])
    actions = np.asarray(replay_buffer.actions[:used_entries])
    if observations.shape[-1] != 4:
        raise RuntimeError("Replay observations must contain exactly four values.")
    if actions.shape[-1] != 1:
        raise RuntimeError("Replay actions must contain exactly one value.")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise RuntimeError("Replay actions left [-1, 1].")
    return int(used_entries)


def _save_training_artifacts(
    model: SAC,
    controller_kind: ControllerKind,
    config: ExperimentConfig,
    seed: int,
    metrics: dict[str, float],
    callback: _EpisodeMetricsCallback,
    replay_entries: int,
) -> str:
    """Save the actor checkpoint, full configuration, seed, and metrics."""

    run_directory = (
        Path(config.training.output_directory)
        / controller_kind.value
        / f"seed_{seed}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_stem = run_directory / "actor_checkpoint"
    model.save(str(checkpoint_stem))
    checkpoint_path = checkpoint_stem.with_suffix(".zip")

    configuration = {
        "controller_kind": controller_kind.value,
        "seed": seed,
        "sac_implementation": {
            "library": "stable_baselines3",
            "version": stable_baselines3.__version__,
            "algorithm": "SAC",
        },
        "experiment_config": asdict(config),
    }
    with (run_directory / "config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)

    training_summary = {
        "controller_kind": controller_kind.value,
        "seed": seed,
        "metrics": metrics,
        "completed_episode_returns": callback.episode_returns,
        "completed_episode_lengths": callback.episode_lengths,
        "unfinished_episode_prefix": callback.current_prefix,
        "replay_entries": replay_entries,
        "interface_checks": {
            "actions_are_scalar": True,
            "actions_are_bounded": True,
            "observation_shape_is_four": True,
            "observation_contains_only_state": True,
            "replay_values_are_finite": True,
            "mass_is_fixed_within_episode": True,
        },
    }
    with (run_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2, sort_keys=True)

    return str(checkpoint_path)


def train_sac(
    controller_kind: ControllerKind,
    config: ExperimentConfig,
    seed: int,
) -> TrainingResult:
    """Train one seeded residual or shielded-residual SAC actor.

    ``config.training.total_timesteps`` defaults to a short interface smoke run.
    It is not enough to claim policy convergence, robustness, local stability,
    safety, or an empirical region of attraction.
    """

    try:
        controller_kind = ControllerKind(controller_kind)
    except ValueError as error:
        raise ValueError(f"Unknown controller kind: {controller_kind}") from error
    if controller_kind == ControllerKind.PHYSICS:
        raise ValueError("The physics baseline is not trained with SAC.")

    using_cuda = config.training.device.startswith("cuda")
    _seed_everything(seed, using_cuda=using_cuda)

    # Local import avoids a module cycle: the environment uses
    # sample_rollout_mass() when it resets an episode.
    from .environment import ResidualCartPoleEnvironment

    project_environment = ResidualCartPoleEnvironment(
        config=config,
        shielded=controller_kind == ControllerKind.SHIELDED_RESIDUAL_SAC,
    )
    environment = _ResidualSACEnvironment(project_environment, config)

    # Seed and validate the environment before handing it to SB3. SAC receives
    # the same seed and seeds the wrapped environment again at learning start.
    environment.reset(seed=seed)

    model = SAC(
        policy="MlpPolicy",
        env=environment,
        learning_rate=config.training.learning_rate,
        buffer_size=config.training.buffer_size,
        learning_starts=config.training.learning_starts,
        batch_size=config.training.batch_size,
        tau=config.training.tau,
        gamma=config.training.gamma,
        train_freq=config.training.train_frequency,
        gradient_steps=config.training.gradient_steps,
        ent_coef=config.training.entropy_coefficient,
        target_entropy=config.training.target_entropy,
        policy_kwargs={"net_arch": list(config.training.hidden_sizes)},
        seed=seed,
        device=config.training.device,
        verbose=0,
    )

    callback = _EpisodeMetricsCallback()
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    if environment.checked_steps < 1:
        raise RuntimeError("SAC did not step the residual environment.")
    replay_entries = _assert_finite_replay_buffer(model)
    metrics = callback.scalar_metrics()
    checkpoint_path = _save_training_artifacts(
        model=model,
        controller_kind=controller_kind,
        config=config,
        seed=seed,
        metrics=metrics,
        callback=callback,
        replay_entries=replay_entries,
    )

    return TrainingResult(
        actor=TrainedSACActor(model),
        metrics=metrics,
        extra={
            "seed": seed,
            "checkpoint_path": checkpoint_path,
        },
    )
