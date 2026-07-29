from dataclasses import dataclass
from typing import Any

import numpy as np

from .actor import Actor
from .config import ExperimentConfig
from .controllers import ControllerKind


@dataclass(frozen=True)
class TrainingResult:
    """Expected output of your future SAC implementation."""

    actor: Actor
    metrics: dict[str, float]
    extra: dict[str, Any]


def sample_rollout_mass(
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> float:
    """Sample once at rollout start; keep this value fixed for that rollout."""

    return float(
        rng.uniform(config.training.mu_min, config.training.mu_max)
    )


def train_sac(
    controller_kind: ControllerKind,
    config: ExperimentConfig,
    seed: int,
) -> TrainingResult:
    """Boundary for an SAC library; intentionally not implemented.

    Inputs:
        controller_kind:
            RESIDUAL_SAC or SHIELDED_RESIDUAL_SAC.
        config:
            All static plant/controller/training settings.
        seed:
            Seed for the actor, replay sampling, rollout RNG, and library.

    Expected behavior:
        1. Sample one mu in [0.8, 1.2] at each rollout start.
        2. Keep that mu fixed throughout the rollout.
        3. Give the actor only normalize_state(state), never mu.
        4. Store Transition-compatible values in replay.
        5. Use identical rewards and data conventions for both residual agents.

    Output:
        TrainingResult containing the frozen actor, final scalar metrics, and
        optional library-specific state.
    """

    if controller_kind == ControllerKind.PHYSICS:
        raise ValueError("The physics baseline is not trained with SAC.")
    del config, seed
    raise NotImplementedError(
        "Connect your chosen SAC implementation at this one boundary."
    )

