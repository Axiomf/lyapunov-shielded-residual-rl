"""Typed configuration with no experiment constants hidden in code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    output_dir: str = "outputs"


@dataclass(frozen=True)
class PlantConfig:
    cart_mass: float = 1.0
    nominal_pole_mass: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.81
    dt: float = 0.02
    force_limit: float = 20.0
    track_limit: float = 2.4


@dataclass(frozen=True)
class ControllerConfig:
    energy_gain: float = 4.0
    cart_position_gain: float = 1.0
    cart_velocity_gain: float = 1.5
    switch_angle: float = 0.55
    residual_force_limit: float = 5.0
    lqr_q: tuple[float, ...] = (10.0, 2.0, 120.0, 8.0)
    lqr_r: float = 0.5


@dataclass(frozen=True)
class ShieldConfig:
    active_angle: float = 0.45
    alpha: float = 0.2
    tolerance: float = 1e-9


@dataclass(frozen=True)
class TrainingConfig:
    pole_mass_range: tuple[float, float] = (0.06, 0.14)
    episode_steps: int = 1000
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    batch_size: int = 256
    gamma: float = 0.99
    start_noise: tuple[float, ...] = (0.05, 0.05, 0.1, 0.1)


@dataclass(frozen=True)
class EvaluationConfig:
    pole_masses: tuple[float, ...] = (0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16)
    episodes_per_mass: int = 10
    episode_steps: int = 1000
    success_hold_steps: int = 100
    success_tolerance: tuple[float, ...] = (0.2, 0.5, 0.15, 0.75)
    fixed_point_tolerance: float = 1e-8
    jacobian_epsilon: float = 1e-5
    basin_angles: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    basin_angular_rates: tuple[float, float] = (-6.0, 6.0)
    basin_points_per_axis: int = 9
    basin_horizon_steps: int = 600


@dataclass(frozen=True)
class ExperimentConfig:
    study: StudyConfig = field(default_factory=StudyConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _tuple_fields(data: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    converted = dict(data)
    for name in names:
        if name in converted:
            converted[name] = tuple(converted[name])
    return converted


def load_config(path: str | Path) -> ExperimentConfig:
    """Load the single YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    return ExperimentConfig(
        study=StudyConfig(**_tuple_fields(raw.get("study", {}), ("seeds",))),
        plant=PlantConfig(**raw.get("plant", {})),
        controller=ControllerConfig(
            **_tuple_fields(raw.get("controller", {}), ("lqr_q",))
        ),
        shield=ShieldConfig(**raw.get("shield", {})),
        training=TrainingConfig(
            **_tuple_fields(
                raw.get("training", {}),
                ("pole_mass_range", "start_noise"),
            )
        ),
        evaluation=EvaluationConfig(
            **_tuple_fields(
                raw.get("evaluation", {}),
                (
                    "pole_masses",
                    "success_tolerance",
                    "basin_angles",
                    "basin_angular_rates",
                ),
            )
        ),
    )
