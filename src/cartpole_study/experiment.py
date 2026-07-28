"""Training, deterministic evaluation, and dynamical-systems diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import root

from .config import ExperimentConfig
from .controllers import ClosedLoopPolicy, NominalController
from .plant import CartPolePlant, wrap_angle
from .shield import LyapunovShield

CONTROLLERS = ("nominal", "residual", "shielded")


def train_sac(
    config: ExperimentConfig,
    controller: str,
    seeds: Iterable[int],
    timesteps: int | None = None,
) -> list[Path]:
    """Train residual or shielded residual SAC and save one model per seed."""
    if controller not in {"residual", "shielded"}:
        raise ValueError("Only residual and shielded controllers are trainable")
    from stable_baselines3 import SAC

    from .env import ResidualCartPoleEnv

    output_root = Path(config.study.output_dir)
    saved: list[Path] = []
    for seed in seeds:
        env = ResidualCartPoleEnv(config, shielded=controller == "shielded")
        model = SAC(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            learning_rate=config.training.learning_rate,
            buffer_size=config.training.buffer_size,
            learning_starts=config.training.learning_starts,
            batch_size=config.training.batch_size,
            gamma=config.training.gamma,
            tensorboard_log=str(output_root / "tensorboard" / controller),
        )
        model.learn(total_timesteps=timesteps or config.training.total_timesteps)
        model_path = output_root / "models" / controller / f"seed_{seed}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        env.close()
        saved.append(model_path.with_suffix(".zip"))
    return saved


def evaluate_study(
    config: ExperimentConfig,
    controllers: Iterable[str],
    seeds: Iterable[int],
    masses: Iterable[float] | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    """Evaluate frozen masses and write raw and summarized CSV tables."""
    selected = tuple(controllers)
    unknown = set(selected) - set(CONTROLLERS)
    if unknown:
        raise ValueError(f"Unknown controllers: {sorted(unknown)}")

    mass_grid = tuple(masses or config.evaluation.pole_masses)
    episodes_per_mass = 2 if quick else config.evaluation.episodes_per_mass
    basin_points = 5 if quick else config.evaluation.basin_points_per_axis
    basin_horizon = 200 if quick else config.evaluation.basin_horizon_steps

    episode_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    for controller in selected:
        for seed in seeds:
            model = _load_model(config, controller, seed)
            for mass_index, pole_mass in enumerate(mass_grid):
                plant = CartPolePlant(config.plant, pole_mass)
                policy = _build_policy(config, controller, model)
                for episode in range(episodes_per_mass):
                    rollout_seed = seed * 100_000 + mass_index * 1_000 + episode
                    episode_rows.append(
                        _rollout(
                            config,
                            controller,
                            seed,
                            pole_mass,
                            rollout_seed,
                            plant,
                            policy,
                        )
                    )

                fixed_point, converged = _find_fixed_point(
                    config, plant, policy
                )
                jacobian = _closed_loop_jacobian(
                    plant,
                    policy,
                    fixed_point,
                    config.evaluation.jacobian_epsilon,
                )
                local_rows.append(
                    {
                        "controller": controller,
                        "seed": seed,
                        "pole_mass": pole_mass,
                        "fixed_point_converged": converged,
                        "fixed_point_displacement": float(
                            np.linalg.norm(fixed_point)
                        ),
                        "jacobian_spectral_radius": float(
                            np.max(np.abs(np.linalg.eigvals(jacobian)))
                        ),
                        **{
                            f"fixed_point_{name}": value
                            for name, value in zip(
                                ("x", "x_dot", "theta", "theta_dot"),
                                fixed_point,
                                strict=True,
                            )
                        },
                    }
                )
                basin_rows.append(
                    _basin_slice(
                        config,
                        controller,
                        seed,
                        plant,
                        policy,
                        basin_points,
                        basin_horizon,
                    )
                )

    output_dir = Path(config.study.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.DataFrame(episode_rows)
    local = pd.DataFrame(local_rows)
    basin = pd.DataFrame(basin_rows)
    metric_columns = [
        "return",
        "success",
        "terminal_displacement",
        "control_effort",
        "track_violation",
        "lyapunov_decrease_violations",
        "shield_activation_rate",
        "shield_change_rate",
        "shield_infeasibility_rate",
    ]
    summary = (
        episodes.groupby(["controller", "pole_mass"], as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    paths = {
        "episodes": output_dir / "episodes.csv",
        "summary": output_dir / "summary.csv",
        "local_diagnostics": output_dir / "local_diagnostics.csv",
        "basin_slices": output_dir / "basin_slices.csv",
    }
    episodes.to_csv(paths["episodes"], index=False)
    summary.to_csv(paths["summary"], index=False)
    local.to_csv(paths["local_diagnostics"], index=False)
    basin.to_csv(paths["basin_slices"], index=False)
    return paths


def _load_model(config: ExperimentConfig, controller: str, seed: int) -> Any | None:
    if controller == "nominal":
        return None
    from stable_baselines3 import SAC

    path = Path(config.study.output_dir) / "models" / controller / f"seed_{seed}.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train the {controller} controller for seed {seed} first."
        )
    return SAC.load(path)


def _build_policy(
    config: ExperimentConfig, controller: str, model: Any | None
) -> ClosedLoopPolicy:
    nominal_plant = CartPolePlant(config.plant, config.plant.nominal_pole_mass)
    nominal = NominalController(nominal_plant, config.controller)
    shield = (
        LyapunovShield(
            nominal.lyapunov_matrix,
            config.controller,
            config.shield,
        )
        if controller == "shielded"
        else None
    )
    return ClosedLoopPolicy(
        name=controller,
        nominal=nominal,
        residual_limit=config.controller.residual_force_limit,
        model=model,
        shield=shield,
    )


def _initial_state(config: ExperimentConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.asarray(config.training.start_noise, dtype=np.float64)
    state = rng.uniform(-noise, noise)
    state[2] = wrap_angle(np.pi + state[2])
    return state


def _is_success(
    states: list[np.ndarray],
    tolerance: ArrayLike,
    hold_steps: int,
) -> bool:
    if len(states) < hold_steps:
        return False
    recent = np.abs(np.asarray(states[-hold_steps:]))
    recent[:, 2] = np.abs(
        [wrap_angle(float(angle)) for angle in np.asarray(states[-hold_steps:])[:, 2]]
    )
    return bool(np.all(recent <= np.asarray(tolerance)))


def _rollout(
    config: ExperimentConfig,
    controller: str,
    seed: int,
    pole_mass: float,
    rollout_seed: int,
    plant: CartPolePlant,
    policy: ClosedLoopPolicy,
) -> dict[str, Any]:
    state = _initial_state(config, rollout_seed)
    states: list[np.ndarray] = []
    total_reward = 0.0
    effort = 0.0
    track_violation = False
    lyapunov_violations = 0
    shield_active = 0
    shield_changed = 0
    shield_infeasible = 0

    for _ in range(config.evaluation.episode_steps):
        result = policy.action(plant, state)
        if abs(wrap_angle(float(state[2]))) <= config.shield.active_angle:
            derivative = policy.nominal.lyapunov_derivative(
                plant, state, result.force
            )
            if derivative > config.shield.tolerance:
                lyapunov_violations += 1

        if result.shield is not None:
            shield_active += int(result.shield.active)
            shield_changed += int(result.shield.changed)
            shield_infeasible += int(result.shield.infeasible)

        state = plant.step(state, result.force)
        states.append(state.copy())
        theta = wrap_angle(float(state[2]))
        reward = (
            np.cos(theta)
            - 0.10 * state[0] ** 2
            - 0.01 * state[1] ** 2
            - 0.005 * state[3] ** 2
            - 0.001 * result.force**2
        )
        total_reward += float(reward)
        effort += result.force**2 * config.plant.dt
        if abs(state[0]) > config.plant.track_limit:
            track_violation = True
            total_reward -= 10.0
            break

    steps = max(len(states), 1)
    tolerance = np.asarray(config.evaluation.success_tolerance)
    terminal = states[-1] if states else state
    scaled_terminal = np.abs(terminal) / tolerance
    scaled_terminal[2] = abs(wrap_angle(float(terminal[2]))) / tolerance[2]
    return {
        "controller": controller,
        "seed": seed,
        "pole_mass": pole_mass,
        "rollout_seed": rollout_seed,
        "steps": len(states),
        "return": total_reward,
        "success": _is_success(
            states,
            tolerance,
            config.evaluation.success_hold_steps,
        ),
        "terminal_displacement": float(np.linalg.norm(scaled_terminal)),
        "control_effort": effort,
        "track_violation": track_violation,
        "lyapunov_decrease_violations": lyapunov_violations,
        "shield_activation_rate": shield_active / steps,
        "shield_change_rate": shield_changed / steps,
        "shield_infeasibility_rate": shield_infeasible / steps,
    }


def _find_fixed_point(
    config: ExperimentConfig,
    plant: CartPolePlant,
    policy: ClosedLoopPolicy,
) -> tuple[np.ndarray, bool]:
    def residual(state: np.ndarray) -> np.ndarray:
        action = policy.action(plant, state).force
        return plant.step(state, action) - state

    solution = root(
        residual,
        np.zeros(4, dtype=np.float64),
        tol=config.evaluation.fixed_point_tolerance,
    )
    point = np.asarray(solution.x, dtype=np.float64)
    point[2] = wrap_angle(float(point[2]))
    return point, bool(solution.success)


def _closed_loop_jacobian(
    plant: CartPolePlant,
    policy: ClosedLoopPolicy,
    state: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    jacobian = np.empty((4, 4), dtype=np.float64)
    for index in range(4):
        offset = np.zeros(4, dtype=np.float64)
        offset[index] = epsilon
        plus = state + offset
        minus = state - offset
        next_plus = plant.step(plus, policy.action(plant, plus).force)
        next_minus = plant.step(minus, policy.action(plant, minus).force)
        difference = next_plus - next_minus
        difference[2] = wrap_angle(float(next_plus[2] - next_minus[2]))
        jacobian[:, index] = difference / (2.0 * epsilon)
    return jacobian


def _basin_slice(
    config: ExperimentConfig,
    controller: str,
    seed: int,
    plant: CartPolePlant,
    policy: ClosedLoopPolicy,
    points_per_axis: int,
    horizon_steps: int,
) -> dict[str, Any]:
    angle_grid = np.linspace(*config.evaluation.basin_angles, points_per_axis)
    rate_grid = np.linspace(
        *config.evaluation.basin_angular_rates, points_per_axis
    )
    tolerance = np.asarray(config.evaluation.success_tolerance)
    converged = 0
    total = points_per_axis**2
    for angle in angle_grid:
        for angular_rate in rate_grid:
            state = np.array([0.0, 0.0, angle, angular_rate], dtype=np.float64)
            history: list[np.ndarray] = []
            for _ in range(horizon_steps):
                force = policy.action(plant, state).force
                state = plant.step(state, force)
                history.append(state.copy())
                if abs(state[0]) > config.plant.track_limit:
                    break
            converged += int(
                _is_success(
                    history,
                    tolerance,
                    min(config.evaluation.success_hold_steps, horizon_steps),
                )
            )

    angle_span = config.evaluation.basin_angles[1] - config.evaluation.basin_angles[0]
    rate_span = (
        config.evaluation.basin_angular_rates[1]
        - config.evaluation.basin_angular_rates[0]
    )
    fraction = converged / total
    return {
        "controller": controller,
        "seed": seed,
        "pole_mass": plant.pole_mass,
        "grid_points": total,
        "converged_points": converged,
        "converged_fraction": fraction,
        "slice_area_estimate": fraction * angle_span * rate_span,
        "angle_min": config.evaluation.basin_angles[0],
        "angle_max": config.evaluation.basin_angles[1],
        "angular_rate_min": config.evaluation.basin_angular_rates[0],
        "angular_rate_max": config.evaluation.basin_angular_rates[1],
        "horizon_steps": horizon_steps,
    }


def config_record(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-friendly configuration record for external logging."""
    return asdict(config)
