"""Run the complete empirical evaluation of the physics baseline only.

The evaluated controller is energy shaping with hysteretic switching to nominal
sampled-data LQR. No actor, residual action, SAC model, or learning is used here.

Full mode evaluates:

* upright fixed points and finite-difference local Jacobians for each plant mass;
* a 61 by 61 ``(theta, theta_dot)`` empirical-basin slice at ``x=x_dot=0``;
* 1,000 shared four-dimensional Latin-hypercube states for each of three seeds;
* success, track violations, settling, control effort, controller mode, and
  realized change of the nominal LQR candidate ``V(s)=s.T P s``.

All results are empirical sampled-data measurements. The saved success sets are
empirical basins on the stated finite initial-state sets, not certified regions of
attraction. Local spectral radii apply only to the selected upright/LQR hybrid
mode and do not imply global stability or track safety.
"""

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from cartpole.config import ExperimentConfig
from cartpole.control_math import LQRData, build_lqr
from cartpole.controllers import PhysicsController
from cartpole.data import State
from cartpole.evaluation import (
    ControllerFactory,
    LocalClosedLoopAnalysis,
    RolloutMetrics,
    SuccessCriteria,
    analyze_local_closed_loop,
    evaluate_rollout,
    make_latin_hypercube_states,
    make_theta_theta_dot_slice,
)
from cartpole.simulation import run_rollout


@dataclass(frozen=True)
class BaselineEvaluationSettings:
    """Fixed mass grid and initial-state designs for the baseline experiment."""

    mass_multipliers: tuple[float, ...]
    evaluation_seeds: tuple[int, ...]
    theta_points: int
    theta_dot_points: int
    theta_min: float
    theta_max: float
    theta_dot_min: float
    theta_dot_max: float
    lhs_sample_count: int
    lhs_bounds: tuple[tuple[float, float], ...]
    quick_mode: bool = False


@dataclass
class SummaryAccumulator:
    """Small streaming accumulator used for success-versus-mass summaries."""

    rollout_count: int = 0
    success_count: int = 0
    track_violation_count: int = 0
    completed_horizon_count: int = 0
    l2_effort_sum: float = 0.0
    rms_force_sum: float = 0.0
    saturation_fraction_sum: float = 0.0
    swing_up_fraction_sum: float = 0.0
    lqr_fraction_sum: float = 0.0
    local_lyapunov_steps: int = 0
    local_condition_satisfied: float = 0.0
    local_decrease_steps: float = 0.0
    successful_settling_times: list[float] = field(default_factory=list)

    def add(self, metrics: RolloutMetrics) -> None:
        """Add one rollout without retaining its full trajectory."""

        self.rollout_count += 1
        self.success_count += int(metrics.success)
        self.track_violation_count += int(metrics.track_violation)
        self.completed_horizon_count += int(metrics.completed_horizon)
        self.l2_effort_sum += metrics.l2_control_effort
        self.rms_force_sum += metrics.rms_force
        self.saturation_fraction_sum += metrics.saturation_fraction
        self.swing_up_fraction_sum += metrics.swing_up_fraction
        self.lqr_fraction_sum += metrics.lqr_fraction

        local_steps = metrics.local_lyapunov_step_count
        self.local_lyapunov_steps += local_steps
        if metrics.realized_v_condition_fraction is not None:
            self.local_condition_satisfied += (
                local_steps * metrics.realized_v_condition_fraction
            )
        if metrics.realized_v_decrease_fraction is not None:
            self.local_decrease_steps += (
                local_steps * metrics.realized_v_decrease_fraction
            )

        if metrics.success and metrics.settling_time_seconds is not None:
            self.successful_settling_times.append(metrics.settling_time_seconds)

    def as_row(self, sample_set: str, mu: float, seed: str | int) -> dict[str, object]:
        """Return one flat CSV row."""

        count = self.rollout_count
        settling = self.successful_settling_times or []
        return {
            "sample_set": sample_set,
            "mu": mu,
            "seed": seed,
            "rollout_count": count,
            "success_count": self.success_count,
            "success_rate": self.success_count / count if count else 0.0,
            "track_violation_count": self.track_violation_count,
            "track_violation_rate": (
                self.track_violation_count / count if count else 0.0
            ),
            "completed_horizon_rate": (
                self.completed_horizon_count / count if count else 0.0
            ),
            "mean_successful_settling_time_seconds": (
                float(np.mean(settling)) if settling else ""
            ),
            "median_successful_settling_time_seconds": (
                float(np.median(settling)) if settling else ""
            ),
            "mean_l2_control_effort": self.l2_effort_sum / count if count else 0.0,
            "mean_rms_force": self.rms_force_sum / count if count else 0.0,
            "mean_saturation_fraction": (
                self.saturation_fraction_sum / count if count else 0.0
            ),
            "mean_swing_up_fraction": (
                self.swing_up_fraction_sum / count if count else 0.0
            ),
            "mean_lqr_fraction": self.lqr_fraction_sum / count if count else 0.0,
            "local_lyapunov_step_count": self.local_lyapunov_steps,
            "realized_v_condition_fraction": (
                self.local_condition_satisfied / self.local_lyapunov_steps
                if self.local_lyapunov_steps
                else ""
            ),
            "realized_v_decrease_fraction": (
                self.local_decrease_steps / self.local_lyapunov_steps
                if self.local_lyapunov_steps
                else ""
            ),
            "shield_statistics": "not_applicable_to_physics_baseline",
        }


def full_settings(config: ExperimentConfig) -> BaselineEvaluationSettings:
    """Return the research evaluation design described in this script."""

    return BaselineEvaluationSettings(
        mass_multipliers=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
        evaluation_seeds=(0, 1, 2),
        theta_points=61,
        theta_dot_points=61,
        theta_min=-math.pi,
        theta_max=math.pi,
        theta_dot_min=-10.0,
        theta_dot_max=10.0,
        lhs_sample_count=1_000,
        lhs_bounds=(
            (-config.plant.x_limit, config.plant.x_limit),
            (-math.pi, math.pi),
            (-config.observation.scales[2], config.observation.scales[2]),
            (-config.observation.scales[3], config.observation.scales[3]),
        ),
    )


def quick_settings(config: ExperimentConfig) -> BaselineEvaluationSettings:
    """Return a small design for checking code paths, not for research claims."""

    settings = full_settings(config)
    return BaselineEvaluationSettings(
        mass_multipliers=(0.6, 1.0, 1.4),
        evaluation_seeds=settings.evaluation_seeds,
        theta_points=9,
        theta_dot_points=9,
        theta_min=settings.theta_min,
        theta_max=settings.theta_max,
        theta_dot_min=settings.theta_dot_min,
        theta_dot_max=settings.theta_dot_max,
        lhs_sample_count=20,
        lhs_bounds=settings.lhs_bounds,
        quick_mode=True,
    )


def _metric_field_names() -> list[str]:
    prefix = ["sample_set", "seed", "initial_state_index", "x", "theta", "x_dot", "theta_dot"]
    return prefix + [item.name for item in fields(RolloutMetrics)]


def _metric_row(
    sample_set: str,
    seed: str | int,
    state_index: int,
    initial_state: State,
    metrics: RolloutMetrics,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_set": sample_set,
        "seed": seed,
        "initial_state_index": state_index,
        "x": initial_state.x,
        "theta": initial_state.theta,
        "x_dot": initial_state.x_dot,
        "theta_dot": initial_state.theta_dot,
    }
    row.update(asdict(metrics))
    return row


def _local_row(analysis: LocalClosedLoopAnalysis) -> dict[str, object]:
    row: dict[str, object] = {
        "mu": analysis.mu,
        "upright_residual_norm": analysis.upright_residual_norm,
        "equilibrium_x": analysis.equilibrium.x,
        "equilibrium_theta": analysis.equilibrium.theta,
        "equilibrium_x_dot": analysis.equilibrium.x_dot,
        "equilibrium_theta_dot": analysis.equilibrium.theta_dot,
        "equilibrium_force": analysis.equilibrium_force,
        "equilibrium_residual_norm": analysis.equilibrium_residual_norm,
        "equilibrium_bias_norm": analysis.equilibrium_bias_norm,
        "root_converged": analysis.root_converged,
        "spectral_radius": analysis.spectral_radius,
        "finite_difference_locally_stable": analysis.spectral_radius < 1.0,
    }
    for row_index in range(4):
        for column_index in range(4):
            row[f"jacobian_{row_index}_{column_index}"] = analysis.jacobian[
                row_index, column_index
            ]
    for index, eigenvalue in enumerate(analysis.eigenvalues):
        row[f"eigenvalue_{index}_real"] = float(eigenvalue.real)
        row[f"eigenvalue_{index}_imag"] = float(eigenvalue.imag)
    return row


def _write_local_analysis(
    output_directory: Path,
    settings: BaselineEvaluationSettings,
    controller_factory: ControllerFactory,
    config: ExperimentConfig,
) -> None:
    analyses: list[LocalClosedLoopAnalysis] = []
    rows: list[dict[str, object]] = []

    print("Local upright/LQR analysis")
    print("mu    fixed-point residual    bias norm    spectral radius")
    for mu in settings.mass_multipliers:
        analysis = analyze_local_closed_loop(controller_factory, mu, config)
        analyses.append(analysis)
        rows.append(_local_row(analysis))
        print(
            f"{mu:3.1f}   {analysis.upright_residual_norm:18.3e}   "
            f"{analysis.equilibrium_bias_norm:9.3e}   "
            f"{analysis.spectral_radius:15.8f}"
        )

    local_csv = output_directory / "local_closed_loop_analysis.csv"
    with local_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output_directory / "local_closed_loop_analysis.npz",
        mass_multipliers=np.asarray(settings.mass_multipliers, dtype=np.float64),
        equilibria=np.asarray(
            [analysis.equilibrium.as_array() for analysis in analyses],
            dtype=np.float64,
        ),
        jacobians=np.asarray(
            [analysis.jacobian for analysis in analyses],
            dtype=np.float64,
        ),
        eigenvalues=np.asarray(
            [analysis.eigenvalues for analysis in analyses],
            dtype=np.complex128,
        ),
        spectral_radius=np.asarray(
            [analysis.spectral_radius for analysis in analyses],
            dtype=np.float64,
        ),
    )
print

def _save_shared_initial_states(
    output_directory: Path,
    settings: BaselineEvaluationSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_values, theta_dot_values, slice_states = make_theta_theta_dot_slice(
        theta_min=settings.theta_min,
        theta_max=settings.theta_max,
        theta_points=settings.theta_points,
        theta_dot_min=settings.theta_dot_min,
        theta_dot_max=settings.theta_dot_max,
        theta_dot_points=settings.theta_dot_points,
    )
    lhs_states = np.asarray(
        [
            make_latin_hypercube_states(
                sample_count=settings.lhs_sample_count,
                bounds=settings.lhs_bounds,
                seed=seed,
            )
            for seed in settings.evaluation_seeds
        ],
        dtype=np.float64,
    )

    np.savez_compressed(
        output_directory / "shared_initial_states.npz",
        state_order=np.asarray(["x", "theta", "x_dot", "theta_dot"]),
        evaluation_seeds=np.asarray(settings.evaluation_seeds, dtype=np.int64),
        lhs_bounds=np.asarray(settings.lhs_bounds, dtype=np.float64),
        lhs_states=lhs_states,
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        theta_theta_dot_slice_states=slice_states,
    )
    return theta_values, theta_dot_values, slice_states, lhs_states


def _evaluate_slice(
    output_directory: Path,
    settings: BaselineEvaluationSettings,
    config: ExperimentConfig,
    lqr: LQRData,
    controller: PhysicsController,
    theta_values: np.ndarray,
    theta_dot_values: np.ndarray,
    slice_states: np.ndarray,
    criteria: SuccessCriteria,
) -> list[dict[str, object]]:
    shape = (
        len(settings.mass_multipliers),
        settings.theta_points,
        settings.theta_dot_points,
    )
    success = np.zeros(shape, dtype=np.bool_)
    track_violation = np.zeros(shape, dtype=np.bool_)
    settling_time = np.full(shape, np.nan, dtype=np.float64)
    l2_effort = np.zeros(shape, dtype=np.float64)
    lqr_fraction = np.zeros(shape, dtype=np.float64)
    realized_v_condition = np.full(shape, np.nan, dtype=np.float64)
    summary_rows: list[dict[str, object]] = []

    csv_path = output_directory / "theta_theta_dot_slice_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_metric_field_names())
        writer.writeheader()

        points_per_mass = settings.theta_points * settings.theta_dot_points
        for mass_index, mu in enumerate(settings.mass_multipliers):
            accumulator = SummaryAccumulator()
            for theta_index in range(settings.theta_points):
                for theta_dot_index in range(settings.theta_dot_points):
                    flat_index = (
                        theta_index * settings.theta_dot_points + theta_dot_index
                    )
                    initial_state = State.from_array(
                        slice_states[theta_index, theta_dot_index]
                    )
                    rollout = run_rollout(
                        controller=controller,
                        initial_state=initial_state,
                        mu=mu,
                        config=config,
                        deterministic=True,
                    )
                    metrics = evaluate_rollout(rollout, config, lqr, criteria)
                    accumulator.add(metrics)
                    writer.writerow(
                        _metric_row(
                            sample_set="theta_theta_dot_slice",
                            seed="deterministic_grid",
                            state_index=flat_index,
                            initial_state=initial_state,
                            metrics=metrics,
                        )
                    )

                    success[mass_index, theta_index, theta_dot_index] = metrics.success
                    track_violation[
                        mass_index, theta_index, theta_dot_index
                    ] = metrics.track_violation
                    if metrics.settling_time_seconds is not None:
                        settling_time[
                            mass_index, theta_index, theta_dot_index
                        ] = metrics.settling_time_seconds
                    l2_effort[
                        mass_index, theta_index, theta_dot_index
                    ] = metrics.l2_control_effort
                    lqr_fraction[
                        mass_index, theta_index, theta_dot_index
                    ] = metrics.lqr_fraction
                    if metrics.realized_v_condition_fraction is not None:
                        realized_v_condition[
                            mass_index, theta_index, theta_dot_index
                        ] = metrics.realized_v_condition_fraction

                    if (flat_index + 1) % 500 == 0:
                        print(
                            f"  slice mu={mu:.1f}: "
                            f"{flat_index + 1}/{points_per_mass}"
                        )

            row = accumulator.as_row("theta_theta_dot_slice", mu, "deterministic_grid")
            summary_rows.append(row)
            print(
                f"  slice mu={mu:.1f} complete: "
                f"success={row['success_rate']:.3f}, "
                f"track_violation={row['track_violation_rate']:.3f}"
            )

    np.savez_compressed(
        output_directory / "theta_theta_dot_slice_results.npz",
        mass_multipliers=np.asarray(settings.mass_multipliers, dtype=np.float64),
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        success=success,
        track_violation=track_violation,
        settling_time_seconds=settling_time,
        l2_control_effort=l2_effort,
        lqr_fraction=lqr_fraction,
        realized_v_condition_fraction=realized_v_condition,
    )
    return summary_rows


def _evaluate_lhs(
    output_directory: Path,
    settings: BaselineEvaluationSettings,
    config: ExperimentConfig,
    lqr: LQRData,
    controller: PhysicsController,
    lhs_states: np.ndarray,
    criteria: SuccessCriteria,
) -> list[dict[str, object]]:
    shape = (
        len(settings.mass_multipliers),
        len(settings.evaluation_seeds),
        settings.lhs_sample_count,
    )
    success = np.zeros(shape, dtype=np.bool_)
    track_violation = np.zeros(shape, dtype=np.bool_)
    settling_time = np.full(shape, np.nan, dtype=np.float64)
    l2_effort = np.zeros(shape, dtype=np.float64)
    lqr_fraction = np.zeros(shape, dtype=np.float64)
    realized_v_condition = np.full(shape, np.nan, dtype=np.float64)
    summary_rows: list[dict[str, object]] = []

    csv_path = output_directory / "latin_hypercube_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_metric_field_names())
        writer.writeheader()

        for mass_index, mu in enumerate(settings.mass_multipliers):
            mass_accumulator = SummaryAccumulator()
            for seed_index, seed in enumerate(settings.evaluation_seeds):
                seed_accumulator = SummaryAccumulator()
                for state_index in range(settings.lhs_sample_count):
                    initial_state = State.from_array(lhs_states[seed_index, state_index])
                    rollout = run_rollout(
                        controller=controller,
                        initial_state=initial_state,
                        mu=mu,
                        config=config,
                        deterministic=True,
                    )
                    metrics = evaluate_rollout(rollout, config, lqr, criteria)
                    seed_accumulator.add(metrics)
                    mass_accumulator.add(metrics)
                    writer.writerow(
                        _metric_row(
                            sample_set="latin_hypercube_4d",
                            seed=seed,
                            state_index=state_index,
                            initial_state=initial_state,
                            metrics=metrics,
                        )
                    )

                    success[mass_index, seed_index, state_index] = metrics.success
                    track_violation[
                        mass_index, seed_index, state_index
                    ] = metrics.track_violation
                    if metrics.settling_time_seconds is not None:
                        settling_time[
                            mass_index, seed_index, state_index
                        ] = metrics.settling_time_seconds
                    l2_effort[
                        mass_index, seed_index, state_index
                    ] = metrics.l2_control_effort
                    lqr_fraction[
                        mass_index, seed_index, state_index
                    ] = metrics.lqr_fraction
                    if metrics.realized_v_condition_fraction is not None:
                        realized_v_condition[
                            mass_index, seed_index, state_index
                        ] = metrics.realized_v_condition_fraction

                seed_row = seed_accumulator.as_row("latin_hypercube_4d", mu, seed)
                summary_rows.append(seed_row)
                print(
                    f"  LHS mu={mu:.1f}, seed={seed} complete: "
                    f"success={seed_row['success_rate']:.3f}, "
                    f"track_violation={seed_row['track_violation_rate']:.3f}"
                )

            summary_rows.append(
                mass_accumulator.as_row("latin_hypercube_4d", mu, "all")
            )

    np.savez_compressed(
        output_directory / "latin_hypercube_results.npz",
        mass_multipliers=np.asarray(settings.mass_multipliers, dtype=np.float64),
        evaluation_seeds=np.asarray(settings.evaluation_seeds, dtype=np.int64),
        success=success,
        track_violation=track_violation,
        settling_time_seconds=settling_time,
        l2_control_effort=l2_effort,
        lqr_fraction=lqr_fraction,
        realized_v_condition_fraction=realized_v_condition,
    )
    return summary_rows


def _write_summary(output_directory: Path, rows: list[dict[str, object]]) -> None:
    with (output_directory / "success_vs_mass_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    output_directory: Path,
    config: ExperimentConfig,
    settings: BaselineEvaluationSettings,
    criteria: SuccessCriteria,
) -> None:
    metadata = {
        "controller": "physics",
        "controller_definition": "energy_shaping_with_hysteretic_switch_to_discrete_lqr",
        "deterministic_actions": True,
        "contains_rl": False,
        "experiment_config": asdict(config),
        "evaluation_settings": asdict(settings),
        "success_criteria": asdict(criteria),
        "state_order": ["x", "theta", "x_dot", "theta_dot"],
        "mass_definition": "plant pole_mass = mu * nominal pole_mass",
        "seed_interpretation": (
            "Seeds randomize only the Latin-hypercube initial-state design. "
            "The physics controller itself is deterministic and has no policy seed."
        ),
        "local_analysis_mode_convention": (
            "Each one-step map starts from controller reset. Perturbations near "
            "the origin satisfy the entry thresholds and use the LQR hybrid mode."
        ),
        "lyapunov_interpretation": (
            "P is the nominal LQR Riccati matrix. Delta V is realized on the "
            "actual mass-mismatched plant and is summarized for V <= rho."
        ),
        "shield_statistics": "not applicable to the physics baseline",
        "claim_limits": (
            "Results are empirical. They do not certify safety, global stability, "
            "or a region of attraction."
        ),
    }
    with (output_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def run_baseline_evaluation(
    output_directory: Path,
    quick: bool = False,
) -> None:
    """Run and save the complete physics-controller evaluation."""

    config = ExperimentConfig()
    settings = quick_settings(config) if quick else full_settings(config)
    criteria = SuccessCriteria()
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_metadata(output_directory, config, settings, criteria)

    lqr = build_lqr(config.plant, config.lqr)

    def controller_factory() -> PhysicsController:
        return PhysicsController(config, lqr)

    controller = controller_factory()
    _write_local_analysis(
        output_directory=output_directory,
        settings=settings,
        controller_factory=controller_factory,
        config=config,
    )
    theta_values, theta_dot_values, slice_states, lhs_states = (
        _save_shared_initial_states(output_directory, settings)
    )

    total_rollouts = len(settings.mass_multipliers) * (
        settings.theta_points * settings.theta_dot_points
        + len(settings.evaluation_seeds) * settings.lhs_sample_count
    )
    if quick:
        print("Quick validation mode: these basin results are not research results.")
    print(f"Running {total_rollouts:,} deterministic baseline rollouts")
    start_time = time.perf_counter()

    summary_rows = _evaluate_slice(
        output_directory=output_directory,
        settings=settings,
        config=config,
        lqr=lqr,
        controller=controller,
        theta_values=theta_values,
        theta_dot_values=theta_dot_values,
        slice_states=slice_states,
        criteria=criteria,
    )
    summary_rows.extend(
        _evaluate_lhs(
            output_directory=output_directory,
            settings=settings,
            config=config,
            lqr=lqr,
            controller=controller,
            lhs_states=lhs_states,
            criteria=criteria,
        )
    )
    _write_summary(output_directory, summary_rows)

    elapsed = time.perf_counter() - start_time
    print(f"Finished in {elapsed:.1f} s")
    print(f"Results: {output_directory}")
    print("Interpret successful initial states only as an empirical basin.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate only the energy-shaping/LQR physics baseline."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small code-path check instead of the full research design.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Directory for CSV, JSON, and NPZ results.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    arguments = _parse_arguments()
    output_directory = arguments.output_directory
    if output_directory is None:
        name = "physics_baseline_quick" if arguments.quick else "physics_baseline"
        output_directory = Path("artifacts") / name
    run_baseline_evaluation(output_directory, quick=arguments.quick)


if __name__ == "__main__":
    main()
