"""Reusable empirical evaluation tools for frozen cart-pole controllers.

The functions in this module use the project's existing ``State``, ``Rollout``,
controller, and RK4 plant interfaces. They evaluate sampled-data trajectories and
local finite-difference models; they do not certify safety, global stability, or a
region of attraction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.stats import qmc

from .config import ExperimentConfig
from .control_math import LQRData
from .controllers import Controller
from .data import Rollout, State
from .plant import step_rk4, wrap_angle

ControllerFactory: TypeAlias = Callable[[], Controller]
FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
print

@dataclass(frozen=True)
class SuccessCriteria:
    """Trajectory-level success limits used by every controller comparison."""

    final_window_seconds: float = 2.0
    theta_limit: float = 0.15
    theta_dot_limit: float = 0.5
    x_limit: float = 0.5


@dataclass(frozen=True)
class LocalClosedLoopAnalysis:
    """Finite-difference evidence for one local closed-loop equilibrium."""

    mu: float
    upright_residual_norm: float
    equilibrium: State
    equilibrium_force: float
    equilibrium_residual_norm: float
    equilibrium_bias_norm: float
    root_converged: bool
    jacobian: FloatArray
    eigenvalues: ComplexArray
    spectral_radius: float


@dataclass(frozen=True)
class RolloutMetrics:
    """Synchronized scalar metrics extracted from one empirical rollout."""

    controller_name: str
    mu: float
    step_count: int
    duration_seconds: float
    completed_horizon: bool
    success: bool
    track_violation: bool
    settling_time_seconds: float | None
    total_reward: float
    mean_absolute_force: float
    rms_force: float
    peak_absolute_force: float
    l1_control_effort: float
    l2_control_effort: float
    saturation_fraction: float
    swing_up_fraction: float
    lqr_fraction: float
    local_lyapunov_step_count: int
    realized_delta_v_mean: float | None
    realized_delta_v_max: float | None
    realized_v_decrease_fraction: float | None
    realized_v_condition_fraction: float | None
    shield_applicable: bool
    shield_activation_fraction: float | None
    shield_projection_fraction: float | None
    shield_infeasibility_fraction: float | None


def make_theta_theta_dot_slice(
    theta_min: float,
    theta_max: float,
    theta_points: int,
    theta_dot_min: float,
    theta_dot_max: float,
    theta_dot_points: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a regular ``(theta, theta_dot)`` slice with ``x=x_dot=0``.

    The returned state array has shape ``(theta_points, theta_dot_points, 4)``
    and keeps the canonical state order ``[x, theta, x_dot, theta_dot]``.
    """

    if theta_points < 2 or theta_dot_points < 2:
        raise ValueError("Each basin-slice axis must contain at least two points.")

    theta_values = np.linspace(theta_min, theta_max, theta_points, dtype=np.float64)
    theta_dot_values = np.linspace(
        theta_dot_min,
        theta_dot_max,
        theta_dot_points,
        dtype=np.float64,
    )
    states = np.zeros((theta_points, theta_dot_points, 4), dtype=np.float64)
    states[:, :, 1] = theta_values[:, np.newaxis]
    states[:, :, 3] = theta_dot_values[np.newaxis, :]
    return theta_values, theta_dot_values, states


def make_latin_hypercube_states(
    sample_count: int,
    bounds: tuple[tuple[float, float], ...],
    seed: int,
) -> FloatArray:
    """Generate reproducible four-dimensional Latin-hypercube initial states."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")
    if len(bounds) != 4:
        raise ValueError("Latin-hypercube bounds must follow the four-state order.")

    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.shape != (4, 2):
        raise ValueError("Latin-hypercube bounds must have shape (4, 2).")
    lower = bounds_array[:, 0]
    upper = bounds_array[:, 1]
    if np.any(lower >= upper):
        raise ValueError("Every lower state bound must be smaller than its upper bound.")

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=sample_count)
    return np.asarray(qmc.scale(unit_samples, lower, upper), dtype=np.float64)


def closed_loop_one_step(
    controller_factory: ControllerFactory,
    state: State,
    mu: float,
    config: ExperimentConfig,
) -> State:
    """Evaluate one deterministic closed-loop step on the mismatched plant.

    A new controller is used for each call. For the physics baseline, sufficiently
    small perturbations around the origin therefore enter the LQR branch from the
    documented reset mode. This defines the local hybrid-mode convention used by
    equilibrium solving and finite differences.
    """

    controller = controller_factory()
    controller.reset()
    decision = controller.act(state, deterministic=True)
    actual_plant = config.plant.with_mass_multiplier(mu)
    return step_rk4(state, decision.force, actual_plant)


def find_closed_loop_equilibrium(
    controller_factory: ControllerFactory,
    mu: float,
    config: ExperimentConfig,
    initial_guess: State | None = None,
    tolerance: float = 1e-10,
) -> tuple[State, bool, float, float]:
    """Numerically solve ``F_cl(s; mu) - s = 0`` near the upright state."""

    guess = State(0.0, 0.0, 0.0, 0.0) if initial_guess is None else initial_guess

    def residual(values: FloatArray) -> FloatArray:
        state = State.from_array(values)
        next_state = closed_loop_one_step(controller_factory, state, mu, config)
        return next_state.as_array() - values

    solution = root(residual, guess.as_array(), method="hybr", tol=tolerance)
    equilibrium = State.from_array(np.asarray(solution.x, dtype=np.float64))
    residual_norm = float(np.linalg.norm(residual(equilibrium.as_array()), ord=2))

    controller = controller_factory()
    controller.reset()
    equilibrium_force = controller.act(equilibrium, deterministic=True).force
    converged = bool(solution.success and residual_norm <= 10.0 * tolerance)
    return equilibrium, converged, residual_norm, float(equilibrium_force)


def finite_difference_closed_loop_jacobian(
    controller_factory: ControllerFactory,
    equilibrium: State,
    mu: float,
    config: ExperimentConfig,
    epsilon: float,
) -> FloatArray:
    """Estimate the Jacobian of the implemented one-step closed-loop map."""

    if epsilon <= 0.0:
        raise ValueError("Finite-difference epsilon must be positive.")

    center = equilibrium.as_array()
    jacobian = np.zeros((4, 4), dtype=np.float64)
    for column in range(4):
        offset = np.zeros(4, dtype=np.float64)
        offset[column] = epsilon
        plus = closed_loop_one_step(
            controller_factory,
            State.from_array(center + offset),
            mu,
            config,
        ).as_array()
        minus = closed_loop_one_step(
            controller_factory,
            State.from_array(center - offset),
            mu,
            config,
        ).as_array()
        jacobian[:, column] = (plus - minus) / (2.0 * epsilon)
    return jacobian


def analyze_local_closed_loop(
    controller_factory: ControllerFactory,
    mu: float,
    config: ExperimentConfig,
) -> LocalClosedLoopAnalysis:
    """Find the nearby fixed point and estimate its local spectral radius."""

    upright = State(0.0, 0.0, 0.0, 0.0)
    upright_next = closed_loop_one_step(controller_factory, upright, mu, config)
    upright_residual_norm = float(
        np.linalg.norm(upright_next.as_array() - upright.as_array(), ord=2)
    )

    equilibrium, converged, residual_norm, force = find_closed_loop_equilibrium(
        controller_factory=controller_factory,
        mu=mu,
        config=config,
        initial_guess=upright,
    )
    jacobian = finite_difference_closed_loop_jacobian(
        controller_factory=controller_factory,
        equilibrium=equilibrium,
        mu=mu,
        config=config,
        epsilon=config.lqr.finite_difference_epsilon,
    )
    eigenvalues = np.asarray(np.linalg.eigvals(jacobian), dtype=np.complex128)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    return LocalClosedLoopAnalysis(
        mu=mu,
        upright_residual_norm=upright_residual_norm,
        equilibrium=equilibrium,
        equilibrium_force=force,
        equilibrium_residual_norm=residual_norm,
        equilibrium_bias_norm=float(np.linalg.norm(equilibrium.as_array(), ord=2)),
        root_converged=converged,
        jacobian=jacobian,
        eigenvalues=eigenvalues,
        spectral_radius=spectral_radius,
    )


def _rollout_states(rollout: Rollout) -> tuple[State, ...]:
    return (rollout.initial_state,) + tuple(
        transition.next_state for transition in rollout.transitions
    )


def _inside_success_limits(state: State, criteria: SuccessCriteria) -> bool:
    return (
        abs(wrap_angle(state.theta)) < criteria.theta_limit
        and abs(state.theta_dot) < criteria.theta_dot_limit
        and abs(state.x) < criteria.x_limit
    )


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_rollout(
    rollout: Rollout,
    config: ExperimentConfig,
    lqr: LQRData,
    criteria: SuccessCriteria | None = None,
) -> RolloutMetrics:
    """Compute success, effort, mode, and realized Lyapunov measurements.

    ``V(s)=s.T P s`` uses the nominal LQR matrix, while every ``Delta V`` is
    measured on the actual rollout transition. The one-step condition fraction is
    restricted to transitions beginning inside ``V <= rho`` and is empirical for
    ``mu != 1``.
    """

    if criteria is None:
        criteria = SuccessCriteria()
    if criteria.final_window_seconds <= 0.0:
        raise ValueError("The final success window must be positive.")

    transitions = rollout.transitions
    step_count = len(transitions)
    expected_steps = round(
        config.rollout.horizon_seconds / config.plant.control_dt
    )
    completed_horizon = step_count == expected_steps and not rollout.track_violation
    states = _rollout_states(rollout)
    inside = np.asarray(
        [_inside_success_limits(state, criteria) for state in states],
        dtype=np.bool_,
    )

    window_steps = math.ceil(
        criteria.final_window_seconds / config.plant.control_dt
    )
    success = bool(
        completed_horizon
        and len(inside) >= window_steps + 1
        and np.all(inside[-(window_steps + 1) :])
    )

    settling_time: float | None = None
    if completed_horizon and inside[-1]:
        outside_indices = np.flatnonzero(~inside)
        settling_index = 0 if outside_indices.size == 0 else int(outside_indices[-1] + 1)
        settling_time = settling_index * config.plant.control_dt

    forces = np.asarray(
        [transition.decision.force for transition in transitions],
        dtype=np.float64,
    )
    if step_count:
        mean_absolute_force = float(np.mean(np.abs(forces)))
        rms_force = float(np.sqrt(np.mean(forces**2)))
        peak_absolute_force = float(np.max(np.abs(forces)))
        l1_control_effort = float(config.plant.control_dt * np.sum(np.abs(forces)))
        l2_control_effort = float(config.plant.control_dt * np.sum(forces**2))
        saturation_fraction = float(
            np.mean(np.isclose(np.abs(forces), config.plant.u_max, atol=1e-9))
        )
    else:
        mean_absolute_force = 0.0
        rms_force = 0.0
        peak_absolute_force = 0.0
        l1_control_effort = 0.0
        l2_control_effort = 0.0
        saturation_fraction = 0.0

    swing_up_count = sum(
        transition.decision.diagnostics.get("physics_mode") == "swing_up"
        for transition in transitions
    )
    lqr_count = sum(
        transition.decision.diagnostics.get("physics_mode") == "lqr"
        for transition in transitions
    )

    local_delta_v: list[float] = []
    local_v_before: list[float] = []
    for transition in transitions:
        state_values = transition.state.as_array()
        next_values = transition.next_state.as_array()
        v_before = float(state_values @ lqr.P @ state_values)
        if v_before <= config.shield.rho:
            v_after = float(next_values @ lqr.P @ next_values)
            local_v_before.append(v_before)
            local_delta_v.append(v_after - v_before)

    if local_delta_v:
        delta_v = np.asarray(local_delta_v, dtype=np.float64)
        v_before = np.asarray(local_v_before, dtype=np.float64)
        realized_delta_v_mean: float | None = float(np.mean(delta_v))
        realized_delta_v_max: float | None = float(np.max(delta_v))
        realized_v_decrease_fraction: float | None = float(np.mean(delta_v < 0.0))
        realized_v_condition_fraction: float | None = float(
            np.mean(
                delta_v
                <= -config.shield.alpha * v_before
                + config.shield.feasibility_tolerance
            )
        )
    else:
        realized_delta_v_mean = None
        realized_delta_v_max = None
        realized_v_decrease_fraction = None
        realized_v_condition_fraction = None

    shield_diagnostics = [
        transition.decision.diagnostics
        for transition in transitions
        if "shield_active" in transition.decision.diagnostics
    ]
    shield_applicable = len(shield_diagnostics) > 0
    if shield_applicable:
        shield_activation_fraction: float | None = float(
            np.mean([bool(item["shield_active"]) for item in shield_diagnostics])
        )
        shield_projection_fraction: float | None = float(
            np.mean([bool(item["shield_projected"]) for item in shield_diagnostics])
        )
        shield_infeasibility_fraction: float | None = float(
            np.mean([bool(item["shield_infeasible"]) for item in shield_diagnostics])
        )
    else:
        shield_activation_fraction = None
        shield_projection_fraction = None
        shield_infeasibility_fraction = None

    controller_name = (
        transitions[0].decision.controller_name if transitions else "unknown"
    )
    return RolloutMetrics(
        controller_name=controller_name,
        mu=rollout.mu,
        step_count=step_count,
        duration_seconds=step_count * config.plant.control_dt,
        completed_horizon=completed_horizon,
        success=success,
        track_violation=rollout.track_violation,
        settling_time_seconds=settling_time,
        total_reward=rollout.total_reward,
        mean_absolute_force=mean_absolute_force,
        rms_force=rms_force,
        peak_absolute_force=peak_absolute_force,
        l1_control_effort=l1_control_effort,
        l2_control_effort=l2_control_effort,
        saturation_fraction=saturation_fraction,
        swing_up_fraction=_safe_fraction(swing_up_count, step_count),
        lqr_fraction=_safe_fraction(lqr_count, step_count),
        local_lyapunov_step_count=len(local_delta_v),
        realized_delta_v_mean=realized_delta_v_mean,
        realized_delta_v_max=realized_delta_v_max,
        realized_v_decrease_fraction=realized_v_decrease_fraction,
        realized_v_condition_fraction=realized_v_condition_fraction,
        shield_applicable=shield_applicable,
        shield_activation_fraction=shield_activation_fraction,
        shield_projection_fraction=shield_projection_fraction,
        shield_infeasibility_fraction=shield_infeasibility_fraction,
    )
