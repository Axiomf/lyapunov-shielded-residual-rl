from dataclasses import dataclass
from math import ceil

import numpy as np

from residual_sac.environment import ResidualCartPoleEnvironment, wrap_angle
from residual_sac.sac import SACAgent


@dataclass
class RolloutResult:
    mass_scale: float
    initial_state: list[float]
    return_value: float
    steps: int
    success: bool
    track_violation: bool
    mean_absolute_force: float
    maximum_absolute_force: float


def rollout(
    environment: ResidualCartPoleEnvironment,
    agent: SACAgent,
    initial_state: np.ndarray,
    mass_scale: float,
) -> RolloutResult:
    observation, _ = environment.reset(initial_state, mass_scale)
    states = [environment.state.copy()]
    forces: list[float] = []
    total_reward = 0.0
    track_violation = False

    for _ in range(environment.config.max_steps):
        action = agent.act(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        states.append(environment.state.copy())
        forces.append(float(info["applied_force"]))
        total_reward += reward
        track_violation = track_violation or bool(info["track_violation"])
        if terminated or truncated:
            break

    success = success_from_trajectory(
        np.asarray(states),
        track_violation,
        environment.config.control_period,
    )
    return RolloutResult(
        mass_scale=float(mass_scale),
        initial_state=np.asarray(initial_state, dtype=float).tolist(),
        return_value=float(total_reward),
        steps=len(forces),
        success=success,
        track_violation=track_violation,
        mean_absolute_force=float(np.mean(np.abs(forces))) if forces else 0.0,
        maximum_absolute_force=float(np.max(np.abs(forces))) if forces else 0.0,
    )


def success_from_trajectory(
    states: np.ndarray,
    track_violation: bool,
    control_period: float,
) -> bool:
    """Apply the success definition from the project prompt."""
    if track_violation:
        return False

    final_interval_count = ceil(2.0 / control_period)
    if len(states) - 1 < final_interval_count:
        return False

    # Include both endpoints of the final two-second interval.
    final_states = states[-(final_interval_count + 1) :]
    theta = np.array([wrap_angle(value) for value in final_states[:, 1]])
    return bool(
        np.all(np.abs(theta) < 0.15)
        and np.all(np.abs(final_states[:, 3]) < 0.5)
        and np.all(np.abs(final_states[:, 0]) < 0.5)
    )


def evaluate_mass_grid(
    environment: ResidualCartPoleEnvironment,
    agent: SACAgent,
    mass_scales: np.ndarray,
    initial_states: np.ndarray,
) -> list[RolloutResult]:
    """Use the same supplied initial states for every mass value."""
    results: list[RolloutResult] = []
    for mass_scale in mass_scales:
        for initial_state in initial_states:
            results.append(
                rollout(
                    environment,
                    agent,
                    np.asarray(initial_state),
                    float(mass_scale),
                )
            )
    return results
