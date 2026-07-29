from collections.abc import Callable

from .config import ExperimentConfig
from .controllers import Controller
from .data import Rollout, State, Transition
from .plant import has_track_violation, step_rk4, wrap_angle


RewardFunction = Callable[[State, float, State], float]


def starter_reward(state: State, force: float, next_state: State) -> float:
    """Placeholder reward.

    Input:
        current State, applied scalar force, next State.

    Output:
        one scalar reward.

    Use the same finalized reward function for both residual agents.
    """

    del state
    angle_cost = wrap_angle(next_state.theta) ** 2
    position_cost = 0.1 * next_state.x**2
    effort_cost = 0.001 * force**2
    return -(angle_cost + position_cost + effort_cost)


def run_rollout(
    controller: Controller,
    initial_state: State,
    mu: float,
    config: ExperimentConfig,
    reward_function: RewardFunction = starter_reward,
    deterministic: bool = True,
) -> Rollout:
    """Run one episode with one fixed plant mass multiplier."""

    actual_plant = config.plant.with_mass_multiplier(mu)
    step_count = round(
        config.rollout.horizon_seconds / config.plant.control_dt
    )
    state = initial_state
    transitions: list[Transition] = []
    track_violation = False
    controller.reset()

    for _ in range(step_count):
        decision = controller.act(state, deterministic=deterministic)
        next_state = step_rk4(state, decision.force, actual_plant)
        reward = reward_function(state, decision.force, next_state)
        track_violation = has_track_violation(next_state, actual_plant)

        transitions.append(
            Transition(
                state=state,
                decision=decision,
                reward=reward,
                next_state=next_state,
                terminated=track_violation,
            )
        )
        state = next_state
        if track_violation:
            break

    return Rollout(
        mu=mu,
        initial_state=initial_state,
        transitions=tuple(transitions),
        track_violation=track_violation,
    )

