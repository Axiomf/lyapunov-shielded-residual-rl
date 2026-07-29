"""Generate synchronized sampled-data trajectories for training and evaluation.

Theoretical role
----------------
This module realizes the finite-horizon closed-loop recursion

    s[k] --controller--> u[k] --ZOH + nonlinear plant--> s[k + 1],

where the plant transition is approximated by RK4 over one control period.  A
single call to :func:`run_rollout` therefore produces an empirical trajectory of
one of the project's three controllers on the actual mass-mismatched plant.

The controller keeps its nominal model, while the simulated plant uses
``m_p = mu * m_p0`` with one fixed ``mu`` for the complete rollout.  This is the
code-level separation between the nominal model used for control or shielding
and the actual model used to gather evidence.  In particular, a rollout can be
used to estimate success, settling time, realized Lyapunov change, control
effort, or empirical basin membership, but it does not certify safety, global
stability, or a region of attraction.

Every sampled step is stored as the shared :class:`Transition` format, and the
complete trajectory is returned as a :class:`Rollout`.  Using this same function
for all controllers helps keep plant settings, time indexing, termination, and
action conventions paired across comparisons.
"""

from collections.abc import Callable

from .config import ExperimentConfig
from .controllers import Controller
from .data import Rollout, State, Transition
from .plant import has_track_violation, step_rk4, wrap_angle

# A reward is evaluated after one plant step as r(s[k], u[k], s[k + 1]).
# Keeping this callable interface independent of the controller ensures that the
# two residual agents can be trained and compared with exactly the same reward.
RewardFunction = Callable[[State, float, State], float]


def starter_reward(state: State, force: float, next_state: State) -> float:
    """Return a simple placeholder reward for one sampled transition.

    The current definition is the negative stage cost

        r[k] = -(theta[k+1]^2 + 0.1*x[k+1]^2 + 0.001*u[k]^2),

    where the angle is represented in ``[-pi, pi)``.  The first two terms favor
    the upright pole and centered cart; the final term penalizes control effort.
    Larger rewards are therefore better, with zero being the ideal value of this
    placeholder expression.

    Parameters
    ----------
    state:
        State ``s[k]`` before applying the control.  It is accepted to preserve
        the general transition-reward interface but is not used by this formula.
    force:
        Final scalar force ``u[k]`` applied to the plant, after controller-side
        clipping or shield projection.
    next_state:
        State ``s[k + 1]`` produced by the sampled nonlinear plant.

    Notes
    -----
    This reward is a training objective, not a Lyapunov function and not the
    project's success criterion.  If it is replaced, the same finalized reward
    must be used for both residual agents so their comparison remains paired.
    """

    # This placeholder is a next-state stage cost; ``state`` remains in the
    # signature so a future shared reward may depend on the full transition.
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
    """Simulate one finite-horizon episode at a fixed mass multiplier.

    For ``h = config.plant.control_dt``, this function generates the sampled
    closed-loop sequence

        u[k] = controller(s[k]),
        s[k + 1] = F_h(s[k], u[k]; mu),

    where :func:`step_rk4` approximates ``F_h`` using zero-order-held control.
    The actual pole mass is ``m_p = mu * m_p0`` and does not change during the
    rollout.  The controller itself is not rebuilt with this mass: it retains
    the nominal model contained in ``config``.

    Parameters
    ----------
    controller:
        Any of the three project controllers through their shared ``reset`` and
        ``act`` interface.
    initial_state:
        Common state ``s[0]`` from which the paired comparison begins.
    mu:
        Fixed plant pole-mass multiplier for this complete rollout.
    config:
        Synchronized experiment settings.  The horizon and control period come
        from this object, and its nominal plant is copied before changing mass.
    reward_function:
        Shared scalar transition reward.  Both residual agents must use the same
        function during a paired experiment.
    deterministic:
        Passed to the controller's actor convention.  Use ``True`` for frozen
        policy evaluation; stochastic actions may be used during training.

    Returns
    -------
    Rollout
        Time-aligned transitions and the episode-level track-violation flag.
        The violating transition is retained, then the rollout terminates.

    Notes
    -----
    This routine detects track violations only after each sampled plant step.
    It does not enforce track safety.  It also does not evaluate the final-2-s
    success condition; that trajectory-level calculation belongs in evaluation
    code.  Results from state grids or sampled initial states describe an
    empirical basin only, not a certified region of attraction.
    """

    # Copy the nominal plant settings and change only the pole mass.  Controller
    # and shield calculations continue to use their nominal model, whereas this
    # plant represents the actual dynamics used for empirical evaluation.
    actual_plant = config.plant.with_mass_multiplier(mu)

    # Convert the physical horizon to a number of zero-order-hold intervals.
    # With the default 10 s horizon and 0.02 s period, this gives 500 steps.
    step_count = round(
        config.rollout.horizon_seconds / config.plant.control_dt
    )
    state = initial_state
    transitions: list[Transition] = []
    track_violation = False

    # Reset hybrid controller memory (notably the hysteretic physics mode) so
    # every paired rollout starts from the same controller-side condition.
    controller.reset()

    for _ in range(step_count):
        # Compute one decision from s[k].  ``decision.force`` is the final action
        # after residual bounding, actuator clipping, and any shield projection.
        decision = controller.act(state, deterministic=deterministic)

        # Apply that scalar force with zero-order hold to obtain s[k + 1] on the
        # actual mass-mismatched plant, then align reward and termination data
        # with this same sampled transition.
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

        # Preserve the first violating transition for analysis, but do not
        # simulate beyond failure because the rollout has terminated.
        if track_violation:
            break

    return Rollout(
        mu=mu,
        initial_state=initial_state,
        transitions=tuple(transitions),
        track_violation=track_violation,
    )

