"""Run a small, synchronized comparison of the three project controllers.

Theoretical role
----------------
This file connects the main mathematical objects used by the project. For each
controller, it generates one finite sampled-data closed-loop trajectory

    u[k] = controller(s[k]),
    s[k + 1] = F_h(s[k], u[k]; mu),

where ``F_h`` is the nonlinear zero-order-hold/RK4 plant map and ``h = 0.02 s``
by default. It therefore matches these parts of the project theory:

1. the nonlinear sampled-data cart-pole model;
2. the hybrid energy-shaping/discrete-time-LQR physics controller;
3. the bounded residual-control composition; and
4. the nominal one-step Lyapunov shield.

This is only a smoke-test entry point, not the full research experiment.
``ZeroActor`` represents ``a_RL = 0``, so no trained SAC policy is evaluated.
The single rollout uses the nominal mass ``mu = 1`` and one initial state; it
does not estimate robustness, an empirical basin, local stability, or the full
success criterion. Its output is empirical simulation data and is not a safety
or region-of-attraction guarantee.
"""

from cartpole.actor import ZeroActor
from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind, build_controller
from cartpole.data import State
from cartpole.simulation import run_rollout


def main() -> None:
    """Simulate one nominal-mass rollout for each controller."""

    # One shared configuration keeps the plant period, limits, nominal model,
    # controller gains, residual bound, and shield settings synchronized.
    config = ExperimentConfig()

    # This placeholder policy implements pi(o) = 0. Thus it tests the residual
    # controller interfaces without claiming to evaluate a trained SAC agent.
    actor = ZeroActor()

    # State order is s = [x, theta, x_dot, theta_dot]^T, with theta = 0 upright.
    # Starting at theta = 2.8 rad, near the downward position pi, exercises the
    # swing-up part of the hybrid physics controller before local LQR balancing.
    initial_state = State(x=0.0, theta=2.8, x_dot=0.0, theta_dot=0.0)

    # ControllerKind contains exactly the three comparators in the study. Each
    # receives the same configuration, actor, initial state, and plant mass.
    for kind in ControllerKind:
        controller = build_controller(kind, config, actor)
        rollout = run_rollout(
            controller=controller,
            initial_state=initial_state,
            mu=1.0,  # Nominal plant: m_p = mu * m_p,0 = m_p,0.
            config=config,
        )

        # A rollout can end before the configured horizon if |x| exceeds the
        # track limit. ``return`` is the shared undiscounted starter reward; it
        # is neither the Lyapunov function nor the project's success criterion.
        print(
            f"{kind.value:24s} "
            f"steps={len(rollout.transitions):4d} "
            f"return={rollout.total_reward:8.2f} "
            f"track_violation={rollout.track_violation}"
        )


# The guard runs this demonstration only when executing ``python main.py``;
# importing ``main`` from a test or another module has no simulation side effect.
if __name__ == "__main__":
    main()

