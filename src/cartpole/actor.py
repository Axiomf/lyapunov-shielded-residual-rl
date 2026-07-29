"""Residual-policy interfaces and small actors used for testing.

The actor represents the learned policy map in the residual-control equations:

    o[k] = normalize_state(s[k]),
    a_RL[k] = pi(o[k]) in [-1, 1],
    u_prop[k] = clip(u_physics[k] + beta * a_RL[k], -u_max, u_max).

Here ``o[k]`` contains the normalized state in canonical order
``[x, theta, x_dot, theta_dot]``.  It deliberately does not contain the plant
mass multiplier ``mu``.  Therefore, a trained actor can respond to the observed
state evolution but cannot directly select an action from the true pole mass.

This module defines only the policy boundary.  Residual scaling by ``beta``,
actuator clipping, and Lyapunov-shield projection are performed by the
controller classes.  In particular, an actor output bound alone is not a
stability guarantee or a certified region of attraction.
"""

from typing import Protocol

import numpy as np

from .data import FloatArray


class Actor(Protocol):
    """Minimal interface between the controllers and an SAC implementation.

    Mathematically, an implementation supplies the dimensionless residual policy
    ``a_RL = pi(o)``.  ``Protocol`` uses structural typing: a third-party SAC
    adapter can be used here when it provides the same ``act`` method, without
    inheriting from this class.

    The two action modes keep training and evaluation conventions synchronized:

    * ``deterministic=False`` requests a stochastic policy sample for training
      data collection.
    * ``deterministic=True`` requests the frozen policy's deterministic action
      for evaluation and dynamical-systems analysis.

    The adapter, not this protocol, decides how its SAC library obtains the
    deterministic action.  It must use that convention consistently across all
    masses, initial states, controllers, and seeds.
    """

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        """Return one normalized residual action.

        Parameters
        ----------
        observation:
            Normalized ``float64`` state with shape ``(4,)`` and order
            ``[x, theta, x_dot, theta_dot]``.  The value ``mu`` is not included.
        deterministic:
            Whether to use the frozen deterministic policy rather than sample
            from the actor's action distribution.

        Returns
        -------
        float
            The dimensionless action ``a_RL`` in ``[-1, 1]``.  The residual
            controller also clips this value defensively before multiplying it
            by the force scale ``beta``.
        """

        ...


class ZeroActor:
    """Stateless placeholder implementing the zero policy ``pi_0(o) = 0``.

    This actor is useful before a trained SAC model is connected and in smoke
    tests that exercise the common controller interface.  In the unshielded
    residual controller, it gives ``beta*a_RL = 0``, so the residual proposal
    reduces to the clipped physics-controller command.

    It is not a learned policy and makes no robustness or stability claim.
    """

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        """Return zero for every observation and in both action modes."""

        # The arguments are required by Actor but cannot affect the zero policy.
        del observation, deterministic
        return 0.0


class RandomActor:
    """Seeded, state-independent actor for stochastic-path smoke tests.

    During data-collection mode it samples ``Uniform(-1, 1)``.  This checks that
    residual scaling, clipping, rollout storage, and seed handling can accept
    nonzero actor actions.  During deterministic mode it returns zero so repeated
    evaluations do not depend on the random-number-generator state.

    This class is not SAC: it has no neural network, performs no learning, and
    does not use the observed state.  Its deterministic output is only a testing
    convention, not the deterministic action of a trained stochastic policy.
    """

    def __init__(self, seed: int) -> None:
        """Create a reproducible random stream local to this actor."""

        self._rng = np.random.default_rng(seed)

    def act(self, observation: FloatArray, deterministic: bool) -> float:
        """Return zero for evaluation or a seeded uniform training action."""

        # This test actor is intentionally state-independent.
        del observation
        if deterministic:
            # Do not advance the RNG during deterministic evaluation.
            return 0.0
        return float(self._rng.uniform(-1.0, 1.0))

