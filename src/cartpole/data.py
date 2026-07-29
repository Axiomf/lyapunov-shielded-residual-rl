"""Shared data objects for simulation, control, and evaluation.

This module fixes one common representation for the main mathematical objects in
this project.  Keeping these definitions in one place prevents different parts
of the code from silently using a different state order or rollout format.

The objects correspond to the sampled-data closed-loop system

    s[k] --controller--> u[k] --ZOH + RK4 plant--> s[k + 1].

Here ``s[k]`` is a :class:`State`, ``u[k]`` and its decomposition are stored in
a :class:`ControlDecision`, and one complete sampled step is a
:class:`Transition`.  A :class:`Rollout` collects transitions generated with a
fixed pole-mass multiplier ``mu``.

These classes only store data.  Plant equations, controller logic, Lyapunov
conditions, and success criteria belong in their own modules.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# Type alias used whenever a NumPy array contains double-precision real values.
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class State:
    """Cart-pole state in the project's canonical coordinate order.

    Mathematically, this represents

        s = [x, theta, x_dot, theta_dot]^T,

    where ``theta = 0`` is the upright pole position.  Positions are measured
    in metres and radians; velocities are measured in metres per second and
    radians per second.

    ``frozen=True`` makes a state immutable after creation.  This is useful for
    dynamical-systems experiments because a stored state cannot be changed by a
    later controller or logging operation.
    """

    x: float
    theta: float
    x_dot: float
    theta_dot: float

    def as_array(self) -> FloatArray:
        """Return ``[x, theta, x_dot, theta_dot]`` as a float64 array.

        Array form is convenient for matrix calculations such as ``z.T @ P @
        z``, finite-difference Jacobians, LQR feedback, and actor observations.
        """
        return np.array(
            [self.x, self.theta, self.x_dot, self.theta_dot],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: FloatArray) -> "State":
        """Create a state from an array with exactly four entries.

        Requiring shape ``(4,)`` catches accidental column vectors ``(4, 1)``,
        row vectors ``(1, 4)``, and arrays with the wrong state dimension.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (4,):
            raise ValueError("State array must have shape (4,).")
        return cls(
            x=float(values[0]),
            theta=float(values[1]),
            x_dot=float(values[2]),
            theta_dot=float(values[3]),
        )


@dataclass(frozen=True)
class ControlDecision:
    """Applied control and its decomposition for one control period.

    For a residual controller, the proposed command has the form

        u = clip(u_physics + u_residual, -u_max, u_max).

    ``force`` is the final force sent to the plant after clipping or shield
    projection.  ``physics_force`` is the model-based controller contribution,
    and ``residual_force`` records the bounded RL contribution.  Keeping all
    three values supports paired comparisons of control effort and how much
    authority the learned residual uses.

    ``diagnostics`` stores controller-specific scalar or Boolean information,
    such as shield activation, projection, infeasibility, ``V``, or ``delta_V``.
    It is intentionally flexible because the three controllers do not produce
    exactly the same diagnostics.
    """

    force: float
    physics_force: float
    residual_force: float
    controller_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One step of the sampled-data closed-loop trajectory.

    This object represents the map

        (s[k], u[k]) -> s[k + 1],

    over one 0.02 s control period.  The control is held constant by zero-order
    hold while the nonlinear plant is integrated.  The associated reward and
    termination flag are stored with the same step so controller comparisons
    remain time-aligned.
    """

    state: State
    decision: ControlDecision
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class Rollout:
    """Complete trajectory for one fixed plant-mass multiplier.

    The plant pole mass is ``m_p = mu * m_p0``.  ``mu`` stays fixed throughout
    a rollout, matching both domain-randomized training and frozen-policy
    evaluation.  ``initial_state`` is stored explicitly to make paired
    controller comparisons reproducible even when no transition is generated.

    The transition sequence supports trajectory-level measurements such as
    success, settling time, realized Lyapunov change, control effort, and shield
    statistics.  ``track_violation`` separately records whether ``|x|`` crossed
    the allowed track limit at any point.

    A rollout is empirical trajectory data.  By itself it does not certify a
    region of attraction, global stability, or safety.
    """

    mu: float
    initial_state: State
    transitions: tuple[Transition, ...]
    track_violation: bool

    @property
    def total_reward(self) -> float:
        """Return the undiscounted sum of rewards over stored transitions."""
        return sum(item.reward for item in self.transitions)

