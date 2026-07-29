"""The three feedback controllers compared in the project.

Theoretical role
----------------
At each control time ``k``, a controller maps the measured state

    s[k] = [x, theta, x_dot, theta_dot]^T

to one scalar zero-order-held force ``u[k]``. The nonlinear plant then produces
``s[k + 1]`` over one control period. This module implements exactly the three
control laws in the study:

1. :class:`PhysicsController` is a hybrid model-based baseline. It uses energy
   shaping away from upright and nominal discrete-time LQR near upright.
2. :class:`ResidualController` adds the bounded learned correction
   ``beta * a_RL`` to that physics action.
3. :class:`ShieldedResidualController` filters the residual proposal inside a
   nominal Lyapunov ellipsoid before the force is applied.

All three return the same :class:`ControlDecision` object so rollouts and
metrics stay synchronized. The controller and shield always use
``config.plant``, whose pole mass is the assumed nominal mass ``m_p,0``. The
mass-mismatched rollout plant ``m_p = mu*m_p,0`` is created elsewhere and is
not passed into these control laws.

The shield checks a nominal-model, one-step decrease condition. It does not
establish track safety, global stability, or a certified region of attraction
for the nonlinear mass-mismatched plant. Realized Lyapunov change and empirical
basins must therefore be measured on the actual rollout plant.
"""

import math
from enum import Enum
from typing import Protocol

import numpy as np

from .actor import Actor
from .config import ExperimentConfig
from .control_math import LQRData, build_lqr, nominal_lqr_force
from .data import ControlDecision, State
from .plant import normalize_state, wrap_angle
from .shield import project_with_lyapunov_shield


class Controller(Protocol):
    """Common structural interface used by simulation and evaluation code.

    A concrete controller need not inherit from this protocol; it only needs
    matching ``reset`` and ``act`` methods. Calling ``reset`` before every
    rollout is essential because the physics controller stores its current
    hysteresis mode. During frozen-policy evaluation, ``deterministic=True``
    fixes the actor's action convention.
    """

    def reset(self) -> None:
        """Restore controller memory to its start-of-rollout state."""
        ...

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        """Return the control decision for the current sampled state."""
        ...


class ControllerKind(str, Enum):
    """Stable names for the study's three—and only three—comparators."""

    PHYSICS = "physics"
    RESIDUAL_SAC = "residual_sac"
    SHIELDED_RESIDUAL_SAC = "shielded_residual_sac"


class PhysicsController:
    """Hybrid energy-shaping/LQR baseline with hysteresis.

    This class matches the model-based part of the project theory:

    * **Global maneuver heuristic:** energy shaping drives the pole's nominal
      mechanical energy toward the energy of the upright configuration while
      proportional-derivative terms keep the cart near the track center.
    * **Local regulator:** near ``s = 0``, the controller applies the nominal
      discrete-time LQR law ``u = -K s``, followed by actuator clipping.
    * **Hybrid switching:** separate entry and exit thresholds provide
      hysteresis. Thus the controller has one discrete memory variable,
      ``_mode``, in addition to the continuous plant state.

    The energy calculation and LQR matrices both use the controller's nominal
    pole mass. Their behavior under ``mu != 1`` is evaluated empirically rather
    than assumed from nominal theory.
    """

    def __init__(self, config: ExperimentConfig, lqr: LQRData) -> None:
        # Store one synchronized experiment configuration and the corresponding
        # nominal matrices A, B, K, and P.
        self._config = config
        self._lqr = lqr
        self._mode = "swing_up"

    def reset(self) -> None:
        """Start a new rollout in swing-up mode.

        Resetting avoids carrying the final LQR/swing-up mode from one paired
        rollout into the next rollout.
        """

        self._mode = "swing_up"

    def _update_mode(self, state: State) -> None:
        """Update the discrete controller mode using hysteretic thresholds.

        LQR is entered only when both ``|theta|`` and ``|theta_dot|`` satisfy
        the tighter entry bounds. It is left when either variable reaches its
        wider exit bound. The angle is wrapped only for this switching test.
        """

        theta = abs(wrap_angle(state.theta))
        theta_dot = abs(state.theta_dot)
        switch = self._config.physics

        if self._mode == "swing_up":
            if theta <= switch.enter_theta and theta_dot <= switch.enter_theta_dot:
                self._mode = "lqr"
        else:
            if theta >= switch.exit_theta or theta_dot >= switch.exit_theta_dot:
                self._mode = "swing_up"

    def _energy_shaping_force(self, state: State) -> float:
        """Compute the nominal swing-up and cart-centering force.

        With ``theta = 0`` upright, the nominal pole energy used here is

        ``E = 0.5*m_p*l^2*theta_dot^2 + m_p*g*l*cos(theta)``.

        The target ``E_des = m_p*g*l`` is the upright potential energy, so the
        shaping error is ``E - E_des``. The implemented starter law is

        ``u = k_E*(E-E_des)*theta_dot*cos(theta) - k_x*x - k_v*x_dot``.

        This is a practical swing-up law to tune and test, not a global
        Lyapunov proof. The returned force respects the common actuator bound.
        """

        plant = self._config.plant
        gains = self._config.physics
        theta = wrap_angle(state.theta)

        # Nominal mechanical energy of the pole point mass. At motionless
        # upright E = +m_p*g*l; at motionless downward E = -m_p*g*l.
        energy = (
            0.5 * plant.pole_mass * plant.pole_length**2 * state.theta_dot**2
            + plant.pole_mass * plant.gravity * plant.pole_length * math.cos(theta)
        )
        desired_energy = plant.pole_mass * plant.gravity * plant.pole_length
        energy_error = energy - desired_energy

        # The first term changes pole energy. The final two terms center and
        # damp the cart, which helps avoid empirical track violations.
        force = (
            gains.energy_gain * energy_error * state.theta_dot * math.cos(theta)
            - gains.cart_position_gain * state.x
            - gains.cart_velocity_gain * state.x_dot
        )

        # Exact downward rest is a symmetry; give the starter a deterministic kick.
        near_downward_rest = (
            abs(abs(theta) - math.pi) < 0.10
            and abs(state.theta_dot) < 0.05
        )
        if near_downward_rest:
            force += gains.kick_force

        return float(np.clip(force, -plant.u_max, plant.u_max))

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        """Select swing-up or LQR and return the baseline decision.

        ``deterministic`` is accepted for a uniform controller interface but is
        unused because this baseline has no stochastic actor. In LQR mode, the
        stored state is used directly as the local displacement from the origin.
        Experiments should therefore represent the nearby upright angle near
        ``theta = 0`` rather than an equivalent value such as ``2*pi``.
        """

        del deterministic
        self._update_mode(state)

        if self._mode == "lqr":
            force = nominal_lqr_force(state, self._lqr, self._config.plant)
        else:
            force = self._energy_shaping_force(state)

        return ControlDecision(
            force=force,
            physics_force=force,
            residual_force=0.0,
            controller_name=ControllerKind.PHYSICS.value,
            diagnostics={"physics_mode": self._mode},
        )


class ResidualController:
    """Add a bounded actor correction to the physics baseline.

    This class matches the residual-RL decomposition

    ``u_prop = clip(u_physics + beta*a_RL, -u_max, u_max)``,

    where ``a_RL`` is clipped to ``[-1, 1]`` and configuration validation
    enforces ``beta <= 0.3*u_max``. The actor receives the normalized full state
    but not the mass multiplier ``mu``. Consequently, compensation for mass
    mismatch must be learned indirectly from state trajectories generated by
    domain randomization.

    This class composes an :class:`Actor`; it does not implement SAC training.
    A trained SAC policy can be connected through the small ``Actor`` protocol.
    Bounded residual authority is an experimental design constraint, not a
    sufficient condition for preserving local stability.
    """

    def __init__(
        self,
        physics: PhysicsController,
        actor: Actor,
        config: ExperimentConfig,
    ) -> None:
        self._physics = physics
        self._actor = actor
        self._config = config

    def reset(self) -> None:
        """Reset the stateful physics component for a new rollout."""

        self._physics.reset()

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        """Compute the physics action, actor residual, and clipped proposal.

        The returned fields preserve the action decomposition:

        * ``physics_force`` is the baseline command ``u_physics``.
        * ``residual_force`` is ``beta*a_RL`` before total-force clipping.
        * ``force`` is the final clipped proposal sent to an unshielded plant,
          or passed onward as the proposal to the shield.

        Thus, at saturation, ``force - physics_force`` can be smaller than the
        recorded residual proposal.
        """

        physics = self._physics.act(state, deterministic)

        # Keep one observation convention for training and frozen evaluation:
        # normalized [x, theta, x_dot, theta_dot], with no mass multiplier mu.
        observation = normalize_state(state, self._config.observation)
        actor_action = float(
            np.clip(self._actor.act(observation, deterministic), -1.0, 1.0)
        )

        # beta converts the dimensionless actor output into a bounded force in N.
        residual_force = self._config.residual.beta * actor_action
        total_force = float(
            np.clip(
                physics.force + residual_force,
                -self._config.plant.u_max,
                self._config.plant.u_max,
            )
        )

        diagnostics = dict(physics.diagnostics)
        diagnostics["actor_action"] = actor_action
        return ControlDecision(
            force=total_force,
            physics_force=physics.force,
            residual_force=residual_force,
            controller_name=ControllerKind.RESIDUAL_SAC.value,
            diagnostics=diagnostics,
        )


class ShieldedResidualController:
    """Filter the bounded residual proposal with the nominal Lyapunov shield.

    Let ``P`` be the nominal LQR Riccati matrix and ``V(s) = s.T P s``. When
    ``V(s) <= rho``, the shield searches for the force nearest to ``u_prop``
    that satisfies, on the nominal nonlinear one-step map ``F_h,0``,

    ``V(F_h,0(s, u)) - V(s) <= -alpha*V(s)``.

    The scalar projection is approximated on the force grid implemented in
    :func:`project_with_lyapunov_shield`, with the exact proposed force also
    included as a candidate. If no candidate is feasible, nominal LQR is used
    as a fallback. Outside ``V <= rho``, the bounded residual proposal is left
    unchanged.

    Important theoretical distinction: ``P``, ``rho``, feasibility, and the
    reported ``nominal_delta_v`` all refer to the nominal controller model. The
    check does not include the track constraint and is not a CBF or global
    safety claim. Lyapunov change on a plant with ``mu != 1`` remains an
    empirical measurement.
    """

    def __init__(
        self,
        residual: ResidualController,
        config: ExperimentConfig,
        lqr: LQRData,
    ) -> None:
        self._residual = residual
        self._config = config
        self._lqr = lqr

    def reset(self) -> None:
        """Reset the wrapped residual and physics-controller state."""

        self._residual.reset()

    def act(self, state: State, deterministic: bool = True) -> ControlDecision:
        """Return the shielded force and nominal-model diagnostics.

        ``physics_force`` and ``residual_force`` retain the unshielded proposal's
        decomposition for analysis. If projection occurs, the applied ``force``
        need not equal their sum. ``shield_infeasible=True`` means no tested grid
        candidate met the nominal inequality; it does not say the fallback LQR
        itself met that inequality.
        """

        # First construct exactly the same proposal as the unshielded residual
        # controller. This keeps the comparison paired before shield projection.
        proposed = self._residual.act(state, deterministic)
        shield = project_with_lyapunov_shield(
            state=state,
            proposed_force=proposed.force,
            nominal_plant=self._config.plant,
            lqr=self._lqr,
            config=self._config.shield,
        )

        # Copy rather than mutate the wrapped controller's dictionary. These
        # fields support activation, projection, and infeasibility statistics.
        diagnostics = dict(proposed.diagnostics)
        diagnostics.update(
            {
                "shield_active": shield.active,
                "shield_projected": shield.projected,
                "shield_infeasible": shield.infeasible,
                "v_before": shield.v_before,
                "nominal_delta_v": shield.nominal_delta_v,
            }
        )
        return ControlDecision(
            force=shield.force,
            physics_force=proposed.physics_force,
            residual_force=proposed.residual_force,
            controller_name=ControllerKind.SHIELDED_RESIDUAL_SAC.value,
            diagnostics=diagnostics,
        )


def build_controller(
    kind: ControllerKind,
    config: ExperimentConfig,
    actor: Actor,
) -> Controller:
    """Construct one of the three controllers from synchronized objects.

    The factory always builds LQR matrices from the nominal ``config.plant``.
    Within a shielded controller, the physics baseline and shield receive the
    same immutable :class:`LQRData`, avoiding inconsistent ``K`` and ``P``
    matrices. The same actor interface and configuration layout are used for all
    controller kinds, which simplifies paired rollouts.

    Args:
        kind: One of the three declared experimental comparators.
        config: Shared nominal plant, controller, observation, and shield data.
        actor: SAC-compatible actor. It is accepted but unused for the physics
            comparator; a ``ZeroActor`` can be supplied in that case.

    Returns:
        A controller implementing the common :class:`Controller` protocol.

    Raises:
        ValueError: If ``kind`` is not a recognized comparator.
    """

    # Build nominal control objects once, then pass them into the composed
    # layers rather than recomputing potentially different matrices.
    lqr = build_lqr(config.plant, config.lqr)
    physics = PhysicsController(config, lqr)

    if kind == ControllerKind.PHYSICS:
        return physics

    residual = ResidualController(physics, actor, config)
    if kind == ControllerKind.RESIDUAL_SAC:
        return residual
    if kind == ControllerKind.SHIELDED_RESIDUAL_SAC:
        return ShieldedResidualController(residual, config, lqr)

    raise ValueError(f"Unknown controller kind: {kind}")

