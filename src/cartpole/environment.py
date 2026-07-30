"""Training environment for bounded residual control of the cart-pole."""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .config import ExperimentConfig
from .control_math import LQRData, build_lqr
from .controllers import ControllerKind, PhysicsController
from .data import ControlDecision, State, Transition
from .plant import (
    clip_force,
    has_track_violation,
    normalize_state,
    step_rk4,
)
from .shield import project_with_lyapunov_shield
from .simulation import RewardFunction, starter_reward
from .training import sample_rollout_mass

InitialStateSampler = Callable[[np.random.Generator], State]


def downward_initial_state(rng: np.random.Generator) -> State:
    """Return the default motionless, centered, downward initial state."""

    del rng
    return State(x=0.0, theta=math.pi, x_dot=0.0, theta_dot=0.0)


class ResidualCartPoleEnvironment:
    """Apply SAC actions as bounded residuals around the physics controller.

    The mass multiplier is sampled once in :meth:`reset` and remains fixed until
    the episode ends. The actor sees only the normalized state in the order
    ``[x, theta, x_dot, theta_dot]``; ``mu`` is available only in ``info`` for
    reproducibility and logging.

    This class intentionally has no actor object. During training, the SAC
    implementation calls ``step(actor_action)`` directly.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        shielded: bool = False,
        initial_state_sampler: InitialStateSampler = downward_initial_state,
        reward_function: RewardFunction = starter_reward,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.shielded = shielded
        self._initial_state_sampler = initial_state_sampler
        self._reward_function = reward_function

        # Both the physics controller and optional shield use this same nominal
        # model. Only the rollout plant receives the sampled mass multiplier.
        self._lqr: LQRData = build_lqr(self.config.plant, self.config.lqr)
        self._physics = PhysicsController(self.config, self._lqr)

        self._rng: np.random.Generator | None = None
        self._mu: float | None = None
        self._actual_plant = None
        self._state: State | None = None
        self._step_count = 0
        self._episode_done = False
        self._max_steps = round(
            self.config.rollout.horizon_seconds / self.config.plant.control_dt
        )
        if self._max_steps < 1:
            raise ValueError("The rollout horizon must contain at least one step.")

    @property
    def mu(self) -> float:
        """Return the mass multiplier for the current episode."""

        if self._mu is None:
            raise RuntimeError("Call reset() before reading mu.")
        return self._mu

    @property
    def state(self) -> State:
        """Return the current physical state."""

        if self._state is None:
            raise RuntimeError("Call reset() before reading the state.")
        return self._state

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        """Start an episode with one fixed randomized plant mass.

        Passing the same seed reproduces both the sampled mass and any randomness
        used by ``initial_state_sampler``. Calling ``reset()`` without a new seed
        continues the environment's existing random-number stream.
        """

        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)

        self._mu = sample_rollout_mass(self._rng, self.config)
        self._actual_plant = self.config.plant.with_mass_multiplier(self._mu)
        self._state = self._initial_state_sampler(self._rng)
        if not isinstance(self._state, State):
            raise TypeError("initial_state_sampler must return a State object.")
        if not np.all(np.isfinite(self._state.as_array())):
            raise ValueError("The initial state must contain only finite values.")

        self._physics.reset()
        self._step_count = 0
        self._episode_done = False

        observation = normalize_state(self._state, self.config.observation)
        info: dict[str, object] = {
            "mu": self._mu,
            "pole_mass": self._actual_plant.pole_mass,
            "state": self._state,
        }
        return observation, info

    def step(
        self,
        actor_action: ArrayLike,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Advance the fixed-mass plant by one control period."""

        if self._state is None or self._actual_plant is None or self._mu is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_done:
            raise RuntimeError("The episode is done; call reset() before step().")

        action = self._clip_actor_action(actor_action)
        current_state = self._state

        physics = self._physics.act(current_state)
        residual_force = self.config.residual.beta * action
        proposed_force = clip_force(
            physics.force + residual_force,
            self.config.plant,
        )

        diagnostics = dict(physics.diagnostics)
        diagnostics["actor_action"] = action

        if self.shielded:
            shield = project_with_lyapunov_shield(
                state=current_state,
                proposed_force=proposed_force,
                nominal_plant=self.config.plant,
                lqr=self._lqr,
                config=self.config.shield,
            )
            applied_force = shield.force
            diagnostics.update(
                {
                    "shield_active": shield.active,
                    "shield_projected": shield.projected,
                    "shield_infeasible": shield.infeasible,
                    "v_before": shield.v_before,
                    "nominal_delta_v": shield.nominal_delta_v,
                }
            )
            controller_name = ControllerKind.SHIELDED_RESIDUAL_SAC.value
        else:
            applied_force = proposed_force
            controller_name = ControllerKind.RESIDUAL_SAC.value

        decision = ControlDecision(
            force=applied_force,
            physics_force=physics.force,
            residual_force=residual_force,
            controller_name=controller_name,
            diagnostics=diagnostics,
        )

        next_state = step_rk4(
            current_state,
            decision.force,
            self._actual_plant,
        )
        reward = self._reward_function(
            current_state,
            decision.force,
            next_state,
        )
        terminated = has_track_violation(next_state, self._actual_plant)

        self._step_count += 1
        truncated = self._step_count >= self._max_steps and not terminated
        self._episode_done = terminated or truncated
        self._state = next_state

        transition = Transition(
            state=current_state,
            decision=decision,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
        )
        observation = normalize_state(next_state, self.config.observation)
        info = {
            "mu": self._mu,
            "pole_mass": self._actual_plant.pole_mass,
            "state": next_state,
            "actor_action": action,
            "physics_force": physics.force,
            "residual_force": residual_force,
            "proposed_force": proposed_force,
            "applied_force": applied_force,
            "track_violation": terminated,
            "step_count": self._step_count,
            "decision": decision,
            "transition": transition,
            **diagnostics,
        }
        return observation, reward, terminated, truncated, info

    @staticmethod
    def _clip_actor_action(actor_action: ArrayLike) -> float:
        """Convert a scalar or one-element SAC action to a bounded float."""

        values = np.asarray(actor_action, dtype=np.float64)
        if values.shape == ():
            action = float(values)
        elif values.shape == (1,):
            action = float(values[0])
        else:
            raise ValueError("actor_action must be a scalar or have shape (1,).")

        if not math.isfinite(action):
            raise ValueError("actor_action must be finite.")
        return float(np.clip(action, -1.0, 1.0))
