"""Central configuration for the cart-pole experiments.

The classes in this module follow the main theoretical parts of the project:

1. ``PlantConfig`` defines the sampled-data nonlinear plant and mass mismatch.
2. ``LQRConfig`` defines the local quadratic regulator near the upright state.
3. ``PhysicsControllerConfig`` defines the hybrid swing-up/LQR baseline.
4. ``ResidualConfig`` defines the bounded learned correction.
5. ``ShieldConfig`` defines the nominal one-step Lyapunov condition.
6. The remaining classes keep observations, rollouts, and training synchronized.

These values specify numerical experiments. In particular, the shield parameters do
not by themselves certify safety or a region of attraction for the mismatched plant.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlantConfig:
    """Physical parameters and sampled-data simulation settings.

    The continuous state is ``s = [x, theta, x_dot, theta_dot]``, with
    ``theta = 0`` at the upright equilibrium. During one control interval, the
    input is held constant (zero-order hold) and the nonlinear dynamics are
    integrated with RK4. This produces the numerical one-step map

    ``s[k + 1] = F_dt(s[k], u[k]; pole_mass)``.

    ``pole_mass`` is the nominal controller-model mass ``m_p,0``. A rollout
    plant with mismatch ``m_p = mu * m_p,0`` is created by
    :meth:`with_mass_multiplier`. The controller should continue using the
    original nominal configuration so that model mismatch remains explicit.

    Units are SI: kilograms, metres, seconds, and newtons.
    """

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.81

    # The controller acts every 0.02 s. RK4 may take smaller internal steps.
    control_dt: float = 0.02
    rk4_substeps: int = 4

    # Actuator saturation and the cart-track boundary used in every comparison.
    u_max: float = 10.0
    x_limit: float = 2.4

    def with_mass_multiplier(self, mu: float) -> "PlantConfig":
        """Return a rollout-plant configuration with ``pole_mass = mu*m_p,0``.

        The dataclass is frozen, so this method creates a new object rather than
        changing the nominal model used by the controller.
        """
        return PlantConfig(
            cart_mass=self.cart_mass,
            pole_mass=mu * self.pole_mass,
            pole_length=self.pole_length,
            gravity=self.gravity,
            control_dt=self.control_dt,
            rk4_substeps=self.rk4_substeps,
            u_max=self.u_max,
            x_limit=self.x_limit,
        )


@dataclass(frozen=True)
class LQRConfig:
    """Weights and numerical settings for local discrete-time LQR.

    Near the upright equilibrium, the nominal sampled-data map is linearized as
    ``z[k + 1] = A z[k] + B u[k]``. LQR minimizes the quadratic objective

    ``sum_k (z[k].T Q z[k] + u[k].T R u[k])``.

    Here ``Q`` is diagonal in state order ``[x, theta, x_dot, theta_dot]`` and
    ``R = r`` because the cart-pole has one scalar input. The resulting Riccati
    matrix ``P`` is also used in the candidate Lyapunov function ``V(z)=z.T P z``.
    These are local, nominal-model constructions; mismatched-plant behavior must
    still be measured empirically.
    """

    q_diagonal: tuple[float, float, float, float] = (2.0, 40.0, 1.0, 4.0)
    r: float = 0.2

    # Perturbation used to approximate Jacobians of the nominal one-step map.
    finite_difference_epsilon: float = 1e-6


@dataclass(frozen=True)
class PhysicsControllerConfig:
    """Parameters for the hybrid physics baseline.

    This controller combines two theoretical ideas:

    * Energy shaping injects or removes mechanical energy during swing-up.
    * Discrete-time LQR locally balances the pole near the upright equilibrium.

    The controller enters LQR mode through the tighter ``enter_*`` bounds and
    leaves through the wider ``exit_*`` bounds. This hysteresis gives the hybrid
    controller memory and reduces rapid switching near a single threshold.
    """

    # Bounds are applied to |theta| and |theta_dot|. Angles are in radians.
    enter_theta: float = 0.20
    enter_theta_dot: float = 0.80
    exit_theta: float = 0.35
    exit_theta_dot: float = 1.50

    # Starter energy-shaping and cart-centering gains; tune them experimentally.
    energy_gain: float = 2.0
    cart_position_gain: float = 0.5
    cart_velocity_gain: float = 1.0

    # Breaks a zero-action symmetry when the pole begins in a motionless state.
    kick_force: float = 1.0


@dataclass(frozen=True)
class ResidualConfig:
    """Bound on the SAC residual added to the physics-controller action.

    For actor output ``a_RL in [-1, 1]``, the applied proposal is

    ``u_proposed = clip(u_physics + beta*a_RL, -u_max, u_max)``.

    Thus ``beta`` converts the normalized actor output to newtons and limits how
    much authority learning has relative to the physics baseline. Boundedness
    alone does not imply local stability.
    """

    beta: float = 3.0  # Project rule: beta <= 0.3 * u_max.


@dataclass(frozen=True)
class ShieldConfig:
    """Settings for the nominal-model Lyapunov action projection.

    Let ``z`` denote displacement from the nominal upright equilibrium and let
    ``V(z) = z.T P z`` use the LQR Riccati matrix. Inside the ellipsoid

    ``V(z) <= rho``,

    the shield seeks the scalar action nearest to the residual proposal that
    satisfies the nominal one-step decrease condition

    ``V(z_next) - V(z) <= -alpha*V(z)``.

    If no candidate is feasible, the shield falls back to nominal LQR. Outside
    the ellipsoid, it leaves the bounded residual proposal unchanged.

    The condition uses the nominal numerical model. Consequently, projection
    statistics and realized ``Delta V`` on a mass-mismatched plant are empirical
    evidence, not a formal safety or region-of-attraction guarantee.
    """

    # Size of the nominal ellipsoid in which the shield is active.
    rho: float = 1.0

    # Requested fractional decrease of V over one control interval.
    alpha: float = 0.05

    # Number of scalar control candidates used by the numerical projection.
    grid_size: int = 401

    # Small allowance for floating-point error in the inequality check.
    feasibility_tolerance: float = 1e-10


@dataclass(frozen=True)
class ObservationConfig:
    """Elementwise scales for the SAC actor's normalized full-state input.

    In state order ``[x, theta, x_dot, theta_dot]``, normalization uses
    ``observation[i] = state[i] / scales[i]``. The angle is first wrapped to
    ``[-pi, pi]``. The mass multiplier ``mu`` is deliberately not observed, so
    robustness must come from the policy learned under domain randomization.
    """

    scales: tuple[float, float, float, float] = (2.4, 3.141592653589793, 5.0, 10.0)


@dataclass(frozen=True)
class RolloutConfig:
    """Common finite time horizon for training and evaluation rollouts."""

    # At control_dt=0.02 s, a 10 s rollout contains 500 control intervals.
    horizon_seconds: float = 10.0


@dataclass(frozen=True)
class TrainingConfig:
    """Domain randomization, SAC settings, seeds, and artifact location.

    One fixed ``mu`` is sampled per rollout from ``[mu_min, mu_max]``; it must
    not drift within that rollout. The small default SAC run is an interface
    smoke test, not a converged research policy. Increase ``total_timesteps``
    only after the training data path has been verified.
    """

    mu_min: float = 0.8
    mu_max: float = 1.2
    seeds: tuple[int, ...] = (0, 1, 2)

    # Deliberately small defaults for the first end-to-end interface check.
    total_timesteps: int = 64 #1_000_000
    learning_starts: int = 16 #10_000
    buffer_size: int = 10_000 #500_000
    batch_size: int = 16 #256
    hidden_sizes: tuple[int, ...] = (64, 64)

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    train_frequency: int = 1
    gradient_steps: int = 1
    entropy_coefficient: str | float = "auto"
    target_entropy: str | float = "auto"
    device: str = "cpu"

    output_directory: str = "artifacts/training"


@dataclass(frozen=True)
class ExperimentConfig:
    """Single synchronized configuration object for a complete experiment.

    Passing this object through plant, controller, training, and evaluation code
    prevents silent differences in limits, time steps, and controller settings.
    ``default_factory`` gives each experiment its own nested configuration.
    """

    plant: PlantConfig = field(default_factory=PlantConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    physics: PhysicsControllerConfig = field(default_factory=PhysicsControllerConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        """Reject settings that violate basic numerical and project rules."""
        if self.residual.beta > 0.3 * self.plant.u_max:
            raise ValueError("Residual beta must be <= 0.3 * u_max.")
        if self.plant.rk4_substeps < 1:
            raise ValueError("rk4_substeps must be at least 1.")
        if self.shield.grid_size < 3:
            raise ValueError("shield.grid_size must be at least 3.")
        if not 0.0 < self.training.mu_min <= self.training.mu_max:
            raise ValueError("Training mass bounds must satisfy 0 < mu_min <= mu_max.")
        if self.training.total_timesteps < 1:
            raise ValueError("training.total_timesteps must be at least 1.")
        if self.training.learning_starts < 0:
            raise ValueError("training.learning_starts cannot be negative.")
        if self.training.buffer_size < 1 or self.training.batch_size < 1:
            raise ValueError("SAC buffer_size and batch_size must be at least 1.")
        if self.training.train_frequency < 1:
            raise ValueError("training.train_frequency must be at least 1.")
        if self.training.gradient_steps < 0:
            raise ValueError("training.gradient_steps cannot be negative.")
        if not self.training.hidden_sizes or any(
            size < 1 for size in self.training.hidden_sizes
        ):
            raise ValueError("training.hidden_sizes must contain positive sizes.")

