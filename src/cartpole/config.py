from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlantConfig:
    """Physical and sampled-data settings.

    `pole_mass` is nominal here. An actual rollout plant is created with
    `with_mass_multiplier(mu)`.
    """

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.81
    control_dt: float = 0.02
    rk4_substeps: int = 4
    u_max: float = 10.0
    x_limit: float = 2.4

    def with_mass_multiplier(self, mu: float) -> "PlantConfig":
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
    # State order: x, theta, x_dot, theta_dot.
    q_diagonal: tuple[float, float, float, float] = (2.0, 40.0, 1.0, 4.0)
    r: float = 0.2
    finite_difference_epsilon: float = 1e-6


@dataclass(frozen=True)
class PhysicsControllerConfig:
    # Entering conditions are tighter than exiting conditions: hysteresis.
    enter_theta: float = 0.20
    enter_theta_dot: float = 0.80
    exit_theta: float = 0.35
    exit_theta_dot: float = 1.50

    # Starter energy-shaping/centering gains. These require project tuning.
    energy_gain: float = 2.0
    cart_position_gain: float = 0.5
    cart_velocity_gain: float = 1.0
    kick_force: float = 1.0


@dataclass(frozen=True)
class ResidualConfig:
    beta: float = 3.0  # Must remain <= 0.3 * u_max.


@dataclass(frozen=True)
class ShieldConfig:
    rho: float = 1.0
    alpha: float = 0.05
    grid_size: int = 401
    feasibility_tolerance: float = 1e-10


@dataclass(frozen=True)
class ObservationConfig:
    # theta is wrapped to [-pi, pi] before normalization.
    scales: tuple[float, float, float, float] = (2.4, 3.141592653589793, 5.0, 10.0)


@dataclass(frozen=True)
class RolloutConfig:
    horizon_seconds: float = 10.0


@dataclass(frozen=True)
class TrainingConfig:
    mu_min: float = 0.8
    mu_max: float = 1.2
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class ExperimentConfig:
    plant: PlantConfig = field(default_factory=PlantConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    physics: PhysicsControllerConfig = field(default_factory=PhysicsControllerConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.residual.beta > 0.3 * self.plant.u_max:
            raise ValueError("Residual beta must be <= 0.3 * u_max.")
        if self.plant.rk4_substeps < 1:
            raise ValueError("rk4_substeps must be at least 1.")
        if self.shield.grid_size < 3:
            raise ValueError("shield.grid_size must be at least 3.")

