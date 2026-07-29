from dataclasses import dataclass


@dataclass(frozen=True)
class CartPoleParams:
    """Physical parameters and numerical settings.

    The pole angle is zero when upright. ``pole_com_length`` is the distance
    from the pivot to the pole center of mass.
    """

    cart_mass: float = 1.0
    pole_mass: float = 0.2
    pole_com_length: float = 0.5
    pole_inertia: float = 0.0
    gravity: float = 9.81
    cart_damping: float = 0.05
    pole_damping: float = 0.0
    control_period: float = 0.02
    integration_step: float = 0.002
    max_force: float = 10.0
    track_limit: float = 2.4

    def __post_init__(self):
        positive = (
            self.cart_mass,
            self.pole_mass,
            self.pole_com_length,
            self.gravity,
            self.control_period,
            self.integration_step,
            self.max_force,
            self.track_limit,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Masses, lengths, time steps, and limits must be positive")
        if self.pole_inertia < 0.0:
            raise ValueError("pole_inertia cannot be negative")
        if self.cart_damping < 0.0 or self.pole_damping < 0.0:
            raise ValueError("Damping cannot be negative")


@dataclass(frozen=True)
class LQRConfig:
    """Weights for sum(z.T Q z + u.T R u)."""

    q_x: float = 2.0
    q_theta: float = 35.0
    q_x_dot: float = 1.5
    q_theta_dot: float = 3.0
    r_force: float = 0.25


@dataclass(frozen=True)
class SwingUpConfig:
    """Gains for the energy-shaping swing-up law."""

    energy_gain: float = 1.2
    cart_position_gain: float = 0.35
    cart_velocity_gain: float = 0.65
    max_cart_acceleration: float = 8.0
    kick_acceleration: float = 1.0
    kick_angle: float = 0.18
    kick_speed: float = 0.08


@dataclass(frozen=True)
class SwitchConfig:
    """Inner thresholds enter LQR; outer thresholds leave it."""

    enter_angle: float = 0.35
    enter_angle_speed: float = 1.2
    enter_position: float = 0.65
    enter_cart_speed: float = 1.5

    exit_angle: float = 0.50
    exit_angle_speed: float = 2.2
    exit_position: float = 1.0
    exit_cart_speed: float = 2.2

    def __post_init__(self):
        inner = (
            self.enter_angle,
            self.enter_angle_speed,
            self.enter_position,
            self.enter_cart_speed,
        )
        outer = (
            self.exit_angle,
            self.exit_angle_speed,
            self.exit_position,
            self.exit_cart_speed,
        )
        if any(value <= 0.0 for value in inner + outer):
            raise ValueError("Switch thresholds must be positive")
        if any(enter >= leave for enter, leave in zip(inner, outer)):
            raise ValueError("Every enter threshold must be below its exit threshold")
