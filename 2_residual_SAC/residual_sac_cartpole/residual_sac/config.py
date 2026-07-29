from dataclasses import dataclass
from math import pi


@dataclass
class EnvironmentConfig:
    control_period: float = 0.02
    u_max: float = 10.0
    beta: float = 3.0
    max_steps: int = 500
    train_mu_min: float = 0.8
    train_mu_max: float = 1.2
    state_scale: tuple[float, float, float, float] = (2.4, pi, 5.0, 10.0)
    observation_clip: float = 5.0
    terminate_on_track_violation: bool = True

    def __post_init__(self) -> None:
        if self.control_period <= 0.0:
            raise ValueError("control_period must be positive")
        if self.u_max <= 0.0:
            raise ValueError("u_max must be positive")
        if not 0.0 <= self.beta <= 0.3 * self.u_max:
            raise ValueError("beta must satisfy 0 <= beta <= 0.3 * u_max")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.train_mu_min <= 0.0 or self.train_mu_max < self.train_mu_min:
            raise ValueError("invalid training mass range")
        if len(self.state_scale) != 4 or any(value <= 0.0 for value in self.state_scale):
            raise ValueError("state_scale must contain four positive values")


@dataclass
class SACConfig:
    hidden_sizes: tuple[int, int] = (128, 128)
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    initial_alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float = -1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if len(self.hidden_sizes) != 2 or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive integers")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")


@dataclass
class TrainingConfig:
    episodes: int = 50
    replay_capacity: int = 200_000
    batch_size: int = 256
    random_steps: int = 1_000
    update_after: int = 1_000
    updates_per_step: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.random_steps < 0 or self.update_after < 0:
            raise ValueError("step counts cannot be negative")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be positive")

