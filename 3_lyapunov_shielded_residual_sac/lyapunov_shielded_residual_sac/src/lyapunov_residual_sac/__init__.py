"""Public API for the Lyapunov-shielded residual controller."""

from .config import ControllerConfig, LQRModel, ShieldConfig
from .controller import ControllerOutput, ShieldedResidualController
from .shield import LyapunovShield, ShieldResult, realized_lyapunov_change
from .stats import ShieldStats

__all__ = [
    "ControllerConfig",
    "ControllerOutput",
    "LQRModel",
    "LyapunovShield",
    "ShieldConfig",
    "ShieldResult",
    "ShieldStats",
    "ShieldedResidualController",
    "realized_lyapunov_change",
]
