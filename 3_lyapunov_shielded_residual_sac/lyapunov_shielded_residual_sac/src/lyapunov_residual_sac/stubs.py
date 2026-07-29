"""Runnable dummy dependencies.

Replace these objects with the real physics/LQR and frozen SAC implementations.
The numbers below are intentionally not claimed to be a cart-pole model.
"""

import numpy as np
from numpy.typing import NDArray

from .config import ControllerConfig, LQRModel, ShieldConfig
from .controller import ShieldedResidualController
from .shield import LyapunovShield
from .stats import ShieldStats


FloatArray = NDArray[np.float64]


def dummy_lqr_model() -> LQRModel:
    """Return fake, correctly shaped matrices for wiring and tests only."""

    return LQRModel(
        A=np.array(
            [
                [0.98, 0.00, 0.00, 0.00],
                [0.00, 0.98, 0.00, 0.00],
                [0.00, 0.00, 0.98, 0.00],
                [0.00, 0.00, 0.00, 0.98],
            ],
            dtype=float,
        ),
        B=np.array([0.01, -0.03, 0.05, -0.08], dtype=float),
        P=np.diag([1.0, 12.0, 0.5, 1.5]),
        equilibrium_state=np.zeros(4, dtype=float),
    )


class DummyPhysicsController:
    """Fake local feedback with the two required physics-controller methods."""

    def __init__(self, u_max: float = 10.0) -> None:
        self.u_max = u_max
        self.k = np.array([0.8, -7.0, 1.2, -1.5], dtype=float)

    def lqr_action(self, state: FloatArray) -> float:
        """TODO: replace with the actual nominal discrete-time LQR action."""

        return float(np.clip(-self.k @ state, -self.u_max, self.u_max))

    def action(self, state: FloatArray) -> float:
        """TODO: replace with energy shaping plus hysteretic LQR switching."""

        return self.lqr_action(state)


class DummyResidualPolicy:
    """Fake deterministic residual actor; replace with the frozen SAC actor."""

    def action(
        self,
        normalized_state: FloatArray,
        deterministic: bool = True,
    ) -> float:
        del deterministic
        return float(np.tanh(0.25 * normalized_state[1]))


class DummyNormalizer:
    """Fake fixed state scaling; replace with SAC's training normalization."""

    def __init__(self) -> None:
        self.scales = np.array([2.4, np.pi, 5.0, 8.0], dtype=float)

    def __call__(self, state: FloatArray) -> FloatArray:
        return state / self.scales


def build_dummy_controller() -> ShieldedResidualController:
    """Wire the real shield to dummy project dependencies."""

    u_max = 10.0
    shield = LyapunovShield(
        model=dummy_lqr_model(),
        config=ShieldConfig(
            rho=1.0,
            alpha=0.01,
            u_max=u_max,
        ),
    )
    return ShieldedResidualController(
        physics=DummyPhysicsController(u_max=u_max),
        residual_policy=DummyResidualPolicy(),
        normalizer=DummyNormalizer(),
        shield=shield,
        config=ControllerConfig(beta=3.0, u_max=u_max),
    )


def main() -> None:
    """Print a few dummy calls to demonstrate the integration surface."""

    controller = build_dummy_controller()
    stats = ShieldStats()

    example_states = [
        np.array([0.0, 0.03, 0.0, 0.0]),
        np.array([0.0, 0.10, 0.0, 0.0]),
        np.array([0.0, 1.50, 0.0, 0.0]),
    ]

    for state in example_states:
        output = controller.act(state, deterministic=True)
        stats.update(output)
        print(
            {
                "state": state.tolist(),
                "action": round(output.action, 5),
                "V": round(output.shield.value, 5),
                "inside": output.shield.inside_region,
                "projected": output.shield.projected,
                "feasible": output.shield.feasible,
            }
        )

    print(stats.as_dict())


if __name__ == "__main__":
    main()
