"""Fast checks that do not train a policy."""

import numpy as np

from cartpole_study.config import ControllerConfig, PlantConfig, ShieldConfig
from cartpole_study.controllers import NominalController
from cartpole_study.plant import CartPolePlant
from cartpole_study.shield import LyapunovShield


def test_upright_is_an_equilibrium() -> None:
    plant = CartPolePlant(PlantConfig())
    np.testing.assert_allclose(plant.derivative(np.zeros(4), 0.0), np.zeros(4))


def test_nominal_lqr_has_stable_continuous_linearization() -> None:
    plant = CartPolePlant(PlantConfig())
    controller = NominalController(plant, ControllerConfig())
    a, b = plant.upright_linear_model()
    eigenvalues = np.linalg.eigvals(a - b @ controller.gain)
    assert np.max(np.real(eigenvalues)) < 0.0


def test_feasible_shield_action_satisfies_derivative_bound() -> None:
    plant = CartPolePlant(PlantConfig())
    controller_config = ControllerConfig()
    nominal = NominalController(plant, controller_config)
    shield_config = ShieldConfig()
    shield = LyapunovShield(
        nominal.lyapunov_matrix, controller_config, shield_config
    )
    state = np.array([0.02, 0.0, 0.04, 0.0])
    nominal_force = nominal.action(state)
    result = shield.project(
        plant,
        state,
        nominal_force=nominal_force,
        candidate_force=nominal_force + controller_config.residual_force_limit,
    )
    assert not result.infeasible
    derivative = nominal.lyapunov_derivative(plant, state, result.force)
    assert derivative <= -shield_config.alpha * float(state @ state) + 1e-7
