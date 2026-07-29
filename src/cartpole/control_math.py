from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from .config import LQRConfig, PlantConfig
from .data import FloatArray, State
from .plant import step_rk4


@dataclass(frozen=True)
class LQRData:
    a: FloatArray
    b: FloatArray
    k: FloatArray
    p: FloatArray


def one_step_array(
    state_array: FloatArray,
    force: float,
    nominal_plant: PlantConfig,
) -> FloatArray:
    state = State.from_array(np.asarray(state_array, dtype=np.float64))
    return step_rk4(state, force, nominal_plant).as_array()


def finite_difference_discrete_model(
    nominal_plant: PlantConfig,
    epsilon: float,
) -> tuple[FloatArray, FloatArray]:
    """Linearize the nominal one-control-period map at upright."""

    equilibrium = np.zeros(4, dtype=np.float64)
    a = np.zeros((4, 4), dtype=np.float64)

    for column in range(4):
        offset = np.zeros(4, dtype=np.float64)
        offset[column] = epsilon
        plus = one_step_array(equilibrium + offset, 0.0, nominal_plant)
        minus = one_step_array(equilibrium - offset, 0.0, nominal_plant)
        a[:, column] = (plus - minus) / (2.0 * epsilon)

    plus_u = one_step_array(equilibrium, epsilon, nominal_plant)
    minus_u = one_step_array(equilibrium, -epsilon, nominal_plant)
    b = ((plus_u - minus_u) / (2.0 * epsilon)).reshape(4, 1)
    return a, b


def build_lqr(nominal_plant: PlantConfig, config: LQRConfig) -> LQRData:
    """Build a nominal discrete-time LQR controller and Lyapunov matrix."""

    a, b = finite_difference_discrete_model(
        nominal_plant,
        config.finite_difference_epsilon,
    )
    q = np.diag(np.asarray(config.q_diagonal, dtype=np.float64))
    r = np.array([[config.r]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return LQRData(a=a, b=b, k=k, p=p)


def nominal_lqr_force(
    state: State,
    lqr: LQRData,
    nominal_plant: PlantConfig,
) -> float:
    force = float(-(lqr.k @ state.as_array()).item())
    return float(np.clip(force, -nominal_plant.u_max, nominal_plant.u_max))

