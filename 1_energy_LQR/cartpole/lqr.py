from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from .config import CartPoleParams, LQRConfig
from .dynamics import mass_matrix


@dataclass(frozen=True)
class LQRResult:
    a_continuous: np.ndarray
    b_continuous: np.ndarray
    a_discrete: np.ndarray
    b_discrete: np.ndarray
    gain: np.ndarray
    riccati: np.ndarray
    closed_loop_eigenvalues: np.ndarray


def continuous_linear_model(params):
    """Linearize the nominal nonlinear plant at upright s=0, u=0."""

    inverse_mass = np.linalg.inv(mass_matrix(0.0, params))
    m = params.pole_mass
    length = params.pole_com_length

    acceleration_from_x_dot = inverse_mass @ np.array(
        [-params.cart_damping, 0.0]
    )
    acceleration_from_theta = inverse_mass @ np.array(
        [0.0, m * params.gravity * length]
    )
    acceleration_from_theta_dot = inverse_mass @ np.array(
        [0.0, -params.pole_damping]
    )
    acceleration_from_force = inverse_mass @ np.array([1.0, 0.0])

    a = np.zeros((4, 4), dtype=float)
    a[0, 2] = 1.0
    a[1, 3] = 1.0
    a[2:, 1] = acceleration_from_theta
    a[2:, 2] = acceleration_from_x_dot
    a[2:, 3] = acceleration_from_theta_dot

    b = np.zeros((4, 1), dtype=float)
    b[2:, 0] = acceleration_from_force
    return a, b


def zoh_discrete_model(a, b, sample_time):
    """Exact zero-order-hold discretization of a continuous linear model."""

    c = np.eye(a.shape[0])
    d = np.zeros((a.shape[0], b.shape[1]))
    a_d, b_d, _, _, _ = cont2discrete(
        (a, b, c, d), sample_time, method="zoh"
    )
    return a_d, b_d


def controllability_rank(a, b):
    blocks = [b]
    for _ in range(1, a.shape[0]):
        blocks.append(a @ blocks[-1])
    return np.linalg.matrix_rank(np.hstack(blocks))


def build_discrete_lqr(params, config=None):
    """Build u[k] = -K z[k] for the nominal sampled linear plant."""

    if config is None:
        config = LQRConfig()

    a, b = continuous_linear_model(params)
    a_d, b_d = zoh_discrete_model(a, b, params.control_period)

    if controllability_rank(a_d, b_d) != a_d.shape[0]:
        raise ValueError("The nominal discrete model is not controllable")

    q = np.diag(
        [
            config.q_x,
            config.q_theta,
            config.q_x_dot,
            config.q_theta_dot,
        ]
    )
    r = np.array([[config.r_force]], dtype=float)

    p = solve_discrete_are(a_d, b_d, q, r)
    gain = np.linalg.solve(r + b_d.T @ p @ b_d, b_d.T @ p @ a_d)
    eigenvalues = np.linalg.eigvals(a_d - b_d @ gain)

    return LQRResult(
        a_continuous=a,
        b_continuous=b,
        a_discrete=a_d,
        b_discrete=b_d,
        gain=gain,
        riccati=p,
        closed_loop_eigenvalues=eigenvalues,
    )
