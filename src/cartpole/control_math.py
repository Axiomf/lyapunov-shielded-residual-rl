"""Nominal sampled-data linearization and discrete-time LQR utilities.

This module connects the nonlinear cart-pole simulation to the local control
and Lyapunov theory used in the project.  If the nominal zero-order-held RK4
transition is written as

    z[k + 1] = F_h(z[k], u[k]),

then :func:`finite_difference_discrete_model` approximates its Jacobians at
the upright equilibrium ``(z, u) = (0, 0)``:

    A = dF_h/dz,    B = dF_h/du.

The resulting local model ``z[k + 1] ~= A z[k] + B u[k]`` is used to construct
the nominal discrete-time LQR law ``u = -K z`` and quadratic function
``V(z) = z.T @ P @ z``.  The shield later evaluates this same ``V`` along the
nominal *nonlinear* one-step map.

All matrices here are nominal-model objects.  Their standard LQR/Lyapunov
properties apply locally to the unsaturated nominal linearization under the
usual stabilizability/detectability assumptions; they do not by themselves
certify the mass-mismatched nonlinear plant or a global region of attraction.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from .config import LQRConfig, PlantConfig
from .data import FloatArray, State
from .plant import step_rk4


@dataclass(frozen=True)
class LQRData:
    """Matrices shared by the nominal LQR controller and Lyapunov shield.

    With state order ``[x, theta, x_dot, theta_dot]`` and one scalar force:

    - ``a`` has shape ``(4, 4)`` and is the nominal state Jacobian.
    - ``b`` has shape ``(4, 1)`` and is the nominal input Jacobian.
    - ``k`` has shape ``(1, 4)`` and defines the feedback ``u = -k @ z``.
    - ``p`` has shape ``(4, 4)`` and defines ``V(z) = z.T @ p @ z``.

    The dataclass is frozen so every controller receives the same unchanged
    nominal matrices during a paired experiment.
    """

    a: FloatArray
    b: FloatArray
    k: FloatArray
    p: FloatArray


def one_step_array(
    state_array: FloatArray,
    force: float,
    nominal_plant: PlantConfig,
) -> FloatArray:
    """Evaluate the nominal nonlinear one-control-period map ``F_h``.

    The force is held constant over the control period by :func:`step_rk4`.
    This small array adapter lets the finite-difference code work with vectors
    while preserving the project's canonical :class:`State` representation.

    Args:
        state_array: State vector ``[x, theta, x_dot, theta_dot]``.
        force: Zero-order-held horizontal force in newtons.
        nominal_plant: Controller-model parameters, not mismatched plant data.

    Returns:
        The next nominal state as a float64 array with shape ``(4,)``.
    """

    state = State.from_array(np.asarray(state_array, dtype=np.float64))
    return step_rk4(state, force, nominal_plant).as_array()


def finite_difference_discrete_model(
    nominal_plant: PlantConfig,
    epsilon: float,
) -> tuple[FloatArray, FloatArray]:
    """Linearize the nominal sampled-data map at the upright equilibrium.

    Central differences approximate

        A[:, j] = dF_h/dz_j at (z, u) = (0, 0),
        B       = dF_h/du  at (z, u) = (0, 0).

    Unlike linearizing continuous-time equations and then discretizing them,
    this function differentiates the implemented RK4 one-period map directly.
    Therefore ``A`` and ``B`` match the simulation's control period and
    zero-order-hold convention.

    Args:
        nominal_plant: Plant parameters assumed by the controller.
        epsilon: Positive central-difference perturbation used for both state
            coordinates and the scalar force.

    Returns:
        ``(A, B)`` with shapes ``(4, 4)`` and ``(4, 1)`` for the local model
        ``z[k + 1] ~= A @ z[k] + B @ u[k]``.

    Notes:
        This is a local nominal-model approximation.  It is not the Jacobian
        of a frozen residual policy or of a mass-mismatched closed loop; those
        must be estimated separately during evaluation.
    """

    # theta = 0 is upright, so the origin with zero force is the nominal
    # equilibrium about which the local controller is designed.
    equilibrium = np.zeros(4, dtype=np.float64)
    a = np.zeros((4, 4), dtype=np.float64)

    # Perturb one state coordinate at a time.  Each derivative becomes one
    # column of the discrete-time state matrix A.
    for column in range(4):
        offset = np.zeros(4, dtype=np.float64)
        offset[column] = epsilon
        plus = one_step_array(equilibrium + offset, 0.0, nominal_plant)
        minus = one_step_array(equilibrium - offset, 0.0, nominal_plant)
        a[:, column] = (plus - minus) / (2.0 * epsilon)

    # The scalar-input derivative is stored as a column matrix so products
    # such as B.T @ P @ B keep their standard LQR dimensions.
    plus_u = one_step_array(equilibrium, epsilon, nominal_plant)
    minus_u = one_step_array(equilibrium, -epsilon, nominal_plant)
    b = ((plus_u - minus_u) / (2.0 * epsilon)).reshape(4, 1)
    return a, b


def build_lqr(nominal_plant: PlantConfig, config: LQRConfig) -> LQRData:
    """Build the nominal discrete-time LQR and its quadratic matrix.

    For the local model ``z[k + 1] = A z[k] + B u[k]``, LQR minimizes the
    infinite-horizon quadratic cost

        sum(z[k].T Q z[k] + u[k].T R u[k]).

    SciPy solves the discrete algebraic Riccati equation (DARE) for ``P``;
    the corresponding feedback gain is

        K = (R + B.T P B)^(-1) B.T P A,

    and this project applies ``u = -K z``.  Under the standard linear LQR
    assumptions, ``V(z) = z.T P z`` decreases for the nominal unsaturated
    linear closed loop away from the origin.  In this project ``P`` is also
    reused as the candidate Lyapunov matrix in the nominal nonlinear shield
    condition; that reuse does not turn ``V`` into a global certificate.

    Args:
        nominal_plant: Controller-model parameters used for linearization.
        config: State/input costs and finite-difference perturbation.

    Returns:
        Immutable matrices ``A``, ``B``, ``K``, and ``P`` in :class:`LQRData`.
    """

    a, b = finite_difference_discrete_model(
        nominal_plant,
        config.finite_difference_epsilon,
    )

    # Q weights state deviations; R weights the scalar control effort.
    q = np.diag(np.asarray(config.q_diagonal, dtype=np.float64))
    r = np.array([[config.r]], dtype=np.float64)

    p = solve_discrete_are(a, b, q, r)
    # Solving a linear system is numerically preferable to forming an inverse.
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return LQRData(a=a, b=b, k=k, p=p)


def nominal_lqr_force(
    state: State,
    lqr: LQRData,
    nominal_plant: PlantConfig,
) -> float:
    """Apply the nominal feedback law and enforce the actuator limit.

    The unconstrained LQR command is ``u = -K z``.  Clipping enforces
    ``|u| <= u_max``, but a saturated command no longer follows the exact
    unconstrained linear LQR dynamics used in the Riccati analysis.

    Args:
        state: Current full cart-pole state.
        lqr: Nominal matrices produced by :func:`build_lqr`.
        nominal_plant: Supplies the common force limit ``u_max``.

    Returns:
        A scalar force in newtons within ``[-u_max, u_max]``.
    """

    force = float(-(lqr.k @ state.as_array()).item())
    return float(np.clip(force, -nominal_plant.u_max, nominal_plant.u_max))

