import numpy as np

from cartpole import CartPoleParams, build_discrete_lqr, derivative
from cartpole.lqr import continuous_linear_model


def finite_difference_jacobian(function, point, step=1e-6):
    point = np.asarray(point, dtype=float)
    output_size = function(point).size
    jacobian = np.zeros((output_size, point.size))
    for column in range(point.size):
        offset = np.zeros_like(point)
        offset[column] = step
        jacobian[:, column] = (
            function(point + offset) - function(point - offset)
        ) / (2.0 * step)
    return jacobian


def test_analytic_linearization_matches_nonlinear_finite_difference():
    params = CartPoleParams()
    a, b = continuous_linear_model(params)
    state_zero = np.zeros(4)

    a_fd = finite_difference_jacobian(
        lambda state: derivative(state, 0.0, params), state_zero
    )
    b_fd = finite_difference_jacobian(
        lambda force: derivative(state_zero, force[0], params),
        np.zeros(1),
    )

    np.testing.assert_allclose(a, a_fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b, b_fd, rtol=1e-6, atol=1e-8)


def test_discrete_lqr_is_locally_stable():
    result = build_discrete_lqr(CartPoleParams())
    assert max(abs(result.closed_loop_eigenvalues)) < 1.0
    np.testing.assert_allclose(result.riccati, result.riccati.T, atol=1e-9)
    assert min(np.linalg.eigvalsh(result.riccati)) > 0.0
