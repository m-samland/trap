"""The per-pixel fit of the temporal model gives the same numbers as the general solver.

`fit_planet_coefficient` factors the weighted normal matrix once by Cholesky and reads
the companion variance from the last diagonal element of the factor, instead of an LU
solve plus a full inverse. Problems that are not positive definite or not finite go
through `solve_linear_equation_simple` unchanged.
"""

import numpy as np
import pytest
from sklearn import preprocessing

from trap.pca_regression import fit_planet_coefficient, matrix_scaling, solve_linear_equation_simple

T, K = 256, 77


def _design(rng, n_time=T, n_components=K):
    basis = np.linalg.qr(rng.standard_normal((n_time, n_components)))[0]
    planet = np.exp(-0.5 * ((np.arange(n_time) - n_time / 2) / 15) ** 2)
    return np.column_stack([basis, np.ones(n_time), planet])


def _reference(design, data, ivar):
    coefficients, variances = solve_linear_equation_simple(design.T, data, ivar)
    return coefficients, variances[-1]


@pytest.mark.parametrize("weighted", [True, False])
def test_matches_the_general_solver(weighted):
    rng = np.random.default_rng(0)
    design = _design(rng)
    data = design @ rng.standard_normal(K + 2) + rng.standard_normal(T)
    ivar = rng.uniform(0.2, 5.0, T) if weighted else None
    if weighted:
        ivar[::17] = 0.0
    coefficients, variance = fit_planet_coefficient(design, data, ivar)
    expected_coefficients, expected_variance = _reference(design, data, ivar)
    np.testing.assert_allclose(coefficients, expected_coefficients, rtol=1e-9, atol=1e-12)
    assert variance == pytest.approx(expected_variance, rel=1e-10)


def test_without_components_matches_the_general_solver():
    rng = np.random.default_rng(1)
    design = _design(rng)[:, K:]
    data = rng.standard_normal(T)
    ivar = rng.uniform(0.5, 2.0, T)
    coefficients, variance = fit_planet_coefficient(design, data, ivar)
    expected_coefficients, expected_variance = _reference(design, data, ivar)
    np.testing.assert_allclose(coefficients, expected_coefficients, rtol=1e-12)
    assert variance == pytest.approx(expected_variance, rel=1e-12)


def test_singular_problem_takes_the_general_solver():
    rng = np.random.default_rng(2)
    design = _design(rng)
    design[:, 1] = design[:, 0]
    data = rng.standard_normal(T)
    coefficients, variance = fit_planet_coefficient(design, data, None)
    expected_coefficients, expected_variance = _reference(design, data, None)
    np.testing.assert_array_equal(coefficients, expected_coefficients)
    assert variance == expected_variance


def test_zero_weight_for_the_planet_frames_takes_the_general_solver():
    rng = np.random.default_rng(3)
    design = _design(rng, n_components=5)
    design[design[:, -1] < 1e-3, -1] = 0.0
    data = rng.standard_normal(T)
    ivar = np.ones(T)
    ivar[design[:, -1] > 0] = 0.0
    coefficients, variance = fit_planet_coefficient(design, data, ivar)
    expected_coefficients, expected_variance = _reference(design, data, ivar)
    np.testing.assert_array_equal(coefficients, expected_coefficients)
    assert variance == expected_variance == np.inf


def test_non_finite_input_does_not_raise_and_matches_the_general_solver():
    rng = np.random.default_rng(4)
    design = _design(rng, n_components=10)
    data = rng.standard_normal(T)
    ivar = np.ones(T)
    ivar[7] = np.nan
    coefficients, variance = fit_planet_coefficient(design, data, ivar)
    expected_coefficients, expected_variance = _reference(design, data, ivar)
    np.testing.assert_array_equal(coefficients, expected_coefficients)
    np.testing.assert_array_equal(variance, expected_variance)


@pytest.mark.parametrize("n_time", [255, 256])
def test_median_centring_matches_scikit_learn(n_time):
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((n_time, 300)) * 10 + rng.uniform(0, 100, 300)
    matrix[3, 11] = np.nan
    expected = preprocessing.robust_scale(matrix, with_centering=True, with_scaling=False)
    np.testing.assert_array_equal(matrix_scaling(matrix, scaling="temp-median"), expected)
