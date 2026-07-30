"""Tests for the weighted linear solver used per reduction pixel."""

import numpy as np

from trap.pca_regression import solve_linear_equation_simple


def test_well_constrained_fit_returns_finite_positive_variance():
    """A normal, invertible fit is unchanged: finite positive parameter variance."""
    rng = np.random.RandomState(0)
    design_matrix = np.vstack([np.ones(20), np.linspace(-1, 1, 20)])  # (2 params, 20 data)
    truth = np.array([3.0, 2.0])
    data = truth @ design_matrix + 0.01 * rng.randn(20)
    inverse_covariance = np.full(20, 1.0 / 0.01**2)

    _, variance = solve_linear_equation_simple(design_matrix, data, inverse_covariance)

    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0)


def test_unconstrained_parameter_gets_infinite_not_zero_variance():
    """A pixel with zero weight everywhere makes the fit singular. The
    unconstrained parameters must report *infinite* variance, not zero, so that
    downstream inverse-variance weighting drops them (1/inf = 0) rather than
    dividing by zero and giving them infinite weight."""
    design_matrix = np.vstack([np.ones(5), np.linspace(0, 1, 5)])  # (2 params, 5 data)
    data = np.arange(5.0)
    inverse_covariance = np.zeros(5)  # no information: singular normal matrix

    _, variance = solve_linear_equation_simple(design_matrix, data, inverse_covariance)

    assert np.all(np.isinf(variance))
    assert not np.any(variance == 0)
    # the honest weight for a no-information parameter is exactly zero
    assert np.all(1.0 / variance == 0.0)
