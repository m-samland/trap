"""Tests for the correlation matrix between residual vectors."""

import numpy as np

from trap.utils import compute_empirical_correlation_matrix


def _loop_correlation(residuals, eps=1e-15):
    """The row-by-row loop that the numba version compiled (trap 2.3.0)."""
    n_vectors = residuals.shape[0]
    norms = np.sqrt([np.dot(row, row) for row in residuals])
    corr_matrix = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        for j in range(i, n_vectors):
            denom = norms[i] * norms[j]
            if denom >= eps or np.isnan(denom):
                corr_matrix[i, j] = np.dot(residuals[i], residuals[j]) / denom
            corr_matrix[j, i] = corr_matrix[i, j]
    return corr_matrix


def test_matches_the_loop_including_a_zero_row():
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=(7, 300))
    residuals[3] = 0.0

    result = compute_empirical_correlation_matrix(residuals)

    np.testing.assert_allclose(result, _loop_correlation(residuals), rtol=1e-12, atol=1e-15)
    assert np.all(result[3] == 0.0) and np.all(result[:, 3] == 0.0)
    np.testing.assert_array_equal(result, result.T)


def test_float32_input_and_correlated_rows():
    rng = np.random.default_rng(2)
    base = rng.normal(size=500)
    residuals = np.vstack([base, 2.0 * base, -base, rng.normal(size=500)]).astype(np.float32)

    result = compute_empirical_correlation_matrix(residuals)

    assert result.dtype == np.float64
    np.testing.assert_allclose(result[0, :3], [1.0, 1.0, -1.0], rtol=1e-6)
    np.testing.assert_allclose(result, _loop_correlation(residuals.astype(np.float64)), rtol=1e-12, atol=1e-15)


def test_non_finite_row_propagates_nan():
    residuals = np.ones((3, 10))
    residuals[1, 4] = np.nan

    result = compute_empirical_correlation_matrix(residuals)

    assert np.all(np.isnan(result[1])) and np.all(np.isnan(result[:, 1]))
    np.testing.assert_allclose(result[np.ix_([0, 2], [0, 2])], 1.0)
