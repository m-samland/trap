"""The temporal basis from the QR route spans the same components as the full SVD.

`leading_left_singular_vectors_qr` replaces ``compute_SVD`` in the map-mode temporal
model. The regression only depends on the subspace of the leading components, so the
tests compare subspaces and the companion fit, not the signs of individual components.
"""

import numpy as np
import pytest

from trap.pca_regression import compute_SVD, fit_planet_coefficient, leading_left_singular_vectors_qr


def _training_matrix(rng, n_time, n_pixel, rank=None):
    """Light curves with a decaying spectrum, like a speckle pool, plus white noise."""
    rank = min(n_time, n_pixel) if rank is None else rank
    time_modes = np.linalg.qr(rng.standard_normal((n_time, rank)))[0]
    pixel_modes = np.linalg.qr(rng.standard_normal((n_pixel, rank)))[0]
    strengths = 100.0 * np.exp(-np.arange(rank) / 10.0)
    noise = 0.0 if rank < min(n_time, n_pixel) else 0.01
    return (time_modes * strengths) @ pixel_modes.T + noise * rng.standard_normal((n_time, n_pixel)) + 5.0


def _subspace_distance(a, b):
    return np.linalg.norm(b - a @ (a.T @ b), 2)


@pytest.mark.parametrize("n_time, n_pixel", [(256, 1800), (256, 120)])
@pytest.mark.parametrize("scaling", ["temp-median", "temp-mean", None])
def test_spans_the_svd_components(n_time, n_pixel, scaling):
    rng = np.random.default_rng(0)
    matrix = _training_matrix(rng, n_time, n_pixel)
    reference = compute_SVD(matrix, n_components=51, scaling=scaling)[0]

    basis = leading_left_singular_vectors_qr(matrix, n_components=51, scaling=scaling)

    assert basis.shape == reference.shape
    np.testing.assert_allclose(basis.T @ basis, np.eye(51), atol=1e-12)
    assert _subspace_distance(reference, basis) < 1e-12


def test_all_components_and_more_than_available():
    rng = np.random.default_rng(1)
    matrix = _training_matrix(rng, 64, 40)

    assert leading_left_singular_vectors_qr(matrix, n_components=None).shape == (64, 40)
    assert leading_left_singular_vectors_qr(matrix, n_components=100).shape == (64, 40)


def test_rank_deficient_pool_gives_the_same_companion_fit():
    """Below the rank of the pool the components are well defined, so the fit agrees."""
    rng = np.random.default_rng(2)
    n_time, n_components = 256, 20
    matrix = _training_matrix(rng, n_time, 900, rank=30)
    planet = np.exp(-0.5 * ((np.arange(n_time) - n_time / 2) / 15) ** 2)
    data = matrix[:, 0] + 0.3 * planet + 0.01 * rng.standard_normal(n_time)
    ivar = rng.uniform(0.5, 2.0, n_time)

    fits = []
    for basis in (
        compute_SVD(matrix, n_components=n_components)[0],
        leading_left_singular_vectors_qr(matrix, n_components=n_components),
    ):
        design = np.column_stack([basis, np.ones(n_time), planet])
        fits.append(fit_planet_coefficient(design, data, ivar))

    (reference, reference_variance), (coefficients, variance) = fits
    assert coefficients[-1] == pytest.approx(reference[-1], rel=1e-10)
    assert variance == pytest.approx(reference_variance, rel=1e-10)
