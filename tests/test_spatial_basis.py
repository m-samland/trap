"""Leading spatial principal components of a subset of frames, from one frame-frame Gram matrix.

The spatial model needs, for every frame, the first few principal components of the
other frames that are far enough from it in parallactic angle. When the scaling acts on
each frame alone, all of these come from one Gram matrix of the frames per position
instead of one SVD per frame.
"""

import numpy as np

from trap.pca_regression import leading_left_singular_vectors

N_PIXEL, N_FRAME, N_COMPONENTS = 400, 120, 5


def _subspace_sine(a, b):
    """Sine of the largest principal angle, from the part of b outside the span of a."""
    return np.linalg.norm(b - a @ (a.T @ b), ord=2)


def test_spans_the_svd_components_of_the_selected_frames():
    rng = np.random.default_rng(0)
    modes = rng.standard_normal((N_PIXEL, 8)) * np.geomspace(100, 5, 8)
    matrix = modes @ rng.standard_normal((8, N_FRAME)) + rng.standard_normal((N_PIXEL, N_FRAME))
    gram = matrix.T @ matrix
    columns = np.flatnonzero(np.abs(np.arange(N_FRAME) - 40) > 10)
    basis = leading_left_singular_vectors(matrix, gram, columns, N_COMPONENTS)
    expected = np.linalg.svd(matrix[:, columns], full_matrices=False)[0][:, :N_COMPONENTS]
    assert basis.shape == (N_PIXEL, N_COMPONENTS)
    np.testing.assert_allclose(basis.T @ basis, np.eye(N_COMPONENTS), atol=1e-10)
    assert _subspace_sine(basis, expected) < 1e-9


def test_fewer_frames_than_components_gives_one_component_per_frame():
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((N_PIXEL, N_FRAME))
    columns = np.array([3, 50, 90])
    basis = leading_left_singular_vectors(matrix, matrix.T @ matrix, columns, N_COMPONENTS)
    assert basis.shape == (N_PIXEL, 3)


def test_rank_deficient_selection_returns_none():
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((N_PIXEL, 2)) @ rng.standard_normal((2, N_FRAME))
    basis = leading_left_singular_vectors(matrix, matrix.T @ matrix, np.arange(N_FRAME), N_COMPONENTS)
    assert basis is None
