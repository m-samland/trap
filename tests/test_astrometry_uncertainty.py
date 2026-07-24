"""Unit tests for the astrometric-uncertainty additions to
trap.detection. See
docs/superpowers/specs/2026-07-23-trap-astrometry-uncertainty-design.md
for the mathematical justification of every assertion in this file.
"""

import numpy as np
import pandas as pd
import pytest

from trap import detection


def test_module_imports():
    assert hasattr(detection, "fit_2d_gaussian")


def _make_isotropic_gaussian(size=21, amplitude=50.0, center=None, sigma=1.5, noise=0.5):
    """Noisy 2D Gaussian on a square image. Position at (cy, cx)."""
    if center is None:
        center = (size / 2 - 0.5, size / 2 - 0.5)
    yy, xx = np.mgrid[:size, :size]
    image = amplitude * np.exp(
        -((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2 * sigma**2)
    )
    rng = np.random.default_rng(0)
    image = image + rng.normal(0.0, noise, size=image.shape)
    return image, center


def test_fit_2d_gaussian_returns_param_cov_on_clean_source():
    size = 21
    image, center = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    yx_center = (size // 2, size // 2)
    result = detection.fit_2d_gaussian(
        image,
        yx_position=(round(center[0]), round(center[1])),
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
    )
    assert "param_cov_xy" in result
    assert "param_names" in result
    cov = result["param_cov_xy"]
    assert cov is not None
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    assert cov[0, 0] > 0 and cov[1, 1] > 0
    # For a symmetric input, ρ_xy should be small
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(rho) < 0.3


def test_fit_2d_gaussian_handles_singular_fit():
    """All-NaN cutout region should raise, unchanged by the covariance additions."""
    size = 21
    image, _ = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    image[9:12, 9:12] = np.nan
    yx_center = (size // 2, size // 2)
    with pytest.raises(RuntimeError):
        detection.fit_2d_gaussian(
            image,
            yx_position=(10, 10),
            yx_center=yx_center,
            x_stddev=1.5,
            y_stddev=1.5,
            box_size=3,
        )


def test_fit_2d_gaussian_fixed_position_clamps_centroid():
    """With fixed_position set, the fit centroid must equal the requested
    sub-pixel position exactly (used by Fits B/C to clamp to Fit A)."""
    size = 21
    image, center = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    yx_center = (size // 2, size // 2)
    clamp_yx = (center[0] + 0.3, center[1] - 0.2)
    result = detection.fit_2d_gaussian(
        image,
        yx_position=(round(center[0]), round(center[1])),
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
        fixed_position=clamp_yx,
    )
    assert np.isclose(result["yx_fit_position_orig"][0], clamp_yx[0])
    assert np.isclose(result["yx_fit_position_orig"][1], clamp_yx[1])
    # Position was held fixed, so it is not a free parameter and has no cov row
    assert "x_mean" not in result["param_names"]
    assert "y_mean" not in result["param_names"]
