"""Candidate position of the template path from the template S/N with the formal uncertainty.

See docs/superpowers/specs/2026-11-23-candidate-position-formal-sigma-design.md.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy import units as u

from trap.detection import DetectionAnalysis
from trap.parameters import TrapReductionConfig


def _template_analysis(monkeypatch, n_channels, contrast, formal, ring, correlation=False):
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig())
    analysis.instrument = SimpleNamespace(wavelengths=np.linspace(1.0, 1.2, n_channels) * u.micron)
    analysis.wavelength_indices = np.arange(n_channels)
    shape = (n_channels, 9, 9)
    analysis.detection_cube = np.stack(
        [np.broadcast_to(contrast, shape), np.broadcast_to(formal, shape), np.ones(shape)], axis=1
    ).astype(float)
    analysis.detection_products = {"uncertainty_cube": np.broadcast_to(ring, shape).astype(float)}
    if correlation:
        analysis.empirical_correlation = {
            "matrices": np.identity(n_channels)[None],
            "fit": pd.DataFrame({"separation": [3.0], "fitted": [False], "flagged": [False]}),
            "wavelengths": np.linspace(1.0, 1.2, n_channels),
        }
    monkeypatch.setattr(analysis, "contrast_table_and_normalization", lambda **kwargs: {})
    template = SimpleNamespace(
        name="flat",
        fit_offset=False,
        fit_slope=False,
        use_spectral_correlation=correlation,
        number_of_pca_regressors=0,
        normalized_contrast_modelbox=SimpleNamespace(flux=np.ones(n_channels)),
    )
    return analysis, template


def _reduced(cube_plane):
    mask = cube_plane != 0
    assert mask.sum() > 70
    return mask


def test_position_image_uses_the_formal_uncertainty(monkeypatch):
    analysis, template = _template_analysis(monkeypatch, 3, contrast=1.0, formal=1.0, ring=2.0)
    cube, products = analysis.template_matching_detection(template, file_paths={}, save=False)
    reduced = _reduced(cube[0, 2])
    assert np.allclose(cube[0, 2][reduced], np.sqrt(3) / 2)
    assert np.allclose(products["position_snr_image"][reduced], np.sqrt(3))


def test_position_image_equals_plane_2_when_ring_factor_is_one(monkeypatch):
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 2.0, size=(3, 9, 9))
    analysis, template = _template_analysis(monkeypatch, 3, contrast=rng.normal(size=(3, 9, 9)), formal=sigma, ring=sigma)
    cube, products = analysis.template_matching_detection(template, file_paths={}, save=False)
    assert np.allclose(products["position_snr_image"], cube[0, 2])


def test_formal_fit_uses_correlation_with_formal_sigma(monkeypatch):
    analysis, template = _template_analysis(monkeypatch, 2, contrast=1.0, formal=1.0, ring=2.0, correlation=True)
    cube, products = analysis.template_matching_detection(template, file_paths={}, save=False)
    reduced = _reduced(cube[0, 2])
    assert np.allclose(products["position_snr_image"][reduced], 2 * cube[0, 2][reduced])
