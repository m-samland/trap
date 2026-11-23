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


def _source_maps(size=41, separation=10.0, sigma_psf=1.5, amplitude=20.0, noise=0.0, seed=0):
    """Template S/N with the formal uncertainty, and the ring-scaled one with a factor falling outward."""
    centre = size // 2
    yy, xx = np.mgrid[:size, :size]
    snr = amplitude * np.exp(-((yy - centre) ** 2 + (xx - centre - separation) ** 2) / (2 * sigma_psf**2))
    snr = snr + np.random.default_rng(seed).normal(0.0, noise, snr.shape)
    radius = np.hypot(yy - centre, xx - centre)
    ring_scaled = snr * np.exp((radius - separation) / 3.0)
    contrast = snr * 1e-5
    detection_image = np.stack([contrast, np.full_like(snr, 1e-5), ring_scaled])
    contrast_table = pd.DataFrame({"sep (pix)": np.arange(size), "snr_normalization": np.ones(size)})
    return detection_image, ring_scaled, snr, contrast_table, (centre, int(centre + separation))


def _fit(detection_image, normalized, contrast_table, seed, position_snr_image=None):
    from trap.detection import fit_planet_parameters

    return fit_planet_parameters(
        detection_image=detection_image,
        normalized_detection_image=normalized,
        contrast_table=contrast_table,
        yx_position=seed,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=11,
        position_snr_image=position_snr_image,
    )


def _separation(result, centre=20):
    return float(np.hypot(result["y"].values[0] - centre, result["x"].values[0] - centre))


def test_fit_a_on_formal_map_removes_the_ring_bias():
    detection_image, ring_scaled, snr, table, seed = _source_maps(noise=0.01)
    ring_results = _fit(detection_image, ring_scaled, table, seed)
    formal_results = _fit(detection_image, ring_scaled, table, seed, position_snr_image=snr)

    assert _separation(ring_results[1]) > 10.2
    assert abs(_separation(formal_results[1]) - 10.0) < 0.05
    for contrast_result, snr_result, norm_result in (ring_results, formal_results):
        for other in (contrast_result, norm_result):
            assert np.isclose(other["x"].values[0], snr_result["x"].values[0])
            assert np.isclose(other["y"].values[0], snr_result["y"].values[0])


def test_fit_a_sigma_does_not_depend_on_the_map_scale():
    detection_image, ring_scaled, snr, table, seed = _source_maps(noise=0.5)
    single = _fit(detection_image, ring_scaled, table, seed, position_snr_image=snr)[1]
    scaled = _fit(detection_image, ring_scaled, table, seed, position_snr_image=3 * snr)[1]
    for column in ("radial_sigma_fit", "tangential_sigma_fit"):
        assert np.isfinite(single[column].values[0])
        assert np.isclose(single[column].values[0], scaled[column].values[0], rtol=1e-6)


def test_fit_a_on_position_image_with_nan_box_fails_cleanly():
    detection_image, ring_scaled, snr, table, seed = _source_maps()
    results = _fit(detection_image, ring_scaled, table, seed, position_snr_image=np.full_like(snr, np.nan))
    for result in results:
        assert not result["fit_ok"].values[0]


def _stubbed_run(tmp_path, monkeypatch, position_snr_image=None):
    from test_template_path_masks import CONFIGURED, _run_template_path, _stubbed_template_analysis

    analysis, _ = _stubbed_template_analysis(tmp_path, monkeypatch, configured=[CONFIGURED])
    if position_snr_image is not None:
        products = {"position_snr_image": position_snr_image}
        monkeypatch.setattr(analysis, "template_matching_detection", lambda *args, **kwargs: (np.empty(0), products))
    seen = {}
    original_complete = analysis.complete_candidate_table
    original_extract = analysis.extract_candidate_spectra

    def complete_candidate_table(**kwargs):
        seen.setdefault("position_snr_cube", []).append(kwargs.get("position_snr_cube"))
        return original_complete(**kwargs)

    def extract_candidate_spectra(**kwargs):
        seen["positions"] = np.asarray(kwargs["yx_candidate_positions"], dtype=float)
        return original_extract(**kwargs)

    monkeypatch.setattr(analysis, "complete_candidate_table", complete_candidate_table)
    monkeypatch.setattr(analysis, "extract_candidate_spectra", extract_candidate_spectra)
    _run_template_path(analysis)
    return analysis, seen


def test_run_template_matching_without_position_image_uses_plane_2(tmp_path, monkeypatch):
    _, seen = _stubbed_run(tmp_path, monkeypatch)
    assert seen["position_snr_cube"] and all(cube is None for cube in seen["position_snr_cube"])


def test_run_template_matching_passes_the_position_image_to_both_passes(tmp_path, monkeypatch):
    image = np.arange(21 * 21, dtype=float).reshape(21, 21)
    _, seen = _stubbed_run(tmp_path, monkeypatch, position_snr_image=image)
    assert len(seen["position_snr_cube"]) == 2 * 2
    for cube in seen["position_snr_cube"]:
        assert cube.shape == (1, 21, 21) and np.array_equal(cube[0], image)


def test_spectrum_is_extracted_at_the_reported_position(tmp_path, monkeypatch):
    analysis, seen = _stubbed_run(tmp_path, monkeypatch)
    for template in analysis.templates.values():
        table = template.companion_table
        assert table is not None and len(table)
        for candidate_id, rows in table.groupby("candidate_id"):
            reported = rows[["y_relative", "x_relative"]].to_numpy(dtype=float)
            assert np.allclose(reported, seen["positions"][int(candidate_id)])


def test_low_snr_flag_per_candidate():
    from trap.detection import _flag_low_snr_spectra

    table = pd.DataFrame(
        {
            "candidate_id": [0, 0, 1, 1, 2, 2],
            "snr": [3.0, 5.0, 3.0, 4.0, np.nan, np.nan],
        }
    )
    flagged = _flag_low_snr_spectra(table, 4.75)
    assert flagged["spectrum_low_snr"].tolist() == [False, False, True, True, True, True]
    assert "spectrum_low_snr" not in table


def test_low_snr_flag_reaches_the_combined_tables(tmp_path):
    from test_astrometry_uncertainty import _fake_template_table

    from trap.detection import _flag_low_snr_spectra

    analysis = DetectionAnalysis.__new__(DetectionAnalysis)
    analysis.reduction_parameters = SimpleNamespace(result_folder=str(tmp_path))
    tables = {}
    for name, x_rel, snr in (("T-type", 8.0, 10.0), ("flat", 8.1, 9.0)):
        table = _fake_template_table(name, x_rel, 0.0, snr=snr).assign(snr=[3.0, 4.0])
        tables[name] = _flag_low_snr_spectra(table, 4.75)
    analysis.templates = {name: SimpleNamespace(companion_table=t, validated_companion_table=t) for name, t in tables.items()}
    analysis.combine_template_matched_companion_tables(search_radius=3.0, validated_only=True)

    folder = tmp_path / "template_matching"
    for name in ("overall_validated_companion_detections.csv", "overall_validated_companion_detections_spectra.csv"):
        written = pd.read_csv(folder / name)
        assert written["spectrum_low_snr"].astype(bool).all()
        assert "per_channel_distance_sigma" in written
