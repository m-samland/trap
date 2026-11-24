"""Candidate position of the template path from the template S/N with frozen ring factors."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy import units as u

from trap.detection import DetectionAnalysis
from trap.parameters import TrapReductionConfig


def _template_analysis(monkeypatch, n_channels, contrast, formal, ring, correlation=False, size=9):
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig())
    analysis.instrument = SimpleNamespace(wavelengths=np.linspace(1.0, 1.2, n_channels) * u.micron)
    analysis.wavelength_indices = np.arange(n_channels)
    shape = (n_channels, size, size)
    analysis.detection_cube = np.stack(
        [np.broadcast_to(contrast, shape), np.broadcast_to(formal, shape), np.ones(shape)], axis=1
    ).astype(float)
    analysis.detection_products = {"uncertainty_cube": np.broadcast_to(ring, shape).astype(float)}
    if correlation is not False:
        matrix = correlation if isinstance(correlation, np.ndarray) else np.identity(n_channels)
        analysis.empirical_correlation = {
            "matrices": matrix[None],
            "fit": pd.DataFrame({"separation": [3.0], "fitted": [False], "flagged": [False]}),
            "wavelengths": np.linspace(1.0, 1.2, n_channels),
        }
    monkeypatch.setattr(analysis, "contrast_table_and_normalization", lambda **kwargs: {})
    template = SimpleNamespace(
        name="flat",
        fit_offset=False,
        fit_slope=False,
        use_spectral_correlation=correlation is not False,
        number_of_pca_regressors=0,
        normalized_contrast_modelbox=SimpleNamespace(flux=np.ones(n_channels)),
    )
    return analysis, template


def _radius(size):
    yy, xx = np.indices((size, size))
    return np.hypot(yy - size // 2, xx - size // 2)


def _box(image):
    return np.isfinite(image)


def test_position_image_freezes_the_ring_factor_at_the_candidate(monkeypatch):
    size = 21
    ring = 1.0 + 0.2 * _radius(size)
    analysis, template = _template_analysis(monkeypatch, 3, contrast=1.0, formal=1.0, ring=ring, size=size)
    cube, _ = analysis.template_matching_detection(template, file_paths={}, save=False)
    image, noise = analysis.template_position_snr_image(template, (10, 16), box_size=5)

    box = _box(image)
    assert box.sum() == 7 * 7
    assert np.ptp(image[box]) < 1e-12
    assert np.allclose(image[box], np.sqrt(3) / (1.0 + 0.2 * 6.0), rtol=0.02)
    assert np.ptp(cube[0, 2][box]) > 0.1
    assert np.allclose(noise[box], ring[box] / (1.0 + 0.2 * 6.0), rtol=0.02)
    assert np.allclose(cube[0, 2][box] * noise[box], image[box])


def test_position_image_keeps_the_channel_weights(monkeypatch):
    rng = np.random.default_rng(1)
    size = 15
    formal = rng.uniform(0.5, 2.0, size=(3, size, size))
    ring = formal * np.array([1.0, 2.5, 1.0])[:, None, None]
    contrast = rng.normal(size=(3, size, size))
    analysis, template = _template_analysis(monkeypatch, 3, contrast=contrast, formal=formal, ring=ring, size=size)
    cube, _ = analysis.template_matching_detection(template, file_paths={}, save=False)
    image, noise = analysis.template_position_snr_image(template, (7, 11), box_size=5)
    box = _box(image)
    assert box.sum() > 20
    assert np.allclose(image[box], cube[0, 2][box])
    assert np.allclose(noise[box], 1.0)


def test_position_image_uses_the_correlation(monkeypatch):
    size = 15
    psi = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    contrast = np.array([1.0, 2.0, 0.5])[:, None, None] * np.ones((3, size, size))
    formal = np.array([1.0, 1.5, 0.8])[:, None, None] * np.ones((3, size, size))
    factors = np.array([1.2, 2.0, 1.0])
    analysis, template = _template_analysis(
        monkeypatch, 3, contrast=contrast, formal=formal, ring=formal * factors[:, None, None], correlation=psi, size=size
    )
    image, noise = analysis.template_position_snr_image(template, (7, 11), box_size=3)
    assert np.allclose(noise[_box(image)], 1.0)

    sigma = np.array([1.0, 1.5, 0.8]) * factors
    inverse = np.linalg.inv(sigma[:, None] * psi * sigma[None, :])
    t = np.ones(3)
    expected = (t @ inverse @ np.array([1.0, 2.0, 0.5])) / np.sqrt(t @ inverse @ t)
    assert np.allclose(image[_box(image)], expected)


def test_channel_without_ring_factor_takes_the_median_of_the_others(monkeypatch, caplog):
    size = 15
    ring = np.stack([np.full((size, size), 2.0), np.full((size, size), 4.0), np.full((size, size), np.nan)])
    analysis, template = _template_analysis(monkeypatch, 3, contrast=1.0, formal=1.0, ring=ring, size=size)
    with caplog.at_level(logging.WARNING):
        image, noise = analysis.template_position_snr_image(template, (7, 11), box_size=3)
    assert "formal uncertainty alone" not in caplog.text
    expected = np.sum(1 / np.array([2.0, 4.0, 3.0]) ** 2) / np.sqrt(np.sum(1 / np.array([2.0, 4.0, 3.0]) ** 2))
    assert np.allclose(image[_box(image)], expected)
    assert np.allclose(noise[_box(image)], 1.0)


def test_position_image_falls_back_to_the_formal_uncertainty(monkeypatch, caplog):
    size = 15
    analysis, template = _template_analysis(monkeypatch, 2, contrast=1.0, formal=1.0, ring=np.nan, size=size)
    with caplog.at_level(logging.WARNING):
        image, noise = analysis.template_position_snr_image(template, (7, 11), box_size=3)
    assert "formal uncertainty alone" in caplog.text
    assert np.allclose(image[_box(image)], np.sqrt(2))
    assert np.allclose(noise[_box(image)], 1.0)


def test_position_image_freezes_the_correlation(monkeypatch):
    size = 15
    near = np.array([[1.0, 0.6], [0.6, 1.0]])
    far = np.array([[1.0, -0.4], [-0.4, 1.0]])
    contrast = np.array([1.0, 2.0])[:, None, None] * np.ones((2, size, size))
    analysis, template = _template_analysis(monkeypatch, 2, contrast=contrast, formal=1.0, ring=1.0, correlation=True,
                                            size=size)
    analysis.empirical_correlation["matrices"] = np.stack([near, far])
    analysis.empirical_correlation["fit"] = pd.DataFrame(
        {"separation": [3.6, 6.0], "fitted": [False, False], "flagged": [False, False]}
    )
    image, noise = analysis.template_position_snr_image(template, (7, 11), box_size=3)
    box = _box(image)
    inverse = np.linalg.inv(near)
    expected = (np.ones(2) @ inverse @ np.array([1.0, 2.0])) / np.sqrt(np.ones(2) @ inverse @ np.ones(2))
    assert np.allclose(image[box], expected)
    radius = _radius(size)
    assert np.allclose(noise[box & (radius < 4.8)], 1.0)
    assert not np.allclose(noise[box & (radius > 4.8)], 1.0)


def test_position_image_skips_unreduced_pixels(monkeypatch):
    size = 15
    contrast = np.ones((2, size, size))
    contrast[0, 7, 11] = 0.0
    analysis, template = _template_analysis(monkeypatch, 2, contrast=contrast, formal=1.0, ring=1.0, size=size)
    image, _ = analysis.template_position_snr_image(template, (7, 11), box_size=3)
    assert np.isnan(image[7, 11])
    assert _box(image).sum() == 5 * 5 - 1


def test_noise_ratio_with_correlation_and_offset():
    from trap.detection import _template_noise_ratio

    rng = np.random.default_rng(3)
    design = np.column_stack([np.ones(4), rng.uniform(0.5, 1.5, 4)])
    psi = np.array([[1, 0.3, 0.1, 0], [0.3, 1, 0.3, 0.1], [0.1, 0.3, 1, 0.3], [0, 0.1, 0.3, 1.0]])
    assumed = np.array([1.0, 2.0, 1.5, 1.0])
    true = assumed * np.array([1.5, 0.7, 1.2, 2.0])
    draws = rng.multivariate_normal(np.zeros(4), true[:, None] * psi * true[None, :], size=40000)
    inverse = np.linalg.inv(assumed[:, None] * psi * assumed[None, :])
    normal_inverse = np.linalg.inv(design.T @ inverse @ design)
    snr = draws @ (inverse @ design @ normal_inverse[:, -1]) / np.sqrt(normal_inverse[-1, -1])
    assert np.isclose(_template_noise_ratio(design, assumed, true, psi, psi), snr.std(), rtol=0.02)
    assert np.isclose(_template_noise_ratio(design, assumed, 3 * assumed, psi, psi), 3.0)
    assert np.isnan(_template_noise_ratio(design, assumed, np.array([1.0, np.nan, 1.0, 1.0]), psi, psi))

    true_psi = np.identity(4)
    draws = rng.multivariate_normal(np.zeros(4), true[:, None] * true_psi * true[None, :], size=40000)
    snr = draws @ (inverse @ design @ normal_inverse[:, -1]) / np.sqrt(normal_inverse[-1, -1])
    assert np.isclose(_template_noise_ratio(design, assumed, true, psi, true_psi), snr.std(), rtol=0.02)


def test_noise_ratio_of_a_singular_design_is_nan():
    from trap.detection import _template_noise_ratio

    design = np.column_stack([np.ones(3), np.ones(3)])
    assert np.isnan(_template_noise_ratio(design, np.ones(3), np.ones(3)))


def test_weighted_fit_ignores_a_noisy_side():
    from trap.detection import fit_2d_gaussian

    yy, xx = np.indices((21, 21))
    image = 10 * np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 1.5**2))
    image[9:12, 12:15] += 4.0
    noise = np.ones_like(image)
    noise[:, 12:] = 50.0
    kwargs = dict(yx_position=(10, 10), x_stddev=1.5, y_stddev=1.5, box_size=11, fix_width=False, fix_orientation=False)
    unweighted = fit_2d_gaussian(image, **kwargs)
    weighted = fit_2d_gaussian(image, weights=1 / noise, **kwargs)
    assert unweighted["yx_fit_position_orig"][1] > 10.2
    assert abs(weighted["yx_fit_position_orig"][1] - 10.0) < 0.05
    no_weights = fit_2d_gaussian(image, weights=np.full_like(image, np.nan), **kwargs)
    assert not no_weights["fit_ok"]


def test_weighted_fit_sigma_does_not_depend_on_the_weight_scale():
    from trap.detection import fit_2d_gaussian

    rng = np.random.default_rng(4)
    yy, xx = np.indices((21, 21))
    noise = 1.0 + 0.1 * (21 - xx)
    image = 10 * np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 1.5**2)) + noise * rng.normal(size=noise.shape)
    kwargs = dict(yx_position=(10, 10), x_stddev=1.5, y_stddev=1.5, box_size=11, fix_width=False, fix_orientation=False)
    covariances = [fit_2d_gaussian(image, weights=scale / noise, **kwargs)["param_cov_xy"] for scale in (1.0, 5.0)]
    assert np.all(np.isfinite(covariances[0]))
    assert np.allclose(covariances[0], covariances[1], rtol=1e-6)


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


def _stubbed_run(tmp_path, monkeypatch, position_stub=None):
    from test_template_path_masks import CONFIGURED, _run_template_path, _stubbed_template_analysis

    analysis, _ = _stubbed_template_analysis(tmp_path, monkeypatch, configured=[CONFIGURED])
    if position_stub is not None:
        monkeypatch.setattr(analysis, "template_position_snr_image", position_stub)
    seen = {}
    original_complete = analysis.complete_candidate_table
    original_extract = analysis.extract_candidate_spectra

    def complete_candidate_table(**kwargs):
        seen.setdefault("position_snr_function", []).append(kwargs.get("position_snr_function"))
        return original_complete(**kwargs)

    def extract_candidate_spectra(**kwargs):
        seen["positions"] = np.asarray(kwargs["yx_candidate_positions"], dtype=float)
        return original_extract(**kwargs)

    monkeypatch.setattr(analysis, "complete_candidate_table", complete_candidate_table)
    monkeypatch.setattr(analysis, "extract_candidate_spectra", extract_candidate_spectra)
    _run_template_path(analysis)
    return analysis, seen


def test_run_template_matching_passes_the_position_function_to_both_passes(tmp_path, monkeypatch):
    calls = []

    def position_stub(template, yx_position, box_size, inner_mask_radius):
        calls.append((template.name, tuple(yx_position), box_size))

    _, seen = _stubbed_run(tmp_path, monkeypatch, position_stub=position_stub)
    functions = seen["position_snr_function"]
    assert len(functions) == 2 * 2
    for function in functions:
        function((3, 4), box_size=11)
    assert sorted(calls) == [("first", (3, 4), 11)] * 2 + [("second", (3, 4), 11)] * 2


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
