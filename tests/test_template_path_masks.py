"""Known positions, noise masks and the channel correlation along the detection entry points."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.io import fits

import trap.detection
from trap.detection import DetectionAnalysis
from trap.noise_mask import companion_noise_masks
from trap.parameters import TrapReductionConfig

FWHM = 4.0
ROTATION = 0.4
CONFIGURED = (30.0, 0.0)
NOISE_BLOB = (0.0, 40.0)
SOURCE_PIXEL = (100, 160)


class _Template:
    def __init__(self, name, use_spectral_correlation=False):
        self.name = name
        self.use_spectral_correlation = use_spectral_correlation
        self.companion_table = None
        self.validated_companion_table = None
        self.validated_companion_table_short = None


def _fit_table(positions):
    """Fit results of round candidates (axis ratio 1), aligned with their own position angle."""
    positions = np.asarray(positions, dtype=float)
    separation = np.hypot(positions[:, 0], positions[:, 1])
    position_angle = np.degrees(np.arctan2(positions[:, 1], positions[:, 0])) % 360
    n = len(positions)
    snr_image = pd.DataFrame(
        {
            "x": positions[:, 1] + 100,
            "y": positions[:, 0] + 100,
            "x_relative": positions[:, 1],
            "x_relative_sigma": 0.1,
            "y_relative": positions[:, 0],
            "y_relative_sigma": 0.1,
            "separation": separation,
            "separation_sigma": 0.1,
            "position_angle": position_angle,
            "position_angle_sigma": 0.1,
            "xy_relative_corr": 0.0,
            "radial_sigma_stat": 0.1,
            "tangential_sigma_stat": 0.1,
            "channels_above_threshold": 1,
            "theta_free": position_angle,
            "x_fwhm": FWHM,
            "y_fwhm": FWHM,
            "fwhm_area": 1.0,
            "x_fwhm_free": np.full(n, FWHM),
            "y_fwhm_free": np.full(n, FWHM),
            "good_fraction": 0.9,
            "good_fraction_free": 0.9,
        }
    )
    norm_snr_image = pd.DataFrame({"amplitude": np.full(n, 7.0), "amplitude_free": np.full(n, 7.0)})
    candidates = pd.DataFrame({"y_relative": positions[:, 0], "x_relative": positions[:, 1], "snr": np.full(n, 7.0)})
    return candidates, {"snr_image": snr_image, "norm_snr_image": norm_snr_image}


def _spectra(n):
    return pd.DataFrame(
        {
            "candidate_id": np.arange(n),
            "wavelength_index": 0,
            "wavelength": 1.0,
            "contrast": 1e-5,
            "uncertainty": 1e-6,
            "snr": 10.0,
        }
    )


def _stubbed_template_analysis(tmp_path, monkeypatch, configured):
    """An analysis whose template path runs everything up to the tables on two round candidates."""
    config = TrapReductionConfig(result_folder=str(tmp_path), yx_known_companion_position=configured)
    analysis = DetectionAnalysis(reduction_parameters=config)
    analysis.instrument = SimpleNamespace(wavelengths=np.array([1.0]) * u.micron, fwhm=np.array([FWHM]) * u.pixel)
    analysis.wavelength_indices = np.array([0])
    analysis.detection_cube = np.ones((1, 3, 21, 21))
    analysis.file_paths = {}
    candidates, fit = _fit_table([NOISE_BLOB, CONFIGURED])
    seen = []

    def detection_summary(**kwargs):
        seen.append(kwargs["yx_known_companion_position"])
        return DetectionAnalysis.detection_summary(analysis, **kwargs)

    def add_default_templates(**kwargs):
        analysis.templates = {"first": _Template("first"), "second": _Template("second")}

    for name, value in {
        "add_default_templates": add_default_templates,
        "template_matching_detection": lambda *args, **kwargs: (np.empty(0), {}),
        "find_candidates_all_wavelengths": lambda **kwargs: candidates.copy(),
        "complete_candidate_table": lambda **kwargs: (candidates.copy(), fit),
        "contrast_table_and_normalization": lambda **kwargs: {},
        "extract_candidate_spectra": lambda **kwargs: _spectra(len(candidates)),
        "contrast_plot": lambda **kwargs: None,
        "plot_template_matched_contrasts": lambda: None,
        "measure_per_channel_astrometry": lambda **kwargs: None,
        "combine_template_matched_companion_tables": lambda **kwargs: None,
        "detection_summary": detection_summary,
    }.items():
        monkeypatch.setattr(analysis, name, value, raising=False)
    return analysis, seen


def _run_template_path(analysis, pa=None, reduction_parameters=None, use_spectral_correlation=False):
    analysis.detection_and_characterization_with_template_matching(
        reduction_parameters=reduction_parameters,
        instrument=None if reduction_parameters is None else analysis.instrument,
        species_database_directory=None,
        stellar_parameters=None,
        data_full=None,
        flux_psf_full=None,
        pa=pa,
        temporal_components_fraction=None,
        wavelength_indices=None,
        use_spectral_correlation=use_spectral_correlation,
    )


def _validation(template):
    table = template.companion_table
    return dict(zip(zip(table["y_relative"], table["x_relative"]), table["validation"]))


def test_template_path_keeps_the_configured_positions_through_all_templates(tmp_path, monkeypatch):
    analysis, seen = _stubbed_template_analysis(tmp_path, monkeypatch, configured=[list(CONFIGURED)])
    _run_template_path(analysis)

    assert len(seen) == 2
    for known in seen:
        np.testing.assert_array_equal(np.atleast_2d(known), [CONFIGURED])
    for template in analysis.templates.values():
        validation = _validation(template)
        assert validation[NOISE_BLOB] == "yx_fwhm_ratio"
        assert validation[CONFIGURED] == "shape_skipped_known"
        assert set(zip(template.validated_companion_table["y_relative"], template.validated_companion_table["x_relative"])) == {
            CONFIGURED
        }


def test_template_path_takes_the_configured_positions_from_the_given_config(tmp_path, monkeypatch):
    analysis, seen = _stubbed_template_analysis(tmp_path, monkeypatch, configured=None)
    config = TrapReductionConfig(result_folder=str(tmp_path), yx_known_companion_position=[list(CONFIGURED)])
    _run_template_path(analysis, reduction_parameters=config)
    np.testing.assert_array_equal(np.atleast_2d(seen[0]), [CONFIGURED])
    assert _validation(analysis.templates["second"])[NOISE_BLOB] == "yx_fwhm_ratio"


def test_detection_and_characterization_passes_the_configured_positions(tmp_path, monkeypatch):
    analysis, seen = _stubbed_template_analysis(tmp_path, monkeypatch, configured=[list(CONFIGURED)])
    candidates, fit = _fit_table([NOISE_BLOB, CONFIGURED])
    monkeypatch.setattr(analysis, "complete_candidate_table", lambda **kwargs: (candidates, fit))
    analysis.detection_and_characterization()
    analysis.detection_and_characterization()

    assert len(seen) == 2
    for known in seen:
        np.testing.assert_array_equal(np.atleast_2d(known), [CONFIGURED])
    assert _validation(analysis)[NOISE_BLOB] == "yx_fwhm_ratio"


def test_read_output_stores_the_configured_positions():
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig(yx_known_companion_position=[[1.0, 2.0]]))
    analysis.reduction_parameters = analysis.reduction_parameters.merge(yx_known_companion_position=[[5.0, 5.0]])
    np.testing.assert_array_equal(analysis.configured_known_positions, [[1.0, 2.0]])

    lazy = DetectionAnalysis()
    lazy.reduction_parameters = TrapReductionConfig(yx_known_companion_position=[[3.0, 4.0]])
    np.testing.assert_array_equal(lazy._configured_known_positions(), [[3.0, 4.0]])


def _footprint_cube():
    from test_noise_mask_normalization import _detection_image, _footprint_map

    return np.array([_detection_image(_footprint_map()), _detection_image(_footprint_map(peak=6.0))])


def _mask_analysis(tmp_path, monkeypatch, configured):
    config = TrapReductionConfig(result_folder=str(tmp_path), yx_known_companion_position=configured)
    analysis = DetectionAnalysis(reduction_parameters=config)
    analysis.instrument = SimpleNamespace(
        pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel),
        telescope_diameter=8.2 * u.m,
        wavelengths=np.array([1.0, 1.1]) * u.micron,
        fwhm=np.array([FWHM, FWHM]) * u.pixel,
    )
    analysis.wavelength_indices = np.array([0, 1])
    analysis.detection_cube = _footprint_cube()
    analysis.file_paths = {
        key: str(tmp_path / f"{key}.fits")
        for key in (
            "norm_detection_image_path",
            "uncertainty_image_path",
            "median_uncertainty_image_path",
            "contrast_table_path",
            "noise_mask_path",
            "spectral_correlation_path",
        )
    }
    calls = {"stack": 0, "matrices": [], "correlation_seen": []}

    original_stack = DetectionAnalysis._spectral_correlation_stack
    original_matrices = DetectionAnalysis.make_spectral_correlation_matrices

    def counting_stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(analysis, *args, **kwargs)

    def recording_matrices(**kwargs):
        calls["matrices"].append(kwargs)
        return original_matrices(analysis, **kwargs)

    def add_default_templates(**kwargs):
        analysis.templates = {
            "first": _Template("first", use_spectral_correlation=True),
            "second": _Template("second", use_spectral_correlation=True),
        }

    def run_template_matching(template, **kwargs):
        calls["correlation_seen"].append(analysis.empirical_correlation)
        # A previous template used to leave its masks in detection_products.
        analysis.detection_products = {}

    for name, value in {
        "_spectral_correlation_stack": counting_stack,
        "make_spectral_correlation_matrices": recording_matrices,
        "add_default_templates": add_default_templates,
        "run_template_matching": run_template_matching,
        "plot_template_matched_contrasts": lambda: None,
        "measure_per_channel_astrometry": lambda **kwargs: None,
        "combine_template_matched_companion_tables": lambda **kwargs: None,
    }.items():
        monkeypatch.setattr(analysis, name, value)
    return analysis, calls


def test_template_path_saves_the_mask_of_the_configured_source_and_builds_the_stack_once(tmp_path, monkeypatch):
    analysis, calls = _mask_analysis(tmp_path, monkeypatch, configured=[[0.0, 60.0]])
    _run_template_path(analysis, pa=np.degrees([0.0, ROTATION]), use_spectral_correlation=True)

    assert len(calls["matrices"]) == 1
    assert calls["matrices"][0]["detected_signal_mask"][SOURCE_PIXEL]
    first, second = calls["correlation_seen"]
    assert first is second and first is not None
    # One stack for template matching and one for the saved file.
    assert calls["stack"] == 2

    with fits.open(analysis.file_paths["noise_mask_path"]) as hdul:
        mask = hdul[0].data.astype(bool)
        sources = hdul["SOURCES"].data
        assert hdul[0].header["NMASKRUL"] == "region"
    assert mask[0][SOURCE_PIXEL] and mask[1][SOURCE_PIXEL]
    assert list(sources["mask_rule"]) == ["region", "disk"]
    assert (tmp_path / "spectral_correlation_path.fits").exists()


def test_template_correlation_masks_a_bright_source_that_is_not_configured(tmp_path, monkeypatch):
    analysis, calls = _mask_analysis(tmp_path, monkeypatch, configured=None)
    _run_template_path(analysis, pa=np.degrees([0.0, ROTATION]), use_spectral_correlation=True)

    assert calls["matrices"][0]["detected_signal_mask"][SOURCE_PIXEL]
    assert not hasattr(analysis, "candidates")
    with fits.open(analysis.file_paths["noise_mask_path"]) as hdul:
        assert not hdul[0].data.any()
        assert len(hdul["SOURCES"].data) == 0


def test_template_matching_reuses_a_given_correlation(monkeypatch):
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig())
    analysis.instrument = SimpleNamespace(wavelengths=np.array([1.0, 1.1]) * u.micron)
    analysis.wavelength_indices = np.array([0, 1])
    rng = np.random.default_rng(0)
    analysis.detection_cube = np.stack([rng.normal(size=(2, 9, 9)), np.ones((2, 9, 9)), np.ones((2, 9, 9))], axis=1)
    analysis.detection_products = {"uncertainty_cube": np.ones((2, 9, 9))}
    analysis.empirical_correlation = {
        "matrices": np.identity(2)[None],
        "fit": pd.DataFrame({"separation": [3.0], "fitted": [False], "flagged": [False]}),
        "wavelengths": np.array([1.0, 1.1]),
    }

    def fail(**kwargs):
        raise AssertionError("the correlation was recomputed")

    monkeypatch.setattr(analysis, "make_spectral_correlation_matrices", fail)
    monkeypatch.setattr(analysis, "contrast_table_and_normalization", lambda **kwargs: {})
    template = SimpleNamespace(
        name="flat",
        fit_offset=False,
        fit_slope=False,
        use_spectral_correlation=True,
        number_of_pca_regressors=0,
        normalized_contrast_modelbox=SimpleNamespace(flux=np.ones(2)),
    )
    cube, _ = analysis.template_matching_detection(template, file_paths={}, save=False)
    assert np.isfinite(cube[0, 2]).any()


def test_extract_candidate_spectra_sets_the_field_rotation(monkeypatch):
    analysis = DetectionAnalysis()
    seen = []
    monkeypatch.setattr(
        analysis, "rereduce_single_position", lambda **kwargs: seen.append(analysis.field_rotation) or pd.DataFrame()
    )
    analysis.extract_candidate_spectra(
        temporal_components_fraction=0.1,
        data_full=None,
        flux_psf_full=None,
        pa=np.array([10.0, 30.0]),
        yx_candidate_positions=[[0.0, 20.0]],
    )
    assert seen == [pytest.approx(np.radians(20.0))]


def test_rereduce_single_position_sets_the_field_rotation(monkeypatch):
    analysis = DetectionAnalysis()
    analysis.reduction_parameters = SimpleNamespace(merge=lambda **kwargs: None)
    analysis.detection_products = {}

    class _Stop(Exception):
        pass

    def stop(**kwargs):
        raise _Stop

    monkeypatch.setattr(trap.detection, "run_complete_reduction", stop)
    with pytest.raises(_Stop):
        analysis.rereduce_single_position(
            candidate_index=0,
            yx_candidate_position=(0.0, 20.0),
            data_full=np.empty(0),
            flux_psf_full=np.empty(0),
            pa=np.array([0.0, 40.0]),
            wavelength_indices=[0],
            temporal_components_fraction=0.1,
        )
    assert analysis.field_rotation == pytest.approx(np.radians(40.0))


def test_collapsed_map_uses_the_mean_fwhm_of_the_reduced_channels(tmp_path, monkeypatch):
    received = []

    def recording(*args, **kwargs):
        received.append(kwargs["fwhm"])
        return companion_noise_masks(*args, **kwargs)

    monkeypatch.setattr(trap.detection, "companion_noise_masks", recording)
    analysis, _ = _mask_analysis(tmp_path, monkeypatch, configured=[[0.0, 60.0]])
    analysis.instrument.fwhm = np.array([4.0, 9.0, 6.0]) * u.pixel
    analysis.wavelength_indices = np.array([0, 2])

    analysis.contrast_table_and_normalization()
    assert received == [4.0, 6.0]
    analysis.contrast_table_and_normalization(detection_cube=analysis.detection_cube[:1].copy(), cube_indices=[0], inplace=False)
    assert received[-1] == pytest.approx(5.0)
