"""Base positions of the template path: independent of the template order, configured positions always masked."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.io import fits

import trap.detection
from trap.detection import DetectionAnalysis
from trap.parameters import TrapReductionConfig

N = 81
FWHM = 4.0
FIRST_SOURCE = (0.0, 25.0)
SECOND_SOURCE = (0.0, -25.0)
CONFIGURED = (25.0, 0.0)
FILE_KEYS = (
    "norm_detection_image_path",
    "uncertainty_image_path",
    "median_uncertainty_image_path",
    "contrast_table_path",
    "noise_mask_path",
    "spectral_correlation_path",
)


def _gaussian(position, peak):
    yy, xx = np.indices((N, N), dtype=float) - N // 2
    sigma = FWHM / 2.355
    return peak * np.exp(-((yy - position[0]) ** 2 + (xx - position[1]) ** 2) / (2 * sigma**2))


def _channel(seed, sources):
    snr = np.random.default_rng(seed).normal(size=(N, N))
    for position in sources:
        snr += _gaussian(position, 30.0)
    return np.stack([snr, np.ones_like(snr), snr])


def _template(name, flux):
    return SimpleNamespace(
        name=name,
        fit_offset=False,
        fit_slope=False,
        use_spectral_correlation=False,
        number_of_pca_regressors=0,
        normalized_contrast_modelbox=SimpleNamespace(flux=np.asarray(flux, dtype=float)),
        companion_table=None,
        validated_companion_table=None,
        validated_companion_table_short=None,
    )


def _reduction(**kwargs):
    reduction_parameters = kwargs["reduction_parameters"]
    channel = SimpleNamespace(measured_contrast=1e-4, contrast_uncertainty=1e-5)
    return {
        str(kwargs["temporal_components_fraction"]): {
            str(index): {"temporal": channel} for index in kwargs["wavelength_indices"]
        },
        "position": reduction_parameters.guess_position,
    }


def _analysis(tmp_path, monkeypatch, template_order, configured=None, extra_sources=()):
    """Two channels, each with its own source, and one template per channel."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = TrapReductionConfig(
        result_folder=str(tmp_path), yx_known_companion_position=configured, shape_validation_max_snr=5.0
    )
    analysis = DetectionAnalysis(reduction_parameters=config)
    analysis.instrument = SimpleNamespace(
        pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel),
        telescope_diameter=8.2 * u.m,
        wavelengths=np.array([1.0, 1.1]) * u.micron,
        fwhm=np.array([FWHM, FWHM]) * u.pixel,
    )
    analysis.wavelength_indices = np.array([0, 1])
    analysis.reduction_type = "temporal"
    analysis.detection_cube = np.array(
        [_channel(0, [FIRST_SOURCE, *extra_sources]), _channel(1, [SECOND_SOURCE, *extra_sources])]
    )
    analysis.file_paths = {key: str(tmp_path / f"{key}.fits") for key in FILE_KEYS}
    templates = {"first": _template("first", [1.0, 0.0]), "second": _template("second", [0.0, 1.0])}

    def add_default_templates(**kwargs):
        analysis.templates = {name: templates[name] for name in template_order}

    monkeypatch.setattr(trap.detection, "run_complete_reduction", _reduction)
    for name, value in {
        "add_default_templates": add_default_templates,
        "plot_template_matched_contrasts": lambda: None,
        "measure_per_channel_astrometry": lambda **kwargs: None,
        "combine_template_matched_companion_tables": lambda **kwargs: None,
        "contrast_plot": lambda **kwargs: None,
    }.items():
        monkeypatch.setattr(analysis, name, value)
    return analysis


def _run(analysis):
    analysis.detection_and_characterization_with_template_matching(
        reduction_parameters=None,
        instrument=None,
        species_database_directory=None,
        stellar_parameters=None,
        data_full=np.zeros(1),
        flux_psf_full=np.zeros(1),
        pa=np.array([0.0, 20.0]),
        temporal_components_fraction=0.1,
        wavelength_indices=[0, 1],
        yx_fwhm_ratio_threshold=[0.5, 4.5],
    )


def _sorted(table):
    return table.sort_values(["y_relative", "x_relative", "wavelength_index"]).reset_index(drop=True)


def test_template_results_do_not_depend_on_the_template_order(tmp_path, monkeypatch):
    forward = _analysis(tmp_path / "forward", monkeypatch, ["first", "second"])
    _run(forward)
    backward = _analysis(tmp_path / "backward", monkeypatch, ["second", "first"])
    _run(backward)

    for name, source in (("first", FIRST_SOURCE), ("second", SECOND_SOURCE)):
        validated = forward.templates[name].validated_companion_table
        assert validated is not None and len(validated)
        assert set(zip(validated["y_relative"].round(), validated["x_relative"].round())) == {source}
        pd.testing.assert_frame_equal(_sorted(validated), _sorted(backward.templates[name].validated_companion_table))
        pd.testing.assert_frame_equal(
            _sorted(forward.templates[name].companion_table), _sorted(backward.templates[name].companion_table)
        )


def _contains(positions, position):
    return any(np.allclose(row, position) for row in np.atleast_2d(positions))


def test_configured_position_is_masked_from_the_first_normalisation_on(tmp_path, monkeypatch):
    analysis = _analysis(
        tmp_path, monkeypatch, ["first", "second"], configured=[list(CONFIGURED)], extra_sources=[CONFIGURED]
    )
    calls = []
    original = DetectionAnalysis.contrast_table_and_normalization

    def recording(**kwargs):
        calls.append(kwargs.get("yx_known_companion_position"))
        return original(analysis, **kwargs)

    monkeypatch.setattr(analysis, "contrast_table_and_normalization", recording)
    _run(analysis)

    # The first call is the initial normalisation whose uncertainty map feeds template matching.
    np.testing.assert_array_equal(calls[0], [CONFIGURED])
    assert calls and all(positions is not None and _contains(positions, CONFIGURED) for positions in calls)
    with fits.open(analysis.file_paths["noise_mask_path"]) as hdul:
        sources = hdul["SOURCES"].data
        assert _contains(np.column_stack([sources["y_relative"], sources["x_relative"]]), CONFIGURED)


def test_final_save_masks_nothing_without_configured_or_validated_positions(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, monkeypatch, ["first"])

    def run_template_matching(template, **kwargs):
        # The merge of a template's raw candidates into the config, which the final save used to read.
        analysis.reduction_parameters = analysis.reduction_parameters.merge(
            yx_known_companion_position=[list(FIRST_SOURCE)]
        )

    monkeypatch.setattr(analysis, "run_template_matching", run_template_matching)
    _run(analysis)

    with fits.open(analysis.file_paths["noise_mask_path"]) as hdul:
        assert not hdul[0].data.any()
        assert len(hdul["SOURCES"].data) == 0


def test_known_and_validated_positions_is_empty_without_positions():
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig())
    assert analysis._known_and_validated_positions(None).shape == (0, 2)
    np.testing.assert_array_equal(analysis._known_and_validated_positions([[1.0, 2.0]]), [[1.0, 2.0]])


@pytest.mark.parametrize("base_positions, expected", [(None, [[5.0, 5.0]]), (np.empty((0, 2)), np.empty((0, 2)))])
def test_base_positions_default_to_the_config(base_positions, expected):
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig(yx_known_companion_position=[[5.0, 5.0]]))
    np.testing.assert_array_equal(analysis._base_positions(base_positions), expected)


def test_configured_positions_follow_a_changed_config():
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig(yx_known_companion_position=[[5.0, 5.0]]))
    analysis.reduction_parameters = analysis.reduction_parameters.merge(yx_known_companion_position=[[7.0, 7.0]])
    np.testing.assert_array_equal(analysis._configured_known_positions(), [[7.0, 7.0]])


def test_configured_positions_survive_candidates_written_into_the_config():
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig(yx_known_companion_position=[[5.0, 5.0]]))
    analysis._configured_known_positions()
    analysis.reduction_parameters = analysis.reduction_parameters.merge(yx_known_companion_position=[[9.0, 9.0]])
    analysis._config_holds_candidates = True
    np.testing.assert_array_equal(analysis._configured_known_positions(), [[5.0, 5.0]])
