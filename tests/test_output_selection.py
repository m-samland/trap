"""Selecting the detection files of exactly one reduction run.

A result folder can hold several runs: temporal fractions, the spatial model after TRAP
at several spatial fractions, the spatial model, correlated-residual and injected-fake
outputs. `detection_file_paths` parses the existing names exactly, returns one file per
wavelength for the run asked for, and refuses to stack files from different runs.
"""

from pathlib import Path

import pytest

from trap.detection import detection_file_paths


def _touch(folder, *names):
    for name in names:
        Path(folder, name).write_text("")


def _names(paths):
    return [Path(path).name for path in paths]


def test_temporal_names_are_selected_as_before(tmp_path):
    _touch(tmp_path, "detection_lam01_ncomp051_frac0.20_temporal.fits", "detection_lam00_ncomp051_frac0.20_temporal.fits")
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2)) == [
        "detection_lam00_ncomp051_frac0.20_temporal.fits",
        "detection_lam01_ncomp051_frac0.20_temporal.fits",
    ]


def test_combined_model_is_found_by_its_temporal_fraction(tmp_path):
    name = "detection_lam00_ncomp051_frac0.20_delta0.50_spatialfrac0.10_temporal_plus_spatial.fits"
    _touch(tmp_path, name)
    assert _names(detection_file_paths(tmp_path, "temporal_plus_spatial", 0.2)) == [name]


def test_selects_one_run_among_several(tmp_path):
    wanted = "detection_lam00_ncomp051_frac0.20_delta0.50_spatialfrac0.10_temporal_plus_spatial.fits"
    _touch(
        tmp_path,
        "detection_lam00_ncomp026_frac0.10_temporal.fits",
        "detection_lam00_ncomp051_frac0.20_temporal.fits",
        "detection_lam00_ncomp026_frac0.10_delta0.50_spatialfrac0.10_temporal_plus_spatial.fits",
        wanted,
        "detection_lam00_ncomp051_frac0.20_delta0.50_spatialfrac0.20_temporal_plus_spatial.fits",
    )
    assert _names(detection_file_paths(tmp_path, "temporal_plus_spatial", 0.2, spatial_component_fraction=0.1)) == [wanted]
    with pytest.raises(ValueError, match="spatial_component_fraction"):
        detection_file_paths(tmp_path, "temporal_plus_spatial", 0.2)
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2)) == ["detection_lam00_ncomp051_frac0.20_temporal.fits"]


def test_spatial_model_is_found_by_its_spatial_fraction(tmp_path):
    name = "detection_lam00_delta0.50_spatialfrac0.30_spatial.fits"
    _touch(tmp_path, name)
    assert _names(detection_file_paths(tmp_path, "spatial", 0.3)) == [name]


def test_prefixed_outputs_are_found(tmp_path):
    name = "detection_run1_lam00_ncomp051_frac0.20_temporal.fits"
    _touch(tmp_path, name)
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2, prefix="run1_")) == [name]
    with pytest.raises(FileNotFoundError):
        detection_file_paths(tmp_path, "temporal", 0.2)


def test_corr_and_injected_files_are_not_mixed_in(tmp_path):
    plain = "detection_lam00_ncomp051_frac0.20_temporal.fits"
    corr = "detection_corr_lam00_ncomp051_frac0.20_temporal.fits"
    _touch(tmp_path, plain, corr, "detection_injectedsigma5.00_lam00_ncomp051_frac0.20_temporal.fits")
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2)) == [plain]
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2, correlated_residuals=True)) == [corr]


def test_duplicate_wavelength_is_an_error(tmp_path):
    first = "detection_lam00_ncomp051_frac0.20_temporal.fits"
    second = "detection_lam00_ncomp052_frac0.20_temporal.fits"
    _touch(tmp_path, first, second)
    with pytest.raises(ValueError) as error:
        detection_file_paths(tmp_path, "temporal", 0.2)
    assert first in str(error.value) and second in str(error.value)


def test_nothing_found_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        detection_file_paths(tmp_path, "temporal_plus_spatial", 0.2)
    assert str(tmp_path) in str(error.value) and "temporal_plus_spatial" in str(error.value)
