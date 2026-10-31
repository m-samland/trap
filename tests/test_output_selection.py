"""Selecting the detection files of exactly one reduction run.

A result folder can hold several runs: temporal fractions, the spatial model after TRAP
at several spatial fractions, the spatial model, correlated-residual and injected-fake
outputs. `detection_file_paths` parses the existing names exactly, returns one file per
wavelength for the run asked for, and refuses to stack files from different runs.
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from trap.detection import DetectionAnalysis, detection_file_paths
from trap.parameters import trap_config_for_irdis
from trap.reduction_wrapper import run_name
from trap.utils import save_object


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


def test_run_name_drops_the_wavelength_and_adds_the_model():
    assert run_name("lam00_ncomp051_frac0.20", "temporal") == "ncomp051_frac0.20_temporal"
    assert run_name("run1_lam00_ncomp051_frac0.20", "temporal") == "run1_ncomp051_frac0.20_temporal"
    assert run_name("lam00_delta0.50_spatialfrac0.30", "spatial") == "delta0.50_spatialfrac0.30_spatial"


def _folder_with_shared_objects(folder):
    config = trap_config_for_irdis()
    instrument = config.get_instrument("DB_K12", wavelengths=np.array([2.11, 2.25]) * u.micron)
    save_object(instrument, os.path.join(folder, "instrument.obj"))
    save_object(config.reduction.merge(annulus_width=5), os.path.join(folder, "reduction_config.obj"))
    fits.writeto(os.path.join(folder, "detection_lam00_ncomp051_frac0.20_temporal.fits"), np.ones((3, 5, 5)))
    return config


def test_read_output_prefers_the_run_objects(tmp_path):
    config = _folder_with_shared_objects(tmp_path)
    save_object(config.reduction.merge(annulus_width=7), os.path.join(tmp_path, "reduction_config_ncomp051_frac0.20_temporal.obj"))
    analysis = DetectionAnalysis()
    analysis.read_output(0.2, result_folder=str(tmp_path))
    assert analysis.reduction_parameters.annulus_width == 7


def test_legacy_folder_falls_back_to_shared_objects(tmp_path):
    _folder_with_shared_objects(tmp_path)
    analysis = DetectionAnalysis()
    analysis.read_output(0.2, result_folder=str(tmp_path))
    assert analysis.reduction_parameters.annulus_width == 5


def test_prefixed_run_finds_its_own_objects(tmp_path):
    config = trap_config_for_irdis()
    instrument = config.get_instrument("DB_K12", wavelengths=np.array([2.11, 2.25]) * u.micron)
    save_object(instrument, os.path.join(tmp_path, "instrument.obj"))
    save_object(config.reduction.merge(prefix="run1_", annulus_width=5), os.path.join(tmp_path, "reduction_config.obj"))
    fits.writeto(os.path.join(tmp_path, "detection_run1_lam00_ncomp051_frac0.20_temporal.fits"), np.ones((3, 5, 5)))
    name = run_name("run1_lam00_ncomp051_frac0.20", "temporal")
    save_object(config.reduction.merge(prefix="run1_", annulus_width=9), os.path.join(tmp_path, f"reduction_config_{name}.obj"))
    analysis = DetectionAnalysis()
    analysis.read_output(0.2, result_folder=str(tmp_path))
    assert analysis.reduction_parameters.annulus_width == 9


def test_component_counts_of_any_length_are_found(tmp_path):
    name = "detection_lam00_ncomp1000_frac0.20_temporal.fits"
    _touch(tmp_path, name)
    assert _names(detection_file_paths(tmp_path, "temporal", 0.2)) == [name]


def test_read_output_takes_an_explicit_prefix(tmp_path):
    """The shared config holds the prefix of whichever run wrote it last."""
    config = trap_config_for_irdis()
    instrument = config.get_instrument("DB_K12", wavelengths=np.array([2.11, 2.25]) * u.micron)
    save_object(instrument, os.path.join(tmp_path, "instrument.obj"))
    save_object(config.reduction.merge(prefix="B_"), os.path.join(tmp_path, "reduction_config.obj"))
    fits.writeto(os.path.join(tmp_path, "detection_lam00_ncomp051_frac0.20_temporal.fits"), np.ones((3, 5, 5)))
    fits.writeto(os.path.join(tmp_path, "detection_B_lam00_ncomp051_frac0.20_temporal.fits"), np.full((3, 5, 5), 2.0))
    analysis = DetectionAnalysis()
    analysis.read_output(0.2, result_folder=str(tmp_path), prefix="")
    assert np.nanmax(analysis.detection_cube) == 1.0
    analysis.read_output(0.2, result_folder=str(tmp_path))
    assert np.nanmax(analysis.detection_cube) == 2.0
