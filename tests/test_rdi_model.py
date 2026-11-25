"""The reference-star differential imaging (RDI) model at one tested position.

Each science frame is fitted with one basis of principal components from planet-free
reference frames, a constant and the companion model. With little field rotation the
spatial ADI model can only build its basis from frames that still contain the companion,
so it subtracts part of it; the reference frames carry no companion and the RDI model
recovers the injected contrast.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from trap import makesource, regression
from trap.detection import DetectionAnalysis
from trap.parameters import Instrument, ReductionRuntimeState, TrapReductionConfig, trap_config_for_irdis
from trap.reduction_wrapper import run_complete_reduction
from trap.regression import run_trap_with_model_rdi, run_trap_with_model_spatial

SIZE = 41
YX_CENTER = (20.0, 20.0)
POSITION = (6.0, 0.0)
CONTRAST = 2e-3
N_SCIENCE, N_REFERENCE = 40, 60


def _gaussian(shape, yx, sigma):
    y, x = np.indices(shape)
    return np.exp(-((y - yx[0]) ** 2 + (x - yx[1]) ** 2) / (2 * sigma**2))


def _speckle_field(rng):
    field = np.zeros((SIZE, SIZE))
    for _ in range(6):
        yx = np.array(YX_CENTER) + rng.uniform(-10, 10, 2)
        field += rng.uniform(0.3, 1.0) * _gaussian(field.shape, yx, rng.uniform(1.5, 3.0))
    return field / field.max() * 1e3


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(42)
    speckles = _speckle_field(rng)
    psf = 1e3 * _gaussian((19, 19), (9, 9), 3.0)
    pa = np.linspace(0.0, 6.0, N_SCIENCE)
    model = makesource.inject_model_into_data(
        flux_arr=np.zeros((N_SCIENCE, SIZE, SIZE)), pos=np.array(POSITION), pa=pa, psf_model=psf,
        image_center=np.array(YX_CENTER), norm=np.ones(N_SCIENCE), subpixel=True, remove_interpolation_artifacts=True)

    def frames(n):
        amplitudes = rng.uniform(0.9, 1.1, n)
        return amplitudes[:, None, None] * speckles + rng.standard_normal((n, SIZE, SIZE))

    data = frames(N_SCIENCE) + CONTRAST * model
    reference = frames(N_REFERENCE)
    radius = np.hypot(*(np.indices((SIZE, SIZE)) - np.array(YX_CENTER)[:, None, None]))
    mask = (radius >= 2) & (radius <= 14)
    return data, model, reference, pa, mask


@pytest.fixture
def config():
    return trap_config_for_irdis().reduction.merge(rdi_components_fraction=0.3)


@pytest.fixture
def runtime():
    return ReductionRuntimeState(yx_anamorphism=np.array([1.0, 1.0]), fwhm=7.0)


def _rdi(synthetic, config, runtime, reference=None):
    data, model, default_reference, _, mask = synthetic
    reference = default_reference if reference is None else reference
    return run_trap_with_model_rdi(
        data, model, reference, config, POSITION, mask, runtime=runtime, yx_center=YX_CENTER)


def test_rdi_recovers_injected_contrast_without_bias(synthetic, config, runtime):
    result = _rdi(synthetic, config, runtime)
    result.compute_contrast_weighted_average(mask_outliers=True)
    assert abs(result.measured_contrast - CONTRAST) < 3 * result.contrast_uncertainty
    assert result.measured_contrast == pytest.approx(CONTRAST, rel=0.05)


def test_spatial_adi_is_biased_low_on_the_same_data(synthetic, config, runtime):
    data, model, _, pa, mask = synthetic
    # The default protection angle leaves fewer than 3 frames at 6 deg of rotation; a small
    # one lets the spatial model run on frames that still contain the companion.
    spatial = run_trap_with_model_spatial(
        data, model, pa, config.merge(protection_angle=0.02), POSITION, mask, runtime=runtime, yx_center=YX_CENTER)
    rdi = _rdi(synthetic, config, runtime)
    rdi.compute_contrast_weighted_average(mask_outliers=True)
    assert spatial is not None
    spatial.compute_contrast_weighted_average(mask_outliers=True)
    assert spatial.measured_contrast < 0.9 * CONTRAST
    assert abs(rdi.measured_contrast - CONTRAST) < abs(spatial.measured_contrast - CONTRAST)


def test_number_of_components_is_the_fraction_of_reference_frames(synthetic, config, runtime, monkeypatch):
    requested = []
    original = regression._reference_basis

    def recording(training_matrix, n_components, scaling):
        requested.append(n_components)
        return original(training_matrix, n_components, scaling)

    monkeypatch.setattr(regression, "_reference_basis", recording)
    result = _rdi(synthetic, config, runtime)
    assert requested == [18]
    assert result.number_of_pca_regressors == 18


def test_number_of_components_is_capped_at_half_the_fitted_pixels(synthetic, config, runtime, monkeypatch, caplog):
    data, model, reference, _, mask = synthetic
    brightest = np.argsort(np.where(mask, model.sum(axis=0), -np.inf), axis=None)[-60:]
    small_mask = np.zeros_like(mask)
    small_mask.flat[brightest] = True
    requested = []
    original = regression._reference_basis

    def recording(training_matrix, n_components, scaling):
        requested.append(n_components)
        return original(training_matrix, n_components, scaling)

    monkeypatch.setattr(regression, "_reference_basis", recording)
    with caplog.at_level("DEBUG", logger=regression.logger.name):
        result = run_trap_with_model_rdi(data, model, reference, config.merge(rdi_components_fraction=0.9), POSITION,
                                         small_mask, runtime=runtime, yx_center=YX_CENTER)
    assert requested == [30]
    assert result.number_of_pca_regressors == 30
    assert "capped" in caplog.text


def test_reference_frames_with_nonfinite_pixels_in_the_mask_are_dropped(synthetic, config, runtime, monkeypatch):
    _, _, reference, _, mask = synthetic
    reference = reference.copy()
    y, x = np.argwhere(mask)[0]
    reference[:10, y, x] = np.nan
    requested = []
    original = regression._reference_basis

    def recording(training_matrix, n_components, scaling):
        requested.append((training_matrix.shape[1], n_components))
        return original(training_matrix, n_components, scaling)

    monkeypatch.setattr(regression, "_reference_basis", recording)
    result = _rdi(synthetic, config, runtime, reference=reference)
    assert requested == [(50, 15)]
    assert np.all(np.isfinite(result.reduced_result[:, 0]))


def test_too_few_finite_reference_frames_returns_none(synthetic, config, runtime):
    reference = synthetic[2].copy()
    reference[:58] = np.nan
    assert _rdi(synthetic, config, runtime, reference=reference) is None


WAVELENGTHS = np.array([1.6, 1.65]) * u.micron


@pytest.fixture(scope="module")
def cube(synthetic):
    data, _, reference, pa, _ = synthetic
    psf = 1e3 * _gaussian((21, 21), (10, 10), 3.0)
    return np.stack([data, 0.9 * data]), np.stack([psf, psf]), np.stack([reference, 0.9 * reference]), pa


def _instrument(n_wavelengths=2):
    return Instrument(
        name="synthetic", pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel), telescope_diameter=8.0 * u.m,
        detector_gain=1.0, readnoise=0.0, instrument_type="photometry", wavelengths=WAVELENGTHS[:n_wavelengths])


def _reduce(cube, result_folder=None, fractions=(0.2,), reference=None, wavelength_indices=None, **settings):
    data, psf, default_reference, pa = cube
    settings = {
        "search_region_inner_bound": 4, "search_region_outer_bound": 8, "data_auto_crop": False,
        "data_crop_size": None, "temporal_model": False, "rdi_model": True, "use_multiprocess": False,
        "ncpus": 2, "result_folder": None if result_folder is None else str(result_folder), **settings}
    return run_complete_reduction(
        data_full=data.copy(), flux_psf_full=psf.copy(), pa=pa.copy(), instrument=_instrument(data.shape[0]),
        reduction_parameters=TrapReductionConfig(**settings), temporal_components_fraction=list(fractions),
        reference_data_full=default_reference.copy() if reference is None else reference,
        wavelength_indices=wavelength_indices, overwrite=True, use_progress_bar=False)


@pytest.fixture
def rdi_calls(monkeypatch):
    calls = []
    original = regression.run_trap_with_model_rdi

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(regression, "run_trap_with_model_rdi", recording)
    return calls


def _rdi_map(folder, wavelength_index=0):
    files = sorted(folder.glob(f"detection_lam{wavelength_index:02d}_*_rdi.fits"))
    assert len(files) == 1, files
    return fits.getdata(files[0])


def test_rdi_reduction_writes_rdi_detection_maps(cube, tmp_path):
    _reduce(cube, tmp_path)
    assert (tmp_path / "detection_lam00_rdifrac0.30_rdi.fits").exists()
    assert (tmp_path / "detection_lam01_rdifrac0.30_rdi.fits").exists()
    contrast = _rdi_map(tmp_path)[0]
    center = contrast.shape[-1] // 2
    assert np.isfinite(contrast).sum() > 100
    assert contrast[center + int(POSITION[0]), center + int(POSITION[1])] == pytest.approx(CONTRAST, rel=0.2)


def test_rdi_without_reference_is_refused(cube, rdi_calls):
    data, psf, _, pa = cube
    with pytest.raises(ValueError, match="reference"):
        run_complete_reduction(
            data_full=data.copy(), flux_psf_full=psf.copy(), pa=pa.copy(), instrument=_instrument(),
            reduction_parameters=TrapReductionConfig(rdi_model=True, reduce_single_position=True, guess_position=POSITION))
    assert rdi_calls == []


def test_reference_shape_mismatch_is_refused(cube, rdi_calls):
    reference = cube[2]
    with pytest.raises(ValueError, match=r"\(1, 60, 41, 41\).*\(2, 40, 41, 41\)"):
        _reduce(cube, reference=reference[:1])
    padded = np.pad(reference, ((0, 0), (0, 0), (1, 1), (1, 1)))
    with pytest.raises(ValueError, match=r"\(2, 60, 43, 43\).*\(2, 40, 41, 41\)"):
        _reduce(cube, reference=padded)
    assert rdi_calls == []


def test_bad_frames_leave_the_reference_untouched(cube, rdi_calls):
    data, psf, reference, pa = cube
    run_complete_reduction(
        data_full=data.copy(), flux_psf_full=psf.copy(), pa=pa.copy(), instrument=_instrument(),
        reduction_parameters=TrapReductionConfig(
            temporal_model=False, rdi_model=True, data_auto_crop=False, reduce_single_position=True,
            guess_position=POSITION),
        bad_frames=[0, 5], reference_data_full=reference.copy(), wavelength_indices=[0], use_progress_bar=False)
    assert len(rdi_calls) == 1
    assert rdi_calls[0]["data"].shape[0] == N_SCIENCE - 2
    np.testing.assert_array_equal(rdi_calls[0]["reference_data"], reference[0])


def test_injection_goes_into_science_only(cube, rdi_calls):
    data, psf, reference, pa = cube
    results = run_complete_reduction(
        data_full=data.copy(), flux_psf_full=psf.copy(), pa=pa.copy(), instrument=_instrument(),
        reduction_parameters=TrapReductionConfig(
            temporal_model=False, rdi_model=True, data_auto_crop=False, reduce_single_position=True,
            guess_position=(-6.0, 0.0), inject_fake=True, true_contrast=1e-2),
        reference_data_full=reference.copy(), wavelength_indices=[0], use_progress_bar=False)
    assert len(rdi_calls) == 1
    np.testing.assert_array_equal(rdi_calls[0]["reference_data"], reference[0])
    assert not np.array_equal(rdi_calls[0]["data"], data[0])
    assert results["0.2"]["0"]["rdi"].measured_contrast == pytest.approx(1e-2, rel=0.2)


def test_nan_reference_pixels_shrink_only_overlapping_reduction_masks(cube, tmp_path, monkeypatch):
    data, _, reference, _ = cube
    center = data.shape[-1] // 2
    disc = np.hypot(*(np.indices(data.shape[-2:]) - np.array([center + 6, center])[:, None, None])) <= 1.5
    nan_reference = reference.copy()
    nan_reference[:, :, disc] = np.nan

    pixels = {}
    original = regression.run_trap_with_model_rdi

    def recording(*args, **kwargs):
        mask = kwargs["reduction_mask"]
        assert np.isfinite(kwargs["reference_data"][:, mask]).all()
        pixels.setdefault(run, {})[tuple(kwargs["planet_relative_yx_pos"])] = (int(mask.sum()), bool(mask[disc].any()))
        return original(*args, **kwargs)

    monkeypatch.setattr(regression, "run_trap_with_model_rdi", recording)
    for run, reference_cube in (("clean", reference), ("nan", nan_reference)):
        (tmp_path / run).mkdir()
        # The 36-pixel masks keep more than 20 pixels after losing the 9-pixel disc.
        _reduce(cube, tmp_path / run, reference=reference_cube, wavelength_indices=[0], reduction_mask_min_pixels=20)

    assert pixels["clean"].keys() == pixels["nan"].keys()
    overlapping = [position for position, (_, overlaps) in pixels["clean"].items() if overlaps]
    assert (6, 0) in overlapping
    assert 0 < len(overlapping) < len(pixels["clean"])
    for position, (n_clean, overlaps) in pixels["clean"].items():
        n_nan, overlaps_nan = pixels["nan"][position]
        assert not overlaps_nan
        assert n_nan < n_clean if overlaps else n_nan == n_clean

    contrast = _rdi_map(tmp_path / "nan")[0]
    np.testing.assert_array_equal(np.isfinite(contrast), np.isfinite(_rdi_map(tmp_path / "clean")[0]))
    assert np.isfinite(contrast[center + 6, center])


def test_rdi_runs_once_for_several_temporal_fractions(cube, rdi_calls):
    # The rdi file name has no temporal fraction, so a second run would only overwrite it; count the fits instead.
    results = _reduce(
        cube, fractions=(0.1, 0.2), temporal_model=True, wavelength_indices=[0], reduce_single_position=True,
        guess_position=POSITION)
    assert len(rdi_calls) == 1
    assert "rdi" in results["0.1"]["0"]
    assert "rdi" not in results["0.2"]["0"]
    assert "temporal" in results["0.2"]["0"]


def test_parallel_and_serial_rdi_maps_agree(cube, tmp_path):
    for folder, use_multiprocess in (("serial", False), ("parallel", True)):
        (tmp_path / folder).mkdir()
        _reduce(cube, tmp_path / folder, use_multiprocess=use_multiprocess)
    for wavelength_index in (0, 1):
        np.testing.assert_allclose(
            _rdi_map(tmp_path / "parallel", wavelength_index), _rdi_map(tmp_path / "serial", wavelength_index),
            rtol=1e-10, atol=0.0, equal_nan=True)


def _rdi_analysis(folder):
    analysis = DetectionAnalysis()
    analysis.read_output(0.3, result_folder=str(folder), reduction_type="rdi")
    return analysis


def test_read_output_reads_rdi_maps(cube, tmp_path):
    _reduce(cube, tmp_path)
    analysis = _rdi_analysis(tmp_path)
    assert list(analysis.wavelength_indices) == [0, 1]
    assert analysis.detection_cube.shape[0] == 2
    assert analysis.reduction_parameters.rdi_model


def _rereduce(analysis, cube, reference):
    data, psf, _, pa = cube
    return analysis.rereduce_single_position(
        candidate_index=0, yx_candidate_position=POSITION, data_full=data, flux_psf_full=psf, pa=pa, wavelength_indices=[0, 1],
        temporal_components_fraction=0.2, reference_data_full=reference)


def test_rereduce_single_position_with_rdi_recovers_the_spectrum(cube, tmp_path, rdi_calls):
    _reduce(cube, tmp_path)
    analysis = _rdi_analysis(tmp_path)
    analysis.contrast_table_and_normalization(save=False)
    rdi_calls.clear()
    spectrum = _rereduce(analysis, cube, cube[2])
    assert len(rdi_calls) == 2
    # The cube fixture scales the second channel's science and reference by 0.9 but not its PSF.
    true_contrast = np.array([CONTRAST, 0.9 * CONTRAST])
    assert np.all(np.abs(spectrum.contrast - true_contrast) < 3 * spectrum.original_unc)
    center = analysis.detection_cube.shape[-1] // 2
    map_contrast = analysis.detection_cube[:, 0, center + int(POSITION[0]), center + int(POSITION[1])]
    np.testing.assert_allclose(spectrum.contrast, map_contrast, rtol=1e-6)


def test_rdi_rereduction_without_reference_is_refused(cube, tmp_path, rdi_calls):
    _reduce(cube, tmp_path)
    analysis = _rdi_analysis(tmp_path)
    analysis.contrast_table_and_normalization(save=False)
    rdi_calls.clear()
    with pytest.raises(ValueError, match="reference"):
        _rereduce(analysis, cube, None)
    assert rdi_calls == []
