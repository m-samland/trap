"""The reference-star differential imaging (RDI) model at one tested position.

Each science frame is fitted with one basis of principal components from planet-free
reference frames, a constant and the companion model. With little field rotation the
spatial ADI model can only build its basis from frames that still contain the companion,
so it subtracts part of it; the reference frames carry no companion and the RDI model
recovers the injected contrast.
"""

import numpy as np
import pytest

from trap import makesource, regression
from trap.parameters import ReductionRuntimeState, trap_config_for_irdis
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
    spatial = run_trap_with_model_spatial(
        data, model, pa, config, POSITION, mask, runtime=runtime, yx_center=YX_CENTER)
    rdi = _rdi(synthetic, config, runtime)
    rdi.compute_contrast_weighted_average(mask_outliers=True)
    # With 6 deg of rotation the protection angle can leave the spatial model too few
    # frames, which counts as failing to recover the companion.
    if spatial is not None:
        spatial.compute_contrast_weighted_average(mask_outliers=True)
        assert spatial.measured_contrast < 0.9 * CONTRAST
        assert abs(rdi.measured_contrast - CONTRAST) < abs(spatial.measured_contrast - CONTRAST)
    assert rdi.measured_contrast == pytest.approx(CONTRAST, rel=0.05)


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
