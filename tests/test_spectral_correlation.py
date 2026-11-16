"""Tests for the stack of channel correlation matrices per separation and its fit."""

import numpy as np
import pandas as pd
import pytest

from trap.spectral_correlation import (
    correlation_at,
    empirical_stack,
    fit_stack,
    fitted_stack,
    greco_brandt,
    read_stack,
    write_stack,
)

IFS_WAVELENGTHS = np.linspace(0.95, 1.65, 39)


def _relative_wavelength_difference(wavelengths):
    return np.abs(wavelengths[:, None] - wavelengths[None, :]) / np.mean(wavelengths)


def _correlated_cube(n_channels=10, size=101, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(size, size))
    return base[None] + 0.5 * rng.normal(size=(n_channels, size, size))


def _symmetric_noise(n_rings, n_channels, scale, rng):
    noise = rng.normal(scale=scale, size=(n_rings, n_channels, n_channels))
    noise = (noise + noise.transpose(0, 2, 1)) / np.sqrt(2.0)
    noise[:, np.arange(n_channels), np.arange(n_channels)] = 0.0
    return noise


def test_empirical_stack_keeps_rings_with_a_bad_pixel():
    cube = _correlated_cube()
    separations = np.array([20, 30, 40])
    _, _, clean_pixels = empirical_stack(cube, None, separations, width=9)

    cube[3, 50 + 30, 50] = np.nan
    original = cube.copy()
    psi, separations_used, n_pixels = empirical_stack(cube, None, separations, width=9)

    assert 30 in separations_used
    ring = list(separations_used).index(30)
    assert n_pixels[ring] == clean_pixels[ring] - 1
    assert np.all(np.isfinite(psi[ring]))
    np.testing.assert_array_equal(cube, original)


def test_empirical_stack_drops_masked_rings():
    cube = _correlated_cube(n_channels=4)
    mask = np.zeros(cube.shape[1:], dtype=bool)
    yy, xx = np.indices(mask.shape)
    distance = np.hypot(yy - 50, xx - 50)
    mask[(distance > 25) & (distance < 35)] = True

    _, separations_used, _ = empirical_stack(cube, mask, np.array([20, 30, 40]), width=9)

    np.testing.assert_array_equal(separations_used, [20, 40])


def test_fit_recovers_the_form():
    rng = np.random.default_rng(0)
    dl = _relative_wavelength_difference(IFS_WAVELENGTHS)
    rho = np.array([3.0, 6.0, 12.0, 24.0])
    psi = np.array([greco_brandt(r, dl, 0.5, 0.3, 1.0, 0.05) for r in rho])
    psi = psi + _symmetric_noise(len(rho), len(IFS_WAVELENGTHS), 0.02, rng)

    table = fit_stack(psi, rho, IFS_WAVELENGTHS)

    assert abs(table["sigma_rho"].iloc[0] - 1.0) < 0.15
    assert abs(table["sigma_lambda"].iloc[0] - 0.05) < 0.01
    assert np.all(table["rms"] < 0.05)
    assert not table["flagged"].any()
    assert table["fitted"].all()
    np.testing.assert_allclose(table["a_rho"] + table["a_lambda"] + table["a_delta"], 1.0)
    for matrix in fitted_stack(table, IFS_WAVELENGTHS):
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_two_channels_write_empirical_only(tmp_path):
    wavelengths = np.array([1.593, 1.667])
    cube = _correlated_cube(n_channels=2)
    psi, separations, n_pixels = empirical_stack(cube, None, np.array([10, 20, 30]), width=9)
    rho = separations / 4.0

    table = fit_stack(psi, rho, wavelengths, separations=separations, n_pixels=n_pixels)

    assert not table["fitted"].any()
    np.testing.assert_array_equal(correlation_at(20.0, psi, table, wavelengths), psi[1])

    path = tmp_path / "spectral_correlation_test.fits"
    write_stack(path, psi, table, wavelengths, pixel_scale_mas=12.25, width=9.0, flag_rms=0.2)
    psi_read, table_read, wavelengths_read = read_stack(path)

    np.testing.assert_array_equal(psi_read, psi)
    np.testing.assert_array_equal(wavelengths_read, wavelengths)
    pd.testing.assert_frame_equal(table_read[list(table.columns)], table, check_dtype=False)
    np.testing.assert_allclose(table_read["separation_mas"], table["separation"] * 12.25)


def test_correlation_at_interpolates_amplitudes():
    dl = _relative_wavelength_difference(IFS_WAVELENGTHS)
    rho = np.array([3.0, 9.0])
    psi = np.array(
        [
            greco_brandt(3.0, dl, 0.6, 0.2, 1.0, 0.05),
            greco_brandt(9.0, dl, 0.2, 0.2, 1.0, 0.05),
        ]
    )
    table = fit_stack(psi, rho, IFS_WAVELENGTHS)
    assert table["a_rho"].to_numpy() == pytest.approx([0.6, 0.2], abs=1e-3)

    matrix = correlation_at(6.0, psi, table, IFS_WAVELENGTHS)

    expected = greco_brandt(
        6.0,
        dl,
        table["a_rho"].mean(),
        table["a_lambda"].mean(),
        table["sigma_rho"].iloc[0],
        table["sigma_lambda"].iloc[0],
    )
    assert matrix[0, 5] == pytest.approx(expected[0, 5], rel=1e-10)
    assert matrix[0, 5] == pytest.approx(greco_brandt(6.0, dl, 0.4, 0.2, 1.0, 0.05)[0, 5], abs=1e-3)


def _analysis(tmp_path, **config):
    from types import SimpleNamespace

    from astropy import units as u

    from trap.detection import DetectionAnalysis
    from trap.parameters import TrapReductionConfig

    wavelengths = np.linspace(0.95, 1.35, 6)
    contrast = _correlated_cube(n_channels=len(wavelengths), size=81, seed=3)
    contrast[:, 40, 40] = 0.0
    analysis = DetectionAnalysis(reduction_parameters=TrapReductionConfig(**config))
    analysis.instrument = SimpleNamespace(
        pixel_scale=u.pixel_scale(7.46 * u.mas / u.pixel),
        telescope_diameter=8.2 * u.m,
        wavelengths=wavelengths * u.micron,
        fwhm=np.full(len(wavelengths), 4.0) * u.pixel,
    )
    analysis.wavelength_indices = np.arange(len(wavelengths))
    analysis.detection_cube = np.stack([contrast, np.ones_like(contrast), contrast], axis=1)
    analysis.file_paths = {
        "norm_detection_image_path": str(tmp_path / "norm_detection_run.fits"),
        "uncertainty_image_path": str(tmp_path / "uncertainty_image_run.fits"),
        "median_uncertainty_image_path": str(tmp_path / "median_uncertainty_image_run.fits"),
        "contrast_table_path": str(tmp_path / "contrast_table_run.fits"),
        "noise_mask_path": str(tmp_path / "noise_mask_run.fits"),
        "spectral_correlation_path": str(tmp_path / "spectral_correlation_run.fits"),
    }
    return analysis


def test_normalisation_writes_the_stack(tmp_path):
    from astropy.io import fits

    analysis = _analysis(tmp_path)
    analysis.contrast_table_and_normalization(save=True)

    psi, table, wavelengths = read_stack(tmp_path / "spectral_correlation_run.fits")
    assert len(table) > 5 and psi.shape == (len(table), 6, 6)
    np.testing.assert_allclose(np.diff(table["separation"]), 3.0)
    assert table["fitted"].all()
    np.testing.assert_allclose(wavelengths, analysis.instrument.wavelengths.value)
    with fits.open(tmp_path / "spectral_correlation_run.fits") as hdul:
        assert hdul[0].header["PSIWIDTH"] == 9.0
        assert hdul[0].header["PSIFIT"]
        assert hdul[0].header["PSIFLAG"] == 0.2

    original = analysis.detection_cube.copy()
    analysis.make_spectral_correlation_matrices()
    correlation = analysis.empirical_correlation
    np.testing.assert_array_equal(correlation["separation"], table["separation"])
    np.testing.assert_allclose(correlation["matrices"], psi)
    assert correlation["fit"]["fitted"].all()
    np.testing.assert_array_equal(analysis.detection_cube, original)


def test_without_the_fit_template_matching_gets_the_empirical_matrix(tmp_path):
    analysis = _analysis(tmp_path, spectral_correlation_fit=False)
    analysis.contrast_table_and_normalization(save=False)
    analysis.make_spectral_correlation_matrices()
    correlation = analysis.empirical_correlation

    assert not correlation["fit"]["fitted"].any()
    matrix = correlation_at(
        correlation["separation"][2], correlation["matrices"], correlation["fit"], correlation["wavelengths"]
    )
    np.testing.assert_array_equal(matrix, correlation["matrices"][2])


def test_saturated_amplitudes_keep_the_matrices_invertible():
    rng = np.random.default_rng(0)
    dl = _relative_wavelength_difference(IFS_WAVELENGTHS)
    rho = np.array([3.0, 6.0, 12.0, 24.0])
    a_rho = np.array([0.8, 0.6, 0.4, 0.2])
    a_lambda = 1.0 - a_rho
    psi = np.array(
        [a * np.exp(-0.5 * (r / 1.0 * dl) ** 2) + b * np.exp(-0.5 * (dl / 0.05) ** 2) for r, a, b in zip(rho, a_rho, a_lambda)]
    )
    psi = psi + _symmetric_noise(len(rho), len(IFS_WAVELENGTHS), 0.03, rng)

    table = fit_stack(psi, rho, IFS_WAVELENGTHS)

    assert np.all(table["a_delta"] >= 0.01 - 1e-12)
    assert np.all(table["free_a_rho"] + table["free_a_lambda"] <= 0.99 + 1e-12)
    matrices = list(fitted_stack(table, IFS_WAVELENGTHS))
    matrices += [correlation_at(separation, psi, table, IFS_WAVELENGTHS) for separation in (4.5, 9.0, 18.0, 30.0)]
    for matrix in matrices:
        assert np.linalg.eigvalsh(matrix).min() >= 0.005
        assert np.linalg.cond(matrix) < 1e4


def test_greco_brandt_floors_a_delta_of_amplitudes_read_from_a_file():
    dl = _relative_wavelength_difference(IFS_WAVELENGTHS)

    matrix = greco_brandt(6.0, dl, 0.7, 0.35, 1.0, 0.05)

    assert np.linalg.eigvalsh(matrix).min() >= 0.005
    np.testing.assert_allclose(matrix, greco_brandt(6.0, dl, 0.66, 0.33, 1.0, 0.05))
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_correlation_at_skips_flagged_rings():
    dl = _relative_wavelength_difference(IFS_WAVELENGTHS)
    rho = np.array([3.0, 6.0, 9.0])
    psi = np.array([greco_brandt(r, dl, a, 0.2, 1.0, 0.05) for r, a in zip(rho, [0.6, 0.1, 0.2])])
    table = fit_stack(psi, rho, IFS_WAVELENGTHS)
    table.loc[1, "flagged"] = True

    matrix = correlation_at(6.0, psi, table, IFS_WAVELENGTHS)

    expected = greco_brandt(6.0, dl, table["a_rho"].iloc[[0, 2]].mean(), table["a_lambda"].iloc[[0, 2]].mean(), 1.0, 0.05)
    np.testing.assert_allclose(matrix, expected, atol=1e-4)

    table["flagged"] = True
    all_flagged = correlation_at(6.0, psi, table, IFS_WAVELENGTHS)
    np.testing.assert_allclose(all_flagged, psi[1], atol=1e-4)


def test_no_ring_gives_the_identity_and_round_trips(tmp_path):
    wavelengths = IFS_WAVELENGTHS[:6]
    psi, separations, n_pixels = empirical_stack(
        _correlated_cube(n_channels=6, size=31), None, np.array([10]), width=9, min_pixels=10_000
    )
    assert psi.shape == (0, 6, 6)

    table = fit_stack(psi, separations / 4.0, wavelengths, separations=separations, n_pixels=n_pixels)

    assert len(table) == 0
    np.testing.assert_array_equal(correlation_at(12.0, psi, table, wavelengths), np.identity(6))
    path = tmp_path / "spectral_correlation_empty.fits"
    write_stack(path, psi, table, wavelengths, pixel_scale_mas=7.46, width=9.0, flag_rms=0.2)
    psi_read, table_read, wavelengths_read = read_stack(path)
    assert psi_read.shape == (0, 6, 6)
    assert len(table_read) == 0 and set(table.columns) <= set(table_read.columns)
    np.testing.assert_array_equal(correlation_at(12.0, psi_read, table_read, wavelengths_read), np.identity(6))
