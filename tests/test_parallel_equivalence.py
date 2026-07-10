"""Serial-vs-parallel equivalence of the reduction on a synthetic cube.

WP1 acceptance test: the joblib/memmap-store parallel path must produce the
same detection maps as the serial loop.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from trap import makesource
from trap.parameters import Instrument, TrapReductionConfig
from trap.reduction_wrapper import run_complete_reduction


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(42)
    n_frames = 16
    image_size = 31

    pa = np.linspace(0.0, 40.0, n_frames)

    # Gaussian PSF stamp (odd size, centered on pixel)
    stamp_size = 21
    yy, xx = np.mgrid[:stamp_size, :stamp_size] - stamp_size // 2
    sigma = 3.5 / 2.355
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    data = rng.normal(loc=0.0, scale=1e-4, size=(n_frames, image_size, image_size))
    # Inject a companion at 5 pix separation
    data = makesource.addsource(
        data,
        pos=(0.0, 5.0),
        pa=pa,
        psf_arr=psf,
        norm=np.full(n_frames, 5e-3),
        jitter=0,
        poisson_noise=False,
        subpixel=True,
        copy=True,
        verbose=False,
    )

    instrument = Instrument(
        name="synthetic",
        pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel),
        telescope_diameter=8.0 * u.m,
        detector_gain=1.0,
        readnoise=0.0,
        instrument_type="photometry",
        wavelengths=np.array([1.6]) * u.micron,
    )
    return data, psf, pa, instrument


def _run_reduction(synthetic_dataset, result_folder, use_multiprocess):
    data, psf, pa, instrument = synthetic_dataset
    config = TrapReductionConfig(
        search_region_inner_bound=4,
        search_region_outer_bound=7,
        data_auto_crop=False,
        data_crop_size=None,
        temporal_model=True,
        temporal_plus_spatial_model=False,
        spatial_model=False,
        use_multiprocess=use_multiprocess,
        ncpus=2,
        result_folder=str(result_folder),
        verbose=False,
    )
    run_complete_reduction(
        data_full=data.copy(),
        flux_psf_full=psf.copy(),
        pa=pa,
        instrument=instrument,
        reduction_parameters=config,
        temporal_components_fraction=[0.25],
        overwrite=True,
        use_progress_bar=False,
    )
    detection_files = sorted(result_folder.glob("detection_*.fits"))
    assert len(detection_files) == 1, (
        f"Expected one detection image, found: {[f.name for f in detection_files]}"
    )
    return fits.getdata(detection_files[0])


def test_serial_and_parallel_reductions_are_equivalent(synthetic_dataset, tmp_path):
    serial_folder = tmp_path / "serial"
    parallel_folder = tmp_path / "parallel"
    serial_folder.mkdir()
    parallel_folder.mkdir()

    detection_serial = _run_reduction(
        synthetic_dataset, serial_folder, use_multiprocess=False
    )
    detection_parallel = _run_reduction(
        synthetic_dataset, parallel_folder, use_multiprocess=True
    )

    assert detection_serial.shape == detection_parallel.shape
    # Maps: 0) contrast, 1) uncertainty, 2) SNR; NaN outside the search region.
    np.testing.assert_allclose(
        detection_parallel,
        detection_serial,
        rtol=1e-10,
        atol=0.0,
        equal_nan=True,
    )


def test_parallel_reduction_detects_injected_companion(synthetic_dataset, tmp_path):
    result_folder = tmp_path / "detection"
    result_folder.mkdir()
    detection = _run_reduction(synthetic_dataset, result_folder, use_multiprocess=True)

    contrast = detection[0]
    center = contrast.shape[0] // 2
    # Injected at (dy, dx) = (0, 5) relative to center with contrast 5e-3
    measured = contrast[center, center + 5]
    assert measured == pytest.approx(5e-3, rel=0.2)
