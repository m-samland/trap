"""Integration coverage for run_complete_reduction with valid_pixel_mask."""

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
    image_size = 41

    pa = np.linspace(0.0, 40.0, n_frames)

    stamp_size = 21
    yy, xx = np.mgrid[:stamp_size, :stamp_size] - stamp_size // 2
    sigma = 3.5 / 2.355
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    data = rng.normal(loc=0.0, scale=1e-4, size=(n_frames, image_size, image_size))
    data = makesource.addsource(
        data,
        pos=(0.0, 6.0),
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


def _base_config(result_folder):
    return TrapReductionConfig(
        search_region_inner_bound=4,
        search_region_outer_bound=9,
        data_auto_crop=False,
        data_crop_size=None,
        temporal_model=True,
        temporal_plus_spatial_model=False,
        spatial_model=False,
        use_multiprocess=False,
        ncpus=1,
        result_folder=str(result_folder),
        verbose=False,
    )


class TestRunCompleteReductionFootprint:
    def test_footprint_argument_is_accepted(self, synthetic_dataset, tmp_path):
        data, psf, pa, instrument = synthetic_dataset
        H, W = data.shape[-2:]
        mask = np.ones((H, W), dtype=bool)
        config = _base_config(tmp_path)
        run_complete_reduction(
            data_full=data.copy(),
            flux_psf_full=psf.copy(),
            pa=pa,
            instrument=instrument,
            reduction_parameters=config,
            temporal_components_fraction=[0.25],
            overwrite=True,
            use_progress_bar=False,
            valid_pixel_mask=mask,
        )
        assert sorted(tmp_path.glob("detection_*.fits"))

    def test_positions_outside_footprint_stay_nan(self, synthetic_dataset, tmp_path):
        data, psf, pa, instrument = synthetic_dataset
        H, W = data.shape[-2:]
        # Left half only; right half is invalid → detection map right of centre
        # must remain NaN (the pre-init sentinel).
        mask = np.zeros((H, W), dtype=bool)
        mask[:, : W // 2] = True
        config = _base_config(tmp_path)
        run_complete_reduction(
            data_full=data.copy(),
            flux_psf_full=psf.copy(),
            pa=pa,
            instrument=instrument,
            reduction_parameters=config,
            temporal_components_fraction=[0.25],
            overwrite=True,
            use_progress_bar=False,
            valid_pixel_mask=mask,
        )
        det_files = sorted(tmp_path.glob("detection_*.fits"))
        assert det_files, "no detection image written"
        det = fits.getdata(det_files[0])
        # Detection image has leading channel axis (contrast/uncertainty/snr).
        # Right of the centre column, all channels for every row must be NaN.
        right_half = det[..., :, W // 2 + 3 :]
        assert np.isnan(right_half).all(), (
            f"expected all-NaN right half, got {np.count_nonzero(~np.isnan(right_half))} non-NaN"
        )


class TestNanSafePeakExtraction:
    def test_argmax_of_series_with_nan_ignores_nan(self):
        import pandas as pd
        s = pd.Series([np.nan, 1.0, 3.0, np.nan, 2.0])
        arr = s.to_numpy()
        idx = int(np.nanargmax(arr)) if not np.isnan(arr).all() else -1
        assert idx == 2
