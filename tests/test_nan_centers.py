"""A non-finite image center costs its frame, not the run (#40)."""

import logging

import astropy.units as u
import numpy as np

from trap.parameters import Instrument, TrapReductionConfig
from trap.reduction_wrapper import run_complete_reduction

N_FRAMES = 16
IMAGE_SIZE = 61


def _psf():
    yy, xx = np.mgrid[:21, :21] - 10
    psf = np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
    return psf / psf.sum()


def _reduce(tmp_path, n_wavelengths, xy_image_centers):
    data = np.random.default_rng(0).normal(0.0, 1e-4, (n_wavelengths, N_FRAMES, IMAGE_SIZE, IMAGE_SIZE))
    instrument = Instrument(
        name="synthetic",
        pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel),
        telescope_diameter=8.0 * u.m,
        detector_gain=1.0,
        readnoise=0.0,
        instrument_type="photometry",
        wavelengths=np.linspace(1.6, 1.7, n_wavelengths) * u.micron,
    )
    # data_auto_crop is left at its default: the crop size used to become NaN.
    config = TrapReductionConfig(
        search_region_inner_bound=4,
        search_region_outer_bound=7,
        use_multiprocess=False,
        result_folder=str(tmp_path),
        verbose=False,
    )
    run_complete_reduction(
        data_full=data,
        flux_psf_full=np.stack([_psf()] * n_wavelengths),
        pa=np.linspace(0.0, 40.0, N_FRAMES),
        instrument=instrument,
        reduction_parameters=config,
        xy_image_centers=xy_image_centers,
        temporal_components_fraction=[0.25],
        overwrite=True,
        use_progress_bar=False,
    )
    return sorted(p.name for p in tmp_path.glob("detection_*.fits"))


def test_frame_with_nan_center_is_dropped(tmp_path, caplog):
    centers = np.full((N_FRAMES, 2), IMAGE_SIZE // 2, dtype=float)  # 2-D: (time, xy)
    centers[3] = np.nan
    with caplog.at_level(logging.WARNING, logger="trap.reduction_wrapper"):
        detections = _reduce(tmp_path, 1, centers)
    assert len(detections) == 1
    assert "Dropping 1 of 16 frames" in caplog.text


def test_channel_without_centers_is_skipped_alone(tmp_path):
    centers = np.full((2, N_FRAMES, 2), IMAGE_SIZE // 2, dtype=float)
    centers[1] = np.nan
    detections = _reduce(tmp_path, 2, centers)
    assert len(detections) == 1
    assert "lam00" in detections[0]



def test_frame_with_nan_center_in_one_wavelength_is_dropped_from_all(tmp_path, caplog):
    centers = np.full((2, N_FRAMES, 2), IMAGE_SIZE // 2, dtype=float)
    centers[1, 3] = np.nan
    with caplog.at_level(logging.WARNING, logger="trap.reduction_wrapper"):
        detections = _reduce(tmp_path, 2, centers)
    assert len(detections) == 2
    assert "Dropping 1 of 16 frames" in caplog.text


def test_single_center_for_all_frames(tmp_path):
    detections = _reduce(tmp_path, 1, np.array([IMAGE_SIZE // 2, IMAGE_SIZE // 2], dtype=float))
    assert len(detections) == 1
