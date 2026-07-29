"""Regression tests for the stacked detection cube written by
``DetectionAnalysis.read_output``.

The cube is derived from the per-wavelength ``detection_lam*`` files, so it must
follow them across re-reductions rather than staying frozen at the first run.
"""

import numpy as np
from astropy.io import fits

from trap.detection import DetectionAnalysis
from trap.parameters import TrapReductionConfig

COMPONENT_FRACTION = 0.15
STACKED_NAME = f"detection_ncomp038_frac{COMPONENT_FRACTION:.2f}_temporal.fits"


class _StubInstrument:
    """read_output only ever calls compute_fwhm() on the instrument."""

    def compute_fwhm(self):
        pass


def _write_per_wavelength_images(folder, values):
    for index, value in enumerate(values):
        fits.writeto(
            folder / f"detection_lam{index:02d}_ncomp038_frac{COMPONENT_FRACTION:.2f}_temporal.fits",
            np.full((2, 4, 4), value, dtype="float32"),
            overwrite=True,
        )


def _read_output(folder):
    analysis = DetectionAnalysis()
    analysis.read_output(
        COMPONENT_FRACTION,
        result_folder=str(folder),
        read_parameters=False,
        read_instrument=False,
        reduction_parameters=TrapReductionConfig(),
        instrument=_StubInstrument(),
    )
    return analysis


def test_read_output_writes_stacked_detection_image(tmp_path):
    _write_per_wavelength_images(tmp_path, [1.0, 2.0])

    analysis = _read_output(tmp_path)

    stacked = tmp_path / STACKED_NAME
    assert stacked.exists()
    assert np.allclose(fits.getdata(stacked), analysis.detection_cube)


def test_read_output_refreshes_stale_stacked_detection_image(tmp_path):
    _write_per_wavelength_images(tmp_path, [1.0, 2.0])
    _read_output(tmp_path)

    _write_per_wavelength_images(tmp_path, [7.0, 8.0])
    analysis = _read_output(tmp_path)

    stacked_data = fits.getdata(tmp_path / STACKED_NAME)
    assert np.allclose(stacked_data, analysis.detection_cube)
    assert np.allclose(np.unique(stacked_data), [7.0, 8.0])
