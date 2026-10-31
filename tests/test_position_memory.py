"""Results kept from a map reduction hold no per-pixel arrays.

In map mode every position's result is collected until the map is assembled, which only
reads scalars from it. A full-size noise model or residual array per position adds up to
tens of gigabytes over a few thousand positions, so they are released when a position is
done.
"""

from types import SimpleNamespace

import numpy as np

from trap.reduction_wrapper import _release_position_arrays

ARRAYS = ("data", "model_cube", "noise_model_cube", "residuals", "reduced_result")


def _result():
    return SimpleNamespace(measured_contrast=1.0, **{name: np.zeros((256, 123, 123)) for name in ARRAYS})


def test_per_pixel_arrays_are_released_and_scalars_kept():
    results = {"temporal": _result(), "temporal_plus_spatial": _result(), "spatial": None}
    released = _release_position_arrays(results)
    for key in ("temporal", "temporal_plus_spatial"):
        assert all(getattr(released[key], name) is None for name in ARRAYS)
        assert released[key].measured_contrast == 1.0
    assert released["spatial"] is None
