"""Padded crop semantics for `_crop_box` / `crop_box_from_*`.

Regression: `_crop_box` used raw numpy slicing on the last two axes; a
requested box that reached past an edge of the input (e.g. star at
(126, 129) in a 262×262 cube with boxsize=261) returned an empty
`(0, 0)` slice because numpy interprets negative slice starts as
wraparound. The padded implementation always returns shape
`(..., boxsize, boxsize)` and fills out-of-bounds regions with a
dtype-appropriate value.
"""

import numpy as np
import pytest

from trap.utils import (
    _padded_crop_box,
    crop_box_from_3D_cube,
    crop_box_from_4D_cube,
    crop_box_from_image,
)


class TestPaddedCropBoxCore:
    def test_interior_matches_naive_slice(self):
        arr = np.arange(100 * 100, dtype="float64").reshape(100, 100)
        out = _padded_crop_box(arr, boxsize=21, center_yx=(50, 50), fill_value=0)
        np.testing.assert_array_equal(out, arr[40:61, 40:61])

    def test_off_array_center_pads_with_fill(self):
        arr = np.ones((10, 10), dtype="float64")
        out = _padded_crop_box(arr, boxsize=11, center_yx=(0, 0), fill_value=np.nan)
        assert out.shape == (11, 11)
        # Only the top-left ~6×6 region overlaps the input.
        interior = out[5:11, 5:11]
        assert np.all(interior == 1.0)
        # The rest is NaN.
        border_mask = np.ones_like(out, dtype=bool)
        border_mask[5:11, 5:11] = False
        assert np.all(np.isnan(out[border_mask]))

    def test_wholly_outside_returns_all_fill(self):
        arr = np.ones((10, 10), dtype="float64")
        out = _padded_crop_box(arr, boxsize=5, center_yx=(-100, -100), fill_value=np.nan)
        assert out.shape == (5, 5)
        assert np.all(np.isnan(out))

    def test_negative_slice_wraparound_case(self):
        # The exact scenario that caused the ValueError: (0, 0):
        # star at (126, 129) in a (262, 262) input, crop boxsize=261.
        arr = np.ones((262, 262), dtype=bool)
        out = _padded_crop_box(arr, boxsize=261, center_yx=(126, 129), fill_value=False)
        assert out.shape == (261, 261)

    def test_higher_dim_input_preserves_leading_axes(self):
        arr = np.ones((7, 224, 262, 262), dtype="float64")
        out = _padded_crop_box(arr, boxsize=261, center_yx=(126, 129), fill_value=np.nan)
        assert out.shape == (7, 224, 261, 261)


class TestCropBoxDtypeDispatch:
    def test_float_fills_with_nan(self):
        arr = np.ones((10, 10), dtype="float64")
        out = crop_box_from_image(arr, boxsize=11, center_yx=(0, 0))
        assert np.isnan(out[0, 0])
        assert out[5, 5] == 1.0

    def test_bool_fills_with_false(self):
        arr = np.ones((10, 10), dtype=bool)
        out = crop_box_from_image(arr, boxsize=11, center_yx=(0, 0))
        assert out.dtype == bool
        assert out[0, 0] == False   # noqa: E712 (explicit bool comparison)
        assert out[5, 5] == True    # noqa: E712

    def test_int_fills_with_zero(self):
        arr = np.ones((10, 10), dtype="int32")
        out = crop_box_from_image(arr, boxsize=11, center_yx=(0, 0))
        assert out[0, 0] == 0
        assert out[5, 5] == 1


class TestPublicHelpersRoutingThrough:
    def test_3d_and_4d_share_semantics(self):
        arr = np.ones((3, 10, 10), dtype="float64")
        out3 = crop_box_from_3D_cube(arr, boxsize=11, center_yx=(0, 0))
        assert out3.shape == (3, 11, 11)
        arr4 = np.ones((2, 3, 10, 10), dtype="float64")
        out4 = crop_box_from_4D_cube(arr4, boxsize=11, center_yx=(0, 0))
        assert out4.shape == (2, 3, 11, 11)
