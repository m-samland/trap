"""Tests for TrapReductionConfig / ReductionRuntimeState edge-FoV additions."""

import logging

import numpy as np
import pytest

from trap.parameters import (
    ReductionRuntimeState,
    TrapReductionConfig,
    _derive_outer_bound,
    build_runtime_state,
)


class TestConfigDefaults:
    def test_search_region_outer_bound_default_is_85(self):
        config = TrapReductionConfig()
        assert config.search_region_outer_bound == 85

    def test_search_region_outer_bound_accepts_none(self):
        config = TrapReductionConfig(search_region_outer_bound=None)
        assert config.search_region_outer_bound is None

    def test_reduction_mask_min_pixels_default_is_30(self):
        config = TrapReductionConfig()
        assert config.reduction_mask_min_pixels == 30

    def test_reduction_mask_min_pixels_override(self):
        config = TrapReductionConfig(reduction_mask_min_pixels=15)
        assert config.reduction_mask_min_pixels == 15


class TestRuntimeStateDefaults:
    def test_valid_pixel_mask_cropped_defaults_none(self):
        rs = ReductionRuntimeState(yx_anamorphism=np.array([1.0, 1.0]))
        assert rs.valid_pixel_mask_cropped is None

    def test_reduction_mask_min_pixels_defaults_30(self):
        rs = ReductionRuntimeState(yx_anamorphism=np.array([1.0, 1.0]))
        assert rs.reduction_mask_min_pixels == 30

    def test_for_iteration_preserves_footprint_and_threshold(self):
        mask = np.ones((5, 5), dtype=bool)
        rs = ReductionRuntimeState(
            yx_anamorphism=np.array([1.0, 1.0]),
            valid_pixel_mask_cropped=mask,
            reduction_mask_min_pixels=17,
        )
        rs2 = rs.for_iteration(
            number_of_pca_regressors=10,
            temporal_components_fraction=0.2,
            fwhm=4.0,
            reduction_mask_psf_size=19,
            signal_mask_psf_size=21,
        )
        assert rs2.valid_pixel_mask_cropped is mask
        assert rs2.reduction_mask_min_pixels == 17


class TestDeriveOuterBound:
    def test_full_footprint_reaches_edge(self):
        H = W = 101
        mask = np.ones((H, W), dtype=bool)
        r = _derive_outer_bound(mask, (H // 2, W // 2), min_pixels=8)
        assert r >= 70

    def test_circular_footprint_matches_radius(self):
        H = W = 101
        yy, xx = np.indices((H, W))
        cy, cx = H // 2, W // 2
        radius = 30
        mask = np.hypot(yy - cy, xx - cx) <= radius
        r = _derive_outer_bound(mask, (cy, cx), min_pixels=8)
        assert radius - 1 <= r <= radius

    def test_empty_footprint_returns_zero(self):
        mask = np.zeros((51, 51), dtype=bool)
        assert _derive_outer_bound(mask, (25, 25), min_pixels=1) == 0

    def test_min_pixels_gate_reduces_radius(self):
        H = W = 101
        cy, cx = 50, 50
        mask = np.zeros((H, W), dtype=bool)
        for dy, dx in [(30, 0), (-30, 0), (0, 30), (0, -30)]:
            mask[cy + dy, cx + dx] = True
        r_low = _derive_outer_bound(mask, (cy, cx), min_pixels=1)
        r_high = _derive_outer_bound(mask, (cy, cx), min_pixels=8)
        assert r_low == 30
        assert r_high == 0


def _default_stamp_sizes():
    return np.array([19])


class TestBuildRuntimeState:
    def _call(self, config, data_shape, valid_pixel_mask=None, yx_center_full=None):
        return build_runtime_state(
            config=config,
            data_shape=data_shape,
            stamp_sizes=_default_stamp_sizes(),
            stamp_sizes_reduction=_default_stamp_sizes(),
            max_shift=0.0,
            mas_per_pixel=None,
            valid_pixel_mask=valid_pixel_mask,
            yx_center_full=yx_center_full,
        )

    def test_no_footprint_reproduces_current_behavior(self):
        config = TrapReductionConfig(search_region_outer_bound=85)
        rs = self._call(config, data_shape=(1, 10, 401, 401))
        assert rs.search_region_outer_bound == 85
        assert rs.valid_pixel_mask_cropped is None
        assert rs.search_region.shape[0] == rs.data_crop_size
        assert rs.reduction_mask_min_pixels == config.reduction_mask_min_pixels

    def test_crop_larger_than_input_is_clamped_not_raised(self, caplog):
        config = TrapReductionConfig(search_region_outer_bound=85)
        with caplog.at_level(logging.INFO, logger="trap.parameters"):
            rs = self._call(config, data_shape=(1, 10, 101, 101))
        assert rs.data_crop_size <= 101
        assert rs.data_crop_size % 2 == 1
        assert any("clamping" in rec.message.lower() for rec in caplog.records)

    def test_auto_outer_bound_from_footprint(self):
        H = W = 101
        yy, xx = np.indices((H, W))
        mask = np.hypot(yy - H // 2, xx - W // 2) <= 40
        config = TrapReductionConfig(search_region_outer_bound=None)
        rs = self._call(
            config,
            data_shape=(1, 10, H, W),
            valid_pixel_mask=mask,
            yx_center_full=np.array([[H / 2.0, W / 2.0]]),
        )
        assert 39 <= rs.search_region_outer_bound <= 40

    def test_auto_outer_bound_without_footprint_raises(self):
        config = TrapReductionConfig(search_region_outer_bound=None)
        with pytest.raises(ValueError, match="valid_pixel_mask"):
            self._call(config, data_shape=(1, 10, 101, 101))

    def test_search_region_intersected_with_footprint(self):
        H = W = 101
        mask = np.zeros((H, W), dtype=bool)
        mask[:, : W // 2] = True
        config = TrapReductionConfig(search_region_outer_bound=40)
        rs = self._call(
            config,
            data_shape=(1, 10, H, W),
            valid_pixel_mask=mask,
            yx_center_full=np.array([[H / 2.0, W / 2.0]]),
        )
        c = rs.data_crop_size
        assert rs.valid_pixel_mask_cropped.shape == (c, c)
        assert not (rs.search_region & ~rs.valid_pixel_mask_cropped).any()
