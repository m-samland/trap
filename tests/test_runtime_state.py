"""Tests for TrapReductionConfig / ReductionRuntimeState edge-FoV additions."""

import numpy as np
import pytest

from trap.parameters import ReductionRuntimeState, TrapReductionConfig


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
