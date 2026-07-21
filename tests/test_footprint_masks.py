"""Footprint intersection in regressor pool and multiwavelength masks."""

import numpy as np
import pytest

from trap.parameters import TrapReductionConfig
from trap.regressor_selection import (
    MultiwavelengthRegressors,
    make_multiwavelength_regressor_masks,
    make_regressor_pool_for_pixel,
    make_signal_mask,
)


def _pool(**kw):
    config = TrapReductionConfig(
        annulus_width=5,
        annulus_offset=0.0,
        add_radial_regressors=False,
        target_pix_mask_radius=None,
    )
    return make_regressor_pool_for_pixel(
        reduction_parameters=config,
        yx_dim=(41, 41),
        yx_center=(20, 20),
        yx_pixel=(20, 30),
        **kw,
    )


class TestRegressorPoolFootprint:
    def test_no_footprint_reproduces_baseline(self):
        baseline = _pool()
        again = _pool(valid_pixel_mask=None)
        np.testing.assert_array_equal(baseline, again)

    def test_footprint_masks_out_disallowed_pixels(self):
        H = W = 41
        mask = np.zeros((H, W), dtype=bool)
        mask[:, : W // 2] = True
        baseline = _pool()
        restricted = _pool(valid_pixel_mask=mask)
        assert not (restricted & ~mask).any()
        np.testing.assert_array_equal(restricted, baseline & mask)


def _mw_ctx(mode, valid_pixel_masks=None):
    return MultiwavelengthRegressors(
        data=np.zeros((3, 41, 41, 4)),
        wavelength_indices=np.array([0, 2]),
        scale_factors=np.array([1.05, 0.95]),
        fwhm=np.array([4.0, 4.2]),
        fwhm_reference=4.0,
        yx_centers=np.array([[20.0, 20.0], [20.0, 20.0]]),
        bad_pixel_masks=[None, None],
        mode=mode,
        max_regressor_pool_size=3.0,
        valid_pixel_masks=valid_pixel_masks,
    )


class TestMultiwavelengthRegressorFootprint:
    def _config(self):
        return TrapReductionConfig(annulus_width=5, annulus_offset=0.0)

    def _run(self, mode, valid_pixel_masks=None):
        ctx = _mw_ctx(mode, valid_pixel_masks=valid_pixel_masks)
        signal_mask = make_signal_mask((41, 41), (20, 30), mask_radius=2.5)
        return make_multiwavelength_regressor_masks(
            ctx,
            reduction_parameters=self._config(),
            yx_pixel=(20, 30),
            yx_dim=(41, 41),
            yx_center=(20, 20),
            signal_mask=signal_mask,
            n_reference_pixels=int(signal_mask.sum()) * 3,
        )

    def test_default_valid_pixel_masks_none_reproduces_baseline(self):
        baseline = self._run("occluded", valid_pixel_masks=None)
        again = self._run("occluded", valid_pixel_masks=[None, None])
        assert set(baseline) == set(again)
        for k in baseline:
            np.testing.assert_array_equal(baseline[k], again[k])

    @pytest.mark.parametrize("mode", ["pool", "occluded", "sdi"])
    def test_footprint_intersected_when_provided(self, mode):
        H = W = 41
        mask_left = np.zeros((H, W), dtype=bool)
        mask_left[:, : W // 2] = True
        result = self._run(mode, valid_pixel_masks=[mask_left, None])
        if 0 in result:
            assert not (result[0] & ~mask_left).any()


class TestTrapOnePositionThreshold:
    def _fake_runtime(self, mask, threshold):
        from trap.parameters import ReductionRuntimeState
        return ReductionRuntimeState(
            yx_anamorphism=np.array([1.0, 1.0]),
            valid_pixel_mask_cropped=mask,
            reduction_mask_min_pixels=threshold,
            reduction_mask_psf_size=5,
            signal_mask_psf_size=5,
        )

    def _fake_inputs(self):
        data = np.zeros((8, 41, 41), dtype="float64")
        psf = np.zeros((5, 5), dtype="float64")
        psf[2, 2] = 1.0
        pa = np.zeros(8, dtype="float64")
        return data, psf, pa

    def test_returns_none_when_survivors_below_threshold(self):
        from trap.reduction_wrapper import trap_one_position
        data, psf, pa = self._fake_inputs()
        mask = np.zeros(data.shape[-2:], dtype=bool)
        runtime = self._fake_runtime(mask, threshold=1)
        config = TrapReductionConfig(
            reduction_mask_min_pixels=1,
            annulus_width=5,
            temporal_model=True,
            spatial_model=False,
            temporal_plus_spatial_model=False,
            fit_planet=False,
            use_multiprocess=False,
        )
        result = trap_one_position(
            guess_position=(0, 15),
            data=data, flux_psf=psf, pa=pa,
            reduction_parameters=config,
            known_companion_mask=np.zeros(data.shape[-2:], dtype=bool),
            runtime=runtime,
            yx_center=(20, 20),
            yx_center_injection=np.array([20.0, 20.0]),
        )
        assert result is None
