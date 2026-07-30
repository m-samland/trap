"""Edge-of-FoV injection tests for makesource.inject_signal."""

import numpy as np
import pytest

from trap.makesource import inject_signal, prepare_injection


def _gaussian_psf(size=9, sigma=1.5):
    yy, xx = np.mgrid[:size, :size]
    cy = cx = size // 2
    psf = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return psf.astype("float64")


def _run_inject(flux_arr, yx_positions, psf, subpixel):
    yx_positions = np.asarray(yx_positions, dtype="float64")
    xdiff, ydiff, x1, x2, y1, y2 = prepare_injection(
        yx_positions, flux_arr.shape[-2:], psf.shape
    )
    norm = np.ones(len(yx_positions), dtype="float64")
    return inject_signal(
        flux_arr, xdiff, ydiff, x1, x2, y1, y2, psf, norm,
        subpixel=subpixel, copy=True,
    )


class TestInteriorRegression:
    @pytest.mark.parametrize("subpixel", [True, False])
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_interior_reproduces_naive_slice(self, subpixel, ndim):
        psf = _gaussian_psf()
        H = W = 41
        ntimes = 4
        yx_positions = np.array([[20.0, 20.0]] * ntimes)
        if ndim == 2:
            flux_arr = np.zeros((H, W), dtype="float64")
        else:
            flux_arr = np.zeros((ntimes, H, W), dtype="float64")
        out = _run_inject(flux_arr.copy(), yx_positions, psf, subpixel=subpixel)
        assert np.isfinite(out).all()
        # Sanity: injection is localized around (20, 20); rows/cols far away are zero.
        assert out[..., 0, :].sum() == 0.0
        assert out[..., :, 0].sum() == 0.0
        assert out[..., -1, :].sum() == 0.0
        assert out[..., :, -1].sum() == 0.0
        # And the stamp region did receive flux.
        assert out[..., 16:25, 16:25].sum() > 0.5 * ntimes * psf.sum()


class TestCornerClipping:
    @pytest.mark.parametrize("subpixel", [True, False])
    def test_corner_position_no_crash_partial_inject(self, subpixel):
        psf = _gaussian_psf()
        H = W = 41
        ntimes = 3
        yx_positions = np.array([[0.0, 0.0]] * ntimes)
        flux_arr = np.zeros((H, W), dtype="float64")
        out = _run_inject(flux_arr, yx_positions, psf, subpixel=subpixel)
        assert out[H // 2 :, :].sum() == 0.0
        assert out[:, W // 2 :].sum() == 0.0
        assert out[: psf.shape[0] // 2 + 1, : psf.shape[1] // 2 + 1].sum() > 0.0


class TestWhollyOutside:
    @pytest.mark.parametrize("subpixel", [True, False])
    def test_position_far_outside_leaves_cube_unchanged(self, subpixel):
        psf = _gaussian_psf()
        H = W = 41
        yx_positions = np.array([[-100.0, -100.0], [200.0, 200.0]])
        flux_arr = np.full((H, W), 3.14, dtype="float64")
        out = _run_inject(flux_arr.copy(), yx_positions, psf, subpixel=subpixel)
        np.testing.assert_array_equal(out, flux_arr)


class TestThreeD:
    @pytest.mark.parametrize("subpixel", [True, False])
    def test_3d_corner_only_target_frame_touched(self, subpixel):
        psf = _gaussian_psf()
        H = W = 41
        ntimes = 4
        yx_positions = np.array([[20.0, 20.0]] * ntimes)
        yx_positions[0] = [0.0, 0.0]
        flux_arr = np.zeros((ntimes, H, W), dtype="float64")
        out = _run_inject(flux_arr, yx_positions, psf, subpixel=subpixel)
        assert np.isfinite(out).all()
        assert out[0].sum() > 0.0
        assert out[0].sum() < out[1].sum()
