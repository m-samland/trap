import numpy as np
import pytest

from trap.noise_mask import companion_noise_masks, disk_mask, noise_mask_settings, region_mask, source_snr


def _gaussian(yy, xx, y0, x0, amplitude, fwhm):
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma**2))


def _grids(n=201):
    yy, xx = np.indices((n, n), dtype=float)
    return yy - n // 2, xx - n // 2


def _footprint_map(separation=60.0, peak=40.0, fwhm=4.0, lobe=-8.0, lobe_offset=14.0):
    yy, xx = _grids()
    image = np.random.default_rng(0).normal(size=yy.shape)
    image += _gaussian(yy, xx, 0, separation, peak, fwhm)
    angle = lobe_offset / separation
    for sign in (-1, 1):
        image += _gaussian(yy, xx, separation * np.sin(sign * angle), separation * np.cos(sign * angle), lobe, fwhm)
    along = np.arctan2(yy, xx) * separation
    ridge = (np.abs(np.hypot(yy, xx) - separation) < 1.5) & (np.abs(along) < lobe_offset + 2)
    image[ridge] -= 6.0
    return image


def test_region_contains_the_lobes():
    image = _footprint_map()
    angle = 14.0 / 60.0
    lobes = [(round(60 * np.sin(s * angle)), round(60 * np.cos(s * angle))) for s in (-1, 1)]
    region = region_mask(image, (0, 60), fwhm=4.0)
    disk = disk_mask(image.shape, (0, 60), 11.0)
    for y, x in lobes:
        assert region[100 + y, 100 + x]
        assert not disk[100 + y, 100 + x]
    assert not region[100, 100 + 100]


def test_faint_blob_gets_a_small_region():
    yy, xx = _grids()
    image = np.random.default_rng(0).normal(size=yy.shape) + _gaussian(yy, xx, 0, 60, 4.0, 4.0)
    region = region_mask(image, (0, 60), fwhm=4.0)
    assert region.sum() < np.pi * (11 + 4) ** 2


@pytest.mark.parametrize("fwhm", [2.0, 4.0])
def test_region_never_covers_a_whole_ring(fwhm):
    yy, xx = _grids()
    image = np.random.default_rng(0).normal(size=yy.shape) + _gaussian(yy, xx, 0, 8, 500.0, fwhm)
    region = region_mask(image, (0, 8), fwhm=fwhm)
    radius = np.hypot(yy, xx)
    for separation in range(2, 21):
        ring = np.abs(radius - separation) < 1.5
        assert (ring & ~region).sum() >= 10, separation


def test_nan_pixels_are_not_in_the_region():
    image = _footprint_map()
    yy, xx = _grids()
    image[np.hypot(yy, xx) > 80] = np.nan
    region = region_mask(image, (0, 60), fwhm=4.0)
    assert not region[np.isnan(image)].any()
    assert region.any()


def test_source_snr_ignores_nan():
    image = np.full((51, 51), np.nan)
    assert np.isnan(source_snr(image, (0, 10), 2.0))
    image[25, 35] = 7.0
    assert source_snr(image, (0, 10), 2.0) == 7.0


def test_companion_noise_masks_applies_the_rule():
    yy, xx = _grids()
    image = np.random.default_rng(0).normal(size=yy.shape)
    image += _gaussian(yy, xx, 0, 60, 500.0, 4.0) + _gaussian(yy, xx, 0, -60, 6.0, 4.0)
    positions = [(0, 60), (0, -60)]
    kwargs = dict(fwhm=4.0, half_arcs=None, threshold=2.0, radial_fwhm=6.0, along_fwhm=8.0, disk_radius=11.0)

    mask, info = companion_noise_masks(image, positions, region_snr=10.0, **kwargs)
    assert [item["mask_rule"] for item in info] == ["region", "disk"]
    assert info[0]["mask_snr"] > 100
    assert info[1]["mask_pixels"] == disk_mask(image.shape, positions[1], 11.0).sum()
    expected = region_mask(image, positions[0], 4.0) | disk_mask(image.shape, positions[1], 11.0)
    assert np.array_equal(mask, expected)

    _, info = companion_noise_masks(image, positions, region_snr=None, **kwargs)
    assert [item["mask_rule"] for item in info] == ["disk", "disk"]


def test_footprint_circling_the_star_leaves_every_ring_pixels():
    yy, xx = _grids()
    radius = np.hypot(yy, xx)
    image = np.where(radius < 30, 10.0, 0.0)
    region = region_mask(image, (0, 4), fwhm=4.0)
    for separation in range(2, 21):
        ring = np.abs(radius - separation) < 1.5
        assert (ring & ~region).sum() >= 10, separation


def _band_map(separation):
    yy, xx = _grids()
    return np.where(np.abs(np.hypot(yy, xx) - separation) < 2, 10.0, 0.0)


def _ring_pixel(separation, along):
    angle = along / separation
    return 100 + round(separation * np.sin(angle)), 100 + round(separation * np.cos(angle))


def test_half_arc_extends_the_region_along_the_ring():
    image = _band_map(60.0)
    far = _ring_pixel(60.0, 25.0)
    assert not region_mask(image, (0, 60), fwhm=2.0, half_arc=None)[far]
    assert region_mask(image, (0, 60), fwhm=2.0, half_arc=20.0)[far]


def test_region_wraps_around_theta_pi():
    image = _band_map(60.0)
    region = region_mask(image, (0, -60), fwhm=2.0)
    for along in (-10.0, 10.0):
        y, x = _ring_pixel(60.0, along)
        assert region[y, 200 - x]


def test_noise_mask_settings_defaults_for_old_configs():
    class OldConfig:
        pass

    assert noise_mask_settings(OldConfig()) == {
        "region_snr": 10.0,
        "threshold": 2.0,
        "radial_fwhm": 6.0,
        "along_fwhm": 8.0,
    }
