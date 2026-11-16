"""Two-pass noise normalisation of the detection maps with the footprint mask of bright sources."""

import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from trap.detection import DetectionAnalysis, _warn_about_unmasked_rings, make_contrast_curve, make_radial_profile
from trap.noise_mask import companion_noise_masks, noise_mask_settings, region_mask
from trap.parameters import TrapReductionConfig


def _gaussian(yy, xx, y0, x0, amplitude, fwhm):
    sigma = fwhm / 2.3548
    return amplitude * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma**2))


# Field rotation whose half arc, 12 px at 60 px, lets the region reach the end of the ridge.
ROTATION = 0.4


# The ridge between the lobes reaches 42 px along the ring, beyond the 32 px azimuthal clip
# of the region without field rotation and inside it with ROTATION, and contaminates enough
# of the ring for the disk to leave a noise excess above 30%.
def _footprint_map(separation=60.0, peak=400.0, fwhm=4.0, lobe=-8.0, lobe_offset=40.0, n=201):
    yy, xx = np.indices((n, n), dtype=float) - n // 2
    image = np.random.default_rng(0).normal(size=yy.shape)
    image += _gaussian(yy, xx, 0, separation, peak, fwhm)
    angle = lobe_offset / separation
    for sign in (-1, 1):
        image += _gaussian(yy, xx, separation * np.sin(sign * angle), separation * np.cos(sign * angle), lobe, fwhm)
    along = np.arctan2(yy, xx) * separation
    ridge = (np.abs(np.hypot(yy, xx) - separation) < 1.5) & (np.abs(along) < lobe_offset + 2)
    image[ridge] -= 6.0
    return image


def _detection_image(snr):
    return np.stack([snr, np.ones_like(snr), snr])


def _noise_at(table, separation):
    return float(table["snr_normalization"][table["sep (pix)"] == separation][0])


def test_second_pass_lowers_the_noise_at_a_bright_source():
    image = _detection_image(_footprint_map())
    first_normalized, disk_table, _, _ = make_contrast_curve(image, yx_known_companion_position=[[0, 60]])
    region = region_mask(first_normalized, (0, 60), fwhm=4.0, half_arc=60 * ROTATION / 2)
    _, region_table, _, _ = make_contrast_curve(image, yx_known_companion_position=[[0, 60]], companion_mask=region)

    assert _noise_at(disk_table, 60) > 1.25
    assert _noise_at(region_table, 60) == pytest.approx(1.0, rel=0.1)


def test_normalisation_without_field_rotation():
    image = _footprint_map()
    kwargs = dict(fwhm=4.0, region_snr=10.0, threshold=2.0, radial_fwhm=6.0, along_fwhm=8.0, disk_radius=11.0)
    without, _ = companion_noise_masks(image, [(0, 60)], half_arcs=None, **kwargs)
    zero, _ = companion_noise_masks(image, [(0, 60)], half_arcs=[0.0], **kwargs)
    assert np.array_equal(without, zero)


def test_old_config_gets_defaults():
    class OldConfig:
        pass

    assert noise_mask_settings(OldConfig()) == {"region_snr": 10.0, "threshold": 2.0, "radial_fwhm": 6.0, "along_fwhm": 8.0}


def _old_pickled_config(monkeypatch, **fields):
    """A config pickled by a trap that had none of the four noise mask fields."""
    full_state = TrapReductionConfig.__getstate__
    monkeypatch.setattr(TrapReductionConfig, "__getstate__", lambda self: full_state(self)[:-4])
    data = pickle.dumps(TrapReductionConfig(**fields))
    monkeypatch.undo()
    return pickle.loads(data)


def test_old_pickled_config_restores_with_defaults(monkeypatch):
    old = _old_pickled_config(monkeypatch, companion_mask_radius=7)
    merged = old.merge(result_folder="x")
    assert merged.result_folder == "x"
    assert merged.companion_mask_radius == 7
    assert (
        merged.noise_mask_region_snr,
        merged.noise_mask_region_threshold,
        merged.noise_mask_radial_fwhm,
        merged.noise_mask_along_fwhm,
    ) == (10.0, 2.0, 6.0, 8.0)
    assert pickle.loads(pickle.dumps(merged)).noise_mask_along_fwhm == 8.0


def _analysis(tmp_path, field_rotation=ROTATION, reduction_parameters=None, **config):
    if reduction_parameters is None:
        reduction_parameters = TrapReductionConfig(yx_known_companion_position=[[0, 60]], **config)
    analysis = DetectionAnalysis(reduction_parameters=reduction_parameters)
    analysis.instrument = SimpleNamespace(pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel), fwhm=np.array([4.0, 4.0]) * u.pixel)
    analysis.wavelength_indices = np.array([0, 1])
    analysis.detection_cube = np.array([_detection_image(_footprint_map()), _detection_image(_footprint_map(peak=6.0))])
    analysis.field_rotation = field_rotation
    analysis.file_paths = {
        "norm_detection_image_path": str(tmp_path / "norm_detection_run.fits"),
        "uncertainty_image_path": str(tmp_path / "uncertainty_image_run.fits"),
        "median_uncertainty_image_path": str(tmp_path / "median_uncertainty_image_run.fits"),
        "contrast_table_path": str(tmp_path / "contrast_table_run.fits"),
        "noise_mask_path": str(tmp_path / "noise_mask_run.fits"),
    }
    return analysis


def test_contrast_table_and_normalization_uses_the_region_of_a_bright_source(tmp_path):
    analysis = _analysis(tmp_path)
    analysis.contrast_table_and_normalization(save=True)
    products = analysis.detection_products

    assert products["noise_mask_cube"].dtype == bool
    assert products["noise_mask_cube"].shape == (2, 201, 201)
    sources = products["noise_mask_sources"]
    assert [(s["wavelength_index"], s["mask_rule"]) for s in sources] == [(0, "region"), (1, "disk")]
    assert (sources[0]["y_relative"], sources[0]["x_relative"]) == (0, 60)
    assert _noise_at(products["contrast_tables"][0], 60) == pytest.approx(1.0, rel=0.1)

    with fits.open(tmp_path / "noise_mask_run.fits") as hdul:
        assert np.array_equal(hdul[0].data.astype(bool), products["noise_mask_cube"])
        assert hdul[0].header["NMASKRUL"] == "region"
        assert hdul[0].header["NMASKTHR"] == 2.0


def test_region_snr_none_keeps_the_disks(tmp_path):
    analysis = _analysis(tmp_path, noise_mask_region_snr=None)
    analysis.contrast_table_and_normalization(save=True)
    products = analysis.detection_products

    _, disk_table, _, _ = make_contrast_curve(analysis.detection_cube[0], yx_known_companion_position=[[0, 60]])
    assert _noise_at(products["contrast_tables"][0], 60) == _noise_at(disk_table, 60)
    assert {s["mask_rule"] for s in products["noise_mask_sources"]} == {"disk"}
    assert fits.getheader(tmp_path / "noise_mask_run.fits")["NMASKRUL"] == "disk"


def test_old_pickled_config_runs_the_normalisation(tmp_path, monkeypatch):
    old = _old_pickled_config(monkeypatch, yx_known_companion_position=[[0, 60]])
    analysis = _analysis(tmp_path, reduction_parameters=old)
    analysis.contrast_table_and_normalization()
    assert analysis.detection_products["noise_mask_sources"][0]["mask_rule"] == "region"


def test_field_rotation_sets_the_half_arcs(tmp_path, monkeypatch):
    import trap.detection

    received = []

    def recording(*args, **kwargs):
        received.append(kwargs["half_arcs"])
        return companion_noise_masks(*args, **kwargs)

    monkeypatch.setattr(trap.detection, "companion_noise_masks", recording)
    _analysis(tmp_path, field_rotation=ROTATION).contrast_table_and_normalization()
    _analysis(tmp_path, field_rotation=None).contrast_table_and_normalization()
    assert received[0] == pytest.approx([60 * ROTATION / 2])
    assert received[-1] is None


def test_field_rotation_widens_the_region(tmp_path):
    still = _analysis(tmp_path, field_rotation=None)
    rotating = _analysis(tmp_path, field_rotation=ROTATION)
    still.contrast_table_and_normalization()
    rotating.contrast_table_and_normalization()
    assert (
        rotating.detection_products["noise_mask_sources"][0]["mask_pixels"]
        > still.detection_products["noise_mask_sources"][0]["mask_pixels"]
    )
    assert _noise_at(rotating.detection_products["contrast_tables"][0], 60) < _noise_at(
        still.detection_products["contrast_tables"][0], 60
    )


def test_missing_fwhm_falls_back_to_disks(tmp_path, caplog):
    analysis = _analysis(tmp_path)
    analysis.instrument.fwhm = None
    with caplog.at_level(logging.INFO, logger="trap.detection"):
        analysis.contrast_table_and_normalization()
    assert {s["mask_rule"] for s in analysis.detection_products["noise_mask_sources"]} == {"disk"}
    assert "FWHM" in caplog.text


def test_make_radial_profile_records_its_fallbacks():
    yy, xx = np.indices((101, 101)) - 50
    data = np.random.default_rng(0).normal(size=(101, 101))
    covered = np.abs(np.hypot(yy, xx) - 30) < 4
    fallbacks = []
    make_radial_profile(data, (1, 50), known_companion_mask=covered, fallback_separations=fallbacks)
    assert fallbacks
    assert all(abs(separation - 30) <= 4 for separation in fallbacks)


def test_warning_when_a_bright_source_leaves_its_ring_too_few_pixels(tmp_path, monkeypatch, caplog):
    import trap.detection

    yy, xx = np.indices((201, 201)) - 100
    whole_ring = np.abs(np.hypot(yy, xx) - 60) < 4

    def ring_mask(normalized_image, positions, **kwargs):
        return whole_ring, [{"mask_snr": 200.0, "mask_rule": "region", "mask_pixels": int(whole_ring.sum())}]

    monkeypatch.setattr(trap.detection, "companion_noise_masks", ring_mask)
    with caplog.at_level(logging.WARNING, logger="trap.detection"):
        _analysis(tmp_path).contrast_table_and_normalization()
    assert "falls back to the unmasked ring" in caplog.text


def test_no_warning_for_fallbacks_far_from_bright_sources(caplog):
    with caplog.at_level(logging.WARNING, logger="trap.detection"):
        _warn_about_unmasked_rings([{"mask_snr": 200.0}], [60.0], [50, 70], 4.0, 0)
        _warn_about_unmasked_rings([{"mask_snr": 50.0}], [60.0], [60], 4.0, 0)
    assert caplog.text == ""
    with caplog.at_level(logging.WARNING, logger="trap.detection"):
        _warn_about_unmasked_rings([{"mask_snr": 200.0}], [60.0], [58, 60], 4.0, 0)
    assert "58, 60" in caplog.text


def test_field_rotation_from_parallactic_angles():
    analysis = DetectionAnalysis()
    assert analysis.field_rotation is None
    analysis._set_field_rotation(np.array([170.0, 179.0, -175.0]))
    assert analysis.field_rotation == pytest.approx(np.radians(15.0))
