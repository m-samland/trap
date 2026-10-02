"""End-to-end reduction and detection of 51 Eri b on real SPHERE-IRDIS data.

The input is the K1 cut-out in ``examples/test_data`` (see
``make_51eri_irdis_db_k12.py`` there for its provenance): all 256 frames of the
2015-09-24 DB_K12 sequence, with the inverse variance, bad-pixel map, per-frame
centres and coronagraph transmission that spherical passes to TRAP. The planet sits
close to the detection limit in this sequence, so neither the frames nor the inverse
variance can be dropped to save time: with every second frame, or without the
inverse variance, the planet is no longer the strongest signal.

The reference values below were frozen from a run of this test on macOS (arm64).
They reproduce the spherical reference reduction of 2026-09-14 (trap 2.0.1 on the
full frames) to all printed digits. The tolerances absorb floating-point differences
between platforms and still catch any change to the reduction or the fit.

Takes about two minutes on four cores, so it is deselected by default; run it with
``pytest -m e2e``.
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from trap.detection import DetectionAnalysis
from trap.parameters import trap_config_for_irdis
from trap.reduction_wrapper import run_complete_reduction

pytestmark = pytest.mark.e2e

DATA = Path(__file__).parents[1] / "examples" / "test_data" / "51eri_irdis_db_k12_2015-09-24_K1.fits"
YX_PLANET = (-35.95, -8.43)
COMPONENT_FRACTION = 0.2

# Frozen reference values.
N_POSITIONS = 2792
MAP_PEAK_SNR = 5.847
MAP_PEAK_CONTRAST = 7.1495e-06
MAX_SNR_AWAY_FROM_PLANET = 3.393
CONTRAST_CURVE_37PX = 1.3199e-06  # contrast_50 column at 37 px
XY_RELATIVE = (-8.5511, -36.1135)
NORM_SNR_FIT_FREE = 6.214
FIT_CONTRAST = 7.539e-06


@pytest.fixture(scope="module")
def inputs():
    if not DATA.exists():
        pytest.skip(f"{DATA.name} is not available (examples/ is not part of the sdist)")
    with fits.open(DATA) as hdul:
        return {
            "data_full": hdul["SCI"].data.astype("f8"),
            "inverse_variance_full": hdul["IVAR"].data.astype("f8"),
            "bad_pixel_mask_full": hdul["BADPIX"].data.astype(bool),
            "xy_image_centers": hdul["CENTERS"].data.astype("f8"),
            "flux_psf_full": hdul["PSF"].data.astype("f8"),
            "pa": hdul["DEROT_ANGLE"].data.astype("f8"),
            "wavelengths": (hdul["WAVELENGTH"].data * u.nm).to(u.micron),
            "transmission": hdul["TRANSMISSION"].data.astype("f8"),
        }


@pytest.fixture(scope="module")
def analysis(inputs, tmp_path_factory):
    """Reduce the sequence and normalise the detection map, as spherical does."""
    inputs = dict(inputs)
    wavelengths = inputs.pop("wavelengths")
    transmission = inputs.pop("transmission")
    config = trap_config_for_irdis()
    reduction_config = config.reduction.merge(
        search_region_inner_bound=31,
        search_region_outer_bound=43,
        yx_known_companion_position=list(YX_PLANET),
        coronagraph_transmission=transmission,
        result_folder=str(tmp_path_factory.mktemp("e2e_51eri")),
        use_multiprocess=True,
        ncpus=min(4, os.cpu_count() or 1),
    )
    run_complete_reduction(
        instrument=config.get_instrument("DB_K12", wavelengths=wavelengths),
        reduction_parameters=reduction_config,
        temporal_components_fraction=[COMPONENT_FRACTION],
        wavelength_indices=[0],
        overwrite=True,
        use_progress_bar=False,
        **inputs,
    )
    analysis = DetectionAnalysis()
    analysis.read_output(
        COMPONENT_FRACTION,
        result_folder=reduction_config.result_folder,
        reduction_type="temporal",
        read_parameters=True,
    )
    analysis.contrast_table_and_normalization(save=False, mask_above_sigma=5.0)
    return analysis


def _offsets_from_planet(shape):
    yy, xx = np.indices(shape)
    return np.hypot(yy - shape[0] // 2 - YX_PLANET[0], xx - shape[1] // 2 - YX_PLANET[1])


def test_detection_map_covers_the_search_annulus(analysis):
    snr = analysis.detection_products["normalized_detection_cube"][0]
    assert np.isfinite(snr).sum() == N_POSITIONS


def test_planet_is_the_only_significant_signal(analysis):
    snr = analysis.detection_products["normalized_detection_cube"][0]
    contrast = analysis.detection_cube[0, 0]
    offsets = _offsets_from_planet(snr.shape)

    peak = np.nanargmax(snr)
    assert offsets.flat[peak] < 1.0
    assert snr.flat[peak] == pytest.approx(MAP_PEAK_SNR, rel=0.05)
    assert contrast.flat[peak] == pytest.approx(MAP_PEAK_CONTRAST, rel=0.05)
    assert np.nanmax(np.where(offsets > 6, snr, np.nan)) == pytest.approx(MAX_SNR_AWAY_FROM_PLANET, abs=0.5)


def test_contrast_curve(analysis):
    table = analysis.detection_products["contrast_table"]
    curve = table[(table["sep (pix)"] >= 31) & (table["sep (pix)"] <= 43)]
    assert np.isfinite(curve["contrast_50"]).all()
    at_37 = curve.loc[curve["sep (pix)"] == 37, "contrast_50"].item()
    assert at_37 == pytest.approx(CONTRAST_CURVE_37PX, rel=0.05)


def test_candidate_fit_recovers_the_planet(analysis, inputs):
    inputs = {k: v for k, v in inputs.items() if k not in ("wavelengths", "transmission")}
    analysis.detection_and_characterization(
        temporal_components_fraction=[COMPONENT_FRACTION],
        candidate_threshold=4.75,
        detection_threshold=5.0,
        **inputs,
    )
    table = analysis.validated_companion_table_short
    assert table is not None and len(table) == 1
    planet = table.iloc[0]

    # 0.25 px is about 0.17 of the K-band PSF sigma.
    assert planet["x_relative"] == pytest.approx(XY_RELATIVE[0], abs=0.25)
    assert planet["y_relative"] == pytest.approx(XY_RELATIVE[1], abs=0.25)
    assert planet["norm_snr_fit_free"] == pytest.approx(NORM_SNR_FIT_FREE, rel=0.05)
    assert planet["contrast"] == pytest.approx(FIT_CONTRAST, rel=0.05)
    for column in ("x_relative_sigma", "y_relative_sigma", "separation_sigma", "position_angle_sigma"):
        assert np.isfinite(planet[column]) and planet[column] > 0
