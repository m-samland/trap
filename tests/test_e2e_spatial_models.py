"""The spatial model and the spatial model after TRAP run on real SPHERE-IRDIS data.

Both models are off by default and had stopped running: the spatial model failed on
numpy 2, and ``temporal_plus_spatial_model`` failed in map mode because the temporal
step returned no noise model (#27). These tests reduce the 51 Eri b K1 cut-out of
``test_e2e_51eri.py`` and check that both models run. The spatial model on its own does
not recover 51 Eri b in this sequence (its contrast at the planet is slightly negative),
so only the spatial model after TRAP is checked against the planet.
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


@pytest.fixture(scope="module")
def data_file():
    if not DATA.exists():
        pytest.skip(f"{DATA.name} is not available (examples/ is not part of the sdist)")
    with fits.open(DATA) as hdul:
        yield hdul


def _reduce(data_file, result_folder, fractions=(0.2,), **settings):
    config = trap_config_for_irdis()
    reduction_config = config.reduction.merge(
        coronagraph_transmission=data_file["TRANSMISSION"].data.astype("f8"),
        result_folder=str(result_folder),
        use_multiprocess=True,
        ncpus=min(4, os.cpu_count() or 1),
        **settings,
    )
    return run_complete_reduction(
        data_full=data_file["SCI"].data.astype("f8"),
        inverse_variance_full=data_file["IVAR"].data.astype("f8"),
        bad_pixel_mask_full=data_file["BADPIX"].data.astype(bool),
        xy_image_centers=data_file["CENTERS"].data.astype("f8"),
        flux_psf_full=data_file["PSF"].data.astype("f8"),
        pa=data_file["DEROT_ANGLE"].data.astype("f8"),
        instrument=config.get_instrument("DB_K12", wavelengths=(data_file["WAVELENGTH"].data * u.nm).to(u.micron)),
        reduction_parameters=reduction_config,
        temporal_components_fraction=list(fractions),
        wavelength_indices=[0],
        overwrite=True,
        use_progress_bar=False,
    )


def test_spatial_model_uses_the_spatial_component_fraction(data_file, tmp_path):
    results = _reduce(
        data_file, tmp_path, temporal_model=False, spatial_model=True, spatial_components_fraction=0.2,
        reduce_single_position=True, guess_position=YX_PLANET,
    )
    spatial = results["0.2"]["0"]["spatial"]
    assert np.isfinite(spatial.measured_contrast)
    assert np.isfinite(spatial.contrast_uncertainty) and spatial.contrast_uncertainty > 0
    # About 0.2 of the roughly 240 frames outside the protection angle, not a fixed 5.
    assert 30 <= spatial.number_of_pca_regressors <= 60


def _contrast_map(folder, suffix):
    files = sorted(folder.glob(f"detection_lam00_*_{suffix}.fits"))
    assert len(files) == 1, files
    data = fits.getdata(files[0])
    return data.reshape(-1, *data.shape[-2:])[0]


def test_spatial_model_after_trap_runs_in_map_mode_and_keeps_the_planet(data_file, tmp_path):
    _reduce(
        data_file, tmp_path, temporal_plus_spatial_model=True,
        search_region_inner_bound=36, search_region_outer_bound=37,
    )
    combined = _contrast_map(tmp_path, "temporal_plus_spatial")
    temporal = _contrast_map(tmp_path, "temporal")
    assert np.isfinite(combined).sum() > 100
    center = combined.shape[-1] // 2
    planet = (center + round(YX_PLANET[0]), center + round(YX_PLANET[1]))
    assert combined[planet] == pytest.approx(temporal[planet], rel=0.25)


def test_spatial_model_runs_once_for_several_fractions(data_file, tmp_path):
    _reduce(
        data_file, tmp_path, fractions=(0.1, 0.2), spatial_model=True,
        search_region_inner_bound=36, search_region_outer_bound=37,
    )
    assert len(list(tmp_path.glob("detection_lam00_*_spatial.fits"))) == 1
    assert len(list(tmp_path.glob("detection_lam00_*_temporal.fits"))) == 2
    assert sorted(path.name for path in tmp_path.glob("reduction_config_*.obj")) == [
        "reduction_config_delta0.50_spatialfrac0.30_spatial.obj",
        "reduction_config_ncomp026_frac0.10_temporal.obj",
        "reduction_config_ncomp051_frac0.20_temporal.obj",
    ]
    for fraction, reduction_type in ((0.1, "temporal"), (0.2, "temporal"), (0.3, "spatial")):
        analysis = DetectionAnalysis()
        analysis.read_output(fraction, result_folder=str(tmp_path), reduction_type=reduction_type)
        assert list(analysis.wavelength_indices) == [0]


def test_spatial_only_reduction_with_several_fractions_runs_the_spatial_model_once(data_file, tmp_path):
    results = _reduce(
        data_file, tmp_path, fractions=(0.1, 0.2), temporal_model=False, spatial_model=True,
        reduce_single_position=True, guess_position=YX_PLANET,
    )
    assert list(results) == ["0.1"]
    assert "spatial" in results["0.1"]["0"]
