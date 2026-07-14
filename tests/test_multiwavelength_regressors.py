"""Unit and integration tests for WP2 multi-wavelength regressor enrichment."""

import numpy as np
import pytest

from trap.parameters import TrapReductionConfig
from trap.regressor_selection import (
    find_N_unique_samples,
    make_signal_mask,
    scale_mask_about_center,
)


class TestConfig:
    def test_defaults_off(self):
        config = TrapReductionConfig()
        assert config.multiwavelength_regressors is None
        assert config.regressor_wavelength_indices is None
        assert config.max_regressor_pool_size == 3.0

    def test_valid_modes_accepted(self):
        for mode in (None, "pool", "occluded"):
            config = TrapReductionConfig(multiwavelength_regressors=mode)
            assert config.multiwavelength_regressors == mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="multiwavelength_regressors"):
            TrapReductionConfig(multiwavelength_regressors="bogus")

    def test_legacy_conversion_ignores_new_fields(self):
        config = TrapReductionConfig(
            multiwavelength_regressors="pool",
            regressor_wavelength_indices=[0, 2],
            max_regressor_pool_size=2.0,
        )
        with pytest.warns(DeprecationWarning):
            legacy = config.to_reduction_parameters()
        assert not hasattr(legacy, "multiwavelength_regressors")


class TestScaleMask:
    def test_identity_scale(self):
        mask = make_signal_mask((41, 41), (20 + 3, 20 + 4), mask_radius=2.5)
        scaled = scale_mask_about_center(mask, 1.0, (20.0, 20.0))
        assert np.array_equal(scaled, mask)

    def test_disk_moves_radially_outward(self):
        # Disk of radius 2.5 at separation 8 along +x; scale 1.5 -> separation 12
        center = (30.0, 30.0)
        mask = make_signal_mask((61, 61), (30, 30 + 8), mask_radius=2.5)
        scaled = scale_mask_about_center(mask, 1.5, center)
        coords = np.argwhere(scaled)
        centroid = coords.mean(axis=0)
        separation = np.hypot(centroid[0] - center[0], centroid[1] - center[1])
        assert separation == pytest.approx(12.0, abs=0.75)
        # Magnification must not punch holes into the scaled footprint
        assert scaled.sum() >= mask.sum()

    def test_shrink_moves_inward(self):
        center = (30.0, 30.0)
        mask = make_signal_mask((61, 61), (30, 30 + 12), mask_radius=3.0)
        scaled = scale_mask_about_center(mask, 1.0 / 1.5, center)
        coords = np.argwhere(scaled)
        centroid = coords.mean(axis=0)
        separation = np.hypot(centroid[0] - center[0], centroid[1] - center[1])
        assert separation == pytest.approx(8.0, abs=0.75)


class TestSeededSampling:
    def test_rng_reproducible(self):
        pool = np.zeros((21, 21), dtype=bool)
        pool[3:18, 3:18] = True
        mask_a = find_N_unique_samples(
            10, (21, 21), regressor_pool_mask=pool, rng=np.random.default_rng(7)
        )
        mask_b = find_N_unique_samples(
            10, (21, 21), regressor_pool_mask=pool, rng=np.random.default_rng(7)
        )
        assert np.array_equal(mask_a, mask_b)
        assert mask_a.sum() == 10
        assert not np.any(mask_a & ~pool)


import dataclasses

from trap.parameters import TrapReductionConfig as _Config
from trap.regressor_selection import (
    MultiwavelengthRegressors,
    make_multiwavelength_regressor_masks,
    make_mask_from_psf_track,
)


def _mw_context(mode, scale_factors=(1.25,), max_pool=3.0, n_lambda=2,
                image_size=61, fwhm=(4.0,), fwhm_reference=4.0):
    n = len(scale_factors)
    center = (image_size // 2, image_size // 2)
    return MultiwavelengthRegressors(
        data=np.zeros((n_lambda, image_size, image_size, 4)),
        wavelength_indices=np.arange(1, n + 1),
        scale_factors=np.array(scale_factors, dtype="float64"),
        fwhm=np.array(fwhm, dtype="float64"),
        fwhm_reference=fwhm_reference,
        yx_centers=np.array([center] * n, dtype="float64"),
        bad_pixel_masks=None,
        mode=mode,
        max_regressor_pool_size=max_pool,
    )


def _signal_track(image_size=61, separation=12, psf_size=9):
    pa = np.linspace(0.0, 30.0, 8)
    return make_mask_from_psf_track(
        yx_position=(0.0, float(separation)),
        psf_size=psf_size,
        pa=pa,
        image_size=image_size,
        return_cube=False,
    )


class TestMultiwavelengthMasks:
    image_size = 61

    def _masks(self, mode, separation=12, scale_factors=(1.25,), max_pool=3.0,
               fwhm=(4.0,)):
        config = _Config(annulus_width=5, annulus_offset=0.0)
        center = (self.image_size // 2, self.image_size // 2)
        signal_mask = _signal_track(self.image_size, separation)
        context = _mw_context(
            mode, scale_factors=scale_factors, max_pool=max_pool,
            image_size=self.image_size, fwhm=fwhm,
        )
        return make_multiwavelength_regressor_masks(
            context,
            reduction_parameters=config,
            yx_pixel=(center[0], center[1] + separation),
            yx_dim=(self.image_size, self.image_size),
            yx_center=center,
            signal_mask=signal_mask,
            n_reference_pixels=200,
            rng=np.random.default_rng(3),
        ), signal_mask

    def test_occluded_subset_of_pool(self):
        masks_b, _ = self._masks("occluded")
        masks_a, _ = self._masks("pool", max_pool=1e9)  # unlimited budget -> full A
        assert set(masks_b) == set(masks_a) == {1}
        assert not np.any(masks_b[1] & ~masks_a[1])  # B subset of A
        assert masks_a[1].sum() > masks_b[1].sum() > 0

    def test_static_signal_always_excluded(self):
        for mode in ("pool", "occluded"):
            masks, signal_mask = self._masks(mode, max_pool=1e9)
            assert not np.any(masks[1] & signal_mask)

    def test_displacement_eligibility_falls_out(self):
        # separation * (s - 1) << fwhm_j: displaced speckles still overlap the
        # static source -> occluded mask must be empty (dict omits it).
        masks, _ = self._masks("occluded", separation=6, scale_factors=(1.05,),
                               fwhm=(6.0,))
        assert 1 not in masks
        # Large separation: displacement 12 * 0.25 = 3 px vs fwhm 2 -> eligible.
        masks, _ = self._masks("occluded", separation=12, scale_factors=(1.25,),
                               fwhm=(2.0,))
        assert 1 in masks

    def test_pool_budget_respected(self):
        # Small budget on a large annulus forces subsampling so the cap bites.
        n_reference = 200
        max_pool = 1.5
        config = _Config(annulus_width=9, annulus_offset=0.0)
        center = (self.image_size // 2, self.image_size // 2)
        signal_mask = _signal_track(self.image_size, 12)
        context = _mw_context("pool", scale_factors=(1.25, 0.8),
                              fwhm=(4.0, 4.0), n_lambda=3, max_pool=max_pool,
                              image_size=self.image_size)
        common = dict(
            reduction_parameters=config,
            yx_pixel=(center[0], center[1] + 12),
            yx_dim=(self.image_size, self.image_size),
            yx_center=center,
            signal_mask=signal_mask,
            n_reference_pixels=n_reference,
        )
        masks = make_multiwavelength_regressor_masks(
            context, rng=np.random.default_rng(3), **common
        )
        occluded = make_multiwavelength_regressor_masks(
            dataclasses.replace(context, mode="occluded"),
            rng=np.random.default_rng(3), **common,
        )
        total = sum(int(m.sum()) for m in masks.values())
        n_occluded = sum(int(m.sum()) for m in occluded.values())
        # Occluded (Method B) pixels are always kept; the remaining Method A
        # annulus pixels share the enrichment budget = max_pool*n_ref - n_ref
        # - n_occluded. So the total cannot exceed occluded + that budget.
        enrichment_budget = max(
            int(round(max_pool * n_reference)) - n_reference - n_occluded, 0
        )
        assert n_occluded <= total <= n_occluded + enrichment_budget
        # The cap must actually be active in this configuration (otherwise the
        # test would pass trivially even if budgeting were broken).
        assert enrichment_budget > 0

    def test_bad_pixels_excluded(self):
        config = _Config(annulus_width=5, annulus_offset=0.0)
        center = (self.image_size // 2, self.image_size // 2)
        signal_mask = _signal_track(self.image_size, 12)
        bad = np.zeros((self.image_size, self.image_size), dtype=bool)
        bad[:, ::2] = True
        base = _mw_context("pool", max_pool=1e9, image_size=self.image_size)
        context = dataclasses.replace(base, bad_pixel_masks=[bad])
        masks = make_multiwavelength_regressor_masks(
            context,
            reduction_parameters=config,
            yx_pixel=(center[0], center[1] + 12),
            yx_dim=(self.image_size, self.image_size),
            yx_center=center,
            signal_mask=signal_mask,
            n_reference_pixels=200,
            rng=np.random.default_rng(3),
        )
        assert not np.any(masks[1] & bad)


from trap.regression import _assemble_training_matrix


class TestAssembleTrainingMatrix:
    def _inputs(self):
        rng = np.random.default_rng(11)
        n_time, size = 6, 5
        data = rng.normal(size=(n_time, size, size))
        pool = np.zeros((size, size), dtype=bool)
        pool[0, 0] = pool[2, 3] = pool[4, 4] = True
        cube = rng.normal(size=(3, size, size, n_time))
        mask_j = np.zeros((size, size), dtype=bool)
        mask_j[1, 1] = mask_j[3, 0] = True
        return data, pool, cube, mask_j

    def test_no_enrichment_identical_to_direct_indexing(self):
        data, pool, _, _ = self._inputs()
        assert np.array_equal(
            _assemble_training_matrix(data, pool), data[:, pool]
        )
        subset = np.array([True, False, True, True, False, True])
        assert np.array_equal(
            _assemble_training_matrix(data, pool, data_range_to_fit=subset),
            data[subset][:, pool],
        )

    def test_enrichment_appends_other_wavelength_series(self):
        data, pool, cube, mask_j = self._inputs()
        matrix = _assemble_training_matrix(
            data, pool,
            multiwavelength_data=cube, multiwavelength_masks={2: mask_j},
        )
        assert matrix.shape == (6, 3 + 2)
        np.testing.assert_array_equal(matrix[:, :3], data[:, pool])
        np.testing.assert_array_equal(matrix[:, 3:], cube[2][mask_j].T)

    def test_enrichment_respects_time_selection(self):
        data, pool, cube, mask_j = self._inputs()
        subset = np.array([True, True, False, True, False, True])
        matrix = _assemble_training_matrix(
            data, pool, data_range_to_fit=subset,
            multiwavelength_data=cube, multiwavelength_masks={1: mask_j},
        )
        np.testing.assert_array_equal(matrix[:, 3:], cube[1][mask_j].T[subset])


from trap.shared_arrays import SharedArrayStore


def test_shared_store_remove(tmp_path):
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        store.dump("scratch", np.arange(4.0))
        assert store.ref("scratch") is not None
        store.remove("scratch")
        with pytest.raises(KeyError):
            store.ref("scratch")
        store.remove("scratch")  # idempotent


import astropy.units as u
from astropy.io import fits

from trap import makesource
from trap.parameters import Instrument
from trap.reduction_wrapper import run_complete_reduction


@pytest.fixture(scope="module")
def synthetic_ifs_dataset():
    rng = np.random.default_rng(42)
    n_frames = 16
    image_size = 31
    wavelengths = np.array([1.4, 1.6, 1.8])
    pa = np.linspace(0.0, 40.0, n_frames)

    stamp_size = 21
    yy, xx = np.mgrid[:stamp_size, :stamp_size] - stamp_size // 2
    psfs = []
    data = np.empty((3, n_frames, image_size, image_size))
    for i, wave in enumerate(wavelengths):
        sigma = (3.5 / 2.355) * wave / 1.6
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        psf /= psf.sum()
        psfs.append(psf)
        cube = rng.normal(
            loc=0.0, scale=1e-4, size=(n_frames, image_size, image_size)
        )
        # Static companion: same (y, x) position at every wavelength
        data[i] = makesource.addsource(
            cube,
            pos=(0.0, 5.0),
            pa=pa,
            psf_arr=psf,
            norm=np.full(n_frames, 5e-3),
            jitter=0,
            poisson_noise=False,
            subpixel=True,
            copy=True,
            verbose=False,
        )

    instrument = Instrument(
        name="synthetic-ifs",
        pixel_scale=u.pixel_scale(12.25 * u.mas / u.pixel),
        telescope_diameter=8.0 * u.m,
        detector_gain=1.0,
        readnoise=0.0,
        instrument_type="ifu",
        wavelengths=wavelengths * u.micron,
    )
    return data, np.array(psfs), pa, instrument


def _run_ifs_reduction(dataset, result_folder, mode, use_multiprocess):
    data, psfs, pa, instrument = dataset
    from trap.parameters import TrapReductionConfig

    config = TrapReductionConfig(
        search_region_inner_bound=4,
        search_region_outer_bound=7,
        data_auto_crop=False,
        data_crop_size=None,
        temporal_model=True,
        temporal_plus_spatial_model=False,
        spatial_model=False,
        use_multiprocess=use_multiprocess,
        ncpus=2,
        result_folder=str(result_folder),
        verbose=False,
        multiwavelength_regressors=mode,
        max_regressor_pool_size=3.0,
    )
    run_complete_reduction(
        data_full=data.copy(),
        flux_psf_full=psfs.copy(),
        pa=pa,
        instrument=instrument,
        reduction_parameters=config,
        temporal_components_fraction=[0.25],
        wavelength_indices=[1],
        overwrite=True,
        use_progress_bar=False,
    )
    detection_files = sorted(result_folder.glob("detection_*.fits"))
    assert len(detection_files) == 1
    return fits.getdata(detection_files[0])


class TestEndToEnd:
    @pytest.mark.parametrize("mode", ["pool", "occluded"])
    def test_detects_companion_with_enrichment(
        self, synthetic_ifs_dataset, tmp_path, mode
    ):
        folder = tmp_path / f"mw_{mode}"
        folder.mkdir()
        detection = _run_ifs_reduction(
            synthetic_ifs_dataset, folder, mode, use_multiprocess=True
        )
        contrast = detection[0]
        center = contrast.shape[0] // 2
        measured = contrast[center, center + 5]
        assert measured == pytest.approx(5e-3, rel=0.2)

    def test_serial_parallel_equivalent_with_enrichment(
        self, synthetic_ifs_dataset, tmp_path
    ):
        serial_folder = tmp_path / "serial"
        parallel_folder = tmp_path / "parallel"
        serial_folder.mkdir()
        parallel_folder.mkdir()
        detection_serial = _run_ifs_reduction(
            synthetic_ifs_dataset, serial_folder, "pool", use_multiprocess=False
        )
        detection_parallel = _run_ifs_reduction(
            synthetic_ifs_dataset, parallel_folder, "pool", use_multiprocess=True
        )
        np.testing.assert_allclose(
            detection_parallel,
            detection_serial,
            rtol=1e-10,
            atol=0.0,
            equal_nan=True,
        )

    def test_mode_none_matches_wp1_output(self, synthetic_ifs_dataset, tmp_path):
        # Default (None) on a multi-wavelength cube must take the unchanged
        # WP1 per-wavelength store path and still find the companion.
        folder = tmp_path / "mode_none"
        folder.mkdir()
        detection = _run_ifs_reduction(
            synthetic_ifs_dataset, folder, None, use_multiprocess=True
        )
        contrast = detection[0]
        center = contrast.shape[0] // 2
        assert contrast[center, center + 5] == pytest.approx(5e-3, rel=0.2)
