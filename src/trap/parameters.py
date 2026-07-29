"""
Routines used in TRAP

Reductions are configured with the dataclasses defined here:

    from trap.parameters import trap_config_for_ifs

    config = trap_config_for_ifs()
    reduction_config = config.reduction.merge(result_folder="./results")

@author: Matthias Samland
         MPIA Heidelberg
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy import units as u

logger = logging.getLogger(__name__)


class Instrument(object):
    """Important information on the instrument.

    Parameters
    ----------
    name : str
        Name of the instrument used.
    pixel_scale : `~astropy.units.Quantity`
        The pixel scale either in units of angle/pixel or pixel/angle.
    telescope_diameter : `~astropy.units.Quantity`
        The diameter of the telescopy in units of length.
    detector_gain : float
        The detector gain (electrons/ADU).
    readnoise : float
        The detector read noise (e rms/pix/readout).
    instrument_type : str, optional
        Can take values 'phot', 'ifu' or None. Only used for spectral
        template matching in detection.
        Default is 'photometry'.
    wavelengths : `~astropy.units.Quantity`
        The (central) wavelengths of the data as sampled by this instrument.
        Effective wavelength for photometric observations.
    spectral_resolution : float, optional
        The spectral resolution of the instrument. Only needed if
        'instrument_type' == 'ifu'.
        Default is None.
    filters : array_like??, optional
        The filter curves for each channel observed. species object?
        Default is None.
    transmission : array_like??, optional
        The common-path instrument and atmospheric transmission profile.

    Attributes
    ----------
    name
    pixel_scale
    telescope_diameter
    detector_gain
    readnoise
    instrument_type
    wavelengths
    spectral_resolution
    filters
    transmission

    """

    def __init__(
            self, name, pixel_scale, telescope_diameter, detector_gain=1.0, readnoise=0.,
            instrument_type='photometry', wavelengths=None, spectral_resolution=None, filters=None,
            transmission=None):
        self.name = name
        self.pixel_scale = pixel_scale
        self.telescope_diameter = telescope_diameter
        self.detector_gain = detector_gain
        self.readnoise = readnoise

        self.instrument_type = instrument_type
        if wavelengths is not None:
            self.wavelengths = np.zeros(wavelengths.shape)
            self.wavelengths[:] = wavelengths.value
            self.wavelengths = self.wavelengths * wavelengths.unit
        else:
            self.wavelengths = None

        self.spectral_resolution = spectral_resolution
        self.filters = filters
        self.transmission = transmission
        if self.wavelengths is not None:
            self.compute_fwhm()

    def compute_fwhm(self):
        """
        Diffraction-limited FWHM in detector pixels.
        Assumes:
        * self.wavelengths  –  Quantity, e.g. µm
        * self.telescope_diameter – Quantity, e.g. m
        * self.pixel_scale  –  u.pixel_scale(<angle/pix>) equivalency
        """
        if self.wavelengths is None:
            self.fwhm = None
            return

        ratio = (self.wavelengths / self.telescope_diameter).to(
            u.dimensionless_unscaled
        )
        # 1.029 λ/D in **radians**
        # full width at half maximum of an Airy PSF rather than the first-null diameter (1.22 λ/D)
        airy_fwhm = 1.029 * ratio * u.rad

        # convert angle → pixels
        # choose .to(...) if you want a Quantity with unit pix,
        # or .to_value(...) to keep only the number.
        self.fwhm = airy_fwhm.to(u.pix, equivalencies=self.pixel_scale)


# ============================================================================
# NEW DATACLASS-BASED CONFIGURATION SYSTEM (PREFERRED)
# ============================================================================

# -------- helpers -----------------------------------------------------------

def _to_dict(maybe_dataclass) -> Dict[str, Any]:
    """Accept either a mapping or one of the *Config dataclasses."""
    if isinstance(maybe_dataclass, dict):
        return maybe_dataclass          # already a dict
    return asdict(maybe_dataclass)       # unwrap dataclass


# -------- instrument configuration ------------------------------------------

# IRDIS DBI / broadband filter modes accepted by InstrumentConfig.to_instrument.
# Kept at module scope because InstrumentConfig uses slots=True.
_IRDIS_OBS_MODES = (
    "DB_K12", "DB_H23", "DB_H34", "DB_Y23", "DB_J23",
    "BB_H", "BB_K", "BB_J", "BB_Y", "BB_Ks",
)


@dataclass(slots=True)
class InstrumentConfig:
    """Configuration for TRAP instrument parameters."""
    name: str = "IFS"
    pixel_scale_arcsec_per_pixel: float = 0.00746
    telescope_diameter_m: float = 7.99
    detector_gain: float = 1.0
    readnoise: float = 0.0
    instrument_type: str = "ifu"
    spectral_resolution_yj: int = 55
    spectral_resolution_h: int = 35

    def merge(self, **kw) -> "InstrumentConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

    def to_instrument(self, obs_mode: str, wavelengths=None) -> Instrument:
        """Create ``Instrument`` from configuration.

        Parameters
        ----------
        obs_mode : str
            Observation mode. IFS: ``"OBS_YJ"`` or ``"OBS_H"``. IRDIS: any of
            ``DB_K12/H23/H34/Y23/J23`` or ``BB_H/K/J/Y/Ks``.
        wavelengths : astropy.units.Quantity, optional
            Wavelength array. If ``None``, ``Instrument.wavelengths`` is left
            unset.

        Returns
        -------
        Instrument
            TRAP ``Instrument`` instance. For IFS modes, ``spectral_resolution``
            comes from ``spectral_resolution_yj``/``spectral_resolution_h``.
            For IRDIS modes, ``spectral_resolution=None`` — DBI has no
            meaningful spectral R (2 discrete filters).
        """
        if obs_mode == 'OBS_YJ':
            spectral_resolution = self.spectral_resolution_yj
        elif obs_mode == 'OBS_H':
            spectral_resolution = self.spectral_resolution_h
        elif obs_mode in _IRDIS_OBS_MODES:
            spectral_resolution = None
        else:
            raise ValueError(f"Unsupported observation mode: {obs_mode}")

        return Instrument(
            name=self.name,
            pixel_scale=u.pixel_scale(self.pixel_scale_arcsec_per_pixel * u.arcsec / u.pixel),
            telescope_diameter=self.telescope_diameter_m * u.m,
            detector_gain=self.detector_gain,
            readnoise=self.readnoise,
            instrument_type=self.instrument_type,
            wavelengths=wavelengths,
            spectral_resolution=spectral_resolution,
            filters=None,
            transmission=None,
        )


# -------- stellar parameters ------------------------------------------------

@dataclass(slots=True)
class StellarParameters:
    """Stellar parameters for host star used in TRAP template matching."""
    teff: float = 8000.0      # Effective temperature [K]
    logg: float = 4.0          # Surface gravity [log g]
    feh: float = 0.0           # Metallicity [Fe/H]
    radius: float = 65.0       # Stellar radius [R_sun]
    distance: float = 30.0     # Distance [pc]

    def merge(self, **kw) -> "StellarParameters":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary format expected by TRAP."""
        return asdict(self)


# -------- detection parameters ----------------------------------------------

@dataclass(slots=True)
class DetectionParameters:
    """TRAP detection and characterization parameters."""
    candidate_threshold: float = 4.75
    detection_threshold: float = 5.0
    use_spectral_correlation: bool = False
    stellar_parameters: StellarParameters = field(default_factory=StellarParameters)
    search_radius: int = 11
    inner_mask_radius: int = 1
    good_fraction_threshold: float = 0.05
    theta_deviation_threshold: float = 25.0
    yx_fwhm_ratio_threshold: Tuple[float, float] = (1.1, 4.5)
    save_initial_detection_products: bool = True
    # Per-channel astrometry replaces the template-collapse position only when the
    # channels that individually clear `candidate_threshold` carry a meaningful
    # share of the data. On a 2-channel DBI observation one good channel (0.5) is
    # the case the override was designed for; on a 39-channel IFS cube a handful of
    # threshold-crossing channels is a noise-selected subset, not a cleaner
    # measurement. See docs/llm_reference/ and the 51 Eri IFS benchmark.
    per_channel_min_channel_fraction: float = 0.5
    # Speckle residuals are strongly correlated between neighbouring IFS channels,
    # so the per-channel inverse-variance combination must not be allowed to shrink
    # sigma below the best contributing channel unless independence is established.
    per_channel_independent_channels: bool = False

    def merge(self, **kw) -> "DetectionParameters":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)


# -------- reduction parameters wrapper --------------------------------------

@dataclass(slots=True, frozen=True)
class TrapReductionConfig:
    """Immutable configuration for TRAP reduction parameters.

    This frozen dataclass holds all user-provided reduction settings.
    Use ``merge(**kw)`` to create a copy with selected fields overridden.
    Pipeline-derived runtime values live in :class:`ReductionRuntimeState`.
    """
    # Search region parameters
    search_region: Optional[Any] = None  # Binary mask of relative position to search for planets
    search_region_inner_bound: int = 1
    search_region_outer_bound: Optional[int] = 85
    reduction_mask_min_pixels: int = 30
    auto_footprint: bool = False
    oversampling: int = 1
    
    # Data preprocessing
    data_auto_crop: bool = True
    data_crop_size: Optional[int] = None
    right_handed: bool = True
    estimate_noise_from_data: bool = False
    use_progress_bar: bool = True

    # Model selection
    temporal_model: bool = True
    temporal_plus_spatial_model: bool = False
    second_stage_trap: bool = False
    remove_model_from_spatial_training: bool = True
    remove_bad_residuals_for_spatial_model: bool = True
    spatial_model: bool = False
    local_temporal_model: bool = False
    local_spatial_model: bool = False
    
    # Angular and spatial parameters
    protection_angle: float = 0.5
    spatial_components_fraction: float = 0.3
    spatial_components_fraction_after_trap: float = 0.1
    highpass_filter: Optional[float] = None
    
    # Known companion parameters
    remove_known_companions: bool = False
    yx_known_companion_position: Optional[Tuple[float, float]] = None
    known_companion_contrast: Optional[float] = None
    
    # Processing parameters
    use_multiprocess: bool = True
    ncpus: int = 4
    scratch_dir: Optional[Path] = None
    prefix: str = ''
    result_folder: str = './'
    
    # Injection and testing parameters
    inject_fake: bool = False
    true_position: Optional[Tuple[float, float]] = None
    true_contrast: Optional[float] = None
    read_injection_files: bool = False
    injection_sigma: float = 5.0
    reduce_single_position: bool = False
    guess_position: Optional[Tuple[float, float]] = None
    plot_all_diagnostics: bool = False
    fit_planet: bool = True
    
    # PCA and regressor parameters
    number_of_pca_regressors: int = 20
    yx_anamorphism: Any = field(default_factory=lambda: np.array([1., 1.]))
    pca_scaling: str = 'temp-median'
    method_of_regressor_selection: Optional[str] = None
    auxiliary_frame: Optional[Any] = None
    variance_prior_scaling: float = 1.0
    compute_inverse_once: bool = True
    
    # Mask parameters
    autosize_masks_in_lambda_over_d: bool = True
    reduction_mask_size_in_lambda_over_d: float = 1.
    signal_mask_size_in_lambda_over_d: float = 2.
    reduction_mask_psf_size: int = 19
    signal_mask_psf_size: int = 21
    threshold_pixel_by_contribution: float = 0.0
    target_pix_mask_radius: Optional[float] = None
    use_relative_position: bool = False
    coronagraph_transmission: Optional[Any] = None
    
    # Regressor selection
    annulus_width: int = 5
    annulus_offset: float = 0.0
    add_radial_regressors: bool = True
    include_opposite_regressors: bool = True
    
    # Multi-wavelength regressor enrichment (WP2, IFS data only)
    multiwavelength_regressors: Optional[str] = None  # None | "pool" | "occluded"
    regressor_wavelength_indices: Optional[Any] = None  # indices into wavelength axis; None = all
    max_regressor_pool_size: float = 3.0  # total pool budget in units of the single-wavelength pool

    # Processing control
    make_reconstructed_lightcurve: bool = True
    compute_residual_correlation: bool = False
    use_residual_correlation: bool = False
    use_signal_weighting: bool = False
    
    # Contrast curve and normalization
    contrast_curve: bool = True
    contrast_curve_sigma: float = 5.0
    normalization_width: int = 3
    companion_mask_radius: int = 11
    
    # Output control
    return_input_data: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.multiwavelength_regressors not in (None, "pool", "occluded", "sdi"):
            raise ValueError(
                "multiwavelength_regressors must be None, 'pool', 'occluded' or 'sdi', "
                f"got {self.multiwavelength_regressors!r}"
            )

    def merge(self, **kw) -> "TrapReductionConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)


# -------- runtime state (derived from config + data + instrument) -----------

@dataclass(slots=True)
class ReductionRuntimeState:
    """Derived runtime values computed from TrapReductionConfig + Instrument + data.

    Built once per ``run_complete_reduction`` call. Per-wavelength/component
    fields are updated via ``for_iteration()`` which returns a new instance.
    """

    # --- Category A: normalized inputs ---
    yx_anamorphism: np.ndarray
    yx_known_companion_position: Optional[np.ndarray] = None   # 2D (N,2) or None
    known_companion_contrast: Optional[np.ndarray] = None       # 2D (n_wave, N) or None

    # --- Category B: derived once ---
    search_region_outer_bound: int = 85
    data_crop_size: Optional[int] = None
    search_region: Optional[np.ndarray] = None       # binary mask
    ncpus: int = 4
    coronagraph_transmission_pix: Optional[np.ndarray] = None
    valid_pixel_mask_cropped: Optional[np.ndarray] = None
    reduction_mask_min_pixels: int = 30

    # --- Category C: per iteration (wavelength x component) ---
    number_of_pca_regressors: int = 20
    temporal_components_fraction: float = 0.15
    fwhm: float = 4.0
    reduction_mask_psf_size: int = 19
    signal_mask_psf_size: int = 21

    def for_iteration(
        self, *,
        number_of_pca_regressors: int,
        temporal_components_fraction: float,
        fwhm: float,
        reduction_mask_psf_size: int,
        signal_mask_psf_size: int,
    ) -> ReductionRuntimeState:
        """Return a new instance with per-iteration fields updated."""
        return replace(
            self,
            number_of_pca_regressors=number_of_pca_regressors,
            temporal_components_fraction=temporal_components_fraction,
            fwhm=fwhm,
            reduction_mask_psf_size=reduction_mask_psf_size,
            signal_mask_psf_size=signal_mask_psf_size,
        )


def _crop_footprint(valid_pixel_mask, data_crop_size, yx_center_full):
    """Crop a footprint mask to ``(data_crop_size, data_crop_size)`` centered
    on ``yx_center_full[0]``.

    Uses :class:`astropy.nddata.Cutout2D` with ``mode='partial'`` so a crop
    that partially extends past the input is padded with ``False`` instead of
    silently returning an empty slice. Returns ``None`` if the mask is
    ``None``, the crop size is ``None``, or the center is not finite (in the
    last case, logs a warning and disables the footprint intersection).
    """
    if valid_pixel_mask is None:
        return None
    mask_bool = np.asarray(valid_pixel_mask).astype("bool")
    if data_crop_size is None or yx_center_full is None:
        return mask_bool
    center = np.asarray(yx_center_full)[0]
    if not np.all(np.isfinite(center)):
        logger.warning(
            "valid_pixel_mask crop disabled: yx_center_full[0]=%s contains NaN/inf.",
            tuple(center),
        )
        return None
    from astropy.nddata import Cutout2D
    cutout = Cutout2D(
        mask_bool.astype(np.uint8),
        position=(float(center[1]), float(center[0])),
        size=(int(data_crop_size), int(data_crop_size)),
        mode="partial",
        fill_value=0,
    )
    return cutout.data.astype(bool)


def _derive_outer_bound(valid_mask, yx_center, min_pixels):
    """Largest integer radius whose annulus holds >= ``min_pixels`` valid pixels.

    Single-pass over the mask via bincount; O(H*W) work, no Python loop.
    """
    if not np.any(valid_mask):
        return 0
    yy, xx = np.indices(valid_mask.shape, dtype=np.float32)
    r = np.hypot(yy - yx_center[0], xx - yx_center[1])
    r_int = r.astype(np.int32)
    counts = np.bincount(
        r_int[valid_mask].ravel(),
        minlength=int(r.max()) + 1,
    )
    valid_radii = np.flatnonzero(counts >= min_pixels)
    return int(valid_radii.max()) if valid_radii.size else 0


def build_runtime_state(
    config: TrapReductionConfig,
    data_shape: tuple,
    stamp_sizes: np.ndarray,
    stamp_sizes_reduction: np.ndarray,
    max_shift: float,
    mas_per_pixel: Optional[float] = None,
    valid_pixel_mask: Optional[np.ndarray] = None,
    yx_center_full: Optional[np.ndarray] = None,
) -> ReductionRuntimeState:
    """Compute all derived values from user config + data properties.

    Encapsulates the normalization and derivation logic that was previously
    scattered through ``run_complete_reduction`` as mutations on
    ``reduction_parameters``.

    Parameters
    ----------
    config : TrapReductionConfig
        Immutable user configuration.
    data_shape : tuple
        Shape of data_full: ``(n_wave, n_time, ny, nx)``.
    stamp_sizes : np.ndarray
        Per-wavelength signal mask stamp sizes.
    stamp_sizes_reduction : np.ndarray
        Per-wavelength reduction mask stamp sizes.
    max_shift : float
        Maximum center shift across frames.

    Returns
    -------
    ReductionRuntimeState
    """
    from trap import regressor_selection  # local import to avoid circular

    coronagraph_transmission_pix = None
    transmission = getattr(config, "coronagraph_transmission", None)
    if transmission is not None:
        if mas_per_pixel is None:
            raise ValueError(
                "mas_per_pixel is required to use coronagraph_transmission."
            )
        from trap.makesource import coronagraph_transmission_to_pixels

        coronagraph_transmission_pix = coronagraph_transmission_to_pixels(
            transmission, mas_per_pixel
        )

    # --- Category A: normalize inputs ---
    yx_anamorphism = np.array(config.yx_anamorphism)

    yx_known_companion_position = None
    if config.yx_known_companion_position is not None:
        yx_known_companion_position = np.array(config.yx_known_companion_position)
        if yx_known_companion_position.ndim == 1:
            yx_known_companion_position = np.expand_dims(
                yx_known_companion_position, axis=0
            )
        elif yx_known_companion_position.ndim > 2:
            raise ValueError(
                "Dimensionality of known companion position array too large."
            )

    known_companion_contrast = None
    if (
        config.known_companion_contrast is not None
        and config.remove_known_companions
    ):
        assert (
            yx_known_companion_position is not None
        ), "No position for known companion given."

        known_companion_contrast = np.atleast_1d(
            np.array(config.known_companion_contrast)
        )
        number_of_wavelengths = data_shape[0]
        number_of_companions = yx_known_companion_position.shape[0]

        assert (
            known_companion_contrast.shape[-1] == number_of_companions
        ), "The same number of known companion position and contrasts need to be provided."

        if known_companion_contrast.ndim == 1 and number_of_wavelengths == 1:
            known_companion_contrast = np.expand_dims(
                known_companion_contrast, axis=0
            )
        elif known_companion_contrast.ndim == 1 and number_of_wavelengths > 1:
            raise ValueError(
                "For multi-wavelength data, a known contrast has to be defined "
                "for every wavelength."
            )
        elif known_companion_contrast.ndim > 2:
            raise ValueError(
                "Dimensionality of known companion contrast array too large."
            )

    # --- Category B: derived once ---
    if config.search_region_outer_bound is None:
        if valid_pixel_mask is None:
            raise ValueError(
                "search_region_outer_bound=None requires valid_pixel_mask "
                "to derive a maximum radius."
            )
        if yx_center_full is None:
            derivation_center = (
                valid_pixel_mask.shape[0] / 2.0,
                valid_pixel_mask.shape[1] / 2.0,
            )
        else:
            derivation_center = tuple(np.asarray(yx_center_full)[0])
        search_region_outer_bound = _derive_outer_bound(
            valid_pixel_mask,
            derivation_center,
            min_pixels=config.reduction_mask_min_pixels,
        )
        logger.info(
            "Auto outer bound (footprint-derived): %d px", search_region_outer_bound
        )
    else:
        search_region_outer_bound = config.search_region_outer_bound

    if config.reduce_single_position and config.guess_position is not None:
        guess_position_separation = np.sqrt(
            config.guess_position[0] ** 2 + config.guess_position[1] ** 2
        )
        logger.info("Adjusting outer bound to fit guess position")
        search_region_outer_bound = int(np.ceil(guess_position_separation) + 5)

    data_crop_size = config.data_crop_size
    if config.data_auto_crop:
        data_crop_size = np.ceil(
            search_region_outer_bound * 2
            + np.max(stamp_sizes) * np.sqrt(2)
            + max_shift
        )
        if config.add_radial_regressors:
            # NOTE: Hardcoded binary dilation used right now.
            data_crop_size += 14
        # Round up to odd number
        data_crop_size = int(data_crop_size // 2 * 2 + 1)

        if data_crop_size > data_shape[-1]:
            logger.info(
                "Requested crop %d exceeds input %d; clamping to input FoV.",
                data_crop_size, data_shape[-1],
            )
            data_crop_size = int(data_shape[-1])
            if data_crop_size % 2 == 0:
                data_crop_size -= 1
        logger.info("Auto crop size cropped data to: %s", data_crop_size)
        yx_dim = (data_crop_size, data_crop_size)
    else:
        if config.search_region is None:
            if config.data_crop_size is None:
                yx_dim = (data_shape[-2], data_shape[-1])
            else:
                yx_dim = (config.data_crop_size, config.data_crop_size)
        else:
            yx_dim = (
                config.search_region.shape[-2],
                config.search_region.shape[-1],
            )

    valid_pixel_mask_cropped = _crop_footprint(
        valid_pixel_mask, data_crop_size, yx_center_full,
    )

    search_region = config.search_region
    if search_region is None:
        search_region = regressor_selection.make_annulus_mask(
            config.search_region_inner_bound,
            search_region_outer_bound,
            yx_dim=yx_dim,
            oversampling=config.oversampling,
            yx_center=None,
        )
    if valid_pixel_mask_cropped is not None:
        search_region = np.logical_and(search_region, valid_pixel_mask_cropped)
        logger.info(
            "Scheduling %d positions inside the footprint.",
            int(search_region.sum()),
        )

    ncpus = config.ncpus
    if ncpus is None:
        ncpus = multiprocessing.cpu_count()

    return ReductionRuntimeState(
        yx_anamorphism=yx_anamorphism,
        yx_known_companion_position=yx_known_companion_position,
        known_companion_contrast=known_companion_contrast,
        search_region_outer_bound=search_region_outer_bound,
        data_crop_size=data_crop_size,
        search_region=search_region,
        ncpus=ncpus,
        coronagraph_transmission_pix=coronagraph_transmission_pix,
        valid_pixel_mask_cropped=valid_pixel_mask_cropped,
        reduction_mask_min_pixels=config.reduction_mask_min_pixels,
        # Category C: initial values (will be overwritten by for_iteration)
        number_of_pca_regressors=config.number_of_pca_regressors,
        temporal_components_fraction=0.0,
        fwhm=0.0,
        reduction_mask_psf_size=config.reduction_mask_psf_size,
        signal_mask_psf_size=config.signal_mask_psf_size,
    )


# -------- conversion helper -------------------------------------------------

def _to_reduction_config(reduction_parameters) -> "TrapReductionConfig":
    """Normalize an accepted config type to TrapReductionConfig.

    Accepts a ``TrapConfig`` or a ``TrapReductionConfig``. Always returns a
    frozen ``TrapReductionConfig`` instance.
    """
    if isinstance(reduction_parameters, TrapReductionConfig):
        return reduction_parameters
    if isinstance(reduction_parameters, TrapConfig):
        return reduction_parameters.reduction
    raise TypeError(
        f"Expected TrapReductionConfig or TrapConfig, "
        f"got {type(reduction_parameters).__name__}"
    )


# -------- resources ---------------------------------------------------------

@dataclass(slots=True)
class TrapResources:
    """Resource management for TRAP processing."""
    ncpu_reduction: int = 1
    ncpu_detection: int = 1
    scratch_dir: Optional[Path] = None

    def apply(self, reduction_config: TrapReductionConfig) -> TrapReductionConfig:
        """Apply resource settings to reduction configuration.

        Returns a new config with ncpus set (frozen config cannot be mutated).
        """
        return reduction_config.merge(
            ncpus=self.ncpu_reduction, scratch_dir=self.scratch_dir
        )


def resolve_scratch_dir(scratch_dir=None, required_bytes=None):
    """Resolve the directory used for the shared-array store.

    Resolution order: an explicit ``scratch_dir`` always wins; otherwise
    ``/dev/shm`` is used if it exists and has headroom for the estimated
    store size; otherwise ``tempfile.gettempdir()``.

    Parameters
    ----------
    scratch_dir : str or Path, optional
        Explicit scratch directory. Used as-is if given.
    required_bytes : int, optional
        Estimated size of the store. Used to check ``/dev/shm`` headroom;
        if None, ``/dev/shm`` is used whenever it exists.

    Returns
    -------
    Path
        Directory in which to create the shared-array store.
    """
    if scratch_dir is not None:
        return Path(scratch_dir)
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        if required_bytes is None:
            return shm
        # Require 20% headroom over the estimated store size.
        if shutil.disk_usage(shm).free > required_bytes * 1.2:
            return shm
    return Path(tempfile.gettempdir())


# -------- wavelength and processing parameters ------------------------------

@dataclass(slots=True)
class ProcessingParameters:
    """TRAP processing parameters including wavelength selection."""
    wavelength_indices: Optional[range] = range(1, 38)
    temporal_components_fraction: list[float] = field(default_factory=lambda: [0.15])
    overwrite_reduction: bool = True
    overwrite_detection: bool = True
    verbose: bool = False
    use_progress_bar: bool = True

    def merge(self, **kw) -> "ProcessingParameters":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)


# -------- master TRAP configuration -----------------------------------------

@dataclass(slots=True)
class TrapConfig:
    """Master TRAP configuration containing all sub-configurations."""
    reduction: TrapReductionConfig = field(default_factory=TrapReductionConfig)
    detection: DetectionParameters = field(default_factory=DetectionParameters)
    processing: ProcessingParameters = field(default_factory=ProcessingParameters)
    resources: TrapResources = field(default_factory=TrapResources)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)

    def as_plain_dicts(self):
        """Return tuple of dictionaries for all sub-configurations."""
        return (
            asdict(self.reduction),
            asdict(self.detection),
            asdict(self.processing),
        )

    def apply_resources(self):
        """Apply resource configuration to all sub-configs."""
        self.reduction = self.resources.apply(self.reduction)

    def get_stellar_parameters(self) -> Dict[str, float]:
        """Get stellar parameters as dictionary for TRAP template matching."""
        return self.detection.stellar_parameters.as_dict()

    def get_instrument(self, obs_mode: str, wavelengths=None) -> Instrument:
        """Get TRAP Instrument instance with current configuration.
        
        Parameters
        ----------
        obs_mode : str
            Observation mode, either 'OBS_YJ' or 'OBS_H'
        wavelengths : astropy.units.Quantity, optional
            Wavelength array. If None, will be set to None in the Instrument.
            
        Returns
        -------
        Instrument
            TRAP Instrument instance configured with the parameters.
        """
        return self.instrument.to_instrument(obs_mode, wavelengths)


# -------- factory functions -------------------------------------------------

def default_trap_config() -> TrapConfig:
    """Create default TRAP configuration."""
    return TrapConfig()


def trap_config_for_ifs() -> TrapConfig:
    """Create TRAP configuration optimized for IFS observations.

    ``yx_anamorphism=[1.0059, 1.0011]`` matches the correction Vigan's ``sphere``
    package applies to IFS science and flux cubes in ``sph_ifs_combine_data``.
    The anamorphism comes from the cylindrical mirrors in the SPHERE common path
    and is therefore shared by all three science subsystems (Maire et al. 2016);
    only the field orientation differs. TRAP compensates in the forward model —
    it distorts the injection position per frame instead of interpolating the
    data — so the spherical IFS conversion must leave the cubes uncorrected, as
    it does. Override to ``[1.0, 1.0]`` if the correction is ever applied
    upstream. Omitting it understates the separation of a source lying near the
    detector y axis by up to ~0.6% (0.30 px / 2.2 mas for 51 Eri b).
    """
    config = TrapConfig(
        reduction=TrapReductionConfig(
            search_region_outer_bound=81,
            temporal_model=True,
            spatial_model=False,
            right_handed=False,
            search_region_inner_bound=1,
            yx_anamorphism=np.array([1.0059, 1.0011]),
            auto_footprint=True,
        ),
        processing=ProcessingParameters(
            wavelength_indices=range(1, 38),
        ),
    )
    return config


def trap_config_for_irdis() -> TrapConfig:
    """Create TRAP configuration optimized for IRDIS observations.

    IRDIS-specific defaults:

    * ``pixel_scale_arcsec_per_pixel = 0.01225`` (vs 0.00746 for IFS).
    * ``instrument_type = "photometry"`` (DBI has discrete filter channels;
      matches the ``SpectralTemplate`` branch that integrates model spectra
      through filter bandpasses via ``species.SyntheticPhotometry``).
    * ``wavelength_indices = range(0, 2)`` (2 discrete filter channels).
    * ``search_region_outer_bound = 200`` (K-band AO cutoff at ~1.4″ /
      0.01225″/px ≈ 115 px; 200 px gives comfortable outer margin).
    * ``search_region_inner_bound = 1`` (coronagraph inner-working-angle is
      enforced by the coronagraph_transmission table injected by
      ``spherical.pipeline.run_trap``).
    * ``temporal_model=True``, ``spatial_model=False``, ``right_handed=False``
      (SPHERE convention; mirrors ``trap_config_for_ifs``).
    * ``yx_anamorphism=[1.0062, 1.0]`` (SPHERE-measured y-stretch). Matches the
      spherical IRDIS preprocessing default where ``correct_anamorphism=False``,
      so the un-corrected geometry is baked into the cubes and the forward model
      compensates. Override to ``[1.0, 1.0]`` when the anamorphism correction is
      applied upstream.
    """
    config = TrapConfig(
        instrument=InstrumentConfig(
            name="IRDIS",
            pixel_scale_arcsec_per_pixel=0.01225,
            telescope_diameter_m=7.99,
            detector_gain=1.0,
            readnoise=0.0,
            instrument_type="photometry",
        ),
        reduction=TrapReductionConfig(
            search_region_outer_bound=200,
            temporal_model=True,
            spatial_model=False,
            right_handed=False,
            search_region_inner_bound=1,
            yx_anamorphism=np.array([1.0062, 1.0]),
            auto_footprint=True,
        ),
        processing=ProcessingParameters(
            wavelength_indices=range(0, 2),
        ),
    )
    return config


def trap_config_for_beta_pic() -> TrapConfig:
    """Create TRAP configuration optimized for Beta Pictoris observations."""
    config = trap_config_for_ifs()
    
    # Beta Pic specific stellar parameters
    config.detection.stellar_parameters = StellarParameters(
        teff=8052.0,    # Beta Pic effective temperature
        logg=4.15,      # Beta Pic surface gravity
        feh=0.05,       # Beta Pic metallicity
        radius=1.8,     # Beta Pic radius in solar radii
        distance=19.44  # Beta Pic distance in pc
    )
    
    return config


# ============================================================================
# LEGACY PARAMETER CLASSES (DEPRECATED)
# ============================================================================
