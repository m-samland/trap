"""
Routines used in TRAP

The TRAP parameter system has been modernized with dataclass-based configuration.
For new code, use the TrapConfig classes instead of the legacy Reduction_parameters:

    # Recommended modern approach:
    from trap.parameters import trap_config_for_ifs, TrapConfig
    
    config = trap_config_for_ifs()
    reduction_params = config.get_reduction_parameters()

The legacy Reduction_parameters class is still available but deprecated.

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


class Reduction_parameters(object):
    """Contains all reduction parameters describing the settings
       for TRAP.

    .. deprecated:: 
        This class is deprecated. Use :class:`TrapConfig` and 
        :class:`TrapReductionConfig` instead for a more modern, 
        type-safe configuration interface.

    For new code, use::

        config = trap_config_for_ifs()  # or default_trap_config()
        reduction_params = config.get_reduction_parameters()

    Parameters
    ----------
    search_region : array_like, optional
        Binary mask of relative position to search for planets.
    search_region_inner_bound : integer
        Separation of inner-edge of reduction region (in pixel).
    search_region_outer_bound : integer
        Separation of outer-edge of reduction region (in pixel).
    oversampling : scalar
        Oversampling factor for detection map. Default is 1.0.
    data_auto_crop : boolean
        Automatically crop images to smallest size necessary for
        the chosen reduction parameters.
    data_crop_size : scalar or None
        Manually chosen size for data cropping. Default is None.
    right_handed : boolean
        Determines the sky rotation direction. True for SPHERE,
        False for most instruments. Best try both on a data set
        with known companion to confirm for your instrument.
    estimate_noise_from_data : boolean
        Fallback trigger for noise weighting when the user does NOT
        provide an ``inverse_variance`` cube to the reduction wrapper.
        If True, TRAP estimates the noise from the data itself
        (shot noise + read noise from the instrument object). If False,
        the fit is unweighted. If an ``inverse_variance`` cube IS
        provided, it is always used regardless of this flag. Default
        is False. (Renamed from ``include_noise``.)
    temporal_model : boolean
        Perform temporal model fit. Default is True.
    temporal_plus_spatial_model : boolean
        Perform spatial model fit on the residuals of the temporal
        systematics subtracted frames.
    second_stage_trap : boolean
        Perform a second temporal model fit without the companion model
        after subtracting the 2d companion PSF with best-fit contrast
        from the first iteration. This can be used as an additional step
        before the `temporal_plus_spatial_mode` reduction.
        Default is False.
    remove_model_from_spatial_training : boolean
        Remove temporal best-fit contrast signal from training set used for
        spatial model fit. Default is True.
    remove_bad_residuals_for_spatial_model : boolean
        Remove pixels with anomalous temporal residuals from spatial model fit.
    spatial_model : boolean
        Perform a spatial model fit. Default is False.
    local_temporal_model : boolean
        If True perform temporal model fit local in time (experimental).
        Not recommended. Default is False.
    local_spatial_model : boolean
        If True perform spatial model fit very locally in space (experimental).
        Not recommended. Default is False.
    protection_angle : scalar
        Protection angle in lambda over D used for spatial model.
    spatial_components_fraction : scalar
        Fraction of total available number of principal components to use
        for spatial model. Must be between 0 and 1. Default is 0.3.
    spatial_components_fraction_after_trap : scalar
        Fraction of total available number of principal components to use
        for spatial model after temporal systematics have been subtracted.
        Must be between 0 and 1. Default is 0.1.
    highpass_filter : scalar or None
        Apply high-pass filter with a given filter fraction to data before
        analysis (experimental). Not recommended: Default is None.
    remove_known_companions : boolean
        Remove known companion signals from data via negative injection.
        Not used for normal TRAP reductions. Default is False.
    yx_known_companion_position : None, tuple, or list of tuples
        Position of known companions to mask out.
        Default is None.
    known_companion_contrast : None or scalar
        Contrast associated with `yx_known_companion_position`.
        Use for 'remove_known_companions'. Not to be confused with
        `true_contrast`, which is associated with injected signals.
        Default is None.
    use_multiprocess : boolean
        Use multiprocessing.
    ncpus : integer
        Number of cores available. Beware linear increase in memory usage.
    prefix : string
        Prefix added in front of output file names.
    result_folder : string
        Path where reduction outputs will be saved.
    inject_fake : boolean
        Inject a fake signal with `true_contrast` into the data
        at `true_position`. If `read_injection_files` is True,
        signals will be injected at every position with `injection_sigma`.
        Default is False.
    true_position : tuple
        Position of injected signal.
    true_contrast : scalar
        Contrast of injected signal.
    read_injection_files : boolean
        Use existing detection map for to determine brightness of signals
        to inject given a `injection_sigma`.
        Default is False.
    injection_sigma : scalar
        Expected significance of injected signal based on detection map.
    reduce_single_position : boolean
        Applies TRAP to a single position described by `guess_position`.
    guess_position : tuple
        Individual position to reduce, if `reduce_single_position` is True.
    fit_planet : boolean
        Include planet signal as forward model in fit. Default is True.
    number_of_pca_regressors : integer
        Number of PCA regressors used. This information is added by the pipeline
        based on the component fraction provided.
    yx_anamorphism : None or tuple
        Description of parameter `yx_anamorphism`.
    pca_scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard',
                   'temp-median', 'spat-median', 'temp-quartile, 'spat-quartile'}
        Chose the method of centering and scaling the data for the
        PCA regressors. The temp(oral) and spat(ial) definition assume that
        time is axis=0 and space is axis=1. Median and quartile are the
        robust version of centering and scaling. Default is 'temp-median'.
    method_of_regressor_selection : {'random', 'auxiliary', None}, optional
        'random' selects a random sample of regressors.
        'auxiliary' regressor selection based on `auxiliary_frame`.
        Not implemented at the moment. Default: None
    auxiliary_frame : array_like
        Auxiliary frame on which to base regressor selection on.
        Default is None.
    annulus_width : scalar
        Width of the regressor annulus (in pixel). Default is 5.
    annulus_offset : scalar
        Radially displace regressor annulus  (by pixel). Default is 0.
    add_radial_regressors : boolean
        Add additional radial regressors around the reduction area.
        Default is True.
    include_opposite_regressors : boolean
        Include reduction area mirrored around origin as regressors.
        Default is True.
    variance_prior_scaling : scalar, optional
        Scaling factor for variance. Not in current implementation.
        Default is 1.0.
    autosize_masks_in_lambda_over_d : boolean
        Adjust reduction area and signal protection area based on
        `reduction_mask_size_in_lambda_over_d` and
        `signal_mask_size_in_lambda_over_d`. Default is True.
    reduction_mask_size_in_lambda_over_d : scalar
        If `autosize_masks_in_lambda_over_d` is True, gives
        size of PSF stamp used to create reduction area in resolution
        elements. Will automatically adjust size based on instrument-object
        and wavelength used. Has to be smaller than
        `signal_mask_size_in_lambda_over_d`. Default is 1.
    signal_mask_size_in_lambda_over_d : scalar
        If `autosize_masks_in_lambda_over_d` is True, gives
        size of PSF stamp used to create signal exclusion area in resolution
        elements. Will automatically adjust size based on instrument-object
        and wavelength used. Has to be larger than
        `reduction_mask_size_in_lambda_over_d`. Default is 2.
    reduction_mask_psf_size : scalar
        Size of PSF stamp used to create reduction area in resolution
        elements in pixel. Has to be smaller than `signal_mask_size`.
        Default is 21.
    signal_mask_psf_size : scalar
        Size of PSF stamp used to create reduction area in resolution
        elements in pixel. Has to be larger than `reduction_mask_size`.
        Default is 21.
    threshold_pixel_by_contribution : scalar
        Include all pixels in reduction for which the overall flux fraction
        of the total integrated flux that a pixel observes is higher than the
        threshold, e.g. for 0.1 only pixels that contribute more than 10% of
        total signal are considered. Default is 0.
    target_pix_mask_radius : scalar, optional
        Exclude pixels within this radius from regressor selection.
        Not used in current forward model based implementation.
        Default is None.
    use_relative_position : boolean
        Use relative position for coordinates.
        True may break functionality in current implementation.
        Default is False.
    compute_inverse_once : boolean
        Do not recompute PCAs for each pixel.
        False may break functionality in current implementation.
        Default is True.
    make_reconstructed_lightcurve : boolean
        Reconstruct model fit lightcurve instead of just determining
        parameters. Necessary for normal functionality of the pipeline.
        Default is True.
    compute_residual_correlation : boolean
        Compute correlations between residuals after model fit (experimental).
        Default is False.
    use_residual_correlation : boolean
        Use correlation between residuals after model fit instead of simple,
        uncorrelated weighted average. Produces additional output files similar
        to the detection_image output.
        Default is False.
    use_signal_weighting : boolean
        Use signal-based weighting in contrast estimation. When True, pixels
        with stronger expected signal contribute more to the final contrast
        estimate, improving signal-to-noise ratio.
        Default is False.
    contrast_curve : boolean
        Automatically generate contrast curve after reduction.
        Default is True.
    contrast_curve_sigma : scalar
        Defines the sigma of the contrast curve.
        Default is 5.
    normalization_width : integer
        Width (in pixel) of radial bin used to normalize the detection map.
        Default is 3.
    companion_mask_radius : integer
        Radius of mask around `yx_known_companion_position` of pixels to be
        ignored for detection map normalization and contrast curve.
        Default is 11.
    return_input_data : boolean
        Include input data for temporal model in `~trap.regression.Result`
        object. Default is False.
    plot_all_diagnostics : boolean
        If `reduce_single_position` is True, this will produce diagnostic
        plots in a folder in the current working directory called
        `diagnostic_plots`. This is very helpful when testing the code
        or get more information on a specific location in parameter space.
    verbose : boolean
        Produce additional output in console. Default is False.
    use_progress_bar : boolean
        Use a progress bar to indicate progress of the reduction.

    Attributes
    ----------
    search_region
    search_region_inner_bound
    search_region_outer_bound
    oversampling
    estimate_noise_from_data
    data_auto_crop
    data_crop_size
    right_handed
    remove_known_companions
    yx_known_companion_position
    known_companion_contrast
    use_multiprocess
    ncpus
    prefix
    result_folder
    reduce_single_position
    true_position
    true_contrast
    read_injection_files
    inject_fake
    injection_sigma
    guess_position
    fit_planet
    number_of_pca_regressors
    yx_anamorphism
    variance_prior_scaling
    pca_scaling
    method_of_regressor_selection
    auxiliary_frame
    annulus_width
    annulus_offset
    reduction_mask_psf_size
    signal_mask_psf_size
    autosize_masks_in_lambda_over_d
    reduction_mask_size_in_lambda_over_d
    signal_mask_size_in_lambda_over_d
    add_radial_regressors
    radial_separation_from_source
    include_opposite_regressors
    threshold_pixel_by_contribution
    target_pix_mask_radius
    use_relative_position
    compute_inverse_once
    temporal_model
    temporal_plus_spatial_model
    second_stage_trap
    spatial_model
    local_temporal_model
    local_spatial_model
    protection_angle
    spatial_components_fraction
    spatial_components_fraction_after_trap
    remove_model_from_spatial_training
    remove_bad_residuals_for_spatial_model
    highpass_filter
    make_reconstructed_lightcurve
    compute_residual_correlation
    use_residual_correlation
    use_signal_weighting
    contrast_curve
    constrast_curve_sigma
    normalization_width
    companion_mask_radius
    return_input_data
    plot_all_diagnostics
    verbose
    use_progress_bar
    """

    def __init__(
            self,
            search_region=None,
            search_region_inner_bound=1,
            search_region_outer_bound=55,
            oversampling=1,
            data_auto_crop=True,
            data_crop_size=None,
            right_handed=True,
            estimate_noise_from_data=False,
            temporal_model=True,
            temporal_plus_spatial_model=False,
            second_stage_trap=False,
            remove_model_from_spatial_training=True,
            remove_bad_residuals_for_spatial_model=True,
            spatial_model=False,
            local_temporal_model=False,
            local_spatial_model=False,
            protection_angle=0.5,
            spatial_components_fraction=0.3,
            spatial_components_fraction_after_trap=0.1,
            highpass_filter=None,
            remove_known_companions=False,
            yx_known_companion_position=None,
            known_companion_contrast=None,
            use_multiprocess=False,
            ncpus=1,
            prefix='',
            result_folder='./',
            inject_fake=False,
            true_position=None,
            true_contrast=None,
            read_injection_files=False,
            injection_sigma=5,
            reduce_single_position=False,
            guess_position=None,
            plot_all_diagnostics=False,
            fit_planet=True,
            number_of_pca_regressors=20,
            yx_anamorphism=np.array([1., 1.]),
            pca_scaling='temp-median',
            method_of_regressor_selection=None,
            auxiliary_frame=None,
            annulus_width=5,
            annulus_offset=0,
            add_radial_regressors=True,
            include_opposite_regressors=True,
            variance_prior_scaling=1.,
            compute_inverse_once=True,
            autosize_masks_in_lambda_over_d=True,
            reduction_mask_size_in_lambda_over_d=1.,
            signal_mask_size_in_lambda_over_d=2.,
            reduction_mask_psf_size=21,
            signal_mask_psf_size=21,
            threshold_pixel_by_contribution=0.,
            make_reconstructed_lightcurve=True,
            target_pix_mask_radius=None,
            use_relative_position=False,
            compute_residual_correlation=False,
            use_residual_correlation=False,
            use_signal_weighting=False,
            contrast_curve=True,
            contrast_curve_sigma=5.,
            normalization_width=3,
            companion_mask_radius=13,
            return_input_data=False,
            verbose=False,
            use_progress_bar=True,):

        import warnings
        warnings.warn(
            "Reduction_parameters is deprecated and will be removed in a future release. "
            "Use TrapReductionConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.search_region = search_region
        self.search_region_inner_bound = search_region_inner_bound
        self.search_region_outer_bound = search_region_outer_bound
        self.oversampling = oversampling
        self.estimate_noise_from_data = estimate_noise_from_data
        self.data_auto_crop = data_auto_crop
        self.data_crop_size = data_crop_size
        self.right_handed = right_handed
        self.remove_known_companions = remove_known_companions
        self.yx_known_companion_position = yx_known_companion_position
        self.known_companion_contrast = known_companion_contrast

        self.use_multiprocess = use_multiprocess
        self.ncpus = ncpus
        self.prefix = prefix
        self.result_folder = result_folder

        self.reduce_single_position = reduce_single_position
        self.true_position = true_position
        self.true_contrast = true_contrast
        self.read_injection_files = read_injection_files
        self.inject_fake = inject_fake
        self.injection_sigma = injection_sigma
        self.guess_position = guess_position
        self.fit_planet = fit_planet
        self.number_of_pca_regressors = number_of_pca_regressors
        self.yx_anamorphism = yx_anamorphism
        self.variance_prior_scaling = variance_prior_scaling
        self.pca_scaling = pca_scaling
        self.method_of_regressor_selection = method_of_regressor_selection
        self.auxiliary_frame = auxiliary_frame

        # Mask settings
        self.annulus_width = annulus_width
        self.annulus_offset = annulus_offset
        self.reduction_mask_psf_size = reduction_mask_psf_size
        self.signal_mask_psf_size = signal_mask_psf_size
        self.autosize_masks_in_lambda_over_d = autosize_masks_in_lambda_over_d
        self.reduction_mask_size_in_lambda_over_d = reduction_mask_size_in_lambda_over_d
        self.signal_mask_size_in_lambda_over_d = signal_mask_size_in_lambda_over_d
        self.add_radial_regressors = add_radial_regressors
        self.include_opposite_regressors = include_opposite_regressors
        self.threshold_pixel_by_contribution = threshold_pixel_by_contribution
        self.target_pix_mask_radius = target_pix_mask_radius
        self.use_relative_position = use_relative_position
        self.compute_inverse_once = compute_inverse_once

        self.temporal_model = temporal_model
        self.temporal_plus_spatial_model = temporal_plus_spatial_model
        self.second_stage_trap = second_stage_trap
        self.spatial_model = spatial_model
        self.local_temporal_model = local_temporal_model
        self.local_spatial_model = local_spatial_model
        self.protection_angle = protection_angle
        self.spatial_components_fraction = spatial_components_fraction
        self.spatial_components_fraction_after_trap = spatial_components_fraction_after_trap
        self.remove_model_from_spatial_training = remove_model_from_spatial_training
        self.remove_bad_residuals_for_spatial_model = remove_bad_residuals_for_spatial_model
        self.highpass_filter = highpass_filter
        self.make_reconstructed_lightcurve = make_reconstructed_lightcurve
        self.compute_residual_correlation = compute_residual_correlation
        self.use_residual_correlation = use_residual_correlation
        self.use_signal_weighting = use_signal_weighting

        self.contrast_curve = contrast_curve
        self.contrast_curve_sigma = contrast_curve_sigma
        self.normalization_width = normalization_width
        self.companion_mask_radius = companion_mask_radius

        self.return_input_data = return_input_data
        self.plot_all_diagnostics = plot_all_diagnostics
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar


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

    def to_reduction_parameters(self) -> "Reduction_parameters":
        """Convert to TRAP Reduction_parameters instance.

        .. deprecated::
            Use ``TrapReductionConfig`` directly. This method will be removed
            in a future release.
        """
        import warnings
        warnings.warn(
            "TrapReductionConfig.to_reduction_parameters() is deprecated and will be removed "
            "in a future release. Use TrapReductionConfig directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        params_dict = asdict(self)
        params_dict.pop("coronagraph_transmission", None)
        params_dict.pop("multiwavelength_regressors", None)
        params_dict.pop("regressor_wavelength_indices", None)
        params_dict.pop("max_regressor_pool_size", None)
        params_dict.pop("reduction_mask_min_pixels", None)
        params_dict.pop("auto_footprint", None)

        # Filter out None values, but keep explicit None defaults where needed
        filtered_params = {}
        for k, v in params_dict.items():
            if v is not None:
                filtered_params[k] = v
            elif k in ['search_region', 'data_crop_size', 'yx_known_companion_position', 
                      'known_companion_contrast', 'true_position', 'true_contrast',
                      'guess_position', 'method_of_regressor_selection', 'auxiliary_frame',
                      'target_pix_mask_radius', 'highpass_filter']:
                # These parameters should be explicitly None
                filtered_params[k] = None
        
        return Reduction_parameters(**filtered_params)


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

    valid_pixel_mask_cropped = None
    if valid_pixel_mask is not None:
        from trap.utils import crop_box_from_image
        mask_bool = np.asarray(valid_pixel_mask).astype("bool")
        if data_crop_size is not None and yx_center_full is not None:
            valid_pixel_mask_cropped = crop_box_from_image(
                mask_bool,
                data_crop_size,
                center_yx=np.round(np.asarray(yx_center_full)[0]),
            )
        else:
            valid_pixel_mask_cropped = mask_bool

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
    """Convert any accepted config type to TrapReductionConfig.

    Accepts TrapConfig, TrapReductionConfig, or legacy Reduction_parameters.
    Always returns a frozen TrapReductionConfig instance.
    """
    import warnings
    if isinstance(reduction_parameters, TrapReductionConfig):
        return reduction_parameters
    if hasattr(reduction_parameters, 'get_reduction_parameters'):
        # TrapConfig — extract TrapReductionConfig directly
        return reduction_parameters.reduction
    if isinstance(reduction_parameters, Reduction_parameters):
        # Legacy Reduction_parameters — convert field-by-field
        warnings.warn(
            "Passing Reduction_parameters is deprecated. Use TrapReductionConfig instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        fields = {f.name for f in TrapReductionConfig.__dataclass_fields__.values()}
        kwargs = {}
        for name in fields:
            if hasattr(reduction_parameters, name):
                kwargs[name] = getattr(reduction_parameters, name)
        return TrapReductionConfig(**kwargs)
    raise TypeError(
        f"Expected TrapReductionConfig, TrapConfig, or Reduction_parameters, "
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

    def get_reduction_parameters(self) -> "Reduction_parameters":
        """Get TRAP Reduction_parameters instance with current configuration.

        .. deprecated::
            Access ``TrapConfig.reduction`` (a ``TrapReductionConfig``) directly
            and pass it to ``run_complete_reduction``. This method will be removed
            in a future release.
        """
        import warnings
        warnings.warn(
            "TrapConfig.get_reduction_parameters() is deprecated and will be removed "
            "in a future release. Use TrapConfig.reduction (a TrapReductionConfig) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.reduction.to_reduction_parameters()

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
    """Create TRAP configuration optimized for IFS observations."""
    config = TrapConfig(
        reduction=TrapReductionConfig(
            search_region_outer_bound=81,
            temporal_model=True,
            spatial_model=False,
            right_handed=False,
            search_region_inner_bound=1,
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
