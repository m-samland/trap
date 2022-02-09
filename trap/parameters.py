"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

import numpy as np
from astropy import units as u


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
    read_noise : float
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
    read_noise
    instrument_type
    wavelengths
    spectral_resolution
    filters
    transmission

    """

    def __init__(
            self, name, pixel_scale, telescope_diameter, detector_gain=1.0, read_noise=0.,
            instrument_type='photometry', wavelengths=None, spectral_resolution=None, filters=None,
            transmission=None):
        self.name = name
        self.pixel_scale = pixel_scale
        self.telescope_diameter = telescope_diameter
        self.detector_gain = detector_gain
        self.read_noise = read_noise

        self.instrument_type = instrument_type
        self.wavelengths = wavelengths
        self.spectral_resolution = spectral_resolution
        self.filters = filters
        self.transmission = transmission
        if self.wavelengths is not None:
            self.compute_fwhm()

    def compute_fwhm(self):
        if self.wavelengths is not None:
            angle = (self.wavelengths / self.telescope_diameter).to(
                u.mas, equivalencies=u.dimensionless_angles())
            self.fwhm = angle.to(u.pixel, self.pixel_scale)


class Reduction_parameters(object):
    """Contains all reduction parameters describing the settings
       for TRAP.

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
    include_noise : boolean
        Take into account variance of the input data when fitting
        models. If True and no explicit variance is provided to
        the reduction wrapper, the data is assumed to represent the
        shot noise in the data and the read noise and gain from
        the instrument object are used. Default is False.
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
        `signal_mask_size_in_lambda_over_d`. Default is 1.1.
    signal_mask_size_in_lambda_over_d : scalar
        If `autosize_masks_in_lambda_over_d` is True, gives
        size of PSF stamp used to create signal exclusion area in resolution
        elements. Will automatically adjust size based on instrument-object
        and wavelength used. Has to be larger than
        `reduction_mask_size_in_lambda_over_d`. Default is 2.1.
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
    contrast_curve : boolean
        Automatically generate contrast curve after reduction.
        Default is True.
    constrast_curve_sigma : scalar
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

    Attributes
    ----------
    search_region
    search_region_inner_bound
    search_region_outer_bound
    oversampling
    include_noise
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
    contrast_curve
    constrast_curve_sigma
    normalization_width
    companion_mask_radius
    return_input_data
    plot_all_diagnostics
    verbose

    """

    def __init__(
            self,
            search_region=None,
            search_region_inner_bound=1,
            search_region_outer_bound=55,
            oversampling=1,
            data_auto_crop=False,
            data_crop_size=None,
            right_handed=True,
            include_noise=False,
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
            reduction_mask_size_in_lambda_over_d=2.1,
            signal_mask_size_in_lambda_over_d=2.1,
            reduction_mask_psf_size=21,
            signal_mask_psf_size=21,
            threshold_pixel_by_contribution=0.,
            make_reconstructed_lightcurve=True,
            target_pix_mask_radius=None,
            use_relative_position=False,
            compute_residual_correlation=False,
            use_residual_correlation=False,
            contrast_curve=True,
            contrast_curve_sigma=5.,
            normalization_width=3,
            companion_mask_radius=11,
            return_input_data=False,
            verbose=False):

        self.search_region = search_region
        self.search_region_inner_bound = search_region_inner_bound
        self.search_region_outer_bound = search_region_outer_bound
        self.oversampling = oversampling
        self.include_noise = include_noise
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

        self.contrast_curve = contrast_curve
        self.contrast_curve_sigma = contrast_curve_sigma
        self.normalization_width = normalization_width
        self.companion_mask_radius = companion_mask_radius

        self.return_input_data = return_input_data
        self.plot_all_diagnostics = plot_all_diagnostics
        self.verbose = verbose
