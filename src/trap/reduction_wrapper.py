"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

import dataclasses
import datetime
import logging
import multiprocessing
import os
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from joblib import Parallel, delayed, parallel_config
from tqdm.auto import tqdm

from trap import (
    image_coordinates,
    makesource,
    regression,
    regressor_selection,
    shared_arrays,
)
from trap.parameters import (
    _to_reduction_config,
    build_runtime_state,
)
from trap.shared_arrays import SharedArrayRef, SharedArrayStore
from trap.utils import (
    crop_box_from_3D_cube,
    crop_box_from_image,
    determine_psf_stampsizes,
    prepare_psf,
    save_object,
    shuffle_and_equalize_relative_positions,
)

logger = logging.getLogger(__name__)

def trap_one_position(
    guess_position,
    data,
    flux_psf,
    pa,
    reduction_parameters,
    known_companion_mask,
    runtime=None,
    inverse_variance=None,
    bad_pixel_mask=None,
    yx_center=None,
    yx_center_injection=None,
    amplitude_modulation=None,
    contrast_map=None,
    readnoise=0.0,
    cross_validation=False,
):
    """Runs TRAP on individual position.

    Parameters
    ----------
    guess_position : tuple
        (yx)-position to be reduced, given relative to center.
    data : array_like
        Temporal image cube. First axis is time.
    flux_psf : array_like
        Image of unsaturated PSF.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    known_companion_mask : array_like
        Boolean mask of image size. True for pixels affected by companion flux.
    inverse_variance : array_like
        Cube containing inverse variances for `data`.
    bad_pixel_mask : array_like
        Bad pixel mask for `data`.
    yx_center : tuple
        Average or median image center position to be used for regressor selection.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    amplitude_modulation : array_like, optional
        Array containing scaling factors for the companion PSF brightness for
        each time (e.g. derived from satellite spots).
    contrast_map : array_like, optional
        Contrast detection map (derived using the same reduction parameters
        and data) to be used for injection retrieval testing to determine
        biases in reduction (see `inject_fake`, `read_injection_files`
        and 'injection_sigma') in `~trap.parameters.Reduction_parameters`.
    readnoise : scalar
        The detector read noise (e rms/pix/readout).

    Returns
    -------
    dictionary
        A dictionary containing the `~trap.regression.Result` object for
        'temporal', 'temporal_plus_spatial', and 'spatial' keywords, depending
        on whether the `temporal_model`, `temporal_plus_spatial_model`,
        and `spatial_model` parameters set in `reduction_parameters` are
        True.

    """

    yx_dim = (data.shape[-2], data.shape[-1])
    if yx_center is None:
        yx_center = (yx_dim[0] // 2, yx_dim[1] // 2)
    if amplitude_modulation is None:
        amplitude_modulation = np.ones(data.shape[0])

    if guess_position is not None:
        position_absolute = image_coordinates.relative_yx_to_absolute_yx(
            guess_position, yx_center
        ).astype("int")
    signal_position = np.array(guess_position)

    if contrast_map is not None:
        true_contrast = contrast_map[
            position_absolute[0], position_absolute[1]
        ]
    else:
        true_contrast = reduction_parameters.true_contrast

    planet_absolute_yx_pos = image_coordinates.relative_yx_to_absolute_yx(
        signal_position, yx_center
    )

    if runtime.coronagraph_transmission_pix is not None:
        separation_pix = np.hypot(signal_position[0], signal_position[1])
        # amplitude_modulation is shared read-only across positions: copy, never mutate.
        amplitude_modulation = amplitude_modulation * makesource.coronagraph_throughput_factor(
            separation_pix, runtime.coronagraph_transmission_pix
        )

    injected_model_cube = np.zeros_like(data)
    injected_model_cube = makesource.inject_model_into_data(
        flux_arr=injected_model_cube,
        pos=signal_position,
        pa=pa,
        psf_model=flux_psf,
        image_center=yx_center_injection,
        norm=amplitude_modulation,
        yx_anamorphism=runtime.yx_anamorphism,
        right_handed=reduction_parameters.right_handed,
        subpixel=True,
        remove_interpolation_artifacts=True,
        copy=False,
    )

    signal_mask_local = injected_model_cube > 0.0
    signal_mask = np.any(signal_mask_local, axis=0)
    if (
        runtime.reduction_mask_psf_size
        == runtime.signal_mask_psf_size
    ):
        reduction_mask = signal_mask
    else:
        reduction_mask = regressor_selection.make_mask_from_psf_track(
            yx_position=signal_position,
            psf_size=runtime.reduction_mask_psf_size,
            pa=pa,
            image_size=data.shape[-1],
            image_center=yx_center_injection,
            yx_anamorphism=runtime.yx_anamorphism,
            right_handed=reduction_parameters.right_handed,
            return_cube=False,
        )

    # Due to interpolation edge effects some pixels at the edge of the trajectory can be negative
    total_flux_in_pixel = np.sum(injected_model_cube, axis=0)
    relative_flux_in_pixel = total_flux_in_pixel / np.max(total_flux_in_pixel)
    # excluding pixels with low contribution to over-all signal
    low_contribution_mask = (
        relative_flux_in_pixel <= reduction_parameters.threshold_pixel_by_contribution
    )
    reduction_mask[low_contribution_mask] = False

    if bad_pixel_mask is not None:
        reduction_mask_wo_badpixels = np.logical_and(reduction_mask, ~bad_pixel_mask)

    # Remove interpolation effects of model PSF
    # injected_model_cube[~signal_mask_local] = 0.

    if reduction_parameters.inject_fake:
        # Add artificial model to data
        data_reduce = data + injected_model_cube * true_contrast
    else:
        data_reduce = data

    assert (
        flux_psf.shape[-1] % 2 == 1
    ), "PSF dimension has to be odd (centered on pixel)"

    if reduction_parameters.fit_planet:
        model = injected_model_cube
    else:
        model = None

    if reduction_parameters.include_noise:
        if inverse_variance is None:
            # NOTE: Uses photon noise from data itself.
            # May not be valid based on pre-processing steps done
            if bad_pixel_mask is not None:
                inverse_variance_reduction_area = 1.0 / (
                    np.abs(data_reduce[:, reduction_mask_wo_badpixels]) + readnoise**2
                )
            else:
                inverse_variance_reduction_area = 1.0 / (
                    np.abs(data_reduce[:, reduction_mask]) + readnoise**2
                )

        else:
            if bad_pixel_mask is not None:
                inverse_variance_reduction_area = inverse_variance[
                    :, reduction_mask_wo_badpixels
                ]
            else:
                inverse_variance_reduction_area = inverse_variance[:, reduction_mask]
    else:
        inverse_variance_reduction_area = None

    if bad_pixel_mask is not None:
        reduction_mask_used = reduction_mask_wo_badpixels.copy()
    else:
        reduction_mask_used = reduction_mask.copy()

    results = {}

    if reduction_parameters.temporal_model:
        if reduction_parameters.include_opposite_regressors:
            # opposite_mask, _ = regressor_selection.make_mask_from_psf_track(
            #     yx_position=-1 * signal_position,
            #     psf_size=reduction_parameters.reduction_mask_psf_size,
            #     pa=pa, image_size=data.shape[-1],
            #     image_center=yx_center_injection,
            #     yx_anamorphism=reduction_parameters.yx_anamorphism,
            #     right_handed=reduction_parameters.right_handed,
            #     return_cube=True)
            # NOTE: use `signal_mask` instead?
            opposite_mask = regressor_selection.make_mirrored_mask(
                reduction_mask, yx_center
            )
        else:
            opposite_mask = None

        regressor_pool_mask = regressor_selection.make_regressor_pool_for_pixel(
            reduction_parameters=reduction_parameters,
            runtime=runtime,
            yx_pixel=planet_absolute_yx_pos,
            yx_dim=yx_dim,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            signal_mask=signal_mask,
            known_companion_mask=known_companion_mask,
            bad_pixel_mask=bad_pixel_mask,
            additional_regressors=opposite_mask,
            right_handed=reduction_parameters.right_handed,
            pa=pa,
        )

        if cross_validation:
            cv_result = regression.temporal_pca_cross_validation(
                data=data_reduce,
                model=model,
                pa=pa,
                reduction_parameters=reduction_parameters,
                runtime=runtime,
                reduction_mask=reduction_mask_used,
                regressor_pool_mask=regressor_pool_mask,
                regressor_matrix=None,
                inverse_variance_reduction_area=inverse_variance_reduction_area,
            )
            return cv_result

        if reduction_parameters.reduce_single_position:
            result = regression.run_trap_with_model_temporal(
                data=data_reduce,
                model=model,
                # model=None,
                pa=pa,
                reduction_parameters=reduction_parameters,
                runtime=runtime,
                planet_relative_yx_pos=signal_position,
                reduction_mask=reduction_mask_used,
                known_companion_mask=known_companion_mask,
                opposite_mask=opposite_mask,
                yx_center=yx_center,
                yx_center_injection=yx_center_injection,
                signal_mask=signal_mask,
                regressor_pool_mask=regressor_pool_mask,
                bad_pixel_mask=bad_pixel_mask,
                regressor_matrix=None,
                true_contrast=true_contrast,
                inverse_variance_reduction_area=inverse_variance_reduction_area,
                plot_all_diagnostics=reduction_parameters.plot_all_diagnostics,
                return_input_data=reduction_parameters.return_input_data,
            )
        else:
            result = regression.run_trap_with_model_temporal_optimized(
                data=data_reduce,
                model=model,
                pa=pa,
                reduction_parameters=reduction_parameters,
                runtime=runtime,
                reduction_mask=reduction_mask_used,
                regressor_pool_mask=regressor_pool_mask,
                regressor_matrix=None,
                inverse_variance_reduction_area=inverse_variance_reduction_area,
            )

        # if result is not None:
        #     if reduction_parameters.fit_planet:
        #         result.compute_contrast_weighted_average(mask_outliers=True)
        #         if reduction_parameters.verbose:
        #             print(result)
        #     result.reduction_parameters = reduction_parameters
        # if reduction_parameters.reduce_single_position:
        results["temporal"] = result
        # else:
        #     results['']

    if reduction_parameters.spatial_model:
        result = regression.run_trap_with_model_spatial(
            data=data_reduce,
            model=model,
            pa=pa,
            reduction_parameters=reduction_parameters,
            runtime=runtime,
            planet_relative_yx_pos=signal_position,
            reduction_mask=reduction_mask_used,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            inverse_variance_reduction_area=inverse_variance_reduction_area,
            true_contrast=true_contrast,
            training_data=None,
            return_input_data=False,
            verbose=reduction_parameters.verbose,
        )

        if result is not None:
            if reduction_parameters.fit_planet:
                result.compute_contrast_weighted_average(mask_outliers=True)
                if reduction_parameters.verbose:
                    logger.debug("%s", result)
            result.reduction_parameters = reduction_parameters
        results["spatial"] = result

    if (
        reduction_parameters.temporal_model
        and reduction_parameters.temporal_plus_spatial_model
    ):
        if reduction_parameters.second_stage_trap:
            data_reduce_psf_subtracted = (
                data_reduce
                - injected_model_cube * results["temporal"].measured_contrast
            )

            results[
                "temporal_psf_subtracted"
            ] = regression.run_trap_with_model_temporal(
                data=data_reduce_psf_subtracted,
                model=None,
                pa=pa,
                reduction_parameters=reduction_parameters,
                runtime=runtime,
                planet_relative_yx_pos=signal_position,
                reduction_mask=reduction_mask_used,
                known_companion_mask=known_companion_mask,
                opposite_mask=opposite_mask,
                yx_center=yx_center,
                yx_center_injection=yx_center_injection,
                signal_mask=signal_mask,
                regressor_pool_mask=regressor_pool_mask,
                bad_pixel_mask=bad_pixel_mask,
                regressor_matrix=None,
                inverse_variance_reduction_area=inverse_variance_reduction_area,
                true_contrast=true_contrast,
                plot_all_diagnostics=reduction_parameters.plot_all_diagnostics,
                return_input_data=False,
                verbose=reduction_parameters.verbose,
            )

            data_reduce_noise_subtracted = data_reduce - np.nan_to_num(
                results["temporal_psf_subtracted"].noise_model_cube
            )
        else:
            data_reduce_noise_subtracted = data_reduce - np.nan_to_num(
                results["temporal"].noise_model_cube
            )

        if reduction_parameters.remove_model_from_spatial_training:
            training_data = (
                data_reduce_noise_subtracted
                - injected_model_cube * results["temporal"].measured_contrast
            )
        else:
            training_data = None

        if reduction_parameters.remove_bad_residuals_for_spatial_model:
            bad_residual_mask = np.zeros(
                (data_reduce.shape[-2], data_reduce.shape[-1])
            ).astype("bool")
            bad_residual_mask[reduction_mask_used] = ~results[
                "temporal"
            ].good_residual_mask

            if bad_pixel_mask is None:
                bad_pixel_mask = bad_residual_mask
            else:
                bad_pixel_mask = np.logical_or(bad_pixel_mask, bad_residual_mask)
            if inverse_variance_reduction_area is not None:
                inverse_variance_reduction_area = inverse_variance_reduction_area[
                    :, ~bad_residual_mask[reduction_mask_used]
                ]
            reduction_mask_used = np.logical_and(
                reduction_mask_used, ~bad_residual_mask
            )

        reduction_parameters_alternative = reduction_parameters.merge(
            spatial_components_fraction=reduction_parameters.spatial_components_fraction_after_trap,
        )

        result = regression.run_trap_with_model_spatial(
            data=data_reduce_noise_subtracted,
            model=model,
            pa=pa,
            reduction_parameters=reduction_parameters_alternative,
            runtime=runtime,
            planet_relative_yx_pos=signal_position,
            reduction_mask=reduction_mask_used,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            inverse_variance_reduction_area=inverse_variance_reduction_area,
            true_contrast=true_contrast,
            training_data=training_data,
            return_input_data=False,
            verbose=reduction_parameters.verbose,
        )

        if result is not None:
            if reduction_parameters.fit_planet:
                result.compute_contrast_weighted_average(mask_outliers=True)
                if reduction_parameters.verbose:
                    logger.debug("%s", result)
            result.reduction_parameters = reduction_parameters
        results["temporal_plus_spatial"] = result

    return results


@dataclasses.dataclass
class OutputPaths:
    """File paths for all outputs associated with one reduction model key."""

    detection_image: str
    norm_detection_image: str
    contrast_table: str
    contrast_image: str
    median_contrast_image: str
    contrast_plot: str
    correlation_matrix_binned: str = ""

    @classmethod
    def make(cls, result_folder, basename, key, sigma, corr=False):
        infix = "_corr" if corr else ""
        name = f"{basename}_{key}"
        sigma_str = f"_sigma{sigma:.2f}"
        return cls(
            detection_image=os.path.join(
                result_folder, f"detection{infix}_{name}.fits"
            ),
            norm_detection_image=os.path.join(
                result_folder, f"norm_detection{infix}_{name}.fits"
            ),
            contrast_table=os.path.join(
                result_folder, f"contrast_table{infix}_{name}.fits"
            ),
            contrast_image=os.path.join(
                result_folder, f"contrast_image{infix}_{name}{sigma_str}.fits"
            ),
            median_contrast_image=os.path.join(
                result_folder, f"median_contrast_image{infix}_{name}{sigma_str}.fits"
            ),
            contrast_plot=os.path.join(
                result_folder, f"contrast_plot{infix}_{name}{sigma_str}.jpg"
            ),
            correlation_matrix_binned=os.path.join(
                result_folder,
                f"correlation_matrix_binned{infix}_{name}{sigma_str}.fits",
            )
            if corr
            else "",
        )


def fill_detection_image(
    detection_image,
    detection_image_corr,
    correlation_matrix_binned,
    result,
    yx,
    inject_fake,
    compute_residual_correlation,
    use_residual_correlation,
):
    """Fill detection image arrays for a single position from a result dict."""
    for key in detection_image:
        if result[key] is not None:
            detection_image[key][0][yx[0], yx[1]] = result[key].measured_contrast
            detection_image[key][1][yx[0], yx[1]] = result[key].contrast_uncertainty
            detection_image[key][2][yx[0], yx[1]] = result[key].snr
            if inject_fake:
                detection_image[key][3][yx[0], yx[1]] = result[key].true_contrast
                detection_image[key][4][yx[0], yx[1]] = (
                    result[key].measured_contrast - result[key].true_contrast
                )
                detection_image[key][5][yx[0], yx[1]] = (
                    result[key].relative_deviation_from_true
                )
                detection_image[key][6][yx[0], yx[1]] = result[key].wrong_in_sigma
            if compute_residual_correlation and use_residual_correlation:
                detection_image_corr[key][0][yx[0], yx[1]] = (
                    result[key].measured_contrast_with_corr
                )
                detection_image_corr[key][1][yx[0], yx[1]] = (
                    result[key].contrast_uncertainty_with_corr
                )
                detection_image_corr[key][2][yx[0], yx[1]] = result[key].snr_with_corr
                detection_image_corr[key][3][yx[0], yx[1]] = (
                    result[key].correlation_info["corr_length_exponential"]
                )
                detection_image_corr[key][4][yx[0], yx[1]] = (
                    result[key].correlation_info["corr_length_matern32"]
                )
                detection_image_corr[key][5][yx[0], yx[1]] = (
                    result[key].correlation_info["corr_length_matern52"]
                )
                correlation_matrix_binned[key][:, yx[0], yx[1]] = (
                    result[key]
                    .correlation_info["summary_dataframe"]
                    .empirical_correlation.values
                )


def run_trap_search(
    data,
    flux_psf,
    pa,
    reduction_parameters,
    known_companion_mask,
    runtime=None,
    inverse_variance=None,
    bad_pixel_mask=None,
    yx_center=None,
    yx_center_injection=None,
    amplitude_modulation=None,
    contrast_map=None,
    readnoise=0.0,
    use_progress_bar=False,
):
    """Iterates TRAP over grid of positions given by the boolean mask
    `search_region` in the `reduction_parameters` object.

    Parameters
    ----------
    data : array_like
        Temporal image cube. First axis is time.
    flux_psf : array_like
        Image of unsaturated PSF.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    known_companion_mask : array_like
        Boolean mask of image size. True for pixels affected by companion flux.
    inverse_variance : array_like
        Cube containing inverse variances for `data`.
    bad_pixel_mask : array_like
        Bad pixel mask for `data`.
    yx_center : tuple
        Average or median image center position to be used for regressor selection.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    amplitude_modulation : array_like, optional
        Array containing scaling factors for the companion PSF brightness for
        each time (e.g. derived from satellite spots).
    contrast_map : array_like, optional
        Contrast detection map (derived using the same reduction parameters
        and data) to be used for injection retrieval testing to determine
        biases in reduction (see `inject_fake`, `read_injection_files`
        and 'injection_sigma') in `~trap.parameters.Reduction_parameters`.
    readnoise : scalar
        The detector read noise (e rms/pix/readout).
    use_progress_bar : bool
        If True, a progress bar is shown during the reduction.

    Returns
    -------
    array_like
        Detection map cube containing the following maps in order:
            0) Measured contrast
            1) Contrast uncertainty
            2) SNR
        If `inject_fake` in `reduction_parameters` is True,
        additionally the following information is added:
            3) True contrast
            4) Measured contrast minus `true_contrast`
            5) Relative deviation from `true_contrast`
            6) Deviation from `true_contrast` in sigma
    """

    # `data`/`inverse_variance` may be shared-store references (dumped as
    # float64); resolve to read-only memmap views for driver-side use.
    data_ref = data if isinstance(data, SharedArrayRef) else None
    inverse_variance_ref = (
        inverse_variance if isinstance(inverse_variance, SharedArrayRef) else None
    )
    data = shared_arrays.resolve(data)
    inverse_variance = shared_arrays.resolve(inverse_variance)

    data = data.astype("float64", copy=data_ref is None)
    flux_psf = flux_psf.astype("float64")
    pa = pa.astype("float64")

    if inverse_variance is not None:
        inverse_variance = inverse_variance.astype(
            "float64", copy=inverse_variance_ref is None
        )

    oversampling = reduction_parameters.oversampling
    yx_dim = (data.shape[-2], data.shape[-1])
    yx_center_output = (yx_dim[0] // 2, yx_dim[1] // 2)
    if yx_center is None:
        yx_center = (yx_dim[0] // 2, yx_dim[1] // 2)

    if amplitude_modulation is None:
        amplitude_modulation = np.ones(data.shape[0])

    if reduction_parameters.inject_fake:
        detection_image_dim = 7  # Include info about injected signal
    else:
        # EDIT: REMEMBER TO CHANGE BACK detection_image_dim to 3
        detection_image_dim = 3  # Only contrast, noise, snr

    detection_image = {}
    if reduction_parameters.temporal_model:
        detection_image["temporal"] = np.empty(
            (
                detection_image_dim,
                int(yx_dim[0] * oversampling),
                int(yx_dim[1] * oversampling),
            )
        )
        detection_image["temporal"][:] = np.nan
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image["temporal_plus_spatial"] = np.empty(
                (
                    detection_image_dim,
                    int(yx_dim[0] * oversampling),
                    int(yx_dim[1] * oversampling),
                )
            )
            detection_image["temporal_plus_spatial"][:] = np.nan
    if reduction_parameters.spatial_model:
        detection_image["spatial"] = np.empty(
            (
                detection_image_dim,
                int(yx_dim[0] * oversampling),
                int(yx_dim[1] * oversampling),
            )
        )
        detection_image["spatial"][:] = np.nan

    # EDIT: ADDED FOR QUICK CORRELATION TESTS
    detection_image_corr = {}
    if reduction_parameters.temporal_model:
        detection_image_corr["temporal"] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
        )
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image_corr["temporal_plus_spatial"] = np.zeros(
                (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
            )
    if reduction_parameters.spatial_model:
        detection_image_corr["spatial"] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
        )

    correlation_matrix_binned = {}
    if reduction_parameters.temporal_model:
        correlation_matrix_binned["temporal"] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
        )
        if reduction_parameters.temporal_plus_spatial_model:
            correlation_matrix_binned["temporal_plus_spatial"] = np.zeros(
                (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
            )
    if reduction_parameters.spatial_model:
        correlation_matrix_binned["spatial"] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling))
        )

    search_region = runtime.search_region if runtime is not None else reduction_parameters.search_region
    search_coordinates = np.argwhere(search_region) * oversampling
    # relative coordinates to output image center (i.e. position of star)
    relative_coords = np.array(
        list(
            map(
                lambda x: image_coordinates.absolute_yx_to_relative_yx(
                    x, yx_center_output
                ),
                search_coordinates.tolist(),
            )
        )
    )
    logger.info("Number of positions: %d", len(relative_coords))
    ncpus = runtime.ncpus if runtime is not None else (reduction_parameters.ncpus or multiprocessing.cpu_count())
    ncpus = min(ncpus, multiprocessing.cpu_count())
    if reduction_parameters.use_multiprocess:
        # Use more chunks than CPUs to prevent long idle time in case one job finishes quicker
        number_of_chunks = round(ncpus * 8)

        (
            search_coordinates,
            relative_coords,
            relative_coords_regions,
            iteration,
            separation_equalized,
        ) = shuffle_and_equalize_relative_positions(
            search_coordinates,
            relative_coords,
            number_of_chunks,
            max_separation_deviation=2,
            max_iterations=50,
            rng=None,
        )
        logger.debug("Number of positions per chunk: %d", len(relative_coords_regions[0]))

        a = datetime.datetime.now()
        own_store = None
        if data_ref is None:
            # Standalone call without a pre-dumped store: dump the large
            # arrays here so workers memmap them instead of unpickling copies.
            required_bytes = data.nbytes
            if inverse_variance is not None:
                required_bytes += inverse_variance.nbytes
            own_store = SharedArrayStore(
                scratch_dir=getattr(reduction_parameters, "scratch_dir", None),
                required_bytes=required_bytes,
            )
            data_ref = own_store.dump("data", data)
            if inverse_variance is not None:
                inverse_variance_ref = own_store.dump(
                    "inverse_variance", inverse_variance
                )
        try:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                result_generator = Parallel(n_jobs=ncpus, return_as="generator")(
                    delayed(trap_search_region)(
                        region,
                        data=data_ref,
                        inverse_variance=inverse_variance_ref,
                        flux_psf=flux_psf,
                        pa=pa,
                        reduction_parameters=reduction_parameters,
                        runtime=runtime,
                        known_companion_mask=known_companion_mask,
                        bad_pixel_mask=bad_pixel_mask,
                        yx_center=yx_center,
                        yx_center_injection=yx_center_injection,
                        amplitude_modulation=amplitude_modulation,
                        contrast_map=contrast_map,
                        readnoise=readnoise,
                    )
                    for region in relative_coords_regions
                )
                results = list(
                    tqdm(
                        result_generator,
                        total=len(relative_coords_regions),
                        unit="chunk",
                        disable=not use_progress_bar,
                    )
                )
        finally:
            if own_store is not None:
                own_store.cleanup()
        results = [item for sublist in results for item in sublist]

        for idx, result in enumerate(results):
            fill_detection_image(
                detection_image,
                detection_image_corr,
                correlation_matrix_binned,
                result,
                search_coordinates[idx],
                reduction_parameters.inject_fake,
                reduction_parameters.compute_residual_correlation,
                reduction_parameters.use_residual_correlation,
            )

        b = datetime.datetime.now()
    else:
        a = datetime.datetime.now()

        for idx, coords in enumerate(tqdm(relative_coords)):
            # if reduction_parameters.inject_fake == True:
            #         reduction_parameters.true_position = image_coordinates.absolute_yx_to_relative_yx(
            #             coords, image_center_yx=yx_center)
            #     reduction_parameters.guess_position = image_coordinates.absolute_yx_to_relative_yx(
            #         coords, image_center_yx=yx_center)

            result = trap_one_position(
                coords,
                data=data,
                inverse_variance=inverse_variance,
                flux_psf=flux_psf,
                pa=pa,
                reduction_parameters=reduction_parameters,
                known_companion_mask=known_companion_mask,
                runtime=runtime,
                bad_pixel_mask=bad_pixel_mask,
                yx_center=yx_center,
                yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation,
                contrast_map=contrast_map,
                readnoise=readnoise,
            )
            fill_detection_image(
                detection_image,
                detection_image_corr,
                correlation_matrix_binned,
                result,
                search_coordinates[idx],
                reduction_parameters.inject_fake,
                reduction_parameters.compute_residual_correlation,
                reduction_parameters.use_residual_correlation,
            )
            del result
        b = datetime.datetime.now()
    c = b - a
    logger.debug("Main reduction computation time: %s", c)

    if (
        not reduction_parameters.compute_residual_correlation
        and not reduction_parameters.use_residual_correlation
    ):
        detection_image_corr = None
        correlation_matrix_binned = None

    return detection_image, detection_image_corr, correlation_matrix_binned


def multi_position_cross_validation(
    relative_coords,
    data,
    flux_psf,
    pa,
    reduction_parameters,
    known_companion_mask,
    runtime=None,
    inverse_variance=None,
    bad_pixel_mask=None,
    result_name=None,
    yx_center=None,
    yx_center_injection=None,
    amplitude_modulation=None,
    contrast_map=None,
    readnoise=0.0,
):
    """Iterates TRAP over grid of positions given by the boolean mask
    `search_region` in the `reduction_parameters` object.

    Parameters
    ----------
    data : array_like
        Temporal image cube. First axis is time.
    flux_psf : array_like
        Image of unsaturated PSF.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    known_companion_mask : array_like
        Boolean mask of image size. True for pixels affected by companion flux.
    inverse_variance : array_like
        Cube containing inverse variances for `data`.
    bad_pixel_mask : array_like
        Bad pixel mask for `data`.
    result_name : str
        Name for output file.
    yx_center : tuple
        Average or median image center position to be used for regressor selection.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    amplitude_modulation : array_like, optional
        Array containing scaling factors for the companion PSF brightness for
        each time (e.g. derived from satellite spots).
    contrast_map : array_like, optional
        Contrast detection map (derived using the same reduction parameters
        and data) to be used for injection retrieval testing to determine
        biases in reduction (see `inject_fake`, `read_injection_files`
        and 'injection_sigma') in `~trap.parameters.Reduction_parameters`.
    readnoise : scalar
        The detector read noise (e rms/pix/readout).

    Returns
    -------
    array_like
        Detection map cube containing the following maps in order:
            0) Measured contrast
            1) Contrast uncertainty
            2) SNR
        If `inject_fake` in `reduction_parameters` is True,
        additionally the following information is added:
            3) True contrast
            4) Measured contrast minus `true_contrast`
            5) Relative deviation from `true_contrast`
            6) Deviation from `true_contrast` in sigma
    """

    # `data`/`inverse_variance` may be shared-store references (dumped as
    # float64); resolve to read-only memmap views for driver-side use.
    data_ref = data if isinstance(data, SharedArrayRef) else None
    inverse_variance_ref = (
        inverse_variance if isinstance(inverse_variance, SharedArrayRef) else None
    )
    data = shared_arrays.resolve(data)
    inverse_variance = shared_arrays.resolve(inverse_variance)

    data = data.astype("float64", copy=data_ref is None)
    flux_psf = flux_psf.astype("float64")
    pa = pa.astype("float64")

    if inverse_variance is not None:
        inverse_variance = inverse_variance.astype(
            "float64", copy=inverse_variance_ref is None
        )

    yx_dim = (data.shape[-2], data.shape[-1])
    if yx_center is None:
        yx_center = (yx_dim[0] // 2, yx_dim[1] // 2)

    if amplitude_modulation is None:
        amplitude_modulation = np.ones(data.shape[0])

    logger.info("Number of positions: %d", len(relative_coords))
    ncpus = runtime.ncpus if runtime is not None else (reduction_parameters.ncpus or multiprocessing.cpu_count())
    ncpus = min(ncpus, multiprocessing.cpu_count())
    if reduction_parameters.use_multiprocess:
        a = datetime.datetime.now()
        own_store = None
        if data_ref is None:
            # Standalone call without a pre-dumped store: dump the large
            # arrays here so workers memmap them instead of unpickling copies.
            required_bytes = data.nbytes
            if inverse_variance is not None:
                required_bytes += inverse_variance.nbytes
            own_store = SharedArrayStore(
                scratch_dir=getattr(reduction_parameters, "scratch_dir", None),
                required_bytes=required_bytes,
            )
            data_ref = own_store.dump("data", data)
            if inverse_variance is not None:
                inverse_variance_ref = own_store.dump(
                    "inverse_variance", inverse_variance
                )
        try:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                result_generator = Parallel(n_jobs=ncpus, return_as="generator")(
                    delayed(trap_search_region)(
                        coords,
                        data=data_ref,
                        inverse_variance=inverse_variance_ref,
                        flux_psf=flux_psf,
                        pa=pa,
                        reduction_parameters=reduction_parameters,
                        runtime=runtime,
                        known_companion_mask=known_companion_mask,
                        bad_pixel_mask=bad_pixel_mask,
                        yx_center=yx_center,
                        yx_center_injection=yx_center_injection,
                        amplitude_modulation=amplitude_modulation,
                        contrast_map=contrast_map,
                        readnoise=readnoise,
                        cross_validation=True,
                    )
                    for coords in relative_coords
                )
                results = list(tqdm(result_generator, total=len(relative_coords)))
        finally:
            if own_store is not None:
                own_store.cleanup()
        results = [item for sublist in results for item in sublist]

        b = datetime.datetime.now()
    else:
        a = datetime.datetime.now()

        results = []
        for _, coords in enumerate(tqdm(relative_coords)):
            # if reduction_parameters.inject_fake:
            #         reduction_parameters.true_position = image_coordinates.absolute_yx_to_relative_yx(
            #             coords, image_center_yx=yx_center)
            #     reduction_parameters.guess_position = image_coordinates.absolute_yx_to_relative_yx(
            #         coords, image_center_yx=yx_center)

            result = trap_one_position(
                coords,
                data=data,
                inverse_variance=inverse_variance,
                flux_psf=flux_psf,
                pa=pa,
                reduction_parameters=reduction_parameters,
                known_companion_mask=known_companion_mask,
                runtime=runtime,
                bad_pixel_mask=bad_pixel_mask,
                yx_center=yx_center,
                yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation,
                contrast_map=contrast_map,
                readnoise=readnoise,
                cross_validation=True,
            )
            results.append(result)
        b = datetime.datetime.now()
    c = b - a
    logger.debug("Main reduction computation time: %s", c)

    return results


def trap_search_region(
    relative_coords,
    data,
    flux_psf,
    pa,
    reduction_parameters,
    known_companion_mask,
    runtime=None,
    inverse_variance=None,
    bad_pixel_mask=None,
    result_name=None,
    yx_center=None,
    yx_center_injection=None,
    amplitude_modulation=None,
    contrast_map=None,
    readnoise=0.0,
    cross_validation=False,
):
    """Iterates TRAP over grid of positions given by the boolean mask
    `search_region` in the `reduction_parameters` object.

    Parameters
    ----------
    data : array_like
        Temporal image cube. First axis is time.
    flux_psf : array_like
        Image of unsaturated PSF.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    known_companion_mask : array_like
        Boolean mask of image size. True for pixels affected by companion flux.
    inverse_variance : array_like
        Cube containing inverse variances for `data`.
    bad_pixel_mask : array_like
        Bad pixel mask for `data`.
    result_name : str
        Name for output file.
    yx_center : tuple
        Average or median image center position to be used for regressor selection.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    amplitude_modulation : array_like, optional
        Array containing scaling factors for the companion PSF brightness for
        each time (e.g. derived from satellite spots).
    contrast_map : array_like, optional
        Contrast detection map (derived using the same reduction parameters
        and data) to be used for injection retrieval testing to determine
        biases in reduction (see `inject_fake`, `read_injection_files`
        and 'injection_sigma') in `~trap.parameters.Reduction_parameters`.
    readnoise : scalar
        The detector read noise (e rms/pix/readout).

    Returns
    -------
    array_like
        Detection map cube containing the following maps in order:
            0) Measured contrast
            1) Contrast uncertainty
            2) SNR
        If `inject_fake` in `reduction_parameters` is True,
        additionally the following information is added:
            3) True contrast
            4) Measured contrast minus `true_contrast`
            5) Relative deviation from `true_contrast`
            6) Deviation from `true_contrast` in sigma
    """

    # In worker processes the large arrays arrive as shared-store references;
    # resolve them to read-only memmaps (plain arrays pass through).
    data = shared_arrays.resolve(data)
    inverse_variance = shared_arrays.resolve(inverse_variance)

    sub_region_results = []
    for idx, coords in enumerate(relative_coords):
        result = trap_one_position(
            coords,
            data=data,
            inverse_variance=inverse_variance,
            flux_psf=flux_psf,
            pa=pa,
            reduction_parameters=reduction_parameters,
            known_companion_mask=known_companion_mask,
            runtime=runtime,
            bad_pixel_mask=bad_pixel_mask,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            amplitude_modulation=amplitude_modulation,
            contrast_map=contrast_map,
            readnoise=readnoise,
            cross_validation=cross_validation,
        )
        sub_region_results.append(result)
    return sub_region_results


def make_reduction_header(
    reduction_parameters,
    instrument,
    bad_frames,
    exclude_bad_pixel,
    oversampling,
    right_handed,
    yx_known_companion_position,
):
    raise NotImplementedError()


def _wavelength_geometry(
    wavelength_index,
    data_shape,
    data_crop_size,
    yx_center_full,
    yx_center_injection_full,
):
    """Output-frame center and per-frame injection centers for one wavelength.

    Single source of truth for the geometry used both when filling the
    shared-array store and inside the reduction loops of
    `run_complete_reduction`.
    """
    # This block defines yx_center which gives the center of the output file
    # based on cropping or no cropping
    if yx_center_full is None:
        if data_crop_size is None:
            yx_center = np.array((data_shape[-2] // 2.0, data_shape[-1] // 2.0))
        else:
            yx_center = np.array((data_crop_size // 2.0, data_crop_size // 2.0))
    else:
        if data_crop_size is None:
            yx_center = np.array(yx_center_full[wavelength_index])
        else:
            yx_center = np.array((data_crop_size // 2.0, data_crop_size // 2.0))

    yx_center_injection = yx_center
    if yx_center_injection_full is not None:
        try:
            if yx_center_injection_full.ndim == 3:
                if data_crop_size is None:
                    yx_center_injection = yx_center_injection_full[
                        wavelength_index, :
                    ]
                else:
                    # Image centers in cropped frame
                    yx_center_injection = (
                        yx_center_injection_full[wavelength_index, :]
                        - np.round(yx_center_full[wavelength_index])
                        + yx_center
                    )
                    # Non-rounded image center in cropped frame
                    yx_center = np.nanmedian(yx_center_injection, axis=0)
        except AttributeError:
            pass
    return yx_center, yx_center_injection


def _prepare_wavelength_slice(
    wavelength_index,
    data_full,
    inverse_variance_full,
    flux_psf,
    pa,
    reduction_parameters,
    runtime,
    data_crop_size,
    yx_center_full,
    yx_center_injection,
    amplitude_modulation,
):
    """Crop the data (and inverse variance) for one wavelength and remove
    known companions if configured.

    Single source of truth for the per-wavelength data consumed by the
    reduction: used to fill the shared-array store before the loops of
    `run_complete_reduction` (parallel path) and inside the loops (serial
    and single-position paths).
    """
    if reduction_parameters.inject_fake:
        # Return copy of data when injecting fake to not contaminate data
        if data_crop_size is not None:
            data = crop_box_from_3D_cube(
                data_full[wavelength_index],
                data_crop_size,
                center_yx=np.round(yx_center_full[wavelength_index]),
            ).copy()
        else:
            data = data_full[wavelength_index].copy()
    else:
        if data_crop_size is not None:
            data = crop_box_from_3D_cube(
                data_full[wavelength_index],
                data_crop_size,
                center_yx=np.round(yx_center_full[wavelength_index]),
            )
        else:
            data = data_full[wavelength_index]

    if inverse_variance_full is not None:
        inverse_variance = crop_box_from_3D_cube(
            inverse_variance_full[wavelength_index],
            data_crop_size,
            center_yx=np.round(yx_center_full[wavelength_index]),
        ).copy()
    else:
        inverse_variance = None

    if reduction_parameters.remove_known_companions:
        # NOTE: This currently doesn't remove photon noise from variance map
        # NOTE: Should change to faster implementation of `inject_model_into_data`

        for companion_index, kc_contrast in enumerate(
            runtime.known_companion_contrast[wavelength_index]
        ):
            # NOTE: Check format of known_companion_contrast and amplitude_modulation
            kc_contrast = kc_contrast * amplitude_modulation

            data = makesource.addsource(
                data,
                runtime.yx_known_companion_position[companion_index],
                pa,
                flux_psf,
                image_center=yx_center_injection,
                norm=-1 * kc_contrast,
                jitter=0,
                poisson_noise=False,
                yx_anamorphism=runtime.yx_anamorphism,
                right_handed=reduction_parameters.right_handed,
                subpixel=True,
                verbose=False,
            )

    return data, inverse_variance


def run_complete_reduction(
    data_full,
    flux_psf_full,
    pa,
    instrument,
    reduction_parameters,
    temporal_components_fraction=[0.2],
    wavelength_indices=None,
    inverse_variance_full=None,
    bad_frames=None,
    bad_pixel_mask_full=None,
    xy_image_centers=None,
    amplitude_modulation_full=None,
    cross_validation=False,
    verbose=False,
    overwrite=False,
    use_progress_bar=True,
):
    """Runs complete TRAP reduction on data and produces contrast and
    normalized detection maps as well as contrast curves. This is the most
    high-level wrapper for the code. The wrapper hierarchy is:
    `run_complete_reduction` > `run_trap_search` > `trap_one_position`.

    Parameters
    ----------
    data_full : array_like
        A spectro-temporal image cube or simple temporal image cube.
        First axis should be wavelength, second axis time in the
        spectro-temporal case (IFU).
    flux_psf_full : array_like
        One model PSF image for each wavelength in for IFU data.
        If monochromatic data is used a single image is sufficient.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    instrument : `~trap.parameters.Instrument`
        An `~trap.parameters.Instrument` object containing parameters intrinsic
        to the instrument used, such as diameter, pixel scale,
        gain and read noise.
    reduction_parameters : `~trap.parameters.Reduction_parameters` or `~trap.parameters.TrapConfig`
        A `~trap.parameters.Reduction_parameters` object or `~trap.parameters.TrapConfig`
        object containing all parameters necessary for the TRAP pipeline.
    temporal_components_fraction : array_like
        List containing the principal component fraction to be used for
        the temporal TRAP analysis. If more than one number is given
        TRAP will loop over them and produce outputs for all numbers.
        Default is [0.3].
    wavelength_indices : array_like, optional
        Vector containing the indices of the wavelength slices that should
        be reduced.
    inverse_variance_full : array_like, optional
        Data cube of the same shape as `data_full` that contains the
        inverse variance of each data point.
    bad_frames : array_like, optional
        Vector containing the indices of bad frames to remove.
    bad_pixel_mask_full : array_like, optional
        One bad pixel binary mask (1 for bad pixel, 0 for good pixel) for
        each wavelength slice. Axis=0 is wavelength. This option should be
        used when running TRAP on non-aligned data.
    xy_image_centers : array_like, optional
        Array containing tuple of xy image center positions for each wavelength
        (axis=0) and time (axis=1). This option should be used when running TRAP on
        non-aligned data.
    amplitude_modulation_full : array_like, optional
        Array containing scaling factors for the companion PSF brightness for
        each wavelength (axis=0) and time (axis=1) (e.g. derived from satellite
        spots).
    cross_validation : bool, optional
        If True, the reduction is performed in cross-validation mode, i.e.
        for the given list of component fractions signals are injected to determine
        the best number of components. Default is False. Returns the scores.
    verbose : bool, optional
        If True, print out additional information. Default is False.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    use_progress_bar : bool, optional
        If True, display a progress bar during processing. Default is False.

    Returns
    -------
    collections.OrderedDict or None
        If `reduce_single_position` in `reduction_parameters` is set True,
        an ordered dictionary is returned.
        The dictionary contains an entry for each temporal_components_fraction
        and each wavelength index tested. Therein contained is another
        dictionary containing the `~trap.regression.Result` object for
        reductions performed under the 'temporal', 'temporal_plus_spatial',
        and 'spatial' key, depending on whether the `temporal_model`,
        `temporal_plus_spatial_model`, and `spatial_model` parameters set
        in `reduction_parameters` are True.

        Otherwise, the return value is None.

    """

    # Convert any input type to TrapReductionConfig (frozen, immutable)
    reduction_parameters = _to_reduction_config(reduction_parameters)

    if bad_frames is None:
        bad_frames = []

    if instrument.detector_gain != 1:
        flux_psf_full *= instrument.detector_gain
        data_full *= instrument.detector_gain

    if flux_psf_full.ndim < 3:
        flux_psf_full = np.expand_dims(flux_psf_full, axis=0)
    if data_full.ndim < 4:
        data_full = np.expand_dims(data_full, axis=0)
    if inverse_variance_full is not None and inverse_variance_full.ndim < 4:
        inverse_variance_full = np.expand_dims(inverse_variance_full, axis=0)

    if reduction_parameters.highpass_filter is not None:
        raise NotImplementedError()
        # for wave_idx, wave_cube in enumerate(data_full):
        #     flux_psf_full[wave_idx] = high_pass_filter(
        #         flux_psf_full[wave_idx],
        #         cutoff_frequency=reduction_parameters.highpass_filter)
        # data_full = high_pass_filter_cube(
        #     data_full,
        #     cutoff_frequency=reduction_parameters.highpass_filter,
        #     verbose=True)

    instrument.compute_fwhm()

    if reduction_parameters.autosize_masks_in_lambda_over_d:
        assert (
            reduction_parameters.signal_mask_size_in_lambda_over_d
            >= reduction_parameters.reduction_mask_size_in_lambda_over_d
        ), "Signal mask size must be >= reduction mask size"
        stamp_sizes = determine_psf_stampsizes(
            instrument.fwhm.value,
            size_in_lamda_over_d=reduction_parameters.signal_mask_size_in_lambda_over_d,
        )
        stamp_sizes_reduction = determine_psf_stampsizes(
            instrument.fwhm.value,
            size_in_lamda_over_d=reduction_parameters.reduction_mask_size_in_lambda_over_d,
        )
    else:
        stamp_sizes = np.repeat(
            reduction_parameters.signal_mask_psf_size, len(instrument.wavelengths)
        )
        stamp_sizes_reduction = np.repeat(
            reduction_parameters.reduction_mask_psf_size, len(instrument.wavelengths)
        )
    if flux_psf_full.shape[-1] < np.max(stamp_sizes):
        raise ValueError(
            "The provided PSF images are too small for the chosen parameters."
        )
    psf_stamps = prepare_psf(flux_psf_full, psf_size=stamp_sizes)
    # Remove bad frames
    if bad_frames is not None:
        data_full = np.delete(data_full, bad_frames, axis=1)
        pa = np.delete(pa, bad_frames, axis=0)
        if inverse_variance_full is not None:
            inverse_variance_full = np.delete(inverse_variance_full, bad_frames, axis=1)

        if xy_image_centers is not None:
            xy_image_centers = np.delete(xy_image_centers, bad_frames, axis=1)

    # Configure image centers
    if xy_image_centers is None:
        # yx_center_full contains one center for each wavelength
        yx_center_full = np.ones((data_full.shape[0], 2))
        yx_center_full[:, 0] = data_full.shape[-2] // 2
        yx_center_full[:, 1] = data_full.shape[-1] // 2
        yx_center_injection_full = None
        max_shift = 0
    else:
        if xy_image_centers.ndim == 1:
            # yx_center_injection_full contains one center for each wavelength and time
            yx_center_injection_full = np.ones(
                (data_full.shape[0], data_full.shape[1], 2)
            )
            yx_center_injection_full[:, :, 0] = xy_image_centers[1]
            yx_center_injection_full[:, :, 1] = xy_image_centers[0]
            yx_center_full = np.ones((data_full.shape[0], 2))
            yx_center_full[:, 0] = xy_image_centers[1]
            yx_center_full[:, 1] = xy_image_centers[0]
            max_shift = 0
        elif xy_image_centers.ndim > 1:
            if xy_image_centers.ndim > 3:
                raise ValueError("Dimensionality of provided centers too large.")
            if xy_image_centers.ndim == 2:
                xy_image_centers = np.expand_dims(xy_image_centers, axis=0)
            yx_center_injection_full = xy_image_centers[..., ::-1]
            yx_center_full = np.median(yx_center_injection_full, axis=1)
            max_shift_x = np.max(xy_image_centers[..., 0]) - np.min(
                xy_image_centers[..., 0]
            )
            max_shift_y = np.max(xy_image_centers[..., 1]) - np.min(
                xy_image_centers[..., 1]
            )
            max_shift = np.max([max_shift_x, max_shift_y]) * 2
            logger.debug("The center varies by a maximum of in x or y: %s", max_shift / 2)
        # print("Center variation: {}".format(np.std(amplitude_modulation_full, axis=0)))

    # Build runtime state (replaces all mutations of reduction_parameters)
    runtime = build_runtime_state(
        config=reduction_parameters,
        data_shape=data_full.shape,
        stamp_sizes=stamp_sizes,
        stamp_sizes_reduction=stamp_sizes_reduction,
        max_shift=max_shift,
        mas_per_pixel=(1 * u.pixel).to(u.mas, equivalencies=instrument.pixel_scale).value,
    )
    data_crop_size = runtime.data_crop_size

    # Configure PSF amplitude variation
    if amplitude_modulation_full is not None:
        if amplitude_modulation_full.ndim < 2:
            amplitude_modulation_full = np.expand_dims(
                amplitude_modulation_full, axis=0
            )
        amplitude_modulation_full = np.delete(
            amplitude_modulation_full, bad_frames, axis=1
        )
        logger.debug("Amplitude variation: %s", np.std(amplitude_modulation_full, axis=1))

    # Configure number of principal components

    number_of_components = np.round(
        data_full.shape[1] * np.array(temporal_components_fraction)
    ).astype("int")

    result_folder = reduction_parameters.result_folder
    prefix = reduction_parameters.prefix
    if result_folder is None:
        result_folder = "./"
    else:
        if not reduction_parameters.reduce_single_position:
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

    # Save parameters
    if not reduction_parameters.reduce_single_position:
        save_object(instrument, os.path.join(result_folder, "instrument.obj"))
        # Save both modern config and legacy format for detection.py backward compat
        save_object(reduction_parameters, os.path.join(result_folder, "reduction_config.obj"))
        save_object(
            reduction_parameters.to_reduction_parameters(),
            os.path.join(result_folder, "reduction_parameters.obj"),
        )

    assert (
        flux_psf_full.shape[0] == data_full.shape[0] == len(instrument.wavelengths)
    ), "Different number of wavelengths in data: Flux {} Data {} Wave {}".format(
        flux_psf_full.shape[0], data_full.shape[0], len(instrument.wavelengths)
    )

    # Loop over reductions for different numbers of components
    # Check if number of components is iterable, if not make it iterable
    try:
        _ = iter(number_of_components)
    except TypeError:
        number_of_components = [number_of_components]
        temporal_components_fraction = [temporal_components_fraction]

    if wavelength_indices is None:
        wavelength_indices = np.arange(data_full.shape[0])

    # Dump the preprocessed per-wavelength data once, before the
    # component/wavelength loops: the preprocessing is identical for every
    # component fraction, and workers memmap the store read-only instead of
    # receiving array copies.
    shared_store = None
    data_refs = {}
    inverse_variance_refs = {}
    if (
        reduction_parameters.use_multiprocess
        and not reduction_parameters.reduce_single_position
    ):
        if data_crop_size is None:
            yx_shape = (data_full.shape[-2], data_full.shape[-1])
        else:
            yx_shape = (data_crop_size, data_crop_size)
        n_arrays = 2 if inverse_variance_full is not None else 1
        required_bytes = int(
            n_arrays
            * len(wavelength_indices)
            * data_full.shape[1]
            * np.prod(yx_shape)
            * np.dtype("float64").itemsize
        )
        shared_store = SharedArrayStore(
            scratch_dir=reduction_parameters.scratch_dir,
            required_bytes=required_bytes,
        )
        print("Shared array store: {}".format(shared_store.directory))
        for wavelength_index in wavelength_indices:
            flux_psf = psf_stamps[wavelength_index].astype("float64")
            yx_center, yx_center_injection = _wavelength_geometry(
                wavelength_index,
                data_shape=data_full.shape,
                data_crop_size=data_crop_size,
                yx_center_full=yx_center_full,
                yx_center_injection_full=yx_center_injection_full,
            )
            # Wavelengths with non-finite PSF or centers are skipped by the
            # reduction loop below; skip dumping them as well.
            if np.any(~np.isfinite(flux_psf)) or np.any(
                ~np.isfinite(yx_center_injection)
            ):
                continue
            if amplitude_modulation_full is None:
                amplitude_modulation = np.ones(data_full.shape[1])
            else:
                amplitude_modulation = amplitude_modulation_full[wavelength_index, :]
            data, inverse_variance = _prepare_wavelength_slice(
                wavelength_index,
                data_full=data_full,
                inverse_variance_full=inverse_variance_full,
                flux_psf=flux_psf,
                pa=pa,
                reduction_parameters=reduction_parameters,
                runtime=runtime,
                data_crop_size=data_crop_size,
                yx_center_full=yx_center_full,
                yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation,
            )
            # Workers consume float64 (the cast `run_trap_search` would
            # otherwise apply); dump that representation directly.
            data_refs[wavelength_index] = shared_store.dump(
                "data_lam{:03d}".format(wavelength_index),
                data.astype("float64", copy=False),
            )
            if inverse_variance is not None:
                inverse_variance_refs[wavelength_index] = shared_store.dump(
                    "inverse_variance_lam{:03d}".format(wavelength_index),
                    inverse_variance.astype("float64", copy=False),
                )
            else:
                inverse_variance_refs[wavelength_index] = None

    try:
        return _run_reduction_loops(
            number_of_components=number_of_components,
            temporal_components_fraction=temporal_components_fraction,
            wavelength_indices=wavelength_indices,
            data_full=data_full,
            inverse_variance_full=inverse_variance_full,
            psf_stamps=psf_stamps,
            pa=pa,
            instrument=instrument,
            reduction_parameters=reduction_parameters,
            runtime=runtime,
            stamp_sizes=stamp_sizes,
            stamp_sizes_reduction=stamp_sizes_reduction,
            yx_center_full=yx_center_full,
            yx_center_injection_full=yx_center_injection_full,
            amplitude_modulation_full=amplitude_modulation_full,
            bad_pixel_mask_full=bad_pixel_mask_full,
            data_crop_size=data_crop_size,
            result_folder=result_folder,
            prefix=prefix,
            cross_validation=cross_validation,
            overwrite=overwrite,
            use_progress_bar=use_progress_bar,
            shared_store=shared_store,
            data_refs=data_refs,
            inverse_variance_refs=inverse_variance_refs,
        )
    finally:
        if shared_store is not None:
            shared_store.cleanup()


def _run_reduction_loops(
    number_of_components,
    temporal_components_fraction,
    wavelength_indices,
    data_full,
    inverse_variance_full,
    psf_stamps,
    pa,
    instrument,
    reduction_parameters,
    runtime,
    stamp_sizes,
    stamp_sizes_reduction,
    yx_center_full,
    yx_center_injection_full,
    amplitude_modulation_full,
    bad_pixel_mask_full,
    data_crop_size,
    result_folder,
    prefix,
    cross_validation,
    overwrite,
    use_progress_bar,
    shared_store,
    data_refs,
    inverse_variance_refs,
):
    """Component/wavelength loops of `run_complete_reduction`.

    Separated out so the shared-array store is cleaned up in a single
    ``try/finally`` regardless of which internal path returns.
    """
    if reduction_parameters.reduce_single_position is True:
        all_results = OrderedDict()

    for comp_index, ncomp in enumerate(number_of_components):
        logger.debug("Number of principal comp. used: %s of %d", ncomp, data_full.shape[1])

        if reduction_parameters.reduce_single_position:
            wavelength_results = OrderedDict()

        # Loop over reduction for different wavelengths
        for (
            _,
            wavelength_index,
        ) in enumerate(wavelength_indices):
            wavelength = instrument.wavelengths[wavelength_index]
            logger.debug("Lambda index: %s Wavelength: %.3f", wavelength_index, wavelength)
            if prefix is None:
                prefix = ""
            # Update per-iteration runtime state
            runtime = runtime.for_iteration(
                number_of_pca_regressors=ncomp,
                temporal_components_fraction=temporal_components_fraction[comp_index],
                fwhm=instrument.fwhm[wavelength_index].value,
                reduction_mask_psf_size=int(stamp_sizes_reduction[wavelength_index]),
                signal_mask_psf_size=int(stamp_sizes[wavelength_index]),
            )
            basename = {}
            if reduction_parameters.inject_fake:
                basename[
                    "temporal"
                ] = "injectedsigma{:.2f}_{}lam{:02d}_ncomp{:03d}_frac{:.2f}".format(
                    reduction_parameters.injection_sigma,
                    prefix,
                    wavelength_index,
                    ncomp,
                    temporal_components_fraction[comp_index],
                )
                basename[
                    "temporal_plus_spatial"
                ] = "injectedsigma{:.2f}_{}lam{:02d}_ncomp{:03d}_frac{:.2f}_delta{:.2f}_spatialfrac{:.2f}".format(
                    reduction_parameters.injection_sigma,
                    prefix,
                    wavelength_index,
                    ncomp,
                    temporal_components_fraction[comp_index],
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction_after_trap,
                )
                basename[
                    "spatial"
                ] = "injectedsigma{:.2f}_{}lam{:02d}_delta{:.2f}_spatialfrac{:.2f}".format(
                    reduction_parameters.injection_sigma,
                    prefix,
                    wavelength_index,
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction,
                )
            else:
                basename["temporal"] = "{}lam{:02d}_ncomp{:03d}_frac{:.2f}".format(
                    prefix,
                    wavelength_index,
                    ncomp,
                    temporal_components_fraction[comp_index],
                )
                basename[
                    "temporal_plus_spatial"
                ] = "{}lam{:02d}_ncomp{:03d}_frac{:.2f}_delta{:.2f}_spatialfrac{:.2f}".format(
                    prefix,
                    wavelength_index,
                    ncomp,
                    temporal_components_fraction[comp_index],
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction_after_trap,
                )
                basename[
                    "spatial"
                ] = "{}lam{:02d}_delta{:.2f}_spatialfrac{:.2f}".format(
                    prefix,
                    wavelength_index,
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction,
                )
            logger.debug("%s", basename["temporal"])

            # Having all the different reductions in the output makes it easier to compare but also more complex to refactor
            # Since individual outputs cannot be checked for existence
            sigma = reduction_parameters.contrast_curve_sigma
            use_corr = (
                reduction_parameters.compute_residual_correlation
                and reduction_parameters.use_residual_correlation
            )
            output_paths = {}
            output_paths_corr = {}
            for key in ["temporal", "temporal_plus_spatial", "spatial"]:
                output_paths[key] = OutputPaths.make(
                    result_folder, basename[key], key, sigma
                )
                if use_corr:
                    output_paths_corr[key] = OutputPaths.make(
                        result_folder, basename[key], key, sigma, corr=True
                    )
                # If output file with basename already exists, skip reduction
                if not overwrite and not reduction_parameters.reduce_single_position:
                    if os.path.exists(output_paths[key].detection_image):
                        logger.info(
                            "Reduction already exists for %s - skipping.",
                            output_paths[key].detection_image,
                        )
                        return None

            # if reduction_parameters.temporal_plus_spatial_model:
            #     contrast_plot_comparison_path = os.path.join(
            #         result_folder,
            #         "contrast_comparison_plot_"
            #         + basename["temporal_plus_spatial"]
            #         + "_sigma{:.2f}.jpg".format(
            #             reduction_parameters.contrast_curve_sigma
            #         ),
            #     )

            # Output-frame center and per-frame injection centers for this
            # wavelength (shared with the store pre-dump pass).
            yx_center, yx_center_injection = _wavelength_geometry(
                wavelength_index,
                data_shape=data_full.shape,
                data_crop_size=data_crop_size,
                yx_center_full=yx_center_full,
                yx_center_injection_full=yx_center_injection_full,
            )

            # Make companion mask before cropping to be consistent
            # Do this for each wavelength separately to account for PSF size
            # and differing center position
            if (
                runtime.yx_known_companion_position is not None
                and len(runtime.yx_known_companion_position) > 0
            ):
                if yx_center_injection_full is not None:
                    yx_center_before_crop = yx_center_injection_full[
                        wavelength_index, :
                    ]
                else:
                    yx_center_before_crop = None

                known_companion_masks = []
                for yx_pos in runtime.yx_known_companion_position:
                    # TODO: CHECK
                    known_companion_mask = regressor_selection.make_mask_from_psf_track(
                        yx_position=yx_pos,
                        psf_size=runtime.signal_mask_psf_size,
                        pa=pa,
                        image_size=data_full.shape[-1],
                        image_center=yx_center_before_crop,
                        yx_anamorphism=runtime.yx_anamorphism,
                        right_handed=reduction_parameters.right_handed,
                        return_cube=False,
                    )
                    known_companion_masks.append(known_companion_mask)
                known_companion_mask = np.logical_or.reduce(known_companion_masks)

                if data_crop_size is not None:
                    known_companion_mask = crop_box_from_image(
                        known_companion_mask,
                        data_crop_size,
                        center_yx=np.round(yx_center_full[wavelength_index]),
                    ).copy()
            else:
                known_companion_mask = None

            flux_psf = psf_stamps[wavelength_index].astype("float64")

            if bad_pixel_mask_full is None:
                bad_pixel_mask = None
            else:
                bad_pixel_mask_full = bad_pixel_mask_full.astype("bool")
                try:
                    if bad_pixel_mask_full.ndim == 3:
                        bad_pixel_mask = bad_pixel_mask_full[wavelength_index]
                        if data_crop_size is not None:
                            bad_pixel_mask = crop_box_from_image(
                                bad_pixel_mask,
                                data_crop_size,
                                center_yx=np.round(yx_center_full[wavelength_index]),
                            )
                    elif bad_pixel_mask_full.ndim == 2:
                        bad_pixel_mask = bad_pixel_mask_full
                    else:
                        raise ValueError(
                            "Bad pixel mask, must either be one image or a number of images corresponding to wavelength"
                        )
                except AttributeError:
                    pass

            if (
                reduction_parameters.inject_fake
                and reduction_parameters.read_injection_files
            ):
                input_contrast_image_path = output_paths["temporal"].contrast_image.replace(
                    "injectedsigma{:.2f}_".format(reduction_parameters.injection_sigma),
                    "",
                )
                input_sigma = float(os.path.splitext(input_contrast_image_path)[0][-4:])
                contrast_map = fits.getdata(input_contrast_image_path)
                contrast_map /= input_sigma  # contrast map is 5 sigma
                contrast_map *= reduction_parameters.injection_sigma

            if (
                not reduction_parameters.inject_fake
                or not reduction_parameters.read_injection_files
            ):
                contrast_map = None

            if contrast_map is not None:
                contrast_map = crop_box_from_image(
                    contrast_map, data_crop_size, center_yx=None
                )

            if amplitude_modulation_full is None:
                amplitude_modulation = np.ones(data_full.shape[1])
            else:
                amplitude_modulation = amplitude_modulation_full[wavelength_index, :]

            # Do not reduce data for wavelength if flux PSF or center position contains NaNs
            flux_psf_not_finite = np.any(~np.isfinite(flux_psf))
            yx_center_injection_not_finite = np.any(~np.isfinite(yx_center_injection))
            if flux_psf_not_finite:
                logger.warning("Skipping wavelength %s. NaNs detected in flux PSF.", wavelength_index)
                continue
            if yx_center_injection_not_finite:
                logger.warning(
                    "Skipping wavelength %s. NaNs detected in provided center position.",
                    wavelength_index,
                )
                continue

            if shared_store is not None and wavelength_index in data_refs:
                # Preprocessed data was dumped to the shared store before the
                # loops; pass references so workers memmap it read-only.
                data = data_refs[wavelength_index]
                inverse_variance = inverse_variance_refs[wavelength_index]
            else:
                data, inverse_variance = _prepare_wavelength_slice(
                    wavelength_index,
                    data_full=data_full,
                    inverse_variance_full=inverse_variance_full,
                    flux_psf=flux_psf,
                    pa=pa,
                    reduction_parameters=reduction_parameters,
                    runtime=runtime,
                    data_crop_size=data_crop_size,
                    yx_center_full=yx_center_full,
                    yx_center_injection=yx_center_injection,
                    amplitude_modulation=amplitude_modulation,
                )

            logger.debug("PSF Size: %s", runtime.reduction_mask_psf_size)
            if reduction_parameters.reduce_single_position:
                results = trap_one_position(
                    reduction_parameters.guess_position,
                    data=data,
                    inverse_variance=inverse_variance,
                    flux_psf=flux_psf,
                    pa=pa,
                    reduction_parameters=reduction_parameters,
                    known_companion_mask=known_companion_mask,
                    runtime=runtime,
                    bad_pixel_mask=bad_pixel_mask,
                    yx_center=yx_center,
                    yx_center_injection=yx_center_injection,
                    amplitude_modulation=amplitude_modulation,
                    contrast_map=contrast_map,
                    readnoise=instrument.readnoise,
                )

                wavelength_results["{}".format(wavelength_index)] = results

            else:
                if cross_validation:
                    relative_coords = np.vstack(
                        [
                            # image_coordinates.rhophi_to_relative_yx(
                            #     [[5, 0], [5, 90], [5, 180], [5, 270]]),
                            image_coordinates.rhophi_to_relative_yx(
                                [
                                    [10.0, 0.0],
                                    [10.0, 90.0],
                                    [10.0, 180.0],
                                    [10.0, 270.0],
                                ]
                            ),
                            image_coordinates.rhophi_to_relative_yx(
                                [
                                    [20.0, 0.0],
                                    [20.0, 90.0],
                                    [20.0, 180.0],
                                    [20.0, 270.0],
                                ]
                            ),
                            # image_coordinates.rhophi_to_relative_yx(
                            #     [[30., 0.], [30., 90.], [30., 180.], [30., 270.]]),
                            image_coordinates.rhophi_to_relative_yx(
                                [[40, 0], [40, 90], [40, 180], [40, 270]]
                            ),
                            image_coordinates.rhophi_to_relative_yx(
                                [[60, 0], [60, 90], [60, 180], [60, 270]]
                            ),
                        ]
                    )

                    cv_results = multi_position_cross_validation(
                        relative_coords=relative_coords,
                        data=data,
                        inverse_variance=inverse_variance,
                        flux_psf=flux_psf,
                        pa=pa,
                        reduction_parameters=reduction_parameters,
                        known_companion_mask=known_companion_mask,
                        runtime=runtime,
                        bad_pixel_mask=bad_pixel_mask,
                        yx_center=yx_center,
                        yx_center_injection=yx_center_injection,
                        amplitude_modulation=amplitude_modulation,
                        contrast_map=contrast_map,
                        readnoise=instrument.readnoise,
                    )
                    ncomp_residuals = []
                    for i in range(len(relative_coords)):
                        ncomp_residuals.append(cv_results[i][0])
                    # rel_pos, ncomp, pixels, residuals
                    scores = mad_std(np.array(ncomp_residuals), axis=3)
                    best_scores = np.argmin(scores, axis=1)
                    median_best_scores = np.median(best_scores, axis=1)

                    return ncomp_residuals, scores, best_scores, median_best_scores
                    
                (
                    detection_image,
                    detection_image_corr,
                    correlation_matrix_binned,
                ) = run_trap_search(
                    data=data,
                    inverse_variance=inverse_variance,
                    flux_psf=flux_psf,
                    pa=pa,
                    reduction_parameters=reduction_parameters,
                    known_companion_mask=known_companion_mask,
                    runtime=runtime,
                    bad_pixel_mask=bad_pixel_mask,
                    yx_center=yx_center,
                    yx_center_injection=yx_center_injection,
                    amplitude_modulation=amplitude_modulation,
                    contrast_map=contrast_map,
                    readnoise=instrument.readnoise,
                    use_progress_bar=use_progress_bar,
                )

                for key in detection_image:
                    fits.writeto(
                        output_paths[key].detection_image,
                        detection_image[key],
                        overwrite=True,
                    )
                    if use_corr:
                        fits.writeto(
                            output_paths_corr[key].detection_image,
                            detection_image_corr[key],
                            overwrite=True,
                        )
                        fits.writeto(
                            output_paths_corr[key].correlation_matrix_binned,
                            correlation_matrix_binned[key],
                            overwrite=True,
                        )

                del detection_image

        if reduction_parameters.reduce_single_position:
            all_results[
                "{}".format(temporal_components_fraction[comp_index])
            ] = wavelength_results

    if reduction_parameters.reduce_single_position:
        return all_results
    else:
        return None