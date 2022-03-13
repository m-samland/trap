"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

import datetime
import multiprocessing
import os
import pickle
from collections import OrderedDict
from copy import copy
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import ray
from astropy import units as u
from astropy.io import fits
from tqdm import tqdm
from trap import (detection, image_coordinates, makesource, regression,
                  regressor_selection)
from trap.embed_shell import ipsh
from trap.utils import (ProgressBar, crop_box_from_3D_cube,
                        crop_box_from_image, determine_psf_stampsizes,
                        high_pass_filter, high_pass_filter_cube, prepare_psf,
                        round_up_to_odd, shuffle_and_equalize_relative_positions)


def trap_one_position(guess_position, data, flux_psf, pa,
                      reduction_parameters, known_companion_mask,
                      variance=None,
                      bad_pixel_mask=None,
                      yx_center=None, yx_center_injection=None,
                      amplitude_modulation=None,
                      contrast_map=None,
                      readnoise=0.):
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
    variance : array_like
        Cube containing variances for `data`.
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
        reduction_parameters.guess_position = guess_position
        position_absolute = image_coordinates.relative_yx_to_absolute_yx(
            guess_position, yx_center).astype('int')
    signal_position = np.array(reduction_parameters.guess_position)

    if contrast_map is not None:
        reduction_parameters.true_contrast = contrast_map[position_absolute[0],
                                                          position_absolute[1]]
    true_contrast = reduction_parameters.true_contrast
    # if reduction_parameters.true_position is not None:
    #     true_rhophi = image_coordinates.relative_yx_to_rhophi(
    #         reduction_parameters.true_position)

    planet_absolute_yx_pos = image_coordinates.relative_yx_to_absolute_yx(
        signal_position, yx_center)

    injected_model_cube = np.zeros_like(data)
    injected_model_cube = makesource.inject_model_into_data(
        flux_arr=injected_model_cube,
        pos=signal_position,
        pa=pa,
        psf_model=flux_psf,
        image_center=yx_center_injection,
        norm=amplitude_modulation,
        yx_anamorphism=reduction_parameters.yx_anamorphism,
        right_handed=reduction_parameters.right_handed,
        subpixel=True,
        remove_interpolation_artifacts=True,
        copy=False)
    # injected_model_cube = makesource.addsource(
    #     injected_model_cube, reduction_parameters.guess_position, pa,
    #     flux_psf,
    #     image_center=yx_center_injection,
    #     norm=amplitude_modulation, jitter=0, poisson_noise=False,
    #     yx_anamorphism=reduction_parameters.yx_anamorphism,
    #     right_handed=reduction_parameters.right_handed,
    #     verbose=False)

    signal_mask_local = injected_model_cube > 0.
    signal_mask = np.any(signal_mask_local, axis=0)
    if reduction_parameters.reduction_mask_psf_size == reduction_parameters.signal_mask_psf_size:
        # reduction_mask_local = signal_mask_local
        reduction_mask = signal_mask
    else:
        reduction_mask = regressor_selection.make_mask_from_psf_track(
            yx_position=signal_position,
            psf_size=reduction_parameters.reduction_mask_psf_size,
            pa=pa, image_size=data.shape[-1],
            image_center=yx_center_injection,
            yx_anamorphism=reduction_parameters.yx_anamorphism,
            right_handed=reduction_parameters.right_handed,
            return_cube=False)

    # Due to interpolation edge effects some pixels at the edge of the trajectory can be negative
    total_flux_in_pixel = np.sum(injected_model_cube, axis=0)
    relative_flux_in_pixel = total_flux_in_pixel / np.max(total_flux_in_pixel)
    # excluding pixels with low contribution to over-all signal
    low_contribution_mask = relative_flux_in_pixel <= reduction_parameters.threshold_pixel_by_contribution
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

    assert flux_psf.shape[-1] % 2 == 1, "PSF dimension has to be odd (centered on pixel)"

    if reduction_parameters.fit_planet:
        model = injected_model_cube
    else:
        model = None

    if reduction_parameters.include_noise:
        if variance is None:
            # NOTE: Uses photon noise from data itself.
            # May not be valid based on pre-processing steps done
            if bad_pixel_mask is not None:
                variance_reduction_area = np.abs(
                    data_reduce[:, reduction_mask_wo_badpixels]) + readnoise**2
            else:
                variance_reduction_area = np.abs(data_reduce[:, reduction_mask]) + readnoise**2

        else:
            if bad_pixel_mask is not None:
                variance_reduction_area = variance[:, reduction_mask_wo_badpixels]
            else:
                variance_reduction_area = variance[:, reduction_mask]
    else:
        variance_reduction_area = None

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
            opposite_mask = regressor_selection.make_mirrored_mask(reduction_mask, yx_center)
        else:
            opposite_mask = None

        regressor_pool_mask = regressor_selection.make_regressor_pool_for_pixel(
            reduction_parameters=reduction_parameters,
            yx_pixel=planet_absolute_yx_pos,
            yx_dim=yx_dim,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            signal_mask=signal_mask,
            known_companion_mask=known_companion_mask,
            bad_pixel_mask=bad_pixel_mask,
            additional_regressors=opposite_mask,
            right_handed=reduction_parameters.right_handed,
            pa=pa)

        if reduction_parameters.reduce_single_position:
            result = regression.run_trap_with_model_temporal(
                data=data_reduce,
                model=model,
                # model=None,
                pa=pa,
                reduction_parameters=reduction_parameters,
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
                variance_reduction_area=variance_reduction_area,
                plot_all_diagnostics=reduction_parameters.plot_all_diagnostics,
                return_input_data=reduction_parameters.return_input_data)
        else:
            result = regression.run_trap_with_model_temporal_optimized(
                data=data_reduce,
                model=model,
                pa=pa,
                reduction_parameters=reduction_parameters,
                reduction_mask=reduction_mask_used,
                regressor_pool_mask=regressor_pool_mask,
                regressor_matrix=None,
                variance_reduction_area=variance_reduction_area)

        # if result is not None:
        #     if reduction_parameters.fit_planet:
        #         result.compute_contrast_weighted_average(mask_outliers=True)
        #         if reduction_parameters.verbose:
        #             print(result)
        #     result.reduction_parameters = reduction_parameters
        # if reduction_parameters.reduce_single_position:
        results['temporal'] = result
        # else:
        #     results['']

    if reduction_parameters.spatial_model:
        result = regression.run_trap_with_model_spatial(
            data=data_reduce,
            model=model,
            pa=pa,
            reduction_parameters=reduction_parameters,
            planet_relative_yx_pos=signal_position,
            reduction_mask=reduction_mask_used,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            variance_reduction_area=variance_reduction_area,
            true_contrast=true_contrast,
            training_data=None,
            return_input_data=False,
            verbose=reduction_parameters.verbose)

        if result is not None:
            if reduction_parameters.fit_planet:
                result.compute_contrast_weighted_average(mask_outliers=True)
                if reduction_parameters.verbose:
                    print(result)
            result.reduction_parameters = reduction_parameters
        results['spatial'] = result

    if reduction_parameters.temporal_model and reduction_parameters.temporal_plus_spatial_model:
        if reduction_parameters.second_stage_trap:
            data_reduce_psf_subtracted = data_reduce - \
                injected_model_cube * results['temporal'].measured_contrast

            results['temporal_psf_subtracted'] = regression.run_trap_with_model_temporal(
                data=data_reduce_psf_subtracted,
                model=None,
                pa=pa,
                reduction_parameters=reduction_parameters,
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
                variance_reduction_area=variance_reduction_area,
                true_contrast=true_contrast,
                plot_all_diagnostics=reduction_parameters.plot_all_diagnostics,
                return_input_data=False,
                verbose=reduction_parameters.verbose)

            data_reduce_noise_subtracted = data_reduce - \
                np.nan_to_num(results['temporal_psf_subtracted'].noise_model_cube)
        else:
            data_reduce_noise_subtracted = data_reduce - \
                np.nan_to_num(results['temporal'].noise_model_cube)

        if reduction_parameters.remove_model_from_spatial_training:
            training_data = data_reduce_noise_subtracted - \
                injected_model_cube * results['temporal'].measured_contrast
        else:
            training_data = None

        if reduction_parameters.remove_bad_residuals_for_spatial_model:
            bad_residual_mask = np.zeros(
                (data_reduce.shape[-2], data_reduce.shape[-1])).astype('bool')
            bad_residual_mask[reduction_mask_used] = ~results['temporal'].good_residual_mask

            if bad_pixel_mask is None:
                bad_pixel_mask = bad_residual_mask
            else:
                bad_pixel_mask = np.logical_or(bad_pixel_mask, bad_residual_mask)
            if variance_reduction_area is not None:
                variance_reduction_area = variance_reduction_area[:,
                                                                  ~bad_residual_mask[reduction_mask_used]]
            reduction_mask_used = np.logical_and(reduction_mask_used, ~bad_residual_mask)

        reduction_parameters_alternative = copy(reduction_parameters)
        reduction_parameters_alternative.spatial_components_fraction = \
            reduction_parameters.spatial_components_fraction_after_trap

        result = regression.run_trap_with_model_spatial(
            data=data_reduce_noise_subtracted,
            model=model,
            pa=pa,
            reduction_parameters=reduction_parameters_alternative,
            planet_relative_yx_pos=signal_position,
            reduction_mask=reduction_mask_used,
            yx_center=yx_center,
            yx_center_injection=yx_center_injection,
            variance_reduction_area=variance_reduction_area,
            true_contrast=true_contrast,
            training_data=training_data,
            return_input_data=False,
            verbose=reduction_parameters.verbose)

        if result is not None:
            if reduction_parameters.fit_planet:
                result.compute_contrast_weighted_average(mask_outliers=True)
                if reduction_parameters.verbose:
                    print(result)
            result.reduction_parameters = reduction_parameters
        results['temporal_plus_spatial'] = result

    return results


def run_trap_search(data, flux_psf, pa, wavelength,
                    reduction_parameters, known_companion_mask,
                    variance=None,
                    bad_pixel_mask=None, result_name=None,
                    yx_center=None, yx_center_injection=None,
                    amplitude_modulation=None,
                    contrast_map=None,
                    readnoise=0.):
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
    variance : array_like
        Cube containing variances for `data`.
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

    data = data.astype('float64')
    flux_psf = flux_psf.astype('float64')
    pa = pa.astype('float64')

    if variance is not None:
        variance = variance.astype('float64')

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
        detection_image['temporal'] = np.zeros(
            (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image['temporal_plus_spatial'] = np.zeros(
                (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        detection_image['spatial'] = np.zeros(
            (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    # EDIT: ADDED FOR QUICK CORRELATION TESTS
    detection_image_corr = {}
    if reduction_parameters.temporal_model:
        detection_image_corr['temporal'] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image_corr['temporal_plus_spatial'] = np.zeros(
                (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        detection_image_corr['spatial'] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    correlation_matrix_binned = {}
    if reduction_parameters.temporal_model:
        correlation_matrix_binned['temporal'] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            correlation_matrix_binned['temporal_plus_spatial'] = np.zeros(
                (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        correlation_matrix_binned['spatial'] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    search_region = reduction_parameters.search_region
    search_coordinates = np.argwhere(search_region) * oversampling
    # relative coordinates to output image center (i.e. position of star)
    relative_coords = np.array(list(map(
        lambda x: image_coordinates.absolute_yx_to_relative_yx(x, yx_center_output),
        search_coordinates.tolist())))
    print('Number of positions: {}'.format(len(relative_coords)))
    if reduction_parameters.use_multiprocess:
        if reduction_parameters.ncpus is None:
            reduction_parameters.ncpus = multiprocessing.cpu_count()
        num_ticks = len(relative_coords)
        pb = ProgressBar(num_ticks)
        actor = pb.actor

        # Use more chunks than CPUs to prevent long idle time in case one job finishes quicker
        number_of_chunks = round(reduction_parameters.ncpus * 2)

        search_coordinates, relative_coords, relative_coords_regions, iteration, separation_equalized = shuffle_and_equalize_relative_positions(
            search_coordinates, relative_coords, number_of_chunks,
            max_separation_deviation=2, max_iterations=50, rng=None)
        print('Number of positions per chunk: {}'.format(len(relative_coords_regions[0])))

        a = datetime.datetime.now()
        data_id = ray.put(data)
        variance_id = ray.put(variance)
        flux_psf_id = ray.put(flux_psf)
        pa_id = ray.put(pa)
        known_companion_mask_id = ray.put(known_companion_mask)
        amplitude_modulation_id = ray.put(amplitude_modulation)
        bad_pixel_mask_id = ray.put(bad_pixel_mask)
        contrast_map_id = ray.put(contrast_map)
        result_ids = []
        for region in relative_coords_regions:
            result_ids.append(trap_search_region.remote(
                region, data=data_id, variance=variance_id, flux_psf=flux_psf_id, pa=pa_id,
                reduction_parameters=reduction_parameters,
                known_companion_mask=known_companion_mask_id,
                bad_pixel_mask=bad_pixel_mask_id,
                yx_center=yx_center, yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation_id,
                contrast_map=contrast_map_id,
                readnoise=readnoise, pba=actor))
        pb.print_until_done()
        results = ray.get(result_ids)
        results == list(range(num_ticks))
        num_ticks == ray.get(actor.get_counter.remote())
        results = [item for sublist in results for item in sublist]
        # multiprocess_trap_region = partial(
        #     trap_search_region, data=data, variance=variance, flux_psf=flux_psf, pa=pa,
        #     reduction_parameters=reduction_parameters,
        #     known_companion_mask=known_companion_mask,
        #     bad_pixel_mask=bad_pixel_mask,
        #     yx_center=yx_center, yx_center_injection=yx_center_injection,
        #     amplitude_modulation=amplitude_modulation,
        #     contrast_map=contrast_map,
        #     readnoise=readnoise)
        # result_ids = []
        # pool = Pool(processes=reduction_parameters.ncpus)
        # for idx, sub_region_results in enumerate(tqdm(pool.imap(multiprocess_trap_region, relative_coords_regions),
        #                                               total=len(relative_coords_regions))):
        #     result_ids.append(sub_region_results)
        # results = [item for sublist in result_ids for item in sublist]

        for idx, result in enumerate(results):
            for key in detection_image:
                if result[key] is not None:
                    detection_image[key][0][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].measured_contrast
                    detection_image[key][1][search_coordinates[idx][0],
                                            search_coordinates[idx][1]] = result[key].contrast_uncertainty
                    detection_image[key][2][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].snr
                    if reduction_parameters.inject_fake:
                        detection_image[key][3][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].true_contrast
                        detection_image[key][4][search_coordinates[idx][0],
                                                search_coordinates[idx][1]] = result[key].measured_contrast - result[key].true_contrast
                        detection_image[key][5][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].relative_deviation_from_true
                        detection_image[key][6][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].wrong_in_sigma

                    # NOTE: Should be include in results class or dictionary to avoid code duplication
                    if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                        detection_image_corr[key][0][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].measured_contrast_with_corr
                        detection_image_corr[key][1][search_coordinates[idx][0],
                                                     search_coordinates[idx][1]] = result[key].contrast_uncertainty_with_corr
                        detection_image_corr[key][2][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].snr_with_corr
                        detection_image_corr[key][3][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_exponential']
                        detection_image_corr[key][4][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern32']
                        detection_image_corr[key][5][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern52']

                        correlation_matrix_binned[key][:, search_coordinates[idx]
                                                       [0], search_coordinates[idx][1]] = result[key].correlation_info['summary_dataframe'].empirical_correlation.values

        #     del result
        # del pool
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
                coords, data=data, variance=variance, flux_psf=flux_psf, pa=pa,
                reduction_parameters=reduction_parameters,
                known_companion_mask=known_companion_mask,
                bad_pixel_mask=bad_pixel_mask,
                yx_center=yx_center, yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation,
                contrast_map=contrast_map,
                readnoise=readnoise)
            for key in detection_image:
                if result[key] is not None:
                    detection_image[key][0][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].measured_contrast
                    detection_image[key][1][search_coordinates[idx][0],
                                            search_coordinates[idx][1]] = result[key].contrast_uncertainty
                    detection_image[key][2][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].snr
                    if reduction_parameters.inject_fake:
                        detection_image[key][3][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].true_contrast
                        detection_image[key][4][search_coordinates[idx][0],
                                                search_coordinates[idx][1]] = result[key].measured_contrast - result[key].true_contrast
                        detection_image[key][5][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].relative_deviation_from_true
                        detection_image[key][6][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].wrong_in_sigma
                    # NOTE: Should be include in results class or dictionary to avoid code duplication
                    if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                        detection_image_corr[key][0][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].measured_contrast_with_corr
                        detection_image_corr[key][1][search_coordinates[idx][0],
                                                     search_coordinates[idx][1]] = result[key].contrast_uncertainty_with_corr
                        detection_image_corr[key][2][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].snr_with_corr
                        detection_image_corr[key][3][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_exponential']
                        detection_image_corr[key][4][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern32']
                        detection_image_corr[key][5][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern52']
                        # ipsh()
                        correlation_matrix_binned[key][:, search_coordinates[idx]
                                                       [0], search_coordinates[idx][1]] = result[key].correlation_info['summary_dataframe'].empirical_correlation.values

            del result
        b = datetime.datetime.now()
    c = b - a
    print("Main reduction computation time:")
    print(c)

    if not reduction_parameters.compute_residual_correlation \
            and not reduction_parameters.use_residual_correlation:
        detection_image_corr = None
        correlation_matrix_binned = None

    return detection_image, detection_image_corr, correlation_matrix_binned


@ ray.remote
def trap_search_region(relative_coords, data, flux_psf, pa,
                       reduction_parameters, known_companion_mask,
                       variance=None,
                       bad_pixel_mask=None, result_name=None,
                       yx_center=None, yx_center_injection=None,
                       amplitude_modulation=None,
                       contrast_map=None,
                       readnoise=0.,
                       pba=None):
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
    variance : array_like
        Cube containing variances for `data`.
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

    sub_region_results = []
    for idx, coords in enumerate(relative_coords):
        result = trap_one_position(
            coords, data=data, variance=variance, flux_psf=flux_psf, pa=pa,
            reduction_parameters=reduction_parameters,
            known_companion_mask=known_companion_mask,
            bad_pixel_mask=bad_pixel_mask,
            yx_center=yx_center, yx_center_injection=yx_center_injection,
            amplitude_modulation=amplitude_modulation,
            contrast_map=contrast_map,
            readnoise=readnoise)
        sub_region_results.append(result)
        if pba is not None:
            pba.update.remote(1)
    return sub_region_results


def run_trap_search_old(data, flux_psf, pa, wavelength,
                        reduction_parameters, known_companion_mask,
                        variance=None,
                        bad_pixel_mask=None, result_name=None,
                        yx_center=None, yx_center_injection=None,
                        amplitude_modulation=None,
                        contrast_map=None,
                        readnoise=0.):
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
    variance : array_like
        Cube containing variances for `data`.
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
        detection_image['temporal'] = np.zeros(
            (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image['temporal_plus_spatial'] = np.zeros(
                (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        detection_image['spatial'] = np.zeros(
            (detection_image_dim, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    # EDIT: ADDED FOR QUICK CORRELATION TESTS
    detection_image_corr = {}
    if reduction_parameters.temporal_model:
        detection_image_corr['temporal'] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            detection_image_corr['temporal_plus_spatial'] = np.zeros(
                (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        detection_image_corr['spatial'] = np.zeros(
            (6, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    correlation_matrix_binned = {}
    if reduction_parameters.temporal_model:
        correlation_matrix_binned['temporal'] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
        if reduction_parameters.temporal_plus_spatial_model:
            correlation_matrix_binned['temporal_plus_spatial'] = np.zeros(
                (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))
    if reduction_parameters.spatial_model:
        correlation_matrix_binned['spatial'] = np.zeros(
            (43, int(yx_dim[0] * oversampling), int(yx_dim[1] * oversampling)))

    search_region = reduction_parameters.search_region
    search_coordinates = np.argwhere(search_region) * oversampling
    # relative coordinates to output image center (i.e. position of star)
    relative_coords = np.array(list(map(
        lambda x: image_coordinates.absolute_yx_to_relative_yx(x, yx_center_output),
        search_coordinates.tolist())))

    if reduction_parameters.use_multiprocess:
        multiprocess_trap_position = partial(
            trap_one_position, data=data, variance=variance, flux_psf=flux_psf, pa=pa,
            reduction_parameters=reduction_parameters,
            known_companion_mask=known_companion_mask,
            bad_pixel_mask=bad_pixel_mask,
            yx_center=yx_center, yx_center_injection=yx_center_injection,
            amplitude_modulation=amplitude_modulation,
            contrast_map=contrast_map,
            readnoise=readnoise)

        if reduction_parameters.ncpus is None:
            reduction_parameters.ncpus = multiprocessing.cpu_count()
        pool = Pool(processes=reduction_parameters.ncpus)
        for idx, result in enumerate(tqdm(pool.imap(multiprocess_trap_position, relative_coords),
                                          total=len(relative_coords))):
            for key in detection_image:
                if result[key] is not None:
                    detection_image[key][0][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].measured_contrast
                    detection_image[key][1][search_coordinates[idx][0],
                                            search_coordinates[idx][1]] = result[key].contrast_uncertainty
                    detection_image[key][2][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].snr
                    if reduction_parameters.inject_fake:
                        detection_image[key][3][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].true_contrast
                        detection_image[key][4][search_coordinates[idx][0],
                                                search_coordinates[idx][1]] = result[key].measured_contrast - result[key].true_contrast
                        detection_image[key][5][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].relative_deviation_from_true
                        detection_image[key][6][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].wrong_in_sigma

                    # NOTE: Should be include in results class or dictionary to avoid code duplication
                    if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                        detection_image_corr[key][0][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].measured_contrast_with_corr
                        detection_image_corr[key][1][search_coordinates[idx][0],
                                                     search_coordinates[idx][1]] = result[key].contrast_uncertainty_with_corr
                        detection_image_corr[key][2][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].snr_with_corr
                        detection_image_corr[key][3][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_exponential']
                        detection_image_corr[key][4][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern32']
                        detection_image_corr[key][5][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern52']

                        correlation_matrix_binned[key][:, search_coordinates[idx]
                                                       [0], search_coordinates[idx][1]] = result[key].correlation_info['summary_dataframe'].empirical_correlation.values

            del result
        del pool
    else:
        for idx, coords in enumerate(tqdm(relative_coords)):
            # if reduction_parameters.inject_fake == True:
            #         reduction_parameters.true_position = image_coordinates.absolute_yx_to_relative_yx(
            #             coords, image_center_yx=yx_center)
            #     reduction_parameters.guess_position = image_coordinates.absolute_yx_to_relative_yx(
            #         coords, image_center_yx=yx_center)

            result = trap_one_position(
                coords, data=data, variance=variance, flux_psf=flux_psf, pa=pa,
                reduction_parameters=reduction_parameters,
                known_companion_mask=known_companion_mask,
                bad_pixel_mask=bad_pixel_mask,
                yx_center=yx_center, yx_center_injection=yx_center_injection,
                amplitude_modulation=amplitude_modulation,
                contrast_map=contrast_map,
                readnoise=readnoise)
            for key in detection_image:
                if result[key] is not None:
                    detection_image[key][0][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].measured_contrast
                    detection_image[key][1][search_coordinates[idx][0],
                                            search_coordinates[idx][1]] = result[key].contrast_uncertainty
                    detection_image[key][2][search_coordinates[idx]
                                            [0], search_coordinates[idx][1]] = result[key].snr
                    if reduction_parameters.inject_fake:
                        detection_image[key][3][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].true_contrast
                        detection_image[key][4][search_coordinates[idx][0],
                                                search_coordinates[idx][1]] = result[key].measured_contrast - result[key].true_contrast
                        detection_image[key][5][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].relative_deviation_from_true
                        detection_image[key][6][search_coordinates[idx]
                                                [0], search_coordinates[idx][1]] = result[key].wrong_in_sigma
                    # NOTE: Should be include in results class or dictionary to avoid code duplication
                    if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                        detection_image_corr[key][0][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].measured_contrast_with_corr
                        detection_image_corr[key][1][search_coordinates[idx][0],
                                                     search_coordinates[idx][1]] = result[key].contrast_uncertainty_with_corr
                        detection_image_corr[key][2][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].snr_with_corr
                        detection_image_corr[key][3][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_exponential']
                        detection_image_corr[key][4][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern32']
                        detection_image_corr[key][5][search_coordinates[idx]
                                                     [0], search_coordinates[idx][1]] = result[key].correlation_info['corr_length_matern52']
                        # ipsh()
                        correlation_matrix_binned[key][:, search_coordinates[idx]
                                                       [0], search_coordinates[idx][1]] = result[key].correlation_info['summary_dataframe'].empirical_correlation.values

            del result
    if not reduction_parameters.compute_residual_correlation \
            and not reduction_parameters.use_residual_correlation:
        detection_image_corr = None
        correlation_matrix_binned = None

    return detection_image, detection_image_corr, correlation_matrix_binned


def make_reduction_header(
        reduction_parameters, instrument, bad_frames,
        exclude_bad_pixel, oversampling, right_handed,
        yx_known_companion_position):

    raise NotImplementedError()


def run_complete_reduction(
        data_full,
        flux_psf_full,
        pa,
        instrument,
        reduction_parameters,
        temporal_components_fraction=[0.3],
        wavelength_indices=None,
        variance_full=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        verbose=False):
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
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    temporal_components_fraction : array_like
        List containing the principal component fraction to be used for
        the temporal TRAP analysis. If more than one number is given
        TRAP will loop over them and produce outputs for all numbers.
        Default is [0.3].
    wavelength_indices : array_like, optional
        Vector containing the indices of the wavelength slices that should
        be reduced.
    variance_full : array_like, optional
        Data cube of the same shape as `data_full` that contains the
        variance of each data point.
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

    if bad_frames is None:
        bad_frames = []

    if instrument.detector_gain != 1:
        flux_psf_full *= instrument.detector_gain
        data_full *= instrument.detector_gain

    if flux_psf_full.ndim < 3:
        flux_psf_full = np.expand_dims(flux_psf_full, axis=0)
    if data_full.ndim < 4:
        data_full = np.expand_dims(data_full, axis=0)
    if variance_full is not None and variance_full.ndim < 4:
        variance_full = np.expand_dims(variance_full, axis=0)

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

    if reduction_parameters.reduce_single_position:
        guess_position_separation = np.sqrt(
            reduction_parameters.guess_position[0]**2 +
            reduction_parameters.guess_position[1]**2)
        print('Adjusting outer bound to fit guess position')
        reduction_parameters.search_region_outer_bound = np.ceil(guess_position_separation) + 5

    if reduction_parameters.autosize_masks_in_lambda_over_d:
        assert reduction_parameters.signal_mask_size_in_lambda_over_d >= reduction_parameters.reduction_mask_size_in_lambda_over_d, \
            "Signal mask size must be >= reduction mask size"
        stamp_sizes = determine_psf_stampsizes(
            instrument.fwhm.value,
            size_in_lamda_over_d=reduction_parameters.signal_mask_size_in_lambda_over_d)
        stamp_sizes_reduction = determine_psf_stampsizes(
            instrument.fwhm.value,
            size_in_lamda_over_d=reduction_parameters.reduction_mask_size_in_lambda_over_d)
    else:
        assert reduction_parameters.signal_mask_size >= reduction_parameters.reduction_mask_size, \
            "Signal mask size must be >= reduction mask size"
        stamp_sizes = np.repeat(
            reduction_parameters.signal_mask_psf_size,
            len(instrument.wavelengths))
        stamp_sizes_reduction = np.repeat(
            reduction_parameters.reduction_mask_psf_size,
            len(instrument.wavelengths))
    if flux_psf_full.shape[-1] < np.max(stamp_sizes):
        raise ValueError("The provided PSF images are too small for the chosen parameters.")
    psf_stamps = prepare_psf(
        flux_psf_full, psf_size=stamp_sizes)

    # Remove bad frames
    if bad_frames is not None:
        data_full = np.delete(data_full, bad_frames, axis=1)
        pa = np.delete(pa, bad_frames, axis=0)
        if variance_full is not None:
            variance_full = np.delete(variance_full, bad_frames, axis=1)

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
            yx_center_injection_full = np.ones((data_full.shape[0], data_full.shape[1], 2))
            yx_center_injection_full[:, :, 0] = xy_image_centers[1]
            yx_center_injection_full[:, :, 1] = xy_image_centers[0]
            yx_center_full = np.ones((data_full.shape[0], 2))
            yx_center_full[:, 0] = xy_image_centers[1]
            yx_center_full[:, 1] = xy_image_centers[0]
            max_shift = 0
        elif xy_image_centers.ndim > 1:
            if xy_image_centers.ndim > 3:
                raise ValueError('Dimensionality of provided centers too large.')
            if xy_image_centers.ndim == 2:
                xy_image_centers = np.expand_dims(xy_image_centers, axis=0)
            yx_center_injection_full = xy_image_centers[..., ::-1]
            yx_center_full = np.median(yx_center_injection_full, axis=1)
            max_shift_x = np.max(xy_image_centers[..., 0]) - np.min(xy_image_centers[..., 0])
            max_shift_y = np.max(xy_image_centers[..., 1]) - np.min(xy_image_centers[..., 1])
            max_shift = np.max([max_shift_x, max_shift_y]) * 2
            print("The center varies by a maximum of in x or y: {}".format(max_shift / 2))
        # print("Center variation: {}".format(np.std(amplitude_modulation_full, axis=0)))
    reduction_parameters.yx_anamorphism = np.array(reduction_parameters.yx_anamorphism)

    # Decide crop size for images
    if reduction_parameters.data_auto_crop:
        # Automatically determine smallest size to crop data
        reduction_parameters.data_crop_size = np.ceil(
            reduction_parameters.search_region_outer_bound * 2 + np.max(stamp_sizes) * np.sqrt(2) + max_shift)
        if reduction_parameters.add_radial_regressors:
            # NOTE: Hardcoded binary dilation used right now.
            reduction_parameters.data_crop_size += 14
        reduction_parameters.data_crop_size = int(round_up_to_odd(
            reduction_parameters.data_crop_size))
        yx_dim = (reduction_parameters.data_crop_size,
                  reduction_parameters.data_crop_size)
        print("Auto crop size cropped data to: {}".format(reduction_parameters.data_crop_size))
    else:
        if reduction_parameters.search_region is None:
            if reduction_parameters.data_crop_size is None:
                yx_dim = (data_full.shape[-2], data_full.shape[-1])
            else:
                yx_dim = (reduction_parameters.data_crop_size,
                          reduction_parameters.data_crop_size)
        else:
            yx_dim = (reduction_parameters.search_region.shape[-2],
                      reduction_parameters.search_region.shape[-1])
    data_crop_size = reduction_parameters.data_crop_size

    if reduction_parameters.search_region is None:
        reduction_parameters.search_region = regressor_selection.make_annulus_mask(
            reduction_parameters.search_region_inner_bound,
            reduction_parameters.search_region_outer_bound,
            yx_dim=yx_dim,
            oversampling=reduction_parameters.oversampling,
            yx_center=None)

    # Configure PSF amplitude variation
    if amplitude_modulation_full is not None:
        if amplitude_modulation_full.ndim < 2:
            amplitude_modulation_full = np.expand_dims(amplitude_modulation_full, axis=0)
        amplitude_modulation_full = np.delete(amplitude_modulation_full, bad_frames, axis=1)
        print("Amplitude variation: {}".format(np.std(amplitude_modulation_full, axis=1)))

    # Configure known companion information
    if reduction_parameters.yx_known_companion_position is not None:
        reduction_parameters.yx_known_companion_position = np.array(
            reduction_parameters.yx_known_companion_position)
        # If ndim == 1, just one companion present, make new dimension for more companions
        if reduction_parameters.yx_known_companion_position.ndim == 1:
            reduction_parameters.yx_known_companion_position = np.expand_dims(
                reduction_parameters.yx_known_companion_position, axis=0)
        elif reduction_parameters.yx_known_companion_position.ndim > 2:
            raise ValueError("Dimensionality of known companion position array too large.")

    if reduction_parameters.known_companion_contrast is not None and reduction_parameters.remove_known_companions:
        assert reduction_parameters.yx_known_companion_position is not None, "No position for known companion given."

        reduction_parameters.known_companion_contrast = np.array(
            reduction_parameters.known_companion_contrast)

        number_of_wavelengths = data_full.shape[0]
        number_of_companions = reduction_parameters.yx_known_companion_position.shape[0]

        assert reduction_parameters.known_companion_contrast.shape[-1] == number_of_companions, \
            "The same number of known companion position and contrasts need to be provided."

        if reduction_parameters.known_companion_contrast.ndim == 1 and number_of_wavelengths == 1:
            reduction_parameters.known_companion_contrast = np.expand_dims(
                reduction_parameters.known_companion_contrast, axis=0)
        elif reduction_parameters.known_companion_contrast.ndim == 1 and number_of_wavelengths > 1:
            raise ValueError(
                "For multi-wavelength data, a known contrast has to be defined for every wavelength.")
        elif reduction_parameters.known_companion_contrast.ndim > 2:
            raise ValueError("Dimensionality of known companion contrast array too large.")

    # Configure number of principal components
    number_of_components = np.round(
        data_full.shape[1] * np.array(temporal_components_fraction)).astype('int')

    result_folder = reduction_parameters.result_folder
    prefix = reduction_parameters.prefix
    if result_folder is None:
        result_folder = './'
    else:
        if not reduction_parameters.reduce_single_position:
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

    # Save parameters
    if not reduction_parameters.reduce_single_position:
        with open(os.path.join(result_folder, "instrument.obj"), 'wb') as handle:
            pickle.dump(instrument, handle, protocol=4)
        with open(os.path.join(result_folder, "reduction_parameters.obj"), 'wb') as handle:
            pickle.dump(reduction_parameters, handle, protocol=4)

    assert flux_psf_full.shape[0] == data_full.shape[0] == len(instrument.wavelengths), \
        "Different number of wavelengths in data: Flux {} Data {} Wave {}".format(
            flux_psf_full.shape[0], data_full.shape[0], len(instrument.wavelengths))

    if reduction_parameters.reduce_single_position is True:
        all_results = OrderedDict()

    if reduction_parameters.use_multiprocess and not reduction_parameters.reduce_single_position:
        ray.init(num_cpus=reduction_parameters.ncpus)

    # Loop over reductions for different numbers of components
    for comp_index, ncomp in enumerate(number_of_components):
        reduction_parameters.number_of_pca_regressors = ncomp
        reduction_parameters.temporal_components_fraction = temporal_components_fraction[comp_index]
        print("Number of principal comp. used: {} of {}".format(ncomp, data_full.shape[1]))

        if reduction_parameters.reduce_single_position:
            wavelength_results = OrderedDict()
        number_of_wavelengths = data_full.shape[0]

        if wavelength_indices is None:
            wavelength_indices = np.arange(number_of_wavelengths)

        # Loop over reduction for different wavelengths
        for _, wavelength_index, in enumerate(wavelength_indices):
            wavelength = instrument.wavelengths[wavelength_index]
            print("Lambda index: {} Wavelength: {:.3f}".format(wavelength_index, wavelength))
            if prefix is None:
                prefix = ''
            reduction_parameters.fwhm = instrument.fwhm[wavelength_index].value
            basename = {}
            if reduction_parameters.inject_fake:
                basename['temporal'] = 'injectedsigma{:.2f}_{}lam{:02d}_ncomp{:03d}_frac{:.2f}'.format(
                    reduction_parameters.injection_sigma, prefix, wavelength_index, ncomp,
                    temporal_components_fraction[comp_index])
                basename['temporal_plus_spatial'] = 'injectedsigma{:.2f}_{}lam{:02d}_ncomp{:03d}_frac{:.2f}_delta{:.2f}_spatialfrac{:.2f}'.format(
                    reduction_parameters.injection_sigma, prefix, wavelength_index,
                    ncomp, temporal_components_fraction[comp_index],
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction_after_trap)
                basename['spatial'] = 'injectedsigma{:.2f}_{}lam{:02d}_delta{:.2f}_spatialfrac{:.2f}'.format(
                    reduction_parameters.injection_sigma, prefix, wavelength_index,
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction)
            else:
                basename['temporal'] = '{}lam{:02d}_ncomp{:03d}_frac{:.2f}'.format(
                    prefix, wavelength_index, ncomp, temporal_components_fraction[comp_index])
                basename['temporal_plus_spatial'] = '{}lam{:02d}_ncomp{:03d}_frac{:.2f}_delta{:.2f}_spatialfrac{:.2f}'.format(
                    prefix, wavelength_index, ncomp, temporal_components_fraction[comp_index],
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction_after_trap)
                basename['spatial'] = '{}lam{:02d}_delta{:.2f}_spatialfrac{:.2f}'.format(
                    prefix, wavelength_index,
                    reduction_parameters.protection_angle,
                    reduction_parameters.spatial_components_fraction)
            print(basename['temporal'])

            detection_image_path = {}
            norm_detection_image_path = {}
            contrast_table_path = {}
            contrast_image_path = {}
            median_contrast_image_path = {}
            contrast_plot_path = {}

            # NOTE: Temporarily added for correlation, complex outputs should be implemented as in separate class
            # or dictionary to reduce code duplication
            if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                detection_image_corr_path = {}
                norm_detection_image_corr_path = {}
                contrast_table_corr_path = {}
                contrast_image_corr_path = {}
                median_contrast_image_corr_path = {}
                contrast_plot_corr_path = {}
                correlation_matrix_binned_path = {}

            for key in ['temporal', 'temporal_plus_spatial', 'spatial']:
                detection_image_path[key] = os.path.join(
                    result_folder, 'detection_' + basename[key] + '_' + key + '.fits')
                norm_detection_image_path[key] = os.path.join(
                    result_folder, 'norm_detection_' + basename[key] + '_' + key + '.fits')
                contrast_table_path[key] = os.path.join(
                    result_folder, 'contrast_table_' + basename[key] + '_' + key + '.fits')
                contrast_image_path[key] = os.path.join(
                    result_folder, 'contrast_image_' + basename[key] + '_' + key + '_sigma{:.2f}.fits'.format(
                        reduction_parameters.contrast_curve_sigma))
                median_contrast_image_path[key] = os.path.join(
                    result_folder, 'median_contrast_image_' + basename[key] + '_' + key + '_sigma{:.2f}.fits'.format(
                        reduction_parameters.contrast_curve_sigma))
                contrast_plot_path[key] = os.path.join(
                    result_folder, 'contrast_plot_' + basename[key] + '_' + key + '_sigma{:.2f}.jpg'.format(
                        reduction_parameters.contrast_curve_sigma))

                # NOTE: Temporarily added for correlation, complex outputs should be implemented as in separate class
                # or dictionary to reduce code duplication
                if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                    detection_image_corr_path[key] = os.path.join(
                        result_folder, 'detection_corr_' + basename[key] + '_' + key + '.fits')
                    norm_detection_image_corr_path[key] = os.path.join(
                        result_folder, 'norm_detection_corr_' + basename[key] + '_' + key + '.fits')
                    contrast_table_corr_path[key] = os.path.join(
                        result_folder, 'contrast_table_corr_' + basename[key] + '_' + key + '.fits')
                    contrast_image_corr_path[key] = os.path.join(
                        result_folder, 'contrast_image_corr_' + basename[key] + '_' + key + '_sigma{:.2f}.fits'.format(
                            reduction_parameters.contrast_curve_sigma))
                    median_contrast_image_corr_path[key] = os.path.join(
                        result_folder, 'median_contrast_image_corr_' + basename[key] + '_' + key + '_sigma{:.2f}.fits'.format(
                            reduction_parameters.contrast_curve_sigma))
                    contrast_plot_corr_path[key] = os.path.join(
                        result_folder, 'contrast_plot_corr_' + basename[key] + '_' + key + '_sigma{:.2f}.jpg'.format(
                            reduction_parameters.contrast_curve_sigma))
                    correlation_matrix_binned_path[key] = os.path.join(
                        result_folder, 'correlation_matrix_binned_' + basename[key] + '_' + key + '_sigma{:.2f}.fits'.format(
                            reduction_parameters.contrast_curve_sigma))

            if reduction_parameters.temporal_plus_spatial_model:
                contrast_plot_comparison_path = os.path.join(
                    result_folder, 'contrast_comparison_plot_' + basename['temporal_plus_spatial'] + '_sigma{:.2f}.jpg'.format(
                        reduction_parameters.contrast_curve_sigma))

            # if reduction_parameters.autosize_masks_in_lambda_over_d:
            reduction_parameters.reduction_mask_psf_size = int(
                stamp_sizes_reduction[wavelength_index])
            reduction_parameters.signal_mask_psf_size = int(stamp_sizes[wavelength_index])
            # reduction_parameters.signal_mask_psf_size = int(stamp_sizes[wavelength_index])

            # This block defines yx_center which gives the center of the output file
            # based on cropping or no cropping
            if yx_center_full is None:
                if data_crop_size is None:
                    yx_center = np.array((data_full.shape[-2] // 2., data_full.shape[-1] // 2.))
                else:
                    yx_center = np.array((data_crop_size // 2., data_crop_size // 2.))
            else:
                # try:
                if data_crop_size is None:
                    yx_center = np.array(yx_center_full[wavelength_index])
                else:
                    yx_center = np.array((data_crop_size // 2., data_crop_size // 2.))

            # Make companion mask before cropping to be consistent
            # Do this for each wavelength separately to account for PSF size
            # and differing center position
            if reduction_parameters.yx_known_companion_position is not None \
                    and len(reduction_parameters.yx_known_companion_position) > 0:
                if yx_center_injection_full is not None:
                    yx_center_before_crop = yx_center_injection_full[wavelength_index, :]
                else:
                    yx_center_before_crop = None

                known_companion_masks = []
                for yx_pos in reduction_parameters.yx_known_companion_position:
                    # TODO: CHECK
                    known_companion_mask = regressor_selection.make_mask_from_psf_track(
                        yx_position=yx_pos,
                        psf_size=reduction_parameters.signal_mask_psf_size,
                        pa=pa, image_size=data_full.shape[-1],
                        image_center=yx_center_before_crop,
                        yx_anamorphism=reduction_parameters.yx_anamorphism,
                        right_handed=reduction_parameters.right_handed,
                        return_cube=False)
                    known_companion_masks.append(known_companion_mask)
                known_companion_mask = np.logical_or.reduce(known_companion_masks)

                if data_crop_size is not None:
                    known_companion_mask = crop_box_from_image(
                        known_companion_mask,
                        data_crop_size,
                        center_yx=np.round(yx_center_full[wavelength_index])).copy()
            else:
                known_companion_mask = None

            known_companion_mask = None

            if reduction_parameters.inject_fake:
                # Return copy of data when injecting fake to not contaminate data
                if data_crop_size is not None:
                    data = crop_box_from_3D_cube(
                        data_full[wavelength_index],
                        data_crop_size,
                        center_yx=np.round(yx_center_full[wavelength_index])).copy()
                else:
                    data = data_full[wavelength_index].copy()
            else:
                if data_crop_size is not None:
                    data = crop_box_from_3D_cube(
                        data_full[wavelength_index],
                        data_crop_size,
                        center_yx=np.round(yx_center_full[wavelength_index]))
                else:
                    data = data_full[wavelength_index]
            data = data.astype('float64')

            if variance_full is not None:
                variance = crop_box_from_3D_cube(
                    variance_full[wavelength_index],
                    data_crop_size,
                    center_yx=np.round(yx_center_full[wavelength_index])).copy()
                variance = variance.astype('float64')
            else:
                variance = None

            flux_psf = psf_stamps[wavelength_index].astype('float64')

            if bad_pixel_mask_full is None:
                bad_pixel_mask = None
            else:
                try:
                    if bad_pixel_mask_full.ndim == 3:
                        bad_pixel_mask = bad_pixel_mask_full[wavelength_index]
                        if data_crop_size is not None:
                            bad_pixel_mask = crop_box_from_image(
                                bad_pixel_mask, data_crop_size,
                                center_yx=np.round(yx_center_full[wavelength_index]))
                    elif bad_pixel_mask_full.ndim == 2:
                        bad_pixel_mask = bad_pixel_mask_full
                    else:
                        raise ValueError(
                            'Bad pixel mask, must either be one image or a number of images corresponding to wavelength')
                except AttributeError:
                    pass

            if yx_center_injection_full is None:
                yx_center_injection = yx_center
            else:
                try:
                    if yx_center_injection_full.ndim == 3:
                        if data_crop_size is None:
                            yx_center_injection = yx_center_injection_full[wavelength_index, :]
                        else:
                            # Image centers in cropped frame
                            yx_center_injection = yx_center_injection_full[wavelength_index, :] - np.round(yx_center_full[wavelength_index]) \
                                + yx_center
                            # np.round(yx_center_full[wavelength_index]) - yx_center_injection_full[:, wavelength_index] \
                            # + yx_center
                            # Non-rounded image center in cropped frame
                            yx_center = np.nanmedian(yx_center_injection, axis=0)
                except AttributeError:
                    pass

            if reduction_parameters.inject_fake and reduction_parameters.read_injection_files:
                input_contrast_image_path = contrast_image_path['temporal'].replace(
                    "injectedsigma{:.2f}_".format(reduction_parameters.injection_sigma), '')
                input_sigma = float(os.path.splitext(input_contrast_image_path)[0][-4:])
                contrast_map = fits.getdata(input_contrast_image_path)
                contrast_map /= input_sigma  # contrast map is 5 sigma
                contrast_map *= reduction_parameters.injection_sigma

            if not reduction_parameters.inject_fake or not reduction_parameters.read_injection_files:
                contrast_map = None

            if contrast_map is not None:
                contrast_map = crop_box_from_image(
                    contrast_map,
                    data_crop_size,
                    center_yx=None)

            if amplitude_modulation_full is None:
                amplitude_modulation = np.ones(data_full.shape[1])
            else:
                amplitude_modulation = amplitude_modulation_full[wavelength_index, :]

            # Do not reduce data for wavelength if flux PSF or center position contains NaNs
            flux_psf_not_finite = np.any(~np.isfinite(flux_psf))
            yx_center_injection_not_finite = np.any(~np.isfinite(yx_center_injection))
            if flux_psf_not_finite:
                print("Skipping wavelength {}. NaNs detected in flux PSF.".format(wavelength_index))
                continue
            if yx_center_injection_not_finite:
                print("Skipping wavelength {}. NaNs detected in provided center position.".format(
                    wavelength_index))
                continue

            if reduction_parameters.remove_known_companions:
                # NOTE: This currently doesn't remove photon noise from variance map
                # NOTE: Should change to faster implementation of `inject_model_into_data`

                for companion_index, known_companion_contrast in enumerate(
                        reduction_parameters.known_companion_contrast[wavelength_index]):
                    # NOTE: Check format of known_companion_contrast and amplitude_modulation
                    known_companion_contrast *= amplitude_modulation

                    data = makesource.addsource(
                        data, reduction_parameters.yx_known_companion_position[companion_index],
                        pa, flux_psf, image_center=yx_center_injection,
                        norm=-1 * known_companion_contrast,
                        jitter=0, poisson_noise=False,
                        yx_anamorphism=reduction_parameters.yx_anamorphism,
                        right_handed=reduction_parameters.right_handed,
                        subpixel=True,
                        verbose=False)

            print('PSF Size: {}'.format(reduction_parameters.reduction_mask_psf_size))
            if reduction_parameters.reduce_single_position:
                results = trap_one_position(
                    reduction_parameters.guess_position,
                    data=data,
                    variance=variance,
                    flux_psf=flux_psf,
                    pa=pa,
                    reduction_parameters=reduction_parameters,
                    known_companion_mask=known_companion_mask,
                    bad_pixel_mask=bad_pixel_mask,
                    yx_center=yx_center, yx_center_injection=yx_center_injection,
                    amplitude_modulation=amplitude_modulation,
                    contrast_map=contrast_map,
                    readnoise=instrument.readnoise)

                wavelength_results['{}'.format(wavelength_index)] = results

            else:
                detection_image, detection_image_corr, correlation_matrix_binned = run_trap_search(
                    data=data.astype('float64'),
                    variance=variance,
                    flux_psf=flux_psf,
                    pa=pa,
                    wavelength=wavelength,
                    reduction_parameters=reduction_parameters,
                    known_companion_mask=known_companion_mask,
                    bad_pixel_mask=bad_pixel_mask,
                    yx_center=yx_center,
                    yx_center_injection=yx_center_injection,
                    amplitude_modulation=amplitude_modulation,
                    contrast_map=contrast_map,
                    readnoise=instrument.readnoise)

                # NOTE: Moved out from run_trap_search
                for key in detection_image:
                    fits.writeto(detection_image_path[key], detection_image[key], overwrite=True)
                    # NOTE: Temporarily added for correlation, complex outputs should be implemented as in separate class
                    # or dictionary to reduce code duplication
                    if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                        fits.writeto(detection_image_corr_path[key],
                                     detection_image_corr[key], overwrite=True)
                        fits.writeto(correlation_matrix_binned_path[key],
                                     correlation_matrix_binned[key], overwrite=True)

                del detection_image
                # pixel_scale_mas = (1 * u.pixel).to(u.mas, instrument.pixel_scale).value
                #
                # contrast_table = {}
                # contrast_table_corr = {}
                # for key in detection_image:
                #     normalized_detection_image, contrast_table[key], contrast_image, median_contrast_image = detection.make_contrast_curve(
                #         detection_image[key], radial_bounds=None, bin_width=reduction_parameters.normalization_width,
                #         companion_mask_radius=reduction_parameters.companion_mask_radius,
                #         pixel_scale=pixel_scale_mas,
                #         yx_known_companion_position=reduction_parameters.yx_known_companion_position)
                #     fits.writeto(norm_detection_image_path[key],
                #                  normalized_detection_image, overwrite=True)
                #     fits.writeto(contrast_image_path[key], contrast_image, overwrite=True)
                #     fits.writeto(median_contrast_image_path[key],
                #                  median_contrast_image, overwrite=True)
                #     contrast_table[key].write(contrast_table_path[key], overwrite=True)
                #
                #     if reduction_parameters.contrast_curve:
                #         detection.plot_contrast_curve(
                #             [contrast_table[key]],
                #             instrument=instrument,
                #             wavelengths=instrument.wavelengths[wavelength_index:wavelength_index + 1],
                #             colors=['#1b1cd5'],  # '#de650a', '#ba174e'],
                #             plot_vertical_lod=True, mirror_axis='mas',
                #             convert_to_mag=False, yscale='log',
                #             savefig=contrast_plot_path[key], show=False)
                #
                #     # NOTE: Temporarily added for correlation, complex outputs should be implemented as in separate class
                #     # or dictionary to reduce code duplication
                #     if reduction_parameters.compute_residual_correlation and reduction_parameters.use_residual_correlation:
                #         normalized_detection_image_corr, contrast_table_corr[key], contrast_image_corr, median_contrast_image_corr = detection.make_contrast_curve(
                #             detection_image_corr[key], radial_bounds=None, bin_width=reduction_parameters.normalization_width,
                #             companion_mask_radius=reduction_parameters.companion_mask_radius,
                #             pixel_scale=pixel_scale_mas,
                #             yx_known_companion_position=reduction_parameters.yx_known_companion_position)
                #         fits.writeto(norm_detection_image_corr_path[key],
                #                      normalized_detection_image_corr, overwrite=True)
                #         fits.writeto(contrast_image_corr_path[key],
                #                      contrast_image_corr, overwrite=True)
                #         fits.writeto(
                #             median_contrast_image_corr_path[key], median_contrast_image_corr, overwrite=True)
                #         contrast_table_corr[key].write(
                #             contrast_table_corr_path[key], overwrite=True)
                #
                #         if reduction_parameters.contrast_curve:
                #             detection.plot_contrast_curve(
                #                 [contrast_table_corr[key]],
                #                 instrument=instrument,
                #                 wavelengths=instrument.wavelengths[wavelength_index:wavelength_index + 1],
                #                 colors=['#1b1cd5'],  # '#de650a', '#ba174e'],
                #                 plot_vertical_lod=True, mirror_axis='mas',
                #                 convert_to_mag=False, yscale='log',
                #                 savefig=contrast_plot_corr_path[key], show=False)
                #
                # if reduction_parameters.temporal_plus_spatial_model:
                #     detection.plot_contrast_curve(
                #         [
                #             contrast_table['temporal'],
                #             contrast_table['temporal_plus_spatial']
                #         ],
                #         instrument=instrument,
                #         wavelengths=instrument.wavelengths[wavelength_index:wavelength_index + 1].repeat(
                #             2),
                #         curvelabels=['temporal', 'temporal + spatial'],
                #         linestyles=['-', '--'],
                #         colors=['#1b1cd5', '#de650a'],  # , '#ba174e'],
                #         plot_vertical_lod=True, mirror_axis='mas',
                #         convert_to_mag=False, yscale='log',
                #         savefig=contrast_plot_comparison_path, show=False)

                # del contrast_table
                # del normalized_detection_image
                # del contrast_image
                # del median_contrast_image
        if reduction_parameters.reduce_single_position:
            all_results['{}'.format(temporal_components_fraction[comp_index])] = wavelength_results

    if reduction_parameters.use_multiprocess and not reduction_parameters.reduce_single_position:
        ray.shutdown()

    if reduction_parameters.reduce_single_position:
        return all_results
    else:
        return None


# def make_contrast_from_output(
#         result_folder,
#         instrument,
#         glob_pattern='detection*fits',
#         yx_known_companion_position=None,
#         sigma=5,
#         radial_bounds=None,
#         bin_width=3.,
#         companion_mask_radius=11):
#
#     detection_files = glob(os.path.join(result_folder, glob_pattern))
#     assert len(detection_files) > 0, "No output files found."
#
#     # Loop over reductions for different numbers of components
#     for idx, detection_file in tqdm(enumerate(detection_files)):
#         # TODO: Add sigma to output filename
#         basename = os.path.basename(detection_file)
#         norm_detection_image_path = os.path.join(
#             result_folder, basename.replace('detection', 'norm_detection'))
#         contrast_table_path = os.path.join(
#             result_folder, basename.replace('detection', 'contrast_table'))
#         contrast_image_path = os.path.join(
#             result_folder, basename.replace('detection', 'contrast_image'))
#         median_contrast_image_path = os.path.join(
#             result_folder, basename.replace('detection', 'median_contrast_image'))
#         contrast_plot_path = os.path.join(
#             result_folder, os.path.splitext(basename)[0].replace('detection', 'contrast_plot') + '.jpg')
#
#         string_index = basename.find('lam') + 3
#         # string_index = basename.find('lam_') + 4
#         wavelength_index = int(basename[string_index:string_index + 2])
#
#         detection_image = fits.getdata(detection_file)
#         pixel_scale_mas = (1 * u.pixel).to(u.mas, instrument.pixel_scale).value
#         normalized_detection_image, contrast_table, contrast_image, median_contrast_image = detection.make_contrast_curve(
#             detection_image, radial_bounds=radial_bounds,
#             bin_width=bin_width,
#             companion_mask_radius=companion_mask_radius,
#             pixel_scale=pixel_scale_mas,
#             yx_known_companion_position=yx_known_companion_position)
#         fits.writeto(norm_detection_image_path, normalized_detection_image, overwrite=True)
#         fits.writeto(contrast_image_path, contrast_image, overwrite=True)
#         fits.writeto(median_contrast_image_path, median_contrast_image, overwrite=True)
#         contrast_table.write(contrast_table_path, overwrite=True)
#
#         detection.plot_contrast_curve(
#             [contrast_table],
#             instrument=instrument,
#             wavelengths=wavelengths[wavelength_index:wavelength_index + 1],
#             colors=['#1b1cd5'],  # '#de650a', '#ba174e'],
#             plot_vertical_lod=True, mirror_axis='mas',
#             convert_to_mag=False, yscale='log',
#             savefig=contrast_plot_path, show=False)
