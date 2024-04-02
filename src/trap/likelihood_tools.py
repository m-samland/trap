"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
# from pandas.plotting import autocorrelation_plot
from scipy.linalg import cho_factor, cho_solve

from trap import regression, makesource
from trap.image_coordinates import rhophi_to_relative_yx

# from statsmodels.graphics.tsaplots import plot_pacf


def lnprior(theta, mcmc_parameters):
    """ min_val = mean for gauss, max_val = variance

    """

    # if np.isfinite(np.sum(theta)):
    prior = 0.0
    for t, min_val, max_val, method, gauss_para in zip(theta, mcmc_parameters['minval'], mcmc_parameters['maxval'], mcmc_parameters['method'], mcmc_parameters['gauss_para']):
        if (t < min_val) or (t > max_val):
            return -np.inf
        if method == 'gauss':
            prior -= (t - gauss_para[0])**2 / (2. * gauss_para[1])**2
    return prior


# def lnlike_for_hyperparam(
#         theta, data, planet_relative_yx_pos, reduction_mask,
#         known_companion_mask, model, yx_center, annulus_offset,
#         overall_signal_mask, target_pix_mask_radius, remove_bad_frames,
#         bad_frame_mad_cutoff, remove_bad_frames_iterations,
#         compute_inverse_once, use_relative_position, use_reference_pixel_map,
#         reference_pixel_map, method_of_regressor_selection,
#         auxiliary_frame, make_reconstructed_lightcurve,
#         true_contrast, verbose):
#     """Calculates likelihood by interpolating model grid."""
#
#     result = regression.run_trap_with_model(
#         data=data,
#         planet_relative_yx_pos=planet_relative_yx_pos,
#         reduction_mask=reduction_mask,
#         known_companion_mask=known_companion_mask,
#         model=model,
#         yx_center=yx_center,
#         number_of_pca_regressors=int(theta[0]),
#         annulus_width=int(theta[2]),
#         annulus_offset=annulus_offset,
#         overall_signal_mask=overall_signal_mask,
#         target_pix_mask_radius=target_pix_mask_radius,
#         variance_prior_scaling=theta[1],
#         remove_bad_frames=remove_bad_frames,
#         bad_frame_mad_cutoff=bad_frame_mad_cutoff,
#         remove_bad_frames_iterations=remove_bad_frames_iterations,
#         compute_inverse_once=compute_inverse_once,
#         use_relative_position=use_relative_position,
#         use_reference_pixel_map=use_reference_pixel_map,
#         reference_pixel_map=reference_pixel_map,
#         method_of_regressor_selection=method_of_regressor_selection,
#         auxiliary_frame=auxiliary_frame,
#         make_reconstructed_lightcurve=make_reconstructed_lightcurve,
#         true_contrast=true_contrast,
#         verbose=verbose)
#
#     result.compute_contrast_weighted_average(mask_outliers=True)
#     chi2 = result.wrong_in_sigma**2
#
#     print(theta)
#     print("lnlike{0:.03E}".format(-0.5 * chi2))
#     return -0.5 * chi2
#
#
# def lnprob_for_hyperparam(
#         theta, mcmc_parameters, data, planet_relative_yx_pos, reduction_mask,
#         known_companion_mask, model, yx_center, annulus_offset,
#         overall_signal_mask, target_pix_mask_radius, remove_bad_frames,
#         bad_frame_mad_cutoff, remove_bad_frames_iterations,
#         compute_inverse_once, use_relative_position, use_reference_pixel_map,
#         reference_pixel_map, method_of_regressor_selection,
#         auxiliary_frame, make_reconstructed_lightcurve,
#         true_contrast, verbose):
#     lp = lnprior(theta, mcmc_parameters)
#     if not np.isfinite(lp):
#         return -np.inf
#     else:
#         return lp + lnlike(
#             theta, data, planet_relative_yx_pos, reduction_mask,
#             known_companion_mask, model, yx_center, annulus_offset,
#             overall_signal_mask, target_pix_mask_radius, remove_bad_frames,
#             bad_frame_mad_cutoff, remove_bad_frames_iterations,
#             compute_inverse_once, use_relative_position, use_reference_pixel_map,
#             reference_pixel_map, method_of_regressor_selection,
#             auxiliary_frame, make_reconstructed_lightcurve,
#             true_contrast, verbose)


# FOR PLANET RETRIEVAL

def lnlike_for_negfake(
        theta, data, reduction_parameters, planet_relative_yx_pos, reduction_mask,
        known_companion_mask, yx_center, overall_signal_mask,
        regressor_matrix,
        auxiliary_frame,
        noise_map, true_contrast, fancy_noise, verbose):
    """Calculates likelihood by interpolating model grid."""

    if reduction_parameters.number_of_pca_regressors is None:
        number_of_pca_regressors = int(theta[3])

    result = regression.run_trap_wo_model(
        data=data,
        reduction_parameters=reduction_parameters,
        planet_relative_yx_pos=planet_relative_yx_pos,
        reduction_mask=reduction_mask,
        known_companion_mask=known_companion_mask,
        yx_center=yx_center,
        overall_signal_mask=overall_signal_mask,
        regressor_matrix=regressor_matrix,
        return_only_noise_model=False,
        auxiliary_frame=auxiliary_frame,
        true_contrast=true_contrast,
        verbose=verbose)

    # assume that pixel are independent -> flatten
    ntime = data.shape[0]
    pixel_lnlike = []

    def faster_lnlike(r, K):
        """
        A faster version of ``lnlike``.
        """
        cho_decomp = cho_factor(K)
        alpha_cho = cho_solve(cho_decomp, r)
        lndet_K = 2. * np.sum(np.log(np.diag(cho_decomp[0])))

        return -0.5 * (np.dot(r, alpha_cho) + lndet_K)  # -0.5 * np.log())

    for i in range(result.n_reduction_pix):
        # Should change this to cho_solve
        r = result.residuals[:, i]
        if noise_map is None:
            cov = np.diag(np.ones(ntime))
        else:
            cov = np.diag(noise_map[i])
        if fancy_noise:
            const_term = np.ones(len(cov)) * theta[-1]
            cov += np.diag(const_term)

        pixel_lnlike.append(faster_lnlike(r, cov))  # - 0.5 * np.log())

    log_likelihood = np.sum(np.array(pixel_lnlike))
    # if verbose:
    print(theta)
    print("{0:.03E}".format(log_likelihood))
    return log_likelihood
    # logdet = 2*np.sum(np.log(np.diag(L_cov)))
    # print(-0.5*chi2)
    # return -0.5*chi2


def lnprob_for_negfake(
        theta, mcmc_parameters, data, flux_psf, pa,
        reduction_parameters, reduction_mask,
        known_companion_mask, yx_center,
        overall_signal_mask, regressor_matrix,
        auxiliary_frame,
        noise_map, true_contrast, verbose,
        use_prior=True, fancy_noise=False):

    lp = 0.
    if use_prior:
        lp = lnprior(theta, mcmc_parameters)
        if not np.isfinite(lp):
            return -np.inf

    # TODO right_handed into parameters
    # relative_yx = [theta[1], theta[0]]
    relative_yx = rhophi_to_relative_yx([theta[0], theta[1]])
    planet_relative_yx_pos = np.array(relative_yx)
    data_minus_model = makesource.addsource(
        data, planet_relative_yx_pos, pa, flux_psf,
        norm=-theta[2], jitter=0, poisson_noise=False,
        right_handed=True, verbose=False)

    # Check if any pixels are negative after subtracting planet model
    # Reject those models
    number_of_values = np.prod(data_minus_model[:, reduction_mask].shape)
    number_of_values_below_zero = len(
        np.where(data_minus_model[:, reduction_mask] < 0.))

    if number_of_values_below_zero / number_of_values > 0.9:
        return -np.inf

    else:
        return lp + lnlike_for_negfake(
            theta,
            data=data_minus_model,
            reduction_parameters=reduction_parameters,
            planet_relative_yx_pos=planet_relative_yx_pos,
            reduction_mask=reduction_mask,
            known_companion_mask=known_companion_mask,
            yx_center=yx_center,
            overall_signal_mask=overall_signal_mask,
            regressor_matrix=regressor_matrix,
            auxiliary_frame=auxiliary_frame,
            noise_map=noise_map,
            true_contrast=true_contrast,
            fancy_noise=fancy_noise,
            verbose=verbose)


def lnlike_for_negfake_corrnoise(
        theta, data, reduction_parameters, planet_relative_yx_pos, reduction_mask,
        known_companion_mask, yx_center, overall_signal_mask,
        regressor_matrix,
        auxiliary_frame,
        noise_map, true_contrast, verbose):
    """Calculates likelihood by interpolating model grid."""

    if reduction_parameters.number_of_pca_regressors is None:
        number_of_pca_regressors = int(theta[3])

    result = regression.run_trap_wo_model(
        data=data,
        reduction_parameters=reduction_parameters,
        planet_relative_yx_pos=planet_relative_yx_pos,
        reduction_mask=reduction_mask,
        known_companion_mask=known_companion_mask,
        yx_center=yx_center,
        overall_signal_mask=overall_signal_mask,
        regressor_matrix=regressor_matrix,
        return_only_noise_model=False,
        auxiliary_frame=auxiliary_frame,
        true_contrast=true_contrast,
        verbose=verbose)

    # assume that pixel are independent -> flatten
    ntime = data.shape[0]
    pixel_lnlike = []

    for i in range(result.n_reduction_pix):
        # Should change this to cho_solve
        r = result.residuals[:, i]
        if noise_map is None:
            cov = np.diag(np.ones(ntime))
        else:
            cov = np.diag(noise_map[i])

        pixel_lnlike.append(lnlike_one_pixel)
        # faster_lnlike(r, cov))  # - 0.5 * np.log())

    log_likelihood = np.sum(np.array(pixel_lnlike))

    print(theta)
    print("{0:.03E}".format(log_likelihood))
    return log_likelihood
    # logdet = 2*np.sum(np.log(np.diag(L_cov)))
    # print(-0.5*chi2)
    # return -0.5*chi2


def lnprob_for_negfake_corrnoise(
        theta, mcmc_parameters, data, flux_psf, pa,
        reduction_parameters, reduction_mask,
        known_companion_mask, yx_center,
        overall_signal_mask, regressor_matrix,
        auxiliary_frame,
        noise_map, true_contrast, verbose):

    lp = lnprior(theta, mcmc_parameters)
    if not np.isfinite(lp):
        return -np.inf

    # TODO right_handed into parameters
    # relative_yx = [theta[1], theta[0]]
    relative_yx = rhophi_to_relative_yx([theta[0], theta[1]])
    planet_relative_yx_pos = np.array(relative_yx)
    data_minus_model = makesource.addsource(
        data, planet_relative_yx_pos, pa, flux_psf,
        norm=-theta[2], jitter=0, poisson_noise=False,
        right_handed=True, verbose=False)

    # Check if any pixels are negative after subtracting planet model
    # Reject those models
    number_of_values = np.prod(data_minus_model[:, reduction_mask].shape)
    number_of_values_below_zero = len(
        np.where(data_minus_model[:, reduction_mask] < 0.))

    if number_of_values_below_zero / number_of_values > 0.9:
        return -np.inf

    else:
        return lp + lnlike_for_negfake_corrnoise(
            theta,
            data=data_minus_model,
            reduction_parameters=reduction_parameters,
            planet_relative_yx_pos=planet_relative_yx_pos,
            reduction_mask=reduction_mask,
            known_companion_mask=known_companion_mask,
            yx_center=yx_center,
            overall_signal_mask=overall_signal_mask,
            regressor_matrix=regressor_matrix,
            auxiliary_frame=auxiliary_frame,
            noise_map=noise_map,
            true_contrast=true_contrast,
            verbose=verbose)
