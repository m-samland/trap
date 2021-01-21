"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.stats import mad_std, sigma_clip

from . import (image_coordinates, pca_regression, plotting_tools,
               regressor_selection)
from trap.utils import det_max_ncomp_specific
from .embed_shell import ipsh


class Result(object):
    """Reduction results for all pixels contained in the `reduction_mask`
    of a single tested planet position produced by `run_trap_with_model_temporal`
    or `run_trap_with_model_spatial` function.

    Handles combining the contrast and uncertainty from each pixel into
    a final value, identifying and masking pixels with bad residuals,
    and various plotting routines for visualizing the results for the
    tested pixels.

    """

    def __init__(
            self, model_cube=None, noise_model_cube=None,
            diagnostic_image=None, reduced_result=None,
            residuals=None, reduction_mask=None,
            data=None, number_of_pca_regressors=None,
            true_contrast=None, yx_center=None):
        """Initializer.

        Parameters
        ----------
        model_cube : array_like, optional
            Reconstructed complete model for each data vector fitted.
        noise_model_cube : array_like, optional
            Reconstructed systematics model for each data vector fitted.
        diagnostic_image : array_like, optional
            Image cube containing companion model fit coefficient,
            variance and SNR.
        reduced_result : array_like, optional
            Vector containing the companion model fit coefficient,
            variance and SNR for reduced pixels.
        residuals : array_like, optional
            Description of parameter `residuals`.
        reduction_mask : array_like, optional
            Boolean mask of data included in the reduction
            (\mathcal{P}_\mathcal{Y} in Samland et al. 2020)
        data : array_like, optional
            Image cube used as input for the reduction.
            Not neccessary other than for diagnostics.
        number_of_pca_regressors : integer
            The number of PCA regressors used.
        true_contrast : scalar, optional
            The true contrast of an injected signal.
        yx_center : array_like, optional
            The center position of the image as used in reduction.

        """

        self.data = data
        self.model_cube = model_cube
        self.noise_model_cube = noise_model_cube
        self.diagnostic_image = diagnostic_image
        self.reduced_result = reduced_result
        self.reduction_mask = reduction_mask
        self.n_reduction_pix = np.sum(reduction_mask)
        self.residuals = residuals
        self.number_of_pca_regressors = number_of_pca_regressors
        self.true_contrast = true_contrast
        self.measured_contrast = None
        self.contrast_uncertainty = None
        self.snr = None
        self.relative_uncertainty = None
        self.relative_deviation_from_true = None
        self.wrong_in_sigma = None
        if self.residuals is not None:
            self.compute_good_residual_mask()
        if reduced_result is not None:
            self.compute_contrast_weighted_average()
        if yx_center is None:
            self.yx_center = (self.model_cube.shape[-2],
                              self.model_cube.shape[-1])
        else:
            self.yx_center = yx_center

    def __str__(self):
        if self.true_contrast is not None:
            descr = "n_pix_reg: {}\ntrue contrast: {}\nmeasured contrast: {} +/- {}\nrel. uncertainty: {}\nSNR: {}\nrel. dev. from true: {}\nwrong in sigma: {}\n\n".format(
                self.number_of_pca_regressors,
                self.true_contrast,
                self.measured_contrast,
                self.contrast_uncertainty,
                self.relative_uncertainty,
                self.snr,
                self.relative_deviation_from_true,
                self.wrong_in_sigma)
        else:
            descr = "n_pix_reg: {}\nmeasured contrast: {} +/- {}\nrel. uncertainty: {}\nSNR: {}\n\n".format(
                self.number_of_pca_regressors,
                self.measured_contrast,
                self.contrast_uncertainty,
                self.relative_uncertainty,
                self.snr)

        return descr

    def __call__(self, sigma=None, clip_median=True, clip_std=True):
        if sigma is None:
            self.compute_contrast_weighted_average(mask_outliers=False)
        else:
            self.compute_good_residual_mask(sigma_for_outlier=sigma)
            self.compute_contrast_weighted_average(mask_outliers=True)
        print(self.__str__())

    def compute_good_residual_mask(self,
                                   sigma=3, clip_median=True, clip_std=False,
                                   clip_snr=False, clip_relative_deviation=False):
        mask = []
        if clip_median:  # Clip pixels based on time-domain median
            mask.append(
                sigma_clip(np.median(self.residuals, axis=1),
                           sigma=sigma, sigma_lower=None,
                           sigma_upper=None, maxiters=None, cenfunc=np.ma.median, stdfunc=mad_std,
                           axis=None, copy=True).mask)
        if clip_std:
            mask.append(
                sigma_clip(mad_std(self.residuals, axis=1),
                           sigma=sigma, sigma_lower=None,
                           sigma_upper=None, maxiters=None, cenfunc=np.ma.median, stdfunc=mad_std,
                           axis=None, copy=True).mask)
        if clip_relative_deviation:
            mask.append(
                sigma_clip(np.std(self.residuals, axis=1) / np.median(self.residuals, axis=1),
                           sigma=sigma, sigma_lower=None,
                           sigma_upper=None, maxiters=None, cenfunc=np.ma.median, stdfunc=mad_std,
                           axis=None, copy=True).mask)
        if clip_snr:
            mask.append(
                sigma_clip(self.reduced_result[:, 0] / np.sqrt(self.reduced_result[:, 1]),
                           sigma=sigma, sigma_lower=None,
                           sigma_upper=None, maxiters=None, cenfunc=np.ma.median, stdfunc=mad_std,
                           axis=None, copy=True).mask)

        mask = np.logical_or.reduce(mask)
        self.good_residual_mask = ~mask

    def compute_contrast_weighted_average(self, contrast=None, variance=None, mask_outliers=False):

        if contrast is None:
            contrast = self.reduced_result[:, 0]
        if variance is None:
            variance = self.reduced_result[:, 1]

        if self.reduced_result is None:
            pass

        if mask_outliers:
            mask = self.good_residual_mask
        else:
            mask = np.ones(self.residuals.shape[0], dtype=bool)
        self.measured_contrast = np.average(
            contrast[mask],
            weights=np.ones_like(contrast[mask]) / variance[mask])

        weight_for_variance = np.ones_like(contrast[mask]) / variance[mask]
        self.contrast_uncertainty = np.sqrt(1 / np.sum(weight_for_variance))
        self.snr = self.measured_contrast / self.contrast_uncertainty
        self.relative_uncertainty = 1. / self.snr
        if self.true_contrast is not None:
            self.relative_deviation_from_true = (self.measured_contrast - self.true_contrast) / self.true_contrast
            self.wrong_in_sigma = (self.measured_contrast - self.true_contrast) / self.contrast_uncertainty

    def plot_median_residual_image(self, savefig=False, outputdir=None):
        residual_image = np.zeros_like(self.reduction_mask, dtype='float32')
        residual_image[self.reduction_mask] = np.median(self.residuals, axis=1)
        mask = residual_image == 0
        residual_image[mask] = np.nan
        plotting_tools.plot_scale(
            residual_image, yx_coords=self.yx_center, point_size1=120,
            plot_star_not_circle=True, show=True)

    def make_mask_of_abnormal_residuals(self):
        residual_mask_image = np.zeros_like(self.reduction_mask, dtype='bool')
        residual_mask_image[self.reduction_mask] = ~self.good_residual_mask
        return residual_mask_image

    def plot_planet_coefficient(self, savefig=False, outputdir=None):
        plotting_tools.plot_scale(
            self.diagnostic_image[0], yx_coords=self.yx_center, point_size1=120,
            plot_star_not_circle=True, show=True)

    def plot_planet_snr(self, savefig=False, outputdir=None):
        plotting_tools.plot_scale(
            self.diagnostic_image[0] / np.sqrt(self.diagnostic_image[1]), yx_coords=self.yx_center, point_size1=120,
            plot_star_not_circle=True, show=True)

    def plot_gauss_for_residuals(self, show=True, savefig=False, outputdir=None):
        """ Eiffel tower plot. Gauss fit of distribution of residuals for each pixel."""

        from astropy.modeling import models, fitting
        x = np.linspace(-5, 5, 1000)
        for i in range(self.n_reduction_pix):
            g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
            fit_g = fitting.LevMarLSQFitter()
            hist = np.histogram(self.residuals[i, :], normed=True)
            centers = np.diff(hist[1]) / 2 + hist[1][:-1]
            g = fit_g(g_init, x=centers, y=hist[0])
            plt.plot(x, g(x))

        if savefig is True and outputdir is not None:
            plt.savefig(outputdir, dpi=300, bbox_inches='tight')
        if show is True:
            plt.show()

    def plot_distribution_of_residuals(self, show=True, savefig=False, outputdir=None, reject_outlier=False):
        """ Plot distribution of mean and std. dev. in time dimension of residuals."""
        if reject_outlier:
            mask = self.good_residual_mask
        else:
            mask = np.ones(self.n_reduction_pix, dtype=bool)

        mean_residual_pix = np.median(self.residuals, axis=1)
        mask = np.logical_and(mask, mean_residual_pix != 0)
        mean_residual_pix = mean_residual_pix[mask]
        stddev_residual_pix = np.std(self.residuals[mask], axis=1) / np.median(self.residuals[mask], axis=1)
        f, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharex=False)
        sns.despine(left=False)
        axes[0].set_title('Residual mean over time')
        axes[1].set_title('Residual std. dev over time')

        sns.distplot(
            mean_residual_pix,
            rug=True,
            color="g",
            ax=axes[0],
            rug_kws={"color": "gray"},
            label='mean')
        sns.distplot(
            stddev_residual_pix,
            rug=True,
            color="g",
            ax=axes[1],
            rug_kws={"color": "gray"},
            label='mean')

        plt.setp(axes, yticks=[])
        plt.tight_layout()
        if savefig is True and outputdir is not None:
            plt.savefig(outputdir, dpi=300, bbox_inches='tight')
        if show is True:
            plt.show()


def run_trap_with_model_temporal(
        data, model, pa, reduction_parameters,
        planet_relative_yx_pos, reduction_mask,
        yx_center=None,
        yx_center_injection=None,
        variance_reduction_area=None,
        regressor_matrix=None,
        signal_mask=None,
        known_companion_mask=None,
        opposite_mask=None,
        bad_pixel_mask=None,
        regressor_pool_mask=None,
        true_contrast=None,
        return_input_data=False,
        plot_all_diagnostics=False,
        verbose=False):
    """Core function of the TRAP analysis. Builds the temporal regression
    model and perform model fitting for all time series vectors in the
    `reduction_mask` as described in Samland et al. 2020.

    Parameters
    ----------
    data : array_like
        Temporal image cube. First axis is time.
    model : array_like, optional
        Temporal image cube containing the companion model.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    planet_relative_yx_pos : tuple
        Relative (yx)-position of pixel to be fit, with respect to `yx_center`
        or `yx_center_injection` if provided.
    reduction_mask : array_like
        Boolean mask of data included in the reduction
        (\mathcal{P}_\mathcal{Y} in Samland et al. 2020)
    yx_center : array_like, optional
        The center position of the image as used in reduction.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    variance_reduction_area : array_like, optional
        Variance for pixels in `reduction_mask`. Use if `include_noise`
        in `reduction_parameters` is True.
    regressor_matrix : array_like, optional
        Pre-computed regressor matrix. If not given regressor_matrix
        will be made using the provided masks.
    signal_mask : array_like
        Boolean mask of pixels affects by companion signal at
        `planet_relative_yx_pos`. Will be excluded from
        `regressor_pool_mask`.
    known_companion_mask : array_like, optional
        Boolean mask of pixels affects by known companion signal. Will be
        excluded from `regressor_pool_mask`.
    opposite_mask : array_like, optional
        Boolean mask of pixels opposite of the star from reduction area.
        Will be included in `regressor_pool_mask`.
    bad_pixel_mask : array_like, optional
        Boolean mask of bad pixels. Will be excluded from
        `regressor_pool_mask` and `reduction_mask`.
    regressor_pool_mask : array_like, optional
        Boolean mask of pixels to include as regressors.
        If None, `regressor_pool_mask` will be constructed from provided
        masks.
    true_contrast : scalar, optional
        The true contrast of an injected signal.
    return_input_data : boolean, optional
        Included input data in returned `Result`-object.
        Default is False.
    plot_all_diagnostics : boolean, optional
        If True, will produce diagnostic plots for central pixel of
        `reduction_mask`. Should only be used if `reduce_single_position`
        is True. Default is False.
    verbose : boolean, optional
        If True, some diagnostic output is printed in the terminal.
        Should only be used if `reduce_single_position` is True.
        Default is False.

    Returns
    -------
    ~trap.analyze_data.Result
        Regression results for all pixels contained in the `reduction_mask`
        for a single tested planet position.

    """

    local_model = reduction_parameters.local_temporal_model
    number_of_pca_regressors = reduction_parameters.number_of_pca_regressors
    pca_scaling = reduction_parameters.pca_scaling
    make_reconstructed_lightcurve = reduction_parameters.make_reconstructed_lightcurve
    compute_inverse_once = reduction_parameters.compute_inverse_once

    yx_dim = (data.shape[-2], data.shape[-1])

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    planet_absolute_yx_pos = image_coordinates.relative_yx_to_absolute_yx(planet_relative_yx_pos, yx_center)
    ntime = data.shape[0]

    if reduction_parameters.reduce_single_position and reduction_parameters.plot_all_diagnostics:
        if not os.path.exists('diagnostic_plots'):
            os.makedirs('diagnostic_plots')
        plt.close()
        plt.imshow(reduction_mask.astype('int'), origin='bottom')
        plt.savefig('diagnostic_plots/single_position_reduction_mask_test.jpg', dpi=300)
        plt.close()

    # Make array to contain stellar PSF model result
    model_cube = np.empty((ntime, yx_dim[0], yx_dim[1]))
    model_cube[:] = np.nan

    noise_model_cube = np.empty((ntime, yx_dim[0], yx_dim[1]))
    noise_model_cube[:] = np.nan

    if model is None:
        diagnostic_image = None
        reduced_result = None
    else:
        maximum_counts = np.max(model)
        # number_of_frames_affected = np.sum(model > 0, axis=0)
        total_flux_in_pixel = np.sum(model, axis=0)
        relative_flux_in_pixel = total_flux_in_pixel / np.max(total_flux_in_pixel)
        # excluding pixels with low contribution to over-all signal
        low_contribution_mask = relative_flux_in_pixel < reduction_parameters.threshold_pixel_by_contribution

        reduction_mask[low_contribution_mask] = False
        n_reduction_pix = np.sum(reduction_mask)

        if model is not None:
            diagnostic_image = np.empty((4, yx_dim[0], yx_dim[1],))
            diagnostic_image[:] = np.nan

            reduced_result = np.empty((n_reduction_pix, 3))
            reduced_result[:] = np.nan

        else:
            diagnostic_image = None
            reduced_result = None

        # coefficients = np.empty((n_reduction_pix, number_of_pca_regressors))
        # coefficients[:] = np.nan

    if regressor_matrix is None:
        if compute_inverse_once:
            if regressor_pool_mask is None:
                # One regressor pool for all iterations
                regressor_pool_mask_global = regressor_selection.make_regressor_pool_for_pixel(
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
            else:
                regressor_pool_mask_global = regressor_pool_mask.copy()
            if verbose:
                print("Number of reference pixel: {}".format(np.sum(regressor_pool_mask)))

            if not local_model:
                if number_of_pca_regressors != 0:
                    training_matrix = data[:, regressor_pool_mask_global]
                    B_full, lambdas_full, S_full, V_full = pca_regression.compute_SVD(
                        training_matrix, n_components=None, scaling=pca_scaling)
                    B = B_full[:, :number_of_pca_regressors]
                    # S = S_full[:number_of_pca_regressors]

    if plot_all_diagnostics:
        plotting_tools.plot_scale(
            np.median(data, axis=0),
            output_path='diagnostic_plots/pixel_to_fit.png',
            yx_coords=np.argwhere(reduction_mask),
            point_size1=0.5,
            plot_star_not_circle=False, scale='zscale', show=True)
        plt.imshow(regressor_pool_mask, origin='lower')
        plt.show()

    reduction_pix_indeces = np.argwhere(reduction_mask)
    test_pixel = np.round(np.mean(reduction_pix_indeces, axis=0))

    for idx, yx_pixel in enumerate(reduction_pix_indeces):
        # Pixel data to fit
        y = data[:, reduction_pix_indeces[idx, 0], reduction_pix_indeces[idx, 1]]
        data_range_to_fit = np.ones_like(y).astype('bool')

        # Lightcurve model fitting planet at pixel
        if model is not None:
            model_for_pixel = model[:, reduction_pix_indeces[idx, 0], reduction_pix_indeces[idx, 1]]
            if local_model:
                # Local models are experimental and should not be used without understanding
                # the code
                threshold_affected_data = 0.  # This threshold is for selecting which data to fit
                data_range_to_fit = model_for_pixel > maximum_counts * threshold_affected_data
                model_for_pixel = model_for_pixel[data_range_to_fit]

                maximum_number_of_components = np.sum(data_range_to_fit)
                number_of_pca_regressors = int(
                    np.round(reduction_parameters.number_of_components_fraction * maximum_number_of_components))
                if number_of_pca_regressors < 1:
                    number_of_pca_regressors = 1
                print("Number of components used: {}".format(number_of_pca_regressors))

        y = y[data_range_to_fit]

        if not compute_inverse_once:
            if local_model:
                regressor_pool_mask = regressor_pool_mask_global.copy()
                locally_unaffected = np.all(model[data_range_to_fit] == 0., axis=0)
                local_regressors = np.logical_and(reduction_mask, locally_unaffected)
                regressor_pool_mask[local_regressors] = True

            training_matrix = data[data_range_to_fit][:, regressor_pool_mask]  # .copy()

            if number_of_pca_regressors != 0:
                B_full, lambdas_full, S_full, V_full = pca_regression.compute_SVD(
                    training_matrix, n_components=None, scaling=pca_scaling)
                B = B_full[:, :number_of_pca_regressors]
        constant_offset = np.ones_like(y)
        if number_of_pca_regressors != 0:
            if model is None:
                A = np.hstack((B, constant_offset[:, None]))
            else:
                model_matrix = np.vstack((constant_offset, model_for_pixel))
                A = np.hstack((B, model_matrix.T))
        else:
            if model is None:
                A = constant_offset[:, None]
            else:
                model_matrix = np.vstack((constant_offset, model_for_pixel))
                A = model_matrix.T

        if reduction_parameters.include_noise:
            if variance_reduction_area is not None:
                variance_vector = variance_reduction_area[:, idx]
                inverse_covariance_matrix = np.identity(len(y)) * (1. / variance_vector)
            else:
                inverse_covariance_matrix = np.identity(len(y)) * (1. / y)
        else:
            inverse_covariance_matrix = None
        # ipsh()
        P_wo_marginalization, P_wo_sigma_squared = pca_regression.solve_linear_equation_simple(
            design_matrix=A.T,
            data=y,
            inverse_covariance_matrix=inverse_covariance_matrix)

        reconstructed_lightcurve = np.dot(A, P_wo_marginalization)

        if model is None:
            reconstructed_systematics = reconstructed_lightcurve
        else:
            reconstructed_systematics = np.dot(A[:, :-1], P_wo_marginalization[:-1])

        P = P_wo_marginalization
        P_sigma_squared = P_wo_sigma_squared

        if make_reconstructed_lightcurve:
            model_cube[data_range_to_fit, reduction_pix_indeces[idx, 0],
                       reduction_pix_indeces[idx, 1]] = reconstructed_lightcurve
            noise_model_cube[data_range_to_fit, reduction_pix_indeces[idx, 0],
                             reduction_pix_indeces[idx, 1]] = reconstructed_systematics
        else:
            model_cube = None
            noise_model_cube = None

        if plot_all_diagnostics and model is not None:
            if np.array_equal(yx_pixel, test_pixel):
                print("Selected pixel: {}".format(yx_pixel))
                mask_coordinates = np.argwhere(regressor_pool_mask)

                plotting_tools.plot_scale(np.median(data, axis=0), yx_coords=mask_coordinates,
                                          yx_coords2=[yx_pixel],  # , yx_center],
                                          output_path='diagnostic_plots/regressor_selection_test2.pdf',
                                          point_size1=0.5,
                                          point_size2=3,
                                          show_cb=False,
                                          show=True)

                plt.close()
                plt.plot(y, label='data for pixel')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/pixel_data_test.jpg', dpi=300)

                plt.close()
                plt.plot(y, label='data for pixel')
                plt.plot(model_for_pixel * reduction_parameters.true_contrast,
                         color='green', label='planet contribution')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/pixel_with_planet_test.jpg', dpi=300)

                plt.close()
                fig, ax = plt.subplots(figsize=(9, 2))
                ax.plot(model_for_pixel * reduction_parameters.true_contrast,
                        label='planet contribution')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/planet_model_test.jpg', dpi=300)

                plt.close()
                fig, ax = plt.subplots(figsize=(9, 2))
                ax.plot(y, label='data for pixel')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/data_for_pixel_long_test.jpg', dpi=300)

                plt.close()
                plt.style.use("paper")
                fig = plt.figure()
                ax = plt.subplot(111)
                alpha = 0.7
                ax.plot(y, label='data', color='blue', alpha=alpha)
                ax.plot(reconstructed_lightcurve, label='complete model', color='orange', alpha=alpha)
                ax.plot(reconstructed_systematics, label='reconstructed systematics',
                        color='mediumseagreen', alpha=alpha)
                ax.plot(model_for_pixel * P_wo_marginalization[-1],
                        label='planet model fit', color='black', alpha=alpha)
                ax.set_xlabel('Frame')
                ax.set_ylabel('Counts (ADU)')
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                          ncol=2, fancybox=False, shadow=False)
                plt.savefig('diagnostic_plots/fitted_data_test.pdf', bbox_inches='tight')  # dpi=300)

                plt.close()
                plt.plot(y - model_for_pixel * reduction_parameters.true_contrast,
                         label='data minus planet model')
                plt.plot(reconstructed_systematics, label='fit of systematics')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/data_minus_model_fit_test.jpg', dpi=300)

                plt.close()
                plt.plot(y - reconstructed_lightcurve, label='residuals')
                plt.xlabel('Frame')
                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.savefig('diagnostic_plots/residuals_test.jpg', dpi=300)

                fig = plt.figure()
                ax = plt.subplot(111)
                for i in range(5):
                    ax.plot(B[:, i] * lambdas_full[i] / np.cumsum(lambdas_full)
                            [-1] + 0.1 * i, label=i)  # + 1 * i, label=i)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='upper right')
                plt.title('Principal component lightcurves')
                # plt.ylim(-1, 5)
                plt.xlim(0, 300)
                plt.savefig('diagnostic_plots/principal_component_lightcurves_test.png', dpi=300)

        if model is not None:
            diagnostic_image[:, reduction_pix_indeces[idx, 0], reduction_pix_indeces[idx, 1]] = (
                P[-1], P_sigma_squared[-1], P[-1] / P_sigma_squared[-1], P[0])
            reduced_result[idx] = (P[-1], P_sigma_squared[-1], P[-1] / P_sigma_squared[-1])

    # This is not correct for local model, the range to fit is different for each pixel and should be added to
    # a uniform cube, and later take into account ignoring Nans!
    if make_reconstructed_lightcurve:
        residuals = (data[data_range_to_fit][:, reduction_mask] - model_cube[data_range_to_fit][:, reduction_mask]).T
    else:
        residuals = None

    if not reduction_parameters.reduce_single_position:
        model_cube = None

    if return_input_data:
        data_save = data
    else:
        data_save = None

    result = Result(
        data=data_save,
        model_cube=model_cube,
        noise_model_cube=noise_model_cube,
        diagnostic_image=diagnostic_image,
        reduced_result=reduced_result,
        reduction_mask=reduction_mask,
        residuals=residuals,
        number_of_pca_regressors=number_of_pca_regressors,
        true_contrast=true_contrast,
        yx_center=yx_center)

    return result


def run_trap_with_model_spatial(
        data, model, pa,
        reduction_parameters,
        planet_relative_yx_pos,
        reduction_mask,
        yx_center=None,
        yx_center_injection=None,
        variance_reduction_area=None,
        bad_pixel_mask=None,
        true_contrast=None,
        training_data=None,
        return_input_data=False,
        verbose=False):
    """Core function of the TRAP analysis. Builds the spatial regression
    model and perform model fits for all image vector in the
    `reduction_mask` as described in Samland et al. 2020.

    Parameters
    ----------
    data : array_like
        Temporal image cube. First axis is time.
    model : array_like, optional
        Temporal image cube containing the companion model.
    pa : array_like
        Vector containing the parallactic angles for each frame.
    reduction_parameters : `~trap.parameters.Reduction_parameters`
        A `~trap.parameters.Reduction_parameters` object all parameters
        necessary for the TRAP pipeline.
    planet_relative_yx_pos : tuple
        Relative (yx)-position of pixel to be fit, with respect to `yx_center`
        or `yx_center_injection` if provided.
    reduction_mask : array_like
        Boolean mask of data included in the reduction
        (\mathcal{P}_\mathcal{Y} in Samland et al. 2020)
    yx_center : array_like, optional
        The center position of the image as used in reduction.
    yx_center_injection : array_like
        Array containing yx_center to be used for forward model position.
    variance_reduction_area : array_like, optional
        Variance for pixels in `reduction_mask`. Use if `include_noise`
        in `reduction_parameters` is True.
    bad_pixel_mask : array_like, optional
        Boolean mask of bad pixels. Will be excluded from
        `regressor_pool_mask` and `reduction_mask`.
    true_contrast : scalar, optional
        The true contrast of an injected signal.
    return_input_data : boolean, optional
        Included input data in returned `Result`-object.
        Default is False.
    verbose : boolean, optional
        If True, some diagnostic output is printed in the terminal.
        Should only be used if `reduce_single_position` is True.
        Default is False.

    Returns
    -------
    ~trap.analyze_data.Result
        Regression results for all images in the observation sequence
        for a single tested planet position produced.

    """

    pca_scaling = reduction_parameters.pca_scaling
    make_reconstructed_lightcurve = reduction_parameters.make_reconstructed_lightcurve

    yx_dim = (data.shape[-2], data.shape[-1])
    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    ntime = data.shape[0]

    local_model = reduction_parameters.local_spatial_model
    # Local model here means only area where PSF is located
    if local_model:
        n_reduction_pix = np.sum(model[0] > 0)  # PSF model always same size
    else:
        n_reduction_pix = np.sum(reduction_mask)  # PSF model always same size

    # Make array to contain stellar PSF model result
    model_cube = np.empty((ntime, yx_dim[0], yx_dim[1]))
    model_cube[:] = np.nan

    noise_model_cube = np.empty((ntime, yx_dim[0], yx_dim[1]))
    noise_model_cube[:] = np.nan

    # Contrast, uncertainty, snr for each fit
    reduced_result = np.empty((ntime, 3))
    reduced_result[:] = np.nan

    psf_amplitude = np.zeros(ntime)
    psf_sigma_squared = np.zeros(ntime)
    sigma_squared_systematics = np.zeros(ntime)

    # Compute other times to use for model
    separation = np.sqrt(
        planet_relative_yx_pos[0]**2 + planet_relative_yx_pos[1]**2)

    max_ncomp, time_masks = det_max_ncomp_specific(
        r_planet=separation,
        fwhm=reduction_parameters.fwhm,
        delta=reduction_parameters.protection_angle,
        pa=pa)

    for idx, psf_model_frame in enumerate(model):
        if local_model:
            reduction_mask = psf_model_frame > 0.
        if bad_pixel_mask is not None:
            reduction_mask = np.logical_and(reduction_mask, ~bad_pixel_mask)

        time_mask = time_masks[idx]
        # Pixel data to fit
        y = data[idx, reduction_mask]  # Array of pixels affected in frame
        model_for_pixel = model[idx, reduction_mask]

        # Add constant offset to fit in model
        constant_offset = np.ones_like(y)
        model_matrix = np.vstack((constant_offset, model_for_pixel))
        # Transpose data with respect to temporal approach
        # flux_overlap_fraction = compute_flux_overlap(idx, model)
        # time_mask = flux_overlap_fraction < reduction_parameters.spatial_exclusion_flux_overlap
        if np.sum(time_mask) < 3:  # require at least 5 reference frames
            print('Returning None for lack of frames')
            return None

        if training_data is None:
            training_matrix = data[time_mask][:, reduction_mask]  # .copy()
        else:
            training_matrix = training_data[time_mask][:, reduction_mask]
        # Transpose training matrix as opposed to temporal approach
        B_full, lambdas_full, S_full, V_full = pca_regression.compute_SVD(
            training_matrix.T, n_components=None, scaling=pca_scaling)
        # cummulative_variance = np.cumsum(S_full) / np.sum(S_full)

        # try:
        #     number_of_pca_regressors = np.where(
        #         cummulative_variance < reduction_parameters.spatial_variance_explained)[0][-1] + 1
        # except:
        #     print('Returning None because nPCA could not be determined')
        #     return None

        number_of_pca_regressors = int(np.round(max_ncomp[idx][0] * reduction_parameters.spatial_components_fraction))
        B = B_full[:, :number_of_pca_regressors]
        # S = S_full[:number_of_pca_regressors]
        # V = V_full[:number_of_pca_regressors, :]

        # if reduction_parameters.verbose:
        #     # print("Time: {} of {}".format(idx + 1, model.shape[0]))
        #     if idx == int(len(model) // 2):
        #         plt.close()
        #         plt.plot(np.cumsum(S_full) / np.sum(S_full))
        #         plt.show()

        if make_reconstructed_lightcurve:
            if model is None:
                A = np.hstack((B, constant_offset[:, None]))
            else:
                A = np.hstack((B, model_matrix.T))

            if reduction_parameters.include_noise:
                if variance_reduction_area is not None:
                    variance_vector = variance_reduction_area[idx]
                    inverse_covariance_matrix = np.identity(len(y)) * (1. / variance_vector)
                else:
                    inverse_covariance_matrix = np.identity(len(y)) * (1. / y)
                P_wo_marginalization, P_wo_sigma_squared = pca_regression.solve_linear_equation_simple(
                    design_matrix=A.T,
                    data=y,
                    inverse_covariance_matrix=inverse_covariance_matrix)
            else:
                P_wo_marginalization, P_wo_sigma_squared = pca_regression.solve_linear_equation_simple(
                    design_matrix=A.T,
                    data=y,
                    inverse_covariance_matrix=None)

            reconstructed_lightcurve = np.dot(A, P_wo_marginalization)

            if model is None:
                reconstructed_systematics = reconstructed_lightcurve
            else:
                reconstructed_systematics = np.dot(A[:, :-1], P_wo_marginalization[:-1])

        P = P_wo_marginalization
        P_sigma_squared = P_wo_sigma_squared
        if model is None:
            pass

        if make_reconstructed_lightcurve:
            model_cube[idx, reduction_mask] = reconstructed_lightcurve
            noise_model_cube[idx, reduction_mask] = reconstructed_systematics
        else:
            model_cube = None
            noise_model_cube = None

        if model is not None:
            reduced_result[idx] = (P[-1], P_sigma_squared[-1], P[-1] / P_sigma_squared[-1])

        refined_speckle_subtraction = np.empty((yx_dim[0], yx_dim[1]))
        refined_speckle_subtraction[:] = np.nan
        refined_speckle_subtraction[reduction_mask] = data[idx][reduction_mask] - reconstructed_systematics

        psf_mask = model[idx] > 0.
        non_psf_mask = np.logical_and(reduction_mask, ~psf_mask)
        psf_model = np.expand_dims(model[idx][psf_mask], 1)
        data_to_fit = refined_speckle_subtraction[psf_mask]
        finite_mask = np.isfinite(data_to_fit)
        systematic_residuals = mad_std(data[idx][non_psf_mask])

        inverse_covariance_matrix_systematics = np.identity(
            len(data_to_fit[finite_mask])) * (1 / (systematic_residuals**2))
        if variance_reduction_area is not None:
            noise_psf = variance_reduction_area[idx][psf_mask[reduction_mask]]
            inverse_covariance_matrix_total = np.identity(
                len(data_to_fit[finite_mask])) * (1 / (systematic_residuals**2 + noise_psf))
        else:
            inverse_covariance_matrix_total = inverse_covariance_matrix_systematics

        _, sigma_squared_systematics[idx] = pca_regression.solve_linear_equation_simple(
            design_matrix=psf_model[finite_mask].T,
            data=data_to_fit[finite_mask],
            inverse_covariance_matrix=inverse_covariance_matrix_systematics)
        psf_amplitude[idx], psf_sigma_squared[idx] = pca_regression.solve_linear_equation_simple(
            design_matrix=psf_model[finite_mask].T,
            data=data_to_fit[finite_mask],
            inverse_covariance_matrix=inverse_covariance_matrix_total)

    # Residuals are transposed with respect to temporal model
    if make_reconstructed_lightcurve:
        residuals = (data[:][:, reduction_mask] - model_cube[:][:, reduction_mask])
    else:
        residuals = None

    if not reduction_parameters.reduce_single_position:
        model_cube = None

    if return_input_data:
        data_save = data
    else:
        data_save = None

    result = Result(
        data=data_save,
        model_cube=model_cube,
        noise_model_cube=noise_model_cube,
        diagnostic_image=None,
        reduced_result=reduced_result,
        reduction_mask=reduction_mask,
        residuals=residuals,
        number_of_pca_regressors=number_of_pca_regressors,
        true_contrast=true_contrast,
        yx_center=yx_center)

    result.psf_amplitude = psf_amplitude
    result.psf_sigma_squared = psf_sigma_squared
    result.sigma_squared_systematics = sigma_squared_systematics

    return result
