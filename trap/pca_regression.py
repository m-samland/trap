"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from sklearn import preprocessing

from . import regressor_selection
from .embed_shell import ipsh

# __all__ = ['matrix_scaling', 'compute_SVD', 'compute_V_inverse',
#    'solve_linear_equation_simple', 'detect_bad_frames']


def matrix_scaling(matrix, scaling):
    """ Scales a matrix using sklearn.preprocessing.scale function.

    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
    With None, no scaling is performed on the input data before SVD. With
    "temp-mean" then temporal px-wise mean subtraction is done, with
    "spat-mean" then the spatial mean is subtracted, with "temp-standard"
    temporal mean centering plus scaling to unit variance is done and with
    "spat-standard" spatial mean centering plus scaling to unit variance is
    performed. Using median instead of mean and quartile instead of standard will
    trigger the robust implementations of those algorithms.
    """

    if scaling is None:
        pass
    if scaling == 'temp-mean':
        matrix = preprocessing.scale(matrix, with_mean=True, with_std=False)
    elif scaling == 'spat-mean':
        matrix = preprocessing.scale(matrix, with_mean=True, with_std=False, axis=1)
    elif scaling == 'temp-standard':
        matrix = preprocessing.scale(matrix, with_mean=True, with_std=True)
    elif scaling == 'spat-standard':
        matrix = preprocessing.scale(matrix, with_mean=True, with_std=True, axis=1)
    elif scaling == 'temp-median':
        matrix = preprocessing.robust_scale(matrix, with_centering=True, with_scaling=False)
    elif scaling == 'spat-median':
        matrix = preprocessing.robust_scale(matrix, with_centering=True, with_scaling=False, axis=1)
    elif scaling == 'temp-quartile':
        matrix = preprocessing.robust_scale(matrix, with_centering=True, with_scaling=True)
    elif scaling == 'spat-quartile':
        matrix = preprocessing.robust_scale(matrix, with_centering=True, with_scaling=True, axis=1)
    else:
        raise ValueError('Scaling mode not recognized')

    return matrix


def compute_SVD(training_matrix, n_components, scaling='temp-median'):
    """ Computes the singular value decomposition of the reference pixel matrix
        after scaling/centering it. It also returns the median squared amplitudes
        used instead of the singular values for the regression.

        Assumes the training matrix to be (n_time, n_pixel) in the context of this package.

    """

    training_matrix = matrix_scaling(training_matrix, scaling=scaling)  # 'temp-standard
    U, S, V = np.linalg.svd(training_matrix, full_matrices=False)
    # Uf, Sf, Vf = np.linalg.svd(training_matrix, full_matrices=True)
    # Ut, St, Vt = np.linalg.svd(training_matrix.T, full_matrices=True)
    # bad_frames = detect_bad_frames(
    #     eigenvector_matrix=U, bad_frame_mad_cutoff=6, plot_bad_frames=True,
    #     number_of_components_to_consider=50)
    # If matrix is (time, pixels), then U zero dimension is also time dimension, 1st is eigenvector
    # np.save('eigenvalues_quart_ann9.npy', S)
    # from scipy.linalg import eigh
    # cov = np.dot(training_matrix, training_matrix.T)
    # w, v2 = linalg.eigh(cov, lower=True, eigvals_only=False, overwrite_a=False,
    #                     overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True)
    # w = w[::-1]
    # v2 = v2[:,::-1] #shape (n_ref, ncomp)

    # S**2 == w

    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        V = V[:n_components, :]
    # lambdas are median squared amplitudes ~ similar to singular values but should be more robust
    n_pixel = training_matrix.shape[-1] - 1
    lambdas = np.median(np.square(np.dot(U.T, training_matrix)), axis=1) * n_pixel

    return U, lambdas, S, V


def compute_V_inverse(training_matrix, lambdas, variance, variance_prior_scaling=1.):
    """ Computes inverse matrix needed to perform regression marginalized over the
        systematics model.

        To marginalize over the eigen-lightcurves we need to solve
        x = (A.T V^(-1) A)^(-1) * (A.T V^(-1) y), where V = C + B.T Lambda B,
        with B matrix containing the eigenlightcurves and lambda
        the median squared amplitudes of the eigenlightcurves.
        For the inversion of V we use the 'matrix inversion lemma'.

        This function is currently not used.

    """

    Cinv = np.diag(1. / variance)
    B = training_matrix

    lambdas *= variance_prior_scaling
    lambda_inv = 1. / lambdas

    # Invert V by matrix inversion lemma
    # V = C + np.dot(B, lambdas[:, None]*B.T)

    lambdaplusBTCinvB_inv = np.linalg.inv(np.diag(lambda_inv) + np.dot(B.T, B / variance[:, None]))
    V_inverse = Cinv - np.linalg.multi_dot((Cinv, B, lambdaplusBTCinvB_inv, B.T, Cinv))

    return V_inverse


def solve_linear_equation_simple(design_matrix, data, inverse_covariance_matrix=None):
    # if data.dtype != 'float64':
    #     data = data.astype('float64')
    # if design_matrix.dtype != 'float64':
    #     design_matrix = design_matrix.astype('float64')

    A = design_matrix
    if inverse_covariance_matrix is None:
        AVinvy = np.dot(A, data)
        AVinvAT = np.dot(A, A.T)
    else:
        Vinv = inverse_covariance_matrix
        AVinvy = np.linalg.multi_dot((A, Vinv, data))
        AVinvAT = np.linalg.multi_dot((A, Vinv, A.T))
    P = np.linalg.solve(AVinvAT, AVinvy)
    # Plstsq = np.linalg.lstsq(AVinvAT, AVinvy)
    P_sigma_squared = np.diag(np.linalg.inv(AVinvAT))

    return P, P_sigma_squared


def detect_bad_frames(
        eigenvector_matrix, bad_frame_mad_cutoff=8, plot_bad_frames=False,
        number_of_components_to_consider=20):
    """ Bad values in the lightcurves used to construct the PCA tend to be incorporated
        into individual principal components, because they cannot be explained by
        normal systematics.
        This routine flags points in the time series where eigenlightcurves have a
        value "bad_frame_mad_cutoff" times the median absolute
        deviation and returns the time series indices.

    """

    B = eigenvector_matrix

    outliers = np.zeros_like(B, dtype=bool)

    for i in range(number_of_components_to_consider):
        outliers[i] = np.abs(B[:, i] - np.median(B[:, i])) > mad_std(B[:, i]) * bad_frame_mad_cutoff

    flat_outliers = np.sum(outliers, axis=0)

    if plot_bad_frames is True:
        # Noise eigenlightcurves
        for i in range(number_of_components_to_consider):
            plt.plot(B[:, i] + 1 * i, label=i)
        plt.legend()
        plt.ylim(-1, number_of_components_to_consider)
        plt.show()

        # Significant outliers
        for i in range(number_of_components_to_consider):
            plt.plot(outliers[i, :] + 1. * i, label=i)
        plt.legend()
        plt.ylim(-1, number_of_components_to_consider)
        plt.show()

    return np.argwhere(flat_outliers)


def remove_bad_frames(
        data, yx_pixel, yx_dim, yx_center, annulus_width, annulus_offset,
        overall_signal_mask, known_companion_mask, target_pix_mask_radius,
        use_relative_position, remove_bad_frames_iterations=3,
        bad_frame_mad_cutoff=8, number_of_components_to_consider=20,
        scaling='temp-median', plot_bad_frames=False, verbose=False):

    bad_frame_indices = []
    for i in range(remove_bad_frames_iterations):
        mask_for_bad_frame_detection = regressor_selection.make_regressor_pool_for_pixel(
            yx_pixel=yx_pixel,
            yx_dim=yx_dim, yx_center=yx_center,
            annulus_width=annulus_width, annulus_offset=annulus_offset,
            overall_signal_mask=overall_signal_mask,
            known_companion_mask=known_companion_mask,
            target_pix_mask_radius=target_pix_mask_radius,
            use_relative_position=use_relative_position)

        training_matrix = data[:, mask_for_bad_frame_detection]  # .copy()
        B, lambdas, _, _ = compute_SVD(training_matrix, n_components=None, scaling=scaling)

        bad_frame_indices_in_iteration = detect_bad_frames(
            B,
            bad_frame_mad_cutoff=bad_frame_mad_cutoff,
            plot_bad_frames=plot_bad_frames)

        bad_frame_indices.append(bad_frame_indices_in_iteration)
        data = np.delete(data, bad_frame_indices_in_iteration, axis=0)
        if verbose:
            print('Iteration: {} Bad frames: {}'.format(i, bad_frame_indices_in_iteration))

    return data, bad_frame_indices
