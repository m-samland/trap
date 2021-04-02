"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from scipy import sparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from numba import jit
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


def compute_SVD(training_matrix, n_components, scaling='temp-median', compute_robust_lambda=False):
    """ Computes the singular value decomposition of the reference pixel matrix
        after scaling/centering it. It also returns the median squared amplitudes
        used instead of the singular values for the regression.

        Assumes the training matrix to be (n_time, n_pixel) in the context of this package.

    """
    if scaling is not None:
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
    if compute_robust_lambda:
        n_pixel = training_matrix.shape[-1] - 1
        lambdas = np.median(np.square(np.dot(U.T, training_matrix)), axis=1) * n_pixel
    else:
        lambdas = None

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
    # ipsh()
    P_sigma_squared = np.diag(np.linalg.inv(AVinvAT))

    return P, P_sigma_squared


@jit(nopython=True)
def solve_ordinary_lsq_optimized(design_matrix, data):

    A = design_matrix
    AVinvy = np.dot(A, data)
    AVinvAT = np.dot(A, A.T)

    P = np.linalg.solve(AVinvAT, AVinvy)
    P_sigma_squared = np.diag(np.linalg.inv(AVinvAT))

    return P, P_sigma_squared


@jit(nopython=True)
def multi_dot_three(A, B, C, out):  # , out):  # , out=None):
    """
    Find the best order for three arrays and do the multiplication.
    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    # out = np.ones((A.shape[0], C.shape[-1]), dtype='float64')

    if cost1 < cost2:
        return np.dot(np.dot(A, B), C, out=out)
    else:
        return np.dot(A, np.dot(B, C), out=out)


# np.random.seed(0)
# A = np.random.random((10000, 100))
# B = np.random.random((1000, 1000))
# Bsp = sparse.diags(np.diag(B), offsets=0)
# C = np.random.random((1000, 1))
# a0, a1b0 = A.shape
# b1c0, c1 = C.shape
# cost1 = a0 * b1c0 * (a1b0 + c1)
# # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
# cost2 = a1b0 * c1 * (a0 + b1c0)
# # print(cost1)
# # print(cost2)
# out = np.ones((A.shape[0], C.shape[-1]), dtype='float64')
# %timeit multi_dot_three(A, B, C])#, out)
#
#
# b = multi_dot_three(A, B, C, out)
# # 1.42 ms ± 16.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# a = multi_dot_three(A, B, C, out)
# d = multi_dot_three(A, B, C, out)
# %timeit np.dot(np.dot(A, B), C)
# %timeit np.dot(A, np.dot(B, C))
# %timeit np.dot(A, Bsp.dot(C))
#
# %timeit np.dot(np.dot(A, B), A.T)
# %timeit np.dot(A, np.dot(B, A.T))
# %timeit np.dot(A, Bsp.dot(A.T))

# %timeit np.dot(A.dot(Bsp), C)
# %timeit np.dot(np.dot(A, B), C)
# ## -- End pasted text --
# 292 µs ± 6.27 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

# In [53]: paste
# %timeit np.dot(A, np.dot(B, C))
# ## -- End pasted text --
# 18.1 µs ± 220 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# In [54]: paste
# %timeit np.dot(A, Bsp.dot(C))
# ## -- End pasted text --
# 11.7 µs ± 88.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# In [55]: paste
# %timeit np.dot(A.dot(Bsp), C)
# ## -- End pasted text --
# 6.17 s ± 55.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


@jit(nopython=True)
def solve_lsq_optimized(design_matrix, data, inverse_covariance_matrix):

    A = design_matrix

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
