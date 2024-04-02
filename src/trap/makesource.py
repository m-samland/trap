"""
Routines used in TRAP

@author: Tim Brandt, Matthias Samland
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from numba import jit, njit
from numba.types import float64, int64
from numpy.random import poisson
from photutils.datasets import make_gaussian_sources_image, make_random_gaussians_table
from scipy import ndimage
from scipy.ndimage import shift, spline_filter

from trap.embed_shell import ipsh

# @jit(float64[:](float64[:], int64, float64[:]), nopython=True)
# def rnd1(x, decimals, out):
#     return np.round_(x, decimals, out)


def yx_position_in_cube(
        image_size, pos, pa, image_center=None, yx_anamorphism=[1., 1.],
        right_handed=True):
    """Compute yx position of source in each frame of ADI sequence.

    Parameters
    ----------
    image_size : tuple
        Size of image in y and x (beware order).
    pos : tuple
        (y, x)-Position of object in derotated cube.
    pa : type
        Vector of parallactic angles.
    image_center : None, tuple, list of tuples
        If None assume center to be floor of half of image size.
        Otherwise use value provide in (y, x)-tuple or list of
        (y, x)-tuples
    yx_anamorphism : tuple
        Relative position of model will be divided by this value to correct
        for anamorphism of instrument.
    right_handed : bool
        Determines rotation direction. If True, rawdata FoV rotates counter-
        clockwise, e.g. True for SPHERE-DC, False for ISPY-NACO.

    Returns
    -------
    array
        Coordinates in each frame of sequence.

    """
    if not right_handed:
        pa = -pa
    if image_center is not None:
        image_center = np.array(image_center)

    pa = pa * (2. * np.pi / 360.)
    if image_center is None:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * \
            1 / yx_anamorphism[1] + image_size[-1] // 2
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * \
            1 / yx_anamorphism[0] + image_size[-2] // 2
    elif image_center.ndim == 1:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[0]
    elif image_center.ndim == 2:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[:, 1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[:, 0]
    else:
        raise ValueError('Invalid value for image_center.')

    coords = np.vstack((y, x)).T
    return coords


# @njit
def yx_position_in_cube_optimized(
        image_size, pos, pa, image_center, yx_anamorphism=np.array([1., 1.]),
        right_handed=True):
    """Compute yx position of source in each frame of ADI sequence.

    Parameters
    ----------
    image_size : array_like
        Size of image in y and x (beware order).
    pos : array_like
        (y, x)-Position of object in derotated cube.
    pa : array_like
        Vector of parallactic angles.
    image_center : array_like
        Value provide in (y, x)-tuple or list of (y, x)-tuples
    yx_anamorphism : tuple
        Relative position of model will be divided by this value to correct
        for anamorphism of instrument.
    right_handed : bool
        Determines rotation direction. If True, rawdata FoV rotates counter-
        clockwise, e.g. True for SPHERE-DC, False for ISPY-NACO.

    Returns
    -------
    array
        Coordinates in each frame of sequence.

    """
    if not right_handed:
        pa = -pa

    pa = pa * (2. * np.pi / 360.)
    if image_center is None:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * \
            1 / yx_anamorphism[1] + image_size[-1] // 2
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * \
            1 / yx_anamorphism[0] + image_size[-2] // 2
    elif image_center.ndim == 1:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[0]
    elif image_center.ndim == 2:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[:, 1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[:, 0]
    else:
        raise ValueError('Invalid value for image_center.')

    coords = np.vstack((y, x)).T
    return coords


def prepare_injection(yx_position, image_shape, psf_shape):

    # yx_position_rounded = np.empty_like(yx_position)
    # rnd1(yx_position, 0, yx_position_rounded)
    # yx_position_rounded = yx_position_rounded.astype('int')
    yx_position_rounded = yx_position.round().astype('int')

    y, x = yx_position_rounded.T
    ydiff, xdiff = (yx_position - yx_position_rounded).T

    psf_half_size_y = int(psf_shape[0] // 2.)
    psf_half_size_x = int(psf_shape[1] // 2.)
    x1 = x - psf_half_size_x
    x2 = x + psf_half_size_x + 1
    y1 = y - psf_half_size_y
    y2 = y + psf_half_size_y + 1

    return xdiff, ydiff, x1, x2, y1, y2


def inject_signal(flux_arr, xdiff, ydiff, x1, x2, y1, y2, psf_model, norm,
                  subpixel=True, remove_interpolation_artifacts=False, copy=False):

    if copy:
        injected_cube = flux_arr.copy()
    else:
        injected_cube = flux_arr

    if remove_interpolation_artifacts:
        psf_size = psf_model.shape[-1]
        oversampling = 1
        x = np.arange(0, psf_size, 1 / oversampling)
        y = np.arange(0, psf_size, 1 / oversampling).reshape(-1, 1)
        y_signal, x_signal = (psf_size // 2., psf_size // 2.)
        dist_from_signal = np.sqrt((x - x_signal)**2 + (y - y_signal)**2)
        psf_mask = dist_from_signal > psf_size // 2

    else:
        psf_mask = np.zeros_like(psf_model).astype('bool')
        # flux_psf_form[~psf_mask] = 0
    ntimes = xdiff.shape[0]
    shifted_psf = np.empty_like(psf_model)

    if flux_arr.ndim == 2:
        if subpixel:
            for idx in range(ntimes):
                shift(input=psf_model,
                      shift=(ydiff[idx], xdiff[idx]),
                      output=shifted_psf,
                      order=3,
                      mode='constant',
                      cval=0.,
                      prefilter=False)
                shifted_psf[psf_mask] = 0
                injected_cube[y1[idx]:y2[idx], x1[idx]:x2[idx]] += shifted_psf * norm[idx]
        else:
            for idx in range(ntimes):
                injected_cube[y1[idx]:y2[idx], x1[idx]:x2[idx]] += psf_model * norm[idx]

    elif flux_arr.ndim == 3:
        if subpixel:
            for idx in range(ntimes):
                shift(input=psf_model,
                      shift=(ydiff[idx], xdiff[idx]),
                      output=shifted_psf,
                      order=3,
                      mode='constant',
                      cval=0.,
                      prefilter=False)
                shifted_psf[psf_mask] = 0
                injected_cube[idx, y1[idx]:y2[idx], x1[idx]:x2[idx]] += shifted_psf * norm[idx]
        else:
            for idx in range(ntimes):
                injected_cube[idx, y1[idx]:y2[idx], x1[idx]:x2[idx]] += psf_model * norm[idx]
    else:
        raise ValueError("Injection only possible for 2D or 3D data.")

    return injected_cube


def inject_model_into_data(
        flux_arr, pos, pa,
        psf_model,
        image_center,
        norm,
        yx_anamorphism=np.array([1., 1.]),
        right_handed=True,
        subpixel=True,
        remove_interpolation_artifacts=False,
        copy=False):

    image_shape = flux_arr.shape[-2:]
    psf_shape = psf_model.shape

    yx_position = yx_position_in_cube_optimized(
        image_shape, pos, pa, image_center, yx_anamorphism=yx_anamorphism,
        right_handed=right_handed)

    xdiff, ydiff, x1, x2, y1, y2 = prepare_injection(yx_position, image_shape, psf_shape)

    flux_arr = inject_signal(
        flux_arr=flux_arr, xdiff=xdiff, ydiff=ydiff, x1=x1, x2=x2, y1=y1, y2=y2,
        psf_model=psf_model,
        norm=norm,
        subpixel=subpixel,
        remove_interpolation_artifacts=remove_interpolation_artifacts,
        copy=copy)

    return flux_arr


def addsource(flux_arr, pos, pa, psf_arr, image_center=None,
              norm=None, jitter=0, poisson_noise=False,
              yx_anamorphism=[1., 1.], right_handed=True, subpixel=True,
              copy=True, verbose=False):
    """Add PSF stamps for one position in derotated frame for the whole,
    ADI sequence. The routine makes use of Cutout2D from astropy.


    Parameters
    ----------
    flux_arr : array
        ADI sequence image cube.
    pos : tuple
        (y, x)-Position of object in derotated cube.
    pa : type
        Vector of parallactic angles.
    psf_arr : type
        Description of parameter `psf_arr`.
    image_center : None, tuple, list of tuples
        If None assume center to be floor of half of image size.
        Otherwise use value provide in (y, x)-tuple or list of
        (y, x)-tuples
    norm : type
        Description of parameter `norm`.
    jitter : type
        Description of parameter `jitter`.
    poisson_noise : type
        Description of parameter `poisson_noise`.
    yx_anamorphism : tuple
        Relative position of model will be divided by this value to correct
        for anamorphism of instrument.
    right_handed : bool
        Determines rotation direction. True for SPHERE,
        False for NACO.
    subpixel : bool
        If True, take into account subpixel shifts.
    verbose : bool
        If True print output.

    Returns
    -------
    array
        ADI sequence image cube with added PSF.

    """

    if copy:
        flux = flux_arr.copy()
    else:
        flux = flux_arr

    # try:
    #     check_if_iterable = iter(norm)
    # except TypeError as te:
    #     norm = np.array([norm])

    psf_arr = psf_arr.copy()  # * norm
    if subpixel:
        filtered_psf = spline_filter(psf_arr)
    else:
        filtered_psf = psf_arr
    stamp_size = psf_arr.shape[-1]
    yx_position = yx_position_in_cube((flux.shape[-2], flux.shape[-1]),
                                      pos, pa, image_center, yx_anamorphism,
                                      right_handed)
    for idx, position in enumerate(yx_position):
        cutout = Cutout2D(flux[idx], (position[-1], position[-2]), stamp_size, copy=False)
        if subpixel:
            subpixel_shift = np.array(cutout.position_original) - \
                np.array(cutout.input_position_original)
            shifted_psf = shift(filtered_psf, (-subpixel_shift[-1], -subpixel_shift[-2]),
                                output=None, order=3,
                                mode='constant', cval=0., prefilter=False)
            if len(norm) == 1:
                cutout.data += shifted_psf * norm[0]
            elif len(norm) > 1:
                cutout.data += shifted_psf * norm[idx]
        else:
            if len(norm) == 1:
                cutout.data += psf_arr * norm[0]
            elif len(norm) > 1:
                cutout.data += psf_arr * norm[idx]
    return flux


def extract_stamps(flux_arr, pos, pa, stamp_size, image_center=None,
                   yx_anamorphism=[1., 1.], right_handed=True,
                   shift_order=1, plot=False):
    """Short summary.

    Parameters
    ----------
    flux_arr : array
        ADI sequence image cube.
    pos : tuple
        (y, x)-Position of object in derotated cube.
    pa : type
        Vector of parallactic angles.
    stamp_size : tuple
        Size of stamp to be extracted.
    image_center : None, tuple, list of tuples
        If None assume center to be floor of half of image size.
        Otherwise use value provide in (y, x)-tuple or list of
        (y, x)-tuples
    right_handed : bool
        Determines rotation direction. True for SPHERE,
        False for NACO.
    plot : bool
        Show extracted stamps.

    Returns
    -------
    tuple
        Array of stamp images and subpixel shifts.

    """

    yx_position = yx_position_in_cube((flux_arr.shape[-2], flux_arr.shape[-1]),
                                      pos, pa, image_center, yx_anamorphism,
                                      right_handed)
    stamps = []
    shifts = []

    for idx, position in enumerate(yx_position):
        cutout = Cutout2D(flux_arr[idx], (position[-1], position[-2]), stamp_size, copy=True)
        if plot:
            plt.imshow(flux_arr[idx], origin='lower')
            cutout.plot_on_original(color='white')
            plt.show()
        subpixel_shift = np.array(cutout.position_original) - \
            np.array(cutout.input_position_original)
        shifts.append(subpixel_shift)
        stamps.append(shift(
            cutout.data, (subpixel_shift[-1], subpixel_shift[-2]), output=None,
            order=shift_order, mode='constant', cval=0.0, prefilter=True))

    stamps = np.array(stamps)
    return stamps, shifts


# stamp_size = np.array([9, 9])
# pad_width = 1
# sources = Table()
# sources = Table()
# sources['flux'] = [10]
# sources['x_mean'] = [stamp_size[-1] // 2]
# sources['y_mean'] = [stamp_size[-2] // 2]
# sources['x_stddev'] = [2]
# sources['y_stddev'] = [2]
# sources['theta'] = [0]
# sources['id'] = [1]
#
# ntimes = 256
#
# flux_arr = np.zeros((ntimes, 100, 100))
# image_size = np.array((flux_arr.shape[-2], flux_arr.shape[-1]))
# psf = make_gaussian_sources_image(stamp_size, sources)
# psf = np.pad(psf, pad_width=1, mode='constant', constant_values=0.)
# stamp_size += pad_width * 2
#
# pa = np.arange(0, ntimes, 2)
# pos = np.array((0, 20))
# image_center = np.array((image_size[0] // 2, image_size[1] // 2))
# norm = np.ones(ntimes)
# spline_filtered_psf_arr = spline_filter(psf)

# yx_position = yx_position_in_cube_optimized(
#     image_size, pos, pa, image_center, yx_anamorphism=np.array([1., 1.]),
#     right_handed=True)
#
# image_shape = image_size
# psf_shape = psf.shape
# # xdiff, ydiff, x1, x2, y1, y2 = prepare_injection(yx_position, image_shape, psf_shape)
# xdiff, ydiff, x1, x2, y1, y2 = prepare_injection(yx_position, image_shape, psf_shape)
#
#
# test = inject_signal(
#     flux_arr=flux_arr, xdiff=xdiff, ydiff=ydiff, x1=x1, x2=x2, y1=y1, y2=y2,
#     psf_image=spline_filtered_psf_arr,
#     norm=norm,
#     copy=False)
#
# test2 = inject_signal(
#     flux_arr=flux_arr, xdiff=xdiff, ydiff=ydiff, x1=x1, x2=x2, y1=y1, y2=y2,
#     psf_image=spline_filtered_psf_arr,
#     norm=norm,
#     copy=False)


# test = inject_model_into_data(
#     flux_arr=flux_arr, pos=pos, pa=pa, psf_model=spline_filtered_psf_arr,
#     image_center=image_center, norm=norm, yx_anamorphism=np.array([1., 1.]),
#     right_handed=True, subpixel=True, copy=True)
#
# %timeit inject_model_into_data(flux_arr=flux_arr, pos=pos, pa=pa, psf_model=spline_filtered_psf_arr, image_center=image_center, norm=norm, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=False, copy=False)

# flux_ arr = addsource_optimized(flux_arr=flux_arr, pos=pos, pa=pa, spline_filtered_psf_arr=spline_filtered_psf_arr, image_center=image_center,
#                     norm=np.array((0.1)), jitter=0, poisson_noise=False, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=True, copy=True, verbose=False)
# %timeit addsource(flux_arr=flux_arr, pos=pos, pa=pa, psf_arr=spline_filtered_psf_arr, image_center=image_center, norm=norm, jitter=0, poisson_noise=False, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=True, copy=True, verbose=False)
# %timeit addsource_optimized(flux_arr=flux_arr, pos=pos, pa=pa, spline_filtered_psf_arr=psf, image_center=image_center, norm=norm, jitter=0, poisson_noise=False, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=True, copy=True, verbose=False)


# test2 = addsource(flux_arr=flux_arr, pos=pos, pa=pa, psf_arr=psf, image_center=image_center, norm=norm,
#                   jitter=0, poisson_noise=False, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=True, copy=True, verbose=False)

# test_optimized = addsource_optimized(flux_arr=flux_arr, pos=pos, pa=pa, spline_filtered_psf_arr=spline_filtered_psf_arr, image_center=image_center,
#                                      norm=norm, jitter=0, poisson_noise=False, yx_anamorphism=np.array((1., 1.)), right_handed=True, subpixel=True, copy=True, verbose=False)


#
# flux_arr = addsource(
#     flux_arr, pos, pa, psf, image_center=None,
#     norm=-0.1, jitter=0,
#     poisson_noise=False,
#     right_handed=True, verbose=False)
#
#
# stamps, shifts = extract_stamps(flux_arr, pos, pa, stamp_size=stamp_size,
#                                 image_center=None, right_handed=True)

# mean_stamp = np.mean(stamps, axis=0)
# residual = psf * 0.1 - mean_stamp
# fractional_residual = residual / psf
#
# fits.writeto('test_psf.fits', psf, overwrite=True)
# fits.writeto('test_new_inject.fits', flux_arr, overwrite=True)
# fits.writeto('test_mean_stamp.fits', mean_stamp, overwrite=True)
# fits.writeto('test_residual.fits', residual, overwrite=True)
# fits.writeto('test_fractional_residual.fits', fractional_residual, overwrite=True)
#
#
# fits.writeto('test_stamp.fits', stamps, overwrite=True)
# plt.imshow(cutout.data, origin='lower')
#
# plt.imshow(flux_arr[1], origin='lower')
# cutout.plot_on_original(color='white')
# stamps = extract_stamps(flux_arr, pos, pa, psf, image_center=None,
#                        right_handed=True, verbose=False)
# fits.writeto('stamps.fits', stamps, overwrite=True)
