"""
Routines used in TRAP

@author: Tim Brandt, Matthias Samland
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from pdb import set_trace

import matplotlib.pyplot as plt
import numpy as np
# from photutils.datasets import (make_random_gaussians_table,
#                                 make_gaussian_sources_image)
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from numpy.random import poisson
from scipy import ndimage
from scipy.ndimage.interpolation import shift, spline_filter

from .embed_shell import ipsh


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
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_size[-1] // 2
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_size[-2] // 2
        # y = pos[1] * np.sin(pa) + pos[0] * np.cos(pa) + image_size[-2] // 2
    elif image_center.ndim == 1:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[0]
        # y = pos[1] * np.sin(pa) + pos[0] * np.cos(pa) + image_center[0]
    elif image_center.ndim == 2:
        x = (pos[1] * np.cos(pa) - pos[0] * np.sin(pa)) * 1 / yx_anamorphism[1] + image_center[:, 1]
        y = (pos[1] * np.sin(pa) + pos[0] * np.cos(pa)) * 1 / yx_anamorphism[0] + image_center[:, 0]
        # y = pos[1] * np.sin(pa) + pos[0] * np.cos(pa) + image_center[:, 0]
    else:
        raise ValueError('Invalid value for image_center.')

    coords = np.vstack((y, x)).T
    return coords


def addsource(flux_arr, pos, pa, psf_arr, image_center=None,
              norm=1., jitter=0, poisson_noise=False,
              yx_anamorphism=[1., 1.], right_handed=True, subpixel=True,
              verbose=False):
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

    flux = flux_arr.copy()
    try:
        check_if_iterable = iter(norm)
    except TypeError as te:
        norm = np.array([norm])
    psf_arr = psf_arr.copy()  # * norm
    filtered_psf = spline_filter(psf_arr)
    stamp_size = psf_arr.shape[-1]
    yx_position = yx_position_in_cube((flux.shape[-2], flux.shape[-1]),
                                      pos, pa, image_center, yx_anamorphism,
                                      right_handed)
    for idx, position in enumerate(yx_position):
        cutout = Cutout2D(flux[idx], (position[-1], position[-2]), stamp_size, copy=False)
        if subpixel:
            subpixel_shift = np.array(cutout.position_original) - np.array(cutout.input_position_original)
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

# EDIT: Still need to account for anamorphism and centers!


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
        subpixel_shift = np.array(cutout.position_original) - np.array(cutout.input_position_original)
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
# flux_arr = np.zeros((10, 100, 100))
# image_size = (flux_arr.shape[-2], flux_arr.shape[-1])
# psf = make_gaussian_sources_image(stamp_size, sources)
# psf = np.pad(psf, pad_width=1, mode='constant', constant_values=0.)
# stamp_size += pad_width * 2
#
# pa = np.arange(10)
# pos = [0, 20]
#
# flux_arr = addsource_new(
#     flux_arr, pos, pa, psf, image_center=None,
#     norm=0.1, jitter=0,
#     poisson_noise=False,
#     right_handed=True, verbose=False)
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
