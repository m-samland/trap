"""
Routines used in TRAP

@author: Tim Brandt, Matthias Samland
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import Cutout2D
from scipy.ndimage import shift, spline_filter


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