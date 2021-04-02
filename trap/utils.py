"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

from asyncio import Event
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import ray
from numba import njit
from numpy.random import default_rng
from ray.actor import ActorHandle
from scipy.ndimage.interpolation import spline_filter
from scipy.signal import medfilt
from tqdm import tqdm
from trap import regressor_selection
from trap.embed_shell import ipsh


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter

# Back on the local node, once you launch your remote Ray tasks, call
# `print_until_done`, which will feed everything back into a `tqdm` counter.


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


def shuffle_and_equalize_relative_positions(
        relative_coords, number_of_chunks, max_separation_deviation=2, max_iterations=50,
        rng=None):
    """ Shuffle position array to try and equalize computation time for each chunk.
        Reshuffle until average separation in each chunk is within 'max_separation_deviation'.
        This is not perfect, but it will prevent cases of very bad luck
        In case the criterion is not reached after 'max_iterations' (it may be impossible)
        stop shuffling and return current array.

    Parameters
    ----------
    relative_coords : type
        Description of parameter `relative_coords`.
    number_of_chunks : type
        Description of parameter `number_of_chunks`.
    max_separation_deviation : type
        Description of parameter `max_separation_deviation`.
    max_iterations : type
        Description of parameter `max_iterations`.
    rng : type
        Description of parameter `rng`.

    Returns
    -------
    type
        Description of returned object.

    """

    if rng is None:
        rng = default_rng(12345)
    separation_equalized = False
    avg_separations = np.empty(number_of_chunks)
    iteration = 0
    while not separation_equalized and iteration < max_iterations:
        rng.shuffle(relative_coords, axis=0)
        relative_coords_regions = np.array_split(relative_coords, number_of_chunks)
        for idx, region in enumerate(relative_coords_regions):
            avg_separations[idx] = np.mean(np.linalg.norm(region, axis=1))
        iteration += 1
        if np.all(np.abs(np.mean(avg_separations) - avg_separations)) < max_separation_deviation:
            separation_equalized = True
    return relative_coords, relative_coords_regions, iteration, separation_equalized


def gen_bad_pix_mask(image, filsize=5, threshold=5.0, return_smoothed_image=False):
    """
    """
    image_sm = medfilt(image, filsize)
    res = image - image_sm
    sigma = np.std(res)
    goodpix = np.abs(res) / sigma < threshold
    return (goodpix, image_sm) if return_smoothed_image else goodpix


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def determine_psf_stampsizes(fwhm, size_in_lamda_over_d=2.2):
    return round_up_to_odd(fwhm * size_in_lamda_over_d * 2.)


def prepare_psf(psf_cube, psf_size, filter=True):
    psf_list = []
    for idx, psf_image in enumerate(psf_cube):
        psf_image = resize_image_cube(psf_image, int(psf_size[idx]))
        mask_negative = psf_image < 0.
        mask_psf = regressor_selection.make_signal_mask(
            (psf_image.shape[-2], psf_image.shape[-1]),
            (psf_image.shape[-2] // 2., psf_image.shape[-1] // 2.),
            mask_radius=psf_size[idx] / 2)
        mask = np.logical_or(mask_negative, ~mask_psf)
        psf_image[mask] = 0.
        psf_image = np.pad(psf_image, pad_width=[(1,), (1,)], mode='constant', constant_values=0.)
        if filter:
            psf_image = spline_filter(psf_image.astype('float64'))
        psf_list.append(psf_image)
    return psf_list


def bin_frames(data_cube, bad_frame_indices, binsize):
    mask = np.zeros(data_cube.shape[1], dtype='bool')
    mask[bad_frame_indices] = True
    data_cube[:, mask, :, :] = np.nan
    data_cube2 = data_cube.reshape(
        data_cube.shape[0], int(data_cube.shape[1] / binsize), binsize,
        data_cube.shape[2], data_cube.shape[3])
    data_cube3 = np.nanmean(data_cube2, axis=2)
    return data_cube3


def determine_maximum_contrast_for_injection(data, flux_psf, reduction_mask, percentile=99.):
    max_planet_flux = np.max(flux_psf)
    data_percentile = np.percentile(data[:, reduction_mask], percentile)

    return data_percentile / max_planet_flux


def resize_arr(arr, newdim):
    assert len(np.asarray(arr).shape) == 2, "Function arr_resize expects a 2D array"
    arr = np.asarray(arr)
    dimy, dimx = arr.shape
    dx1 = dimx // 2
    dy1 = dimy // 2
    dx2 = dy2 = newdim // 2

    if newdim % 2 == 0:
        return arr[dy1 - dy2:dy1 + dy2,
                   dx1 - dx2:dx1 + dx2]
    else:
        return arr[dy1 - dy2:dy1 + dy2 + 1,
                   dx1 - dx2:dx1 + dx2 + 1]


def crop_box_from_4D_cube(flux_arr, boxsize, center_yx=None):
    """Extract cropped area centered on given coordinates"""

    if center_yx is None:
        dx1 = flux_arr.shape[-1] // 2
        dy1 = flux_arr.shape[-2] // 2
    else:
        dx1 = center_yx[1]
        dy1 = center_yx[0]

    dx2 = dy2 = boxsize // 2

    if boxsize % 2 == 0:
        return flux_arr[:, :, int(dy1 - dy2):int(dy1 + dy2),
                        int(dx1 - dx2):int(dx1 + dx2)]
    else:
        return flux_arr[:, :, int(dy1 - dy2):int(dy1 + dy2 + 1),
                        int(dx1 - dx2):int(dx1 + dx2 + 1)]


def crop_box_from_3D_cube(flux_arr, boxsize, center_yx=None):
    """Extract cropped area centered on given coordinates"""

    if center_yx is None:
        dx1 = flux_arr.shape[-1] // 2
        dy1 = flux_arr.shape[-2] // 2
    else:
        dx1 = center_yx[1]
        dy1 = center_yx[0]

    dx2 = dy2 = boxsize // 2

    if boxsize % 2 == 0:
        return flux_arr[:, int(dy1 - dy2):int(dy1 + dy2),
                        int(dx1 - dx2):int(dx1 + dx2)]
    else:
        return flux_arr[:, int(dy1 - dy2):int(dy1 + dy2 + 1),
                        int(dx1 - dx2):int(dx1 + dx2 + 1)]


def crop_box_from_image(flux_arr, boxsize, center_yx=None):
    """Extract cropped area centered on given coordinates"""

    if center_yx is None:
        dx1 = flux_arr.shape[-1] // 2
        dy1 = flux_arr.shape[-2] // 2
    else:
        dx1 = center_yx[1]
        dy1 = center_yx[0]

    dx2 = dy2 = boxsize // 2

    if boxsize % 2 == 0:
        return flux_arr[int(dy1 - dy2):int(dy1 + dy2),
                        int(dx1 - dx2):int(dx1 + dx2)]
    else:
        return flux_arr[int(dy1 - dy2):int(dy1 + dy2 + 1),
                        int(dx1 - dx2):int(dx1 + dx2 + 1)]


def resize_image_cube(arr, new_dim):
    """ Takes a two or three dimensional data cube (N, x, y)
        and returns the same data cube with resized frames)
    """
    assert len(np.asarray(arr).shape) in set([2, 3]), "Resize input has to be image or image cube."
    if len(np.asarray(arr).shape) == 2:
        resized_arr = np.zeros((new_dim, new_dim))
        resized_arr = resize_arr(arr, new_dim)
    elif len(np.asarray(arr).shape) == 3:
        resized_arr = np.zeros((arr.shape[0], new_dim, new_dim))
        for i in range(arr.shape[0]):
            resized_arr[i] = resize_arr(arr[i], new_dim)
    return resized_arr


def resize_cube(arr, new_dim):
    """ Takes a four dimensional data cube (wavelength, frame number, x, y)
        and returns the same data cube with resized frames)
    """
    # if new_dim % 2 == 0:
    # new_dim = new_dim + 1
    assert len(np.asarray(arr).shape) == 4, "Function arr_resize expects a 4D data cube"
    resized_arr = np.zeros((arr.shape[0], arr.shape[1], new_dim, new_dim))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # print("Resizing Channel {0}: Frame {1}".format(i+1, j+1))
            resized_arr[i, j] = resize_arr(arr[i, j], new_dim)
    return resized_arr


def derotate_cube(flux_arr, pa, right_handed, verbose=False):
    for i in range(flux_arr.shape[0]):
        if verbose is True:
            print("Derotating Spectral Channel: Frame: {:02d} by {:02.03f} degree.".format(i + 1, pa[i]))
        flux_arr[i] = ndimage.rotate(flux_arr[i], pa[i], reshape=False)

    return flux_arr


def mask_center(arr, dim, r_init=3, fwhm=5, maskvalue=0.):
    mask_size = r_init * fwhm
    xcen = dim / 2. - 1  # Check
    ycen = dim / 2. - 1  # Check
    x = np.arange(arr.shape[0])
    y = np.arange(arr.shape[1]).reshape(-1, 1)
    dist = np.sqrt((x - xcen)**2 + (y - ycen)**2)
    mask1 = dist < mask_size
    # mask2 = dist < r1
    np.putmask(arr, mask1, maskvalue)
    return arr


def mask_cube(arr, dim, r_init=3, fwhm=5):
    """ Takes a four dimensional data cube (wavelength, frame number, x, y)
        and returns the same data cube with resized frames (odd pixel number, centered on the middle pixel)
    """
    assert len(np.asarray(arr).shape) == 4, "Function arr_resize expects a 4D data cube"
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print("Masking Center of Channel {0}: Frame {1}".format(i + 1, j + 1))
            arr[i, j] = mask_center(arr[i, j], dim, r_init, fwhm)
    return arr


def scale_images(arr, lam, newdim):
    """Scales images of different wavelength to the same resolution element, by zooming them to the largest wavelength
    """
    magnification = np.zeros_like(lam)
    flux = arr.copy()
    print(arr.shape)
    print(lam.shape)
    for i in range(lam.shape[0] - 1):
        magnification[i] = np.divide(lam[-1], lam[i])
        for j in range(arr.shape[1]):
            print("Magnifing Channel {0}: Frame {1}".format(i + 1, j + 1))
            # Crop Image before saving into flux is necessary!
            flux[i, j] = resize_arr(ndimage.interpolation.zoom(arr[i, j], float(magnification[i]), order=3), newdim)
    return flux


def scale_images_sdi(arr, lam, newdim):
    """Scales images of different wavelength to the same resolution element, by zooming them to the largest wavelength
    """
    magnification = np.zeros_like(lam)
    flux = arr.copy()
    print(arr.shape)
    print(lam.shape)
    for i in range(lam.shape[0] - 1):
        magnification[i] = np.divide(lam[-1], lam[i])
        print("Magnifing Channel {0}".format(i + 1))
        # Crop Image before saving into flux is necessary!
        flux[i] = resize_arr(ndimage.interpolation.zoom(arr[i], float(magnification[i]), order=3), newdim)
    return flux


def high_pass_filter(image, cutoff_frequency=0.25):
    image_size = image.shape[0]
    cutoff = image_size / 2. * cutoff_frequency

    if cutoff > image_size/2. - 1:
        cutoff = image_size/2. - 1

    cutoff_inside = np.round(cutoff)
    winsize = 2*cutoff_inside + 1

    r = np.round(winsize / 4.)  # how narrower the window is
    hann = np.hanning(image_size)[:, None]  # 1D hamming
    hann2d = 1. - np.sqrt(np.dot(hann, hann.T)) ** r  # expand to 2D hamming

    original = np.fft.fft2(image)
    center = np.fft.fftshift(original) * hann2d
    center = np.fft.ifftshift(center)
    filtered_image = np.real(np.fft.ifft2(center))

    return filtered_image


def high_pass_filter_cube(data, cutoff_frequency=0.25, verbose=False):
    filtered_data = np.zeros_like(data)
    for wave_idx, wave in enumerate(data):
        if verbose:
            wave = tqdm(wave)
        for frame_idx, frame in enumerate(wave):
            filtered_data[wave_idx][frame_idx] = high_pass_filter(
                frame, cutoff_frequency)
    return filtered_data


def high_pass_filter(img, filtersize=10):
    """
    Implementation from pyKLIP package

    A FFT implmentation of high pass filter.

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
        filtered: the filtered image
    """
    # mask NaNs if there are any
    nan_index = np.where(np.isnan(img))
    if np.size(nan_index) > 0:
        good_index = np.where(~np.isnan(img))
        y, x = np.indices(img.shape)
        good_coords = np.array([x[good_index], y[good_index]]).T  # shape of Npix, ndimage
        nan_fixer = sinterp.NearestNDInterpolator(good_coords, img[good_index])
        fixed_dat = nan_fixer(x[nan_index], y[nan_index])
        img[nan_index] = fixed_dat

    transform = fft.fft2(img)

    # coordinate system in FFT image
    u, v = np.meshgrid(fft.fftfreq(transform.shape[1]), fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = np.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - np.exp(-(rho**2/filtersize**2))

    filtered = np.real(fft.ifft2(transform*filt))

    # restore NaNs
    filtered[nan_index] = np.nan
    img[nan_index] = np.nan

    return filtered


def plot_pa(parameters, angle):
    """Takes np.array containing the paralactic angles and plots the difference between adjacent elements.
    """
    plt.close()
    # Two subplots, unpack the axes array immediately
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x = np.arange(angle.shape[0])
    y = angle
    x_diff = np.arange(angle.shape[0] - 1)
    y_diff = np.abs(np.diff(angle))

    ax1.plot(x, y)
    ax2.plot(x_diff, y_diff)
    # ax2.plot(x_space, f_y2, 'k', lw=3, color=colors[i], alpha=0.6)
    # plt.plot(x, y, color="black", alpha=0.6)
    # plt.ylabel('$dPA$')
    # plt.xlabel('$Frame$')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('PA')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('dPA')
    ax1.set_title('PA')
    ax2.set_title('Differential PA')

    fig.tight_layout()
    plt.savefig(parameters.outputdir + '{0}_PA.png'.format(parameters.target_name), dpi=300)
    # fig.show()


def compute_flux_overlap(frame_number, model):
    mask = model[frame_number] > 0
    total_flux = np.sum(model[frame_number])
    flux_overlap_fraction = np.sum(model[:, mask], axis=1) / total_flux
    return flux_overlap_fraction


def det_max_ncomp_specific(r_planet, fwhm, delta, pa):
    """ Determines the maximum number of possible principles components that can be used in a certain annulus.
        This number is determined by the minimum number of references frames for a given annulus, frame, FoV rotation
        and protection angle.

        Returns: Array containing [max_ncomp for frame n, max_ncomp for annulus]
    """
    delta_deg = delta / (r_planet / fwhm) / np.pi * 180.
    max_ncomp = np.zeros((pa.shape[0], 2))
    pa_mask = np.zeros((pa.shape[0], pa.shape[0])).astype('bool')
    for j in range(pa.shape[0]):
        pa_lower = pa < (pa[j] - delta_deg)
        pa_upper = pa > (pa[j] + delta_deg)
        pa_mask[j] = np.logical_or(pa_lower, pa_upper)
        max_ncomp[j, 0] = np.sum(pa_mask[j])
    max_ncomp[:, 1] = np.min(max_ncomp[:, 0])
    return max_ncomp, pa_mask


def combine_reduction_regions(small_image, large_image):
    small_image_size = small_image.shape[0]
    large_image_size = large_image.shape[0]

    assert small_image_size <= large_image_size, \
        "The small image needs to be smaller or equal the size of the larger image."
    if small_image_size < large_image_size:
        padding_size = int(round(large_image_size - small_image_size) / 2)
        small_image = np.pad(small_image, padding_size, constant_values=np.nan)
    mask = np.isfinite(small_image)
    large_image[mask] = small_image[mask]

    return large_image


@njit
def compute_empirical_correlation_matrix(residuals):
    n_vectors = residuals.shape[0]
    psi_ij = np.zeros((n_vectors, n_vectors), dtype='float64')
    for i in range(n_vectors):
        for j in range(i, n_vectors):
            c_ij = np.dot(residuals[i, :], residuals[j, :].T)
            cov1 = np.dot(residuals[i, :], residuals[i, :].T)
            cov2 = np.dot(residuals[j, :], residuals[j, :].T)
            psi_ij[i, j] = c_ij / np.sqrt(cov1 * cov2)
    psi_ij = psi_ij + psi_ij.T - np.diag(np.ones(n_vectors))

    return psi_ij


def matern32_kernel(distance, length_scale):
    r = distance/length_scale
    psi_ij_model = \
        (1 + np.sqrt(3 * r**2)) * np.exp(-(np.sqrt(3 * r**2)))
    return psi_ij_model


def matern52_kernel(distance, length_scale):
    r = distance/length_scale
    psi_ij_model = \
        (1 + np.sqrt(5 * r**2) + (5 * r**2) / 3) * \
        np.exp(-(np.sqrt(5 * r**2)))
    return psi_ij_model


def exponential_kernel(distance, length_scale):
    r = distance/length_scale
    psi_ij_model = np.exp(-1*np.sqrt(r**2))
    return psi_ij_model


def exponential_squared_kernel(distance, length_scale):
    r = distance/length_scale
    psi_ij_model = np.exp(-1/2 * r**2)
    return psi_ij_model


def sinc_kernel(distance):
    return np.sinc(distance)


def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


def get_data_base(arr):
    """For a given NumPy array, find the base array
    that owns the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def arrays_share_data(x, y):
    return get_data_base(x) is get_data_base(y)
