"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from photutils import CircularAnnulus
from scipy.ndimage.morphology import binary_dilation

from .embed_shell import ipsh
from .image_coordinates import (absolute_yx_to_relative_yx,
                                relative_yx_to_absolute_yx,
                                relative_yx_to_rhophi, rhophi_to_relative_yx)
from .makesource import inject_model_into_data
from .plotting_tools import plot_scale

__all__ = [
    'make_signal_mask', 'make_annulus_mask', 'make_annulus_mask_by_separation',
    'make_regressor_pool_for_pixel', 'find_N_closest_values_in_image', 'find_N_unique_samples',
    'select_regressors_for_pixel', 'make_mask_for_known_companions']


def make_signal_mask(yx_dim, pos_yx, mask_radius, oversampling=1, relative_pos=False, yx_center=None):
    """ Masks all pixel inside of mask_radius in an image. If relative_pos is True
        pos_yx corresponds to the relative position to the center, else the absolute position.

    """

    x = np.arange(0, yx_dim[1], 1 / oversampling)
    y = np.arange(0, yx_dim[0], 1 / oversampling).reshape(-1, 1)

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    if relative_pos:
        x_signal = (yx_center[1] + pos_yx[1])
        y_signal = (yx_center[0] + pos_yx[0])
    else:
        x_signal = pos_yx[1]
        y_signal = pos_yx[0]
    dist_from_signal = np.sqrt((x - x_signal)**2 + (y - y_signal)**2)
    signal_mask = dist_from_signal <= mask_radius

    return signal_mask


def make_mask_from_psf_track(
        yx_position, psf_size, pa, image_size, image_center=None,
        yx_anamorphism=np.array([1., 1.]), right_handed=True, return_cube=False):
    # Make reduction mask, should be the same for all steps!
    ntime = len(pa)
    flux_psf_form = np.ones((psf_size, psf_size))
    psf_mask = make_signal_mask(
        (psf_size, psf_size),
        (psf_size // 2., psf_size // 2.),
        mask_radius=psf_size // 2)
    flux_psf_form[~psf_mask] = 0
    if return_cube:
        signal_pix_mask = np.zeros((ntime, image_size, image_size))
    else:
        signal_pix_mask = np.zeros((image_size, image_size))
    # signal_pix_mask_per_frame = addsource(
    #     flux_arr=signal_pix_mask_per_frame,
    #     pos=yx_position,
    #     pa=pa,
    #     psf_arr=flux_psf_form,
    #     image_center=image_center,
    #     norm=np.ones(ntime),
    #     jitter=0,
    #     poisson_noise=False,
    #     yx_anamorphism=yx_anamorphism,
    #     right_handed=right_handed,
    #     subpixel=False,
    #     copy=False,
    #     verbose=False)
    signal_pix_mask = inject_model_into_data(
        flux_arr=signal_pix_mask,
        pos=yx_position,
        pa=pa,
        psf_model=flux_psf_form,
        image_center=image_center,
        norm=np.ones(ntime),
        yx_anamorphism=yx_anamorphism,
        right_handed=right_handed,
        subpixel=False,
        copy=False)

    signal_pix_mask = signal_pix_mask.astype('bool')

    if return_cube:
        return np.any(signal_pix_mask, axis=0), signal_pix_mask
    else:
        return signal_pix_mask


def make_mirrored_mask(mask, yx_center):
    mirrored_indices = relative_yx_to_absolute_yx(
        absolute_yx_to_relative_yx(
            np.argwhere(mask), image_center_yx=yx_center) * (-1),
        image_center_yx=yx_center).astype('int')
    mirrored_mask = np.zeros_like(mask).astype('bool')
    for idx in mirrored_indices:
        mirrored_mask[idx[0], idx[1]] = True
    return mirrored_mask


def alternative_signal_positions(signal_position, fov_rotation, number_of_areas):
    positions = []
    polar_position = relative_yx_to_rhophi(signal_position)
    for i in range(int(number_of_areas)):
        pos_new = polar_position.copy()
        pos_new[1] += (i + 1) * fov_rotation
        # pos_new[1] = pos_new[1] % 360
        pos_new = rhophi_to_relative_yx(pos_new)
        positions.append(pos_new)
    return np.array(positions)


def plot_alternative_signal_positions(signal_position, alt_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(alt_positions[:, 0], alt_positions[:, 1], color='b')
    ax.scatter(signal_position[0], signal_position[1], color='r')
    ax.set_aspect('equal')
    plt.show()


def alternative_reduction_areas(
        positions, pa,
        reduction_parameters, image_size, yx_anamorphism, right_handed):
    reduction_areas = []
    for pos in positions:
        area = make_mask_from_psf_track(
            yx_position=pos,
            psf_size=reduction_parameters.reduction_mask_psf_size,
            pa=pa, image_size=image_size,
            yx_anamorphism=yx_anamorphism,
            right_handed=right_handed,
            return_cube=False)
        reduction_areas.append(area)
    return reduction_areas


def plot_alternative_reduction_areas(reduction_areas, yx_dim):
    image_of_areas = np.zeros((yx_dim[0], yx_dim[1]))
    for area in reduction_areas:
        for yx_pix in np.argwhere(area):
            image_of_areas[yx_pix[0], yx_pix[1]] = 1
    plt.imshow(image_of_areas, origin='bottom')
    plt.show()


def make_annulus_mask(inner_edge, outer_edge, yx_dim, oversampling=1, yx_center=None):
    """ Masks all pixel inside of an annulus of an image.

    """
    x = np.arange(0, yx_dim[1], 1 / oversampling)
    y = np.arange(0, yx_dim[0], 1 / oversampling).reshape(-1, 1)
    # x = np.arange(yx_dim[1])
    # y = np.arange(yx_dim[0]).reshape(-1, 1)

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    dist = np.sqrt((x - yx_center[1])**2 + (y - yx_center[0])**2)

    inner_mask = dist >= inner_edge
    outer_mask = dist <= outer_edge
    annulus_mask = np.logical_and(inner_mask, outer_mask)

    return annulus_mask


def make_annulus_mask_by_separation_old(separation, width, yx_dim, yx_center=None):
    """ Masks all pixel inside of an annulus of an image.

    """
    x = np.arange(yx_dim[1])
    y = np.arange(yx_dim[0]).reshape(-1, 1)

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    dist = np.sqrt((x - yx_center[1])**2 + (y - yx_center[0])**2)

    # Check if this corresponds to intended behaviour
    if width == 1:
        inner_mask = dist > (separation - width / 2)
        outer_mask = dist < (separation + width / 2)
    elif width % 2 == 0:
        inner_mask = dist >= (separation - width // 2)
        outer_mask = dist <= (separation + width // 2)
    else:
        inner_mask = dist >= (separation - width // 2 - 1)
        outer_mask = dist <= (separation + width // 2 - 1)
    annulus_mask = np.logical_and(inner_mask, outer_mask)

    return annulus_mask


def make_annulus_mask_by_separation(separation, width, yx_dim, yx_center=None):
    """ Masks all pixel inside of an annulus of an image.

    """
    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    r_in = separation - width / 2.
    r_out = separation + width / 2.
    if r_in < 0.5:
        r_in = 0.5
    annulus_aperture = CircularAnnulus(
        yx_center[::-1], r_in=r_in, r_out=r_out)
    annulus_mask = annulus_aperture.to_mask(method='center')
    # Make sure only pixels are used for which data exists
    annulus_mask = annulus_mask.to_image(yx_dim[::-1]) > 0
    annulus_mask[int(yx_center[0]), int(yx_center[1])] = False

    return annulus_mask


def plot_annulus_pixels(yx_dim=(101, 101), yx_center=None):
    number_of_pixels = np.zeros((4, 50))
    for i, annulus_width in enumerate([5, 7, 9, 11]):
        for j, separation in enumerate(range(30)):
            annulus_mask = make_annulus_mask_by_separation_new(
                separation, annulus_width, yx_dim, yx_center=None)
            number_of_pixels[i][j] = np.sum(annulus_mask)
    for i, annulus_width in enumerate([5, 7, 9, 11]):
        plt.plot(np.arange(50)/4.22, number_of_pixels[i], label='width {}'.format(annulus_width))
    plt.legend()
    plt.show()


def make_regressor_pool_for_pixel(
        reduction_parameters, yx_pixel, yx_dim, yx_center=None,
        yx_center_injection=None,
        signal_mask=None, known_companion_mask=None,
        bad_pixel_mask=None, additional_regressors=None,
        right_handed=True, pa=None, **kwargs):
    """ Given a certain pixel position, an array with the dimension
    of the image is returned marking the for this pixelregressors as True.

    If yx_center=Noneposition is given the floor of half the image size is
    assumed. If annulus_width=None, pixels from the whole image are
    possible regressors, otherwise they will be selected from an annulus
    of same distance as the target pixel.

    """
    # TODO: use relative position is True doesn't seem to work. Not used though.
    # Doesn't take into account anamorphism for annulus
    annulus_width = reduction_parameters.annulus_width
    annulus_offset = reduction_parameters.annulus_offset
    target_pix_mask_radius = reduction_parameters.target_pix_mask_radius
    use_relative_position = reduction_parameters.use_relative_position

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    separation = np.sqrt(
        (yx_pixel[0] - yx_center[0])**2 +
        (yx_pixel[1] - yx_center[1])**2) + annulus_offset

    if annulus_width is None:
        annulus_mask = np.ones((yx_dim), dtype=bool)
    else:
        annulus_mask = make_annulus_mask_by_separation(
            separation=separation,
            width=annulus_width, yx_dim=yx_dim,
            yx_center=yx_center)

    if target_pix_mask_radius is None:
        target_pix_mask = np.zeros((yx_dim), dtype=bool)  # True for pixels to be excluded
    else:
        target_pix_mask = make_signal_mask(
            yx_dim=yx_dim, pos_yx=yx_pixel, mask_radius=target_pix_mask_radius,
            relative_pos=use_relative_position, yx_center=yx_center)

    if signal_mask is None:
        signal_mask = np.zeros((yx_dim), dtype=bool)

    if known_companion_mask is None:
        known_companion_mask = np.zeros((yx_dim), dtype=bool)

    if bad_pixel_mask is None:
        bad_pixel_mask = np.zeros((yx_dim), dtype=bool)

    radial_regressor_mask = np.zeros((yx_dim), dtype=bool)
    if reduction_parameters.add_radial_regressors:
        # radial_mask = []
        # for i, radial_shift in enumerate(
        #         reduction_parameters.radial_separation_from_source):
        #     if radial_shift is not None:
        #         rhophi = relative_yx_to_rhophi(
        #             absolute_yx_to_relative_yx(yx_pixel, yx_center))
        #         rhophi[0] += radial_shift
        #         track_position = rhophi_to_relative_yx(rhophi)
        #         radial_shifted_psf_track = make_mask_from_psf_track(
        #             yx_position=track_position,
        #             psf_size=reduction_parameters.reduction_mask_psf_size,
        #             pa=pa, image_size=yx_dim[0],
        #             image_center=yx_center_injection,
        #             yx_anamorphism=reduction_parameters.yx_anamorphism,
        #             right_handed=right_handed,
        #             return_cube=False)
        #         radial_mask.append(radial_shifted_psf_track)
        # radial_regressor_mask = np.logical_or.reduce(radial_mask)
        radial_regressor_mask = binary_dilation(signal_mask, iterations=7)

    inclusion = np.logical_or(annulus_mask, radial_regressor_mask)
    if additional_regressors is not None:
        inclusion = np.logical_or(inclusion, additional_regressors)
    regressor_pool_mask = np.logical_and.reduce(([inclusion,
                                                  ~target_pix_mask,
                                                  ~signal_mask,
                                                  ~known_companion_mask,
                                                  ~bad_pixel_mask]))
    return regressor_pool_mask


def find_N_closest_values_in_image(target_value, image, N, regressor_pool_mask=None):
    """ Returns boolean mask with True for the N closest values in image to given
    target value. """

    if regressor_pool_mask is not None:
        temp_arr = image[regressor_pool_mask].copy()
        ref_pix_indeces = np.argwhere(regressor_pool_mask)
        reference_pixel_map = np.zeros_like(image, dtype=bool)
    else:
        temp_arr = image.flatten().copy()
        reference_pixel_map = np.zeros_like(temp_arr, dtype=bool)

    original_image_size = image.shape
    temp_arr = np.abs(temp_arr - target_value)
    closest_value_indices_in_pool = np.argpartition(temp_arr, N)[:N]

    if regressor_pool_mask is not None:
        for closest_val in closest_value_indices_in_pool:
            ref_pos = tuple(ref_pix_indeces[closest_val])
            reference_pixel_map[ref_pos[0], ref_pos[1]] = True
    else:
        for closest_val in closest_value_indices_in_pool:
            reference_pixel_map[closest_val] = True
        reference_pixel_map = reference_pixel_map.reshape(original_image_size)

    return reference_pixel_map


def find_N_unique_samples(N, yx_dim, regressor_pool_mask=None):
    """ Pix N unique random numbers from image or pool.

    """
    if regressor_pool_mask is not None:
        ref_pix_indeces = np.argwhere(regressor_pool_mask)
        number_reference_pixel = np.sum(regressor_pool_mask)
        reference_pixel_map = np.zeros_like(regressor_pool_mask, dtype=bool)
    else:
        number_reference_pixel = np.product(yx_dim)
        reference_pixel_map = np.zeros(number_reference_pixel, dtype=bool)

    random_sample = np.random.choice(number_reference_pixel, size=N, replace=False)
    if regressor_pool_mask is not None:
        for sample in random_sample:
            ref_pos = tuple(ref_pix_indeces[sample])
            reference_pixel_map[ref_pos[0], ref_pos[1]] = True
    else:
        for sample in random_sample:
            reference_pixel_map[sample] = True
        reference_pixel_map = reference_pixel_map.reshape(yx_dim)

    return reference_pixel_map


def select_regressors_for_pixel(
        reduction_parameters, yx_pixel, yx_dim, yx_center,
        signal_mask, auxiliary_frame=None, show_plot=False,
        save_plot=False, outputdir=None):
    """ Function to produce regressor selection mask for a certain pixel.
        Currently not in use / maintained.
    """

    number_of_pca_regressors = reduction_parameters.number_of_pca_regressors
    method_of_regressor_selection = reduction_parameters.method_of_regressor_selection

    assert method_of_regressor_selection == 'random' or method_of_regressor_selection == 'auxiliary', 'Non valid method for regressor selection chosen.'

    if yx_center is None:
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

    reference_pixel_map = np.zeros((yx_dim[0], yx_dim[1]), dtype=bool)

    regressor_pool_mask = make_regressor_pool_for_pixel(
        reduction_parameters=reduction_parameters,
        yx_pixel=yx_pixel, yx_dim=yx_dim, yx_center=yx_center,
        signal_mask=signal_pix_mask)

    if auxiliary_frame is not None:
        if method_of_regressor_selection == 'auxiliary':
            target_pix_val = auxiliary_frame[yx_pixel[0], yx_pixel[1]]

            reference_pixel_map = find_N_closest_values_in_image(
                target_value=target_pix_val,
                image=auxiliary_frame, N=number_of_pca_regressors,
                regressor_pool_mask=regressor_pool_mask)

        if method_of_regressor_selection == 'random':
            # Make mask from drawing random indeces from pool
            reference_pixel_map = find_N_unique_samples(
                N=number_of_pca_regressors,
                yx_dim=yx_dim, regressor_pool_mask=regressor_pool_mask)

        if method_of_regressor_selection == 'correlation':
            raise NotImplementedError

    ref_pixel_positions = np.argwhere(reference_pixel_map)
    if show_plot and save_plot and outputdir is not None:
        plot_scale(auxiliary_frame, ref_pixel_positions, signal_pix_indeces, output_path=outputdir, show=True)
    elif save_plot and outputdir is not None:
        plot_scale(auxiliary_frame, ref_pixel_positions, signal_pix_indeces, output_path=outputdir, show=False)
    elif show_plot:
        plot_scale(auxiliary_frame, ref_pixel_positions, signal_pix_indeces, show=True)

    return reference_pixel_map
