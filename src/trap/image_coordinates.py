"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

import numpy as np
from astropy import units as u


class ImageCoordinates(object):
    """ Defines objects containing pixel coordinates in images and transforms
       thereof. Can also include platescale and telescope information.

       Could be a higher abstraction layer in the future. Currently not used.
    """

    def __init__(
            self, coordinates,
            image_center_yx,
            coordinate_system='absolute_cartesian',
            pixel_scale=None):
        """ Possible coordinate initializations are:
            absolute_cartesian, relative_cartesian, polar
        """

        self.absolute_coordinates = absolute_coordinates
        self.image_center = image_center_yx
        self.pixel_scale = pixel_scale

    # def relative_yx_to_absolute_yx(relative_yx):


def relative_yx_to_absolute_yx(relative_yx, image_center_yx):
    """Convert relative cartesian position to absolute position.

    Relative position is converted to absolute position.

    >>> relative_yx_to_absolute_yx((10, 0), (50, 50))
    array([ 60.,  50.])

    Parameters
    ----------
    relative_yx : tuple
        Value pair specifying y and x position relative
        to center.
    image_center_yx : tuple
        Value pair specifying y and x position of image
        center.

    Returns
    -------
    type : array
        Value pair specifying y and x in absolute image
        coordinates.

    """
    relative_yx = np.array(relative_yx, dtype='float64')
    image_center_yx = np.array(image_center_yx)
    absolute_yx = relative_yx + image_center_yx
    return absolute_yx


def absolute_yx_to_relative_yx(absolute_yx, image_center_yx):
    """Convert absolute cartesian position to relative position.

    Absolute position in an image is converted to
    relative position.

    >>> absolute_yx_to_relative_yx((60., 50.), (50., 50.))
    array([ 10.,   0.])

    Parameters
    ----------
    absolute_yx : tuple
        Value pair specifying absolute y and x position.
    image_center_yx : tuple
        Value pair specifying y and x position of image
        center.

    Returns
    -------
    type : array
        Value pair specifying y and x in coordinates
        relative to center.

    """

    absolute_yx = np.array(absolute_yx, dtype='float64')
    image_center_yx = np.array(image_center_yx)
    relative_yx = absolute_yx - image_center_yx
    return relative_yx


def relative_yx_to_rhophi(relative_yx):
    """Convert relative cartesian to polar coordinates.

    Conversion from relative cartesion position
    to position in polar coordinates. The definition of
    the polar coordinates position angle starts counting
    above the center (dy > 0, dx = 0) and continues
    increasing counter-clockwise, up to 360 degrees.

    >>> relative_yx_to_rhophi((10., 0.))
    array([ 10.,   0.])

    Example for using multiple coordinates at constant
    separation, but increasing angle (counter-clockwise).

    >>> relative_yx_to_rhophi(np.array([(10., 0.), (0., -10.), (-10., 0.), (0., 10.)]))
    array([[  10.,    0.],
           [  10.,   90.],
           [  10.,  180.],
           [  10.,  270.]])

    Parameters
    ----------
    relative_yx : tuple or array
        Either value-pair specifying y and x position relative
        to center or list of multiple value pairs.
    Returns
    -------
    type : array
        Array of polar coordinates (separation, angle [degree])
        pairs.

    """

    relative_yx = np.array(relative_yx, dtype='float64')
    if len(relative_yx.shape) == 1:
        relative_yx = np.expand_dims(relative_yx, axis=0)
    if len(relative_yx.shape) == 2:
        rho = np.linalg.norm(relative_yx, axis=1)
        phi = np.arctan2(relative_yx[:, 1], relative_yx[:, 0])
    else:
        raise ValueError("Coordinates to a single tuple or list of tuples.")

    phi *= -1.
    mask = phi < 0
    # Add 2pi to negative value to change range from [-pi, pi] to [0, 2pi]
    phi[mask] += 2 * np.pi
    phi = phi * 360. / (2. * np.pi) % 360.
    rhophi = np.vstack([rho, phi]).T.squeeze()
    return rhophi


def rhophi_to_relative_yx(rhophi):
    """Convert polar coordinates to cartesian.

    Conversion from position in polar coordinates
    relative position in cartesian coordinates.
    The definition of the polar coordinates position
    angle starts counting above the center
    (dy > 0, dx = 0) and continues increasing
    counter-clockwise, up to 360 degrees.

    >>> rhophi_to_relative_yx([[10, 0], [10, 90]])
    array([[  1.00000000e+01,  -0.00000000e+00],
           [  6.12323400e-16,  -1.00000000e+01]])

    Parameters
    ----------
    rhophi : tuple or array
        Either value-pair specifying radial distance
        and position angle in degree or list of multiple
        such value pairs.

    Returns
    -------
    type
        Array of value pairs specifying y and x
        position relative to center.

    """

    rhophi = np.array(rhophi, dtype='float64')
    if len(rhophi.shape) == 1:
        rhophi = np.expand_dims(rhophi, axis=0)
    if len(rhophi.shape) == 2:
        rhophi[:, 1] = -1. * (rhophi[:, 1]) * (2. * np.pi / 360.)
        y = rhophi[:, 0] * np.cos(rhophi[:, 1])
        x = rhophi[:, 0] * np.sin(rhophi[:, 1])
    else:
        raise ValueError("Coordinates to a single tuple or list of tuples.")

    relative_yx = np.vstack([y, x]).T.squeeze()
    return relative_yx


def separation_pa_to_relative_xy(separation, position_angle, pixel_scale):
    """Converts separation and position angle to relative postion in pixel.
        All input values need to be astropy quantities.

    """

    separation_pix = separation.to(u.pixel, pixel_scale)
    position_angle = position_angle.to(u.radian) + np.pi / 2 * u.radian
    x = separation_pix * np.cos(position_angle)
    y = separation_pix * np.sin(position_angle)

    return separation_pix, x, y


def protection_angle(separation, delta, fwhm):
    delta_deg = delta / (separation / fwhm) / np.pi * 180.
    return delta_deg


if __name__ == "__main__":
    import doctest
    doctest.testmod()
