"""
Routines used for plotting in TRAP.

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (AsymmetricPercentileInterval,
                                   ImageNormalize, LinearStretch, LogStretch,
                                   MinMaxInterval, PercentileInterval,
                                   ZScaleInterval)
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from trap.embed_shell import ipsh
from trap.image_coordinates import absolute_yx_to_relative_yx

# __all__ = ['plot_scale']


def mark_coordinates(fig, ax, yx_coords, color, point_size=10, plot_star_not_circle=False):
    # Quick hack to check whether more than one pair of coordinates is given
    yx_coords = np.array(yx_coords)
    if len(yx_coords.shape) == 1:
        # Because origin is 'bottom' Circle takes x first
        if plot_star_not_circle:
            symbols = plt.scatter(yx_coords[1], yx_coords[0],
                                  s=point_size, color=color, marker='x')  # (5, 1))
        else:
            symbols = plt.Circle((yx_coords[1], yx_coords[0]), point_size, color=color)
        ax.add_artist(symbols)
    elif len(yx_coords.shape) == 2:
        for yx in yx_coords:
            if plot_star_not_circle:
                symbols = plt.scatter(yx[1], yx[0], point_size, color=color,
                                      marker='+')  # marker=(5, 1))
            else:
                symbols = plt.Circle((yx[1], yx[0]), point_size, color=color)
            ax.add_artist(symbols)
    elif len(yx_coords.shape) > 2:
        raise ValueError("Shape of provided coordinate list wrong.")


def plot_scale(
        image, yx_coords=None, yx_coords2=None, point_size1=1,
        scale='zscale', point_size2=1, plot_star_not_circle=False,
        relative_to_center=False,
        output_path=None, normalization=None, show=False,
        cb_label=None, show_cb=True, figsize=(3, 3)):

    plt.close()
    # plt.style.use("paper")
    image_size = np.shape(image)[0]
    r_max = image_size // 2

    # relative_to_center = True
    if relative_to_center:
        dx = np.shape(image)[1]
        dy = np.shape(image)[0]
        extent = [-dx//2+0.5, dx//2+0.5,
                  -dy//2+0.5, dy//2+0.5]
        if yx_coords is not None:
            yx_coords = absolute_yx_to_relative_yx(yx_coords, [dy//2, dx//2])
        if yx_coords2 is not None:
            yx_coords2 = absolute_yx_to_relative_yx(yx_coords2, [dy//2, dx//2])
    else:
        extent = None

    rcParams['xtick.major.size'] = 6
    rcParams['xtick.minor.size'] = 3
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.minor.size'] = 3

    rcParams['xtick.major.width'] = 1
    rcParams['xtick.minor.width'] = 1
    rcParams['ytick.major.width'] = 1
    rcParams['ytick.minor.width'] = 1
    rcParams['lines.markeredgewidth'] = 1

    if normalization is None:
        if scale == 'zscale':
            norm = ImageNormalize(image, interval=ZScaleInterval())
        elif scale == 'minmax':
            norm = ImageNormalize(image, interval=MinMaxInterval())
        elif isinstance(scale, collections.Iterable):
            norm = None
        else:
            raise ValueError("scale parameter must be zscale or minmax.")
    else:
        norm = normalization

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])

    if norm is None:
        cax = ax.imshow(
            image, origin='lower', vmin=scale[0], vmax=scale[1], extent=extent, interpolation='nearest')
    else:
        cax = ax.imshow(
            image, origin='lower', norm=norm, extent=extent, interpolation='nearest')

    if yx_coords is not None:
        mark_coordinates(fig, ax, yx_coords, 'white', point_size1, plot_star_not_circle)

    if yx_coords2 is not None:
        mark_coordinates(fig, ax, yx_coords2, 'pink', point_size2, plot_star_not_circle=True)

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    ax.tick_params(direction='in', color='black')
    if relative_to_center:
        ax.set_xticks(np.linspace(-r_max, r_max, num=5))
        ax.set_yticks(np.linspace(-r_max, r_max, num=5))
    else:
        ax.set_xticks(np.linspace(0, 2*r_max, num=5))
        ax.set_yticks(np.linspace(0, 2*r_max, num=5))
    ax.set_xlabel('Offset (pixel)')
    ax.set_ylabel('Offset (pixel)')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', color='black', which='both')
    ax.minorticks_on()
    ax.set_aspect('equal')
    if relative_to_center:
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
    else:
        ax.set_xlim(0, 2 * r_max)
        ax.set_ylim(0, 2 * r_max)

    # cur_axes = plt.gca()
    # cur_axes.axes.get_xaxis().set_visible(False)
    # cur_axes.axes.get_yaxis().set_visible(False)
    # ax.tick_params(direction='in', color='black')
    # # ax.set_xticks(np.linspace(-1.0, 1.0, num=5))
    # # ax.set_yticks(np.linspace(-1.0, 1.0, num=5))
    # # ax.set_xlabel('Offset (arcsec)')
    # # ax.set_ylabel('Offset (arcsec)')
    # ax.yaxis.set_ticks_position('both')
    # ax.xaxis.set_ticks_position('both')
    # ax.tick_params(direction='in', color='black', which='both')
    # ax.minorticks_on()
    # ax.set_aspect('equal')
    # ax.set_xlim(1.0 * r_max, -1.0 * r_max)
    # ax.set_ylim(-1.0 * r_max, 1.0 * r_max)
    # ax.set_visible(False)
    if show_cb:
        cb = plt.colorbar(cax, ax=ax, pad=0.08, fraction=0.045)  # , format='%.2f')

        # divider = make_axes_locatable(plt.gca())
        # cax2 = divider.append_axes("right", "5%", pad="3%")
        # plt.colorbar(cax, cax=cax2)
        if cb_label is not None:
            cb.set_label(cb_label, rotation=270, labelpad=10)

    fig.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')  # , dpi=120)
    if show:
        plt.show()
    return ax
