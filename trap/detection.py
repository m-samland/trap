"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import pickle
import copy
from collections import OrderedDict
from glob import glob

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.stats import mad_std
from astropy.table import Table
# import seaborn as sns
from matplotlib import rc, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from natsort import natsorted
from photutils import CircularAnnulus
from scipy import linalg, stats
from scipy.interpolate import interp1d
from tqdm import tqdm
from trap import image_coordinates, regressor_selection
from trap.embed_shell import ipsh
from trap import (regressor_selection, image_coordinates, pca_regression)
from trap.image_coordinates import (absolute_yx_to_relative_yx,
                                    relative_yx_to_rhophi)
from trap.reduction_wrapper import run_complete_reduction
from trap.utils import (compute_empirical_correlation_matrix,
                        remove_channel_from_correlation_matrix,
                        find_nearest, subtract_angles)
from trap.template import SpectralTemplate

import species

# plt.style.use("paper")

# rcParams['font.size'] = 12
# rc('font', **{'family': "DejaVu Sans", 'size': "12"})
rc('legend', **{'fontsize': "11"})

# rc('text', usetex=True)
# rc('font', **{'family': "sans-serif"})
# params = {'text.latex.preamble': [r'\usepackage{siunitx}',
#                                   r'\usepackage{sfmath}', r'\sisetup{detect-family = true}', r'\usepackage{amsmath}']}
# plt.rcParams.update(params)


def make_radial_profile(data, radial_bounds=None, bin_width=3.,
                        operation='mad_std',
                        yx_center=None, known_companion_mask=None):
    profile = np.empty_like(data)
    profile[:] = np.nan
    values = []

    data[data == 0.] = np.nan

    if radial_bounds is None:
        separation_max = data.shape[-1] // 2 * np.sqrt(2)
        radial_bounds = [1, int(separation_max)]

    if yx_center is None:
        yx_center = (data.shape[-2] // 2., data.shape[-1] // 2.)
    xy_center = yx_center[::-1]

    # Determine first non-zero separation, to prevent results below IWA
    inner_bound_index = int(yx_center[0] + radial_bounds[0])
    try:
        non_zero_separation = radial_bounds[0] + np.max(
            np.argwhere(np.isnan(data[inner_bound_index:inner_bound_index + 15,
                                      int(yx_center[1])]))) + 1
    except ValueError:
        non_zero_separation = 0
    if non_zero_separation > radial_bounds[0] + 13:
        non_zero_separation = 0

    for idx, separation in enumerate(range(radial_bounds[0], radial_bounds[1])):

        if separation < non_zero_separation:
            if operation == 'percentiles':
                values.append(np.ones(7) * np.nan)
            else:
                values.append(np.nan)
        else:
            # annulus_data = annulus_mask[0].multiply(data)
            # mask = annulus_mask[0].to_image(data.shape) > 0
            r_in = separation - bin_width / 2.
            r_out = separation + bin_width / 2.
            if r_in < 0.5:
                r_in = 0.5
            annulus_aperture = CircularAnnulus(
                xy_center, r_in=r_in, r_out=r_out)
            annulus_mask = annulus_aperture.to_mask(method='center')
            # Make sure only pixels are used for which data exists
            mask = annulus_mask.to_image(data.shape) > 0
            mask[int(xy_center[1]), int(xy_center[0])] = False

            if known_companion_mask is None:
                mask_wo_companion = mask
            else:
                mask_wo_companion = np.logical_and(mask, ~known_companion_mask)

            # Data on which statistic is applied
            annulus_data_1d = data[mask_wo_companion]
            all_nan = False
            if bn.allnan(annulus_data_1d):
                if operation == 'percentiles':
                    azimuthal_quantity = np.ones(7) * np.nan
                else:
                    azimuthal_quantity = np.nan
                all_nan = True
            else:
                # Profile made for each pixel in separation
                # For profile we need another mask that is only 1 pixel wide
                if bin_width > 1:
                    r_in = separation - 0.5
                    r_out = separation + 0.5
                    if r_in < 0.5:
                        r_in = 0.5
                    annulus_aperture = CircularAnnulus(
                        xy_center, r_in=r_in, r_out=r_out)
                    annulus_mask = annulus_aperture.to_mask(method='center')
                    # Make sure only pixels are used for which data exists
                    mask = annulus_mask.to_image(data.shape) > 0
                    mask[int(xy_center[1]), int(xy_center[0])] = False

                if operation == 'mad_std':
                    azimuthal_quantity = mad_std(annulus_data_1d, ignore_nan=True)
                elif operation == 'median':
                    azimuthal_quantity = bn.nanmedian(annulus_data_1d)
                elif operation == 'mean':
                    azimuthal_quantity = bn.nanmean(annulus_data_1d)
                elif operation == 'min':
                    azimuthal_quantity = bn.nanmin(annulus_data_1d)
                elif operation == 'std':
                    azimuthal_quantity = bn.nanstd(annulus_data_1d)
                elif operation == 'percentiles':
                    azimuthal_quantity = np.nanpercentile(
                        annulus_data_1d, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
                else:
                    raise ValueError('Unknown operation: use mad_std, median, or mean')
            if operation != 'percentiles':
                profile[mask] = azimuthal_quantity
            else:
                if all_nan:
                    profile[mask] = np.nan
                else:
                    profile[mask] = azimuthal_quantity[3]

            values.append(azimuthal_quantity)

    return profile, np.array(values)


def make_contrast_curve(detection_image, radial_bounds=None,
                        bin_width=3.,
                        companion_mask_radius=9,
                        pixel_scale=12.25,
                        yx_known_companion_position=None,
                        mask_above_sigma=None):
    yx_dim = (detection_image.shape[-2], detection_image.shape[-1])

    if radial_bounds is None:
        separation_max = detection_image.shape[-1] // 2 * np.sqrt(2)
        radial_bounds = [1, int(separation_max)]

    if yx_known_companion_position is not None:
        yx_known_companion_position = np.array(yx_known_companion_position)
        if yx_known_companion_position.ndim == 1:
            detected_signal_mask = regressor_selection.make_signal_mask(
                yx_dim, yx_known_companion_position, companion_mask_radius,
                relative_pos=True, yx_center=None)
        elif yx_known_companion_position.ndim == 2:
            detected_signal_masks = []
            for yx_pos in yx_known_companion_position:
                detected_signal_masks.append(regressor_selection.make_signal_mask(
                    yx_dim, yx_pos, companion_mask_radius,
                    relative_pos=True, yx_center=None))
            detected_signal_mask = np.logical_or.reduce(detected_signal_masks)
        else:
            raise ValueError(
                "Dimensionality of known companion positions for contrast curve too large.")
    else:
        detected_signal_mask = None
        # detected_signal_mask = np.zeros(detection_image[0].shape, dtype='bool')

    snr_norm_profile, snr_norm = make_radial_profile(
        detection_image[2], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='mad_std', known_companion_mask=detected_signal_mask)
    normalized_detection_image = detection_image[2] / snr_norm_profile

    # Repeat normalization without higher than 'mask_above_sigma' values
    # Removes most contamination by clear companion signals and binaries
    if mask_above_sigma is not None:
        local_detection_image = detection_image.copy()
        mask_high_values = normalized_detection_image > mask_above_sigma
        local_detection_image[:, mask_high_values] = np.nan

        snr_norm_profile, snr_norm = make_radial_profile(
            local_detection_image[2], (radial_bounds[0], radial_bounds[1]),
            bin_width=bin_width,
            operation='mad_std', known_companion_mask=detected_signal_mask)
        normalized_detection_image = detection_image[2] / snr_norm_profile
    else:
        local_detection_image = detection_image

    median_flux_profile, percentile_flux_values = make_radial_profile(
        local_detection_image[0], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='percentiles', known_companion_mask=detected_signal_mask)
    stddev_flux_profile, stddev_flux_values = make_radial_profile(
        local_detection_image[0], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='mad_std', known_companion_mask=detected_signal_mask)
    median_uncertainty_profile, percentile_uncertainty_values = make_radial_profile(
        local_detection_image[1], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='percentiles', known_companion_mask=detected_signal_mask)
    min_uncertainty_profile, min_uncertainty_values = make_radial_profile(
        local_detection_image[1], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='min', known_companion_mask=detected_signal_mask)

    # contrast_norm_profile, contrast_norm_values = make_radial_profile(
    #     detection_image[1], (radial_bounds[0], radial_bounds[1]),
    #     bin_width=bin_width,
    #     operation='std', known_companion_mask=detected_signal_mask)

    # contrast_curve_table = np.vstack([np.arange(radial_bounds[0], radial_bounds[1])])
    uncertainty_image = detection_image[1] * snr_norm_profile
    median_uncertainty_image = median_uncertainty_profile * snr_norm_profile
    percentile_contrast_curve = percentile_uncertainty_values * snr_norm[:, None]
    min_contrast_curve = min_uncertainty_values * snr_norm

    separation_pix = np.arange(radial_bounds[0], radial_bounds[1])
    separation_mas = np.arange(radial_bounds[0], radial_bounds[1]) * pixel_scale

    cols = [separation_pix,
            separation_mas,
            min_contrast_curve]
    for column in percentile_contrast_curve.T:
        cols.append(column)
    cols.append(snr_norm)

    col_names = [
        'sep (pix)', 'sep (mas)',
        'contrast_min', 'contrast_0.15', 'contrast_2.5',
        'contrast_16', 'contrast_50', 'contrast_84',
        'contrast_97.5', 'contrast_99.85', 'snr_normalization']
    contrast_table = Table(cols, names=col_names)

    return normalized_detection_image, contrast_table, uncertainty_image, median_uncertainty_image


def ratio_contrast_tables(table1, table2):
    ratio_table = table1.copy()
    ratio_table['contrast_50'] = table1['contrast_50'] / table2['contrast_50']
    ratio_table['contrast_16'] = table1['contrast_16'] / table2['contrast_16']
    ratio_table['contrast_84'] = table1['contrast_84'] / table2['contrast_84']
    ratio_table['contrast_2.5'] = table1['contrast_2.5'] / table2['contrast_2.5']
    ratio_table['contrast_97.5'] = table1['contrast_97.5'] / table2['contrast_97.5']
    return ratio_table


def sep_pix_to_mas(sep_pix, instrument):
    return (sep_pix * u.pixel).to(u.mas, instrument.pixel_scale).value


def sep_pix_to_lod(separation, instrument):
    fwhm = instrument.fwhm
    lod = separation / fwhm[0]
    return lod.value


def contrast_to_magnitude(contrast):
    return -2.5 * np.log10(contrast)


def convert_flux(contrast, convert=False):
    if convert:
        return contrast_to_magnitude(contrast)
    else:
        return contrast


def add_contrast_curve_to_ax(
        ax0, contrast_table, sigma=5, color='#1b1cd5',
        linestyle='-', curvelabel=None,
        convert_to_mag=False, plot_percentiles=True,
        plot_dashed_outline=True,
        percentile_1sigma_alpha=0.6,
        percentile_2sigma_alpha=0.3,
        percentile_3sigma_alpha=0.):

    if curvelabel is not None:
        label = curvelabel
    else:
        label = None

    if linestyle is None:
        linestyle = '-'

    ax0.plot(contrast_table['sep (pix)'],
             convert_flux(contrast_table['contrast_50']*sigma, convert=convert_to_mag).data,
             color=color, linestyle=linestyle, label=label.format(curvelabel))  # plotting y versus x
    if plot_percentiles:
        ax0.fill_between(
            contrast_table['sep (pix)'],
            convert_flux(contrast_table['contrast_16']*sigma, convert=convert_to_mag),
            convert_flux(contrast_table['contrast_84']*sigma, convert=convert_to_mag),
            alpha=percentile_1sigma_alpha, color=color)  # shade the area between +- 1 sigma
        if plot_dashed_outline:
            ax0.plot(contrast_table['sep (pix)'],
                     convert_flux(contrast_table['contrast_16']*sigma, convert=convert_to_mag).data,
                     color=color, alpha=percentile_1sigma_alpha, linestyle='--', label=None)  # plotting y versus x
            ax0.plot(contrast_table['sep (pix)'],
                     convert_flux(contrast_table['contrast_84']*sigma, convert=convert_to_mag).data,
                     color=color, alpha=percentile_1sigma_alpha, linestyle='--', label=None)  # plotting y versus x

        if percentile_2sigma_alpha > 0:
            ax0.fill_between(
                contrast_table['sep (pix)'],
                convert_flux(contrast_table['contrast_84']*sigma, convert=convert_to_mag),
                convert_flux(contrast_table['contrast_97.5']*sigma, convert=convert_to_mag),
                alpha=percentile_2sigma_alpha, color=color)
        if percentile_3sigma_alpha > 0:
            ax0.fill_between(
                contrast_table['sep (pix)'],
                convert_flux(contrast_table['contrast_2.5']*sigma, convert=convert_to_mag),
                convert_flux(contrast_table['contrast_16']*sigma, convert=convert_to_mag),
                alpha=percentile_3sigma_alpha, color=color)
    return ax0


def plot_contrast_curve(
        contrast_table, instrument=None, wavelengths=[None],
        companion_table=None,
        template_fitted=False,
        colors=['#1b1cd5'],
        linestyles=['-'],
        curvelabels=[None],
        add_wavelength_label=False,
        plot_vertical_lod=False,
        mirror_axis='mas', convert_to_mag=False,
        yscale='linear', savefig=None,
        sigma=5,
        radial_bound=None,
        plot_percentiles=True,
        plot_dashed_outline=True,
        percentile_1sigma_alpha=0.6,
        percentile_2sigma_alpha=0.3,
        percentile_3sigma_alpha=0.,
        set_xlim=None,
        set_ylim=None,
        plot_iwa=False,
        title=None,
        cmap=plt.cm.viridis,
        show=False):

    try:
        wavelengths = wavelengths.to(u.micron)
    except:
        raise TypeError("Wavelengths must be a quantity array.")

    if savefig is not None:
        result_folder = os.path.dirname(savefig)
        contrast_plot_path_pdf = os.path.join(
            result_folder, os.path.splitext(os.path.basename(savefig))[0] + '.pdf')
        pdf = PdfPages(contrast_plot_path_pdf)

    plt.close()
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(1, 1)
    ax0 = fig.add_subplot(grid[0, 0])

    if colors is None:
        colors = colors = cmap(np.linspace(0, 1, len(contrast_table)))

        if wavelengths[0] is not None:
            wavelength_range = np.max(wavelengths - np.min(wavelengths))
            scaled_values = (wavelengths - np.min(wavelengths)) / wavelength_range
            colors = cmap(scaled_values)
        else:
            colors = cmap(np.linspace(0, 1, len(contrast_table)))

    # Add contrast curve(s) to axis
    for idx, contrast_curve in enumerate(contrast_table):
        if curvelabels[idx] is None:
            curvelabel = ''
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = curvelabel + "{:.2f} ".format(wavelengths[idx].value) \
                + wavelengths.unit.to_string() + ""

        ax0 = add_contrast_curve_to_ax(
            ax0, contrast_curve,
            sigma=sigma,
            color=colors[idx],
            linestyle=linestyles[idx], curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            plot_dashed_outline=plot_dashed_outline,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha
        )

        if companion_table is not None:
            if len(companion_table) > 0:
                wavelength_indices = np.unique(companion_table['wavelength_index'])
                temp_table = companion_table[companion_table['wavelength_index']
                                             == wavelength_indices[idx]]
                if template_fitted:
                    contrast_indices_closest_to_candidate = []
                    contrast_closest_to_candidate = []
                    for separation in temp_table['separation'].values:
                        contrast_index_closest_to_candidate = \
                            find_nearest(
                                separation,
                                contrast_curve['sep (pix)'])
                        contrast_indices_closest_to_candidate.append(
                            contrast_index_closest_to_candidate)
                        contrast_closest_to_candidate.append(
                            contrast_curve['contrast_50'][contrast_index_closest_to_candidate])

                    contrast = temp_table['norm_snr_fit'].values * contrast_closest_to_candidate
                    contrast_uncertainty = None
                else:
                    contrast = temp_table['contrast'].values
                    contrast_uncertainty = temp_table['uncertainty'].values
                if np.all(np.isfinite(contrast)):
                    ax0.errorbar(
                        x=temp_table['separation'].values,
                        y=contrast,
                        yerr=contrast_uncertainty,
                        fmt='o',
                        color=colors[idx],
                        markeredgecolor='k',
                        markeredgewidth=1,
                        capsize=3)

            for separation in np.unique(companion_table['separation']):
                ax0.axvline(x=separation, color='k', linestyle=':', alpha=0.3)

    ax0.set_yscale(yscale)

    if set_xlim is not None:
        ax0.set_xlim(set_xlim)

    if set_ylim is not None:
        ax0.set_ylim(set_ylim)

    ymin, ymax = ax0.get_ylim()
    xmin, xmax = ax0.get_xlim()

    x_text_shift = np.abs((xmax - xmin) / 100.) * 1.1
    y_text_shift = np.abs((ymax - ymin) / 100.)

    # 10% padding at bottom for text
    # if convert_to_mag:
    #     ax0.set_ylim(ymin, ymax + y_text_shift * 10)
    # else:
    #     ax0.set_ylim(ymin - y_text_shift * 15, ymax)
    # if yscale == 'log':
    #     ax0.set_ylim(ymin - y_text_shift * 20, ymax)
    if yscale == 'linear':
        ax0.set_ylim(ymin - y_text_shift, ymax)
    ymin, ymax = ax0.get_ylim()
    if yscale == 'log':
        y_text_shift = np.abs((np.log10(ymax) - np.log10(ymin))) / 100 * 6
        # y_text_pos = ymin  # 10**(np.log10(ymin) + y_text_shift)
        ax0.set_ylim(10**(np.log10(ymin) - y_text_shift), ymax)
        y_text_pos = 10**(np.log10(ymin) - y_text_shift / 2)
    elif yscale == 'linear':
        y_text_pos = ymin

    if plot_vertical_lod:
        fwhm = instrument.fwhm[0]
        xposition = (np.array([1, 2, 3, 5, 10]) * fwhm).value
        mask = np.logical_and(xposition > xmin, xposition < xmax)
        xposition = xposition[mask]
        vert_labels = np.array([
            "$1 \lambda/D$", "$2 \lambda/D$", "$3 \lambda/D$",
            "$5 \lambda/D$", "$10 \lambda/D$"])[mask]

        if np.sum(mask) > 0:
            if convert_to_mag:
                text_y = ymax
                y_text_shift *= -1.
            else:
                text_y = ymin

            for idx, xc in enumerate(xposition):
                ax0.axvline(x=xc, color='k', linestyle='--', alpha=0.3)  # , label=vert_labels[idx])
                plt.text(
                    xc + x_text_shift, y_text_pos,
                    vert_labels[idx], rotation=90, verticalalignment='bottom')

    ax0.set_xlabel("Separation (pixel)")
    if convert_to_mag:
        ax0.set_ylabel("{}$\sigma$ contrast (mag)".format(sigma))
    else:
        ax0.set_ylabel("{}$\sigma$ contrast".format(sigma))

    # set ticks visible, if using sharex = True. Not needed otherwise
    ax0.set_xlim(xmin, xmax + x_text_shift)

    if plot_iwa is not False:
        if plot_iwa > xmin:
            if set_ylim is not None:
                ymin, ymax = set_ylim
            else:
                ymin, ymax = ax0.get_ylim()

            ax0.axvspan(xmin, plot_iwa, alpha=0.3, color='black')
            if yscale == 'log':
                ytext_iwa = 10**(np.log10(ymax) + (np.log10(ymin) - np.log10(ymax)) / 2.)
            else:
                ytext_iwa = ymax - (ymax - ymin) * 0.05
            ax0.text(
                plot_iwa, ytext_iwa,
                r"IWA",
                rotation=90,
                horizontalalignment='right',
                verticalalignment='center',
                color='black',
                family='monospace',
                fontsize=12)

    # ax0.set_ylim(np.min(contrast_table['contrast_min']), np.max(contrast_table['contrast_99.85']))
    # for tick in axes[0].get_xticklabels():
    ax0.tick_params(right=0, top=0, which='both', direction='out')  # bottom='on')
    # ax0.ticklabel_format(which='both', style='sci')
    # ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    ax2 = ax0.twiny()  # .twinx()
    # ax2.set_ylabel("Contrast (mag)")
    ax2.tick_params(which='both', direction='out')
    ax2.tick_params(left=0, right=0, bottom=0, which='both', direction='out')

    # get left axis limits
    # xmin, xmax = ax0.get_xlim()
    if mirror_axis == 'lod':
        ax2.set_xlabel("Separation ($\lambda / D$)")
        ax2.set_xlim((
            sep_pix_to_lod(xmin * u.pixel, wavelengths[0], instrument),
            sep_pix_to_lod(xmax * u.pixel, wavelengths[0], instrument)))
        ax2.plot([], [])
    elif mirror_axis == 'mas':
        ax2.set_xlim((sep_pix_to_mas(xmin, instrument), sep_pix_to_mas(xmax, instrument)))
        ax2.set_xlabel("Separation (mas)")
        ax2.plot([], [])
    else:
        raise ValueError("Only 'lod' and 'mas' possible")

    ax3 = ax0.twinx()
    ax3.set_ylim((contrast_to_magnitude(ymin), contrast_to_magnitude(ymax)))
    ax3.set_ylabel("$\Delta \,$magnitude")
    ax3.plot([], [])
    ax0.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    # ymin, ymax = ax0.get_ylim()
    # ax2.set_xlim((contrast_to_magnitude(ymin), contrast_to_magnitude(ymax)))
    # apply function and set transformed values to right axis limits
    # ax2.set_xlim(contrast_table['sep (mas)'][0], contrast_table['sep (mas)'][-1])
    # set an invisible artist to twin axes
    # to prevent falling back to initial values on rescale events

    if title is not None:
        plt.title(title)

    if len(contrast_table) < 5:
        ax0.legend(loc=1)
    else:
        if wavelengths is not None:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(
                    vmin=wavelengths.value[0], vmax=wavelengths.value[-1]))
            cb = plt.colorbar(
                sm, ax=ax3, pad=0.13, use_gridspec=True, fraction=0.045)  # , format='%.2f')
            cb.set_label('wavelength (micron)', rotation=90, labelpad=10)

    # fig.tight_layout()
    if convert_to_mag:
        plt.gca().invert_yaxis()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
        try:
            pdf.savefig(bbox_inches='tight')
        except RuntimeError:
            print("Could not output pdf-version of contrast curve (this may be a Mac issue).")
        pdf.close()

    if show is True:
        plt.show()

    return fig


def plot_contrast_curve_ratio(
        contrast_table, instrument=None, wavelengths=[None],
        colors=['#1b1cd5'],
        linestyles=['-'],
        curvelabels=[None],
        add_wavelength_label=False,
        plot_vertical_lod=False,
        mirror_axis='mas', convert_to_mag=False,
        yscale='linear', savefig=None,
        radial_bound=None,
        plot_percentiles=True,
        percentile_1sigma_alpha=0.6,
        percentile_2sigma_alpha=0.3,
        percentile_3sigma_alpha=0.,
        set_xlim=None,
        set_ylim=None,
        plot_iwa=False,
        show=False):

    try:
        wavelengths = wavelengths.to(u.micron)
    except:
        raise TypeError("Wavelengths must be a quantity array.")

    if savefig is not None:
        result_folder = os.path.dirname(savefig)
        contrast_plot_path_pdf = os.path.join(
            result_folder, os.path.splitext(os.path.basename(savefig))[0] + '.pdf')
        pdf = PdfPages(contrast_plot_path_pdf)

    plt.close()
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(1, 1)
    ax0 = fig.add_subplot(grid[0, 0])

    # Add contrast curve(s) to axis
    for idx, contrast_curve in enumerate(contrast_table):
        if curvelabels[idx] is None:
            curvelabel = ''
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = "{:.2f}".format(wavelengths[idx].value) \
                + wavelengths.unit.to_string() + ""
        ax0 = add_contrast_curve_to_ax(
            ax0, contrast_curve, color=colors[idx],
            linestyle=linestyles[idx], curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha)

    ax0.set_yscale(yscale)

    if set_xlim is not None:
        ax0.set_xlim(set_xlim)

    ymin, ymax = ax0.get_ylim()
    xmin, xmax = ax0.get_xlim()

    x_text_shift = np.abs((xmax - xmin) / 100.) * 1.1
    y_text_shift = np.abs((ymax - ymin) / 100.)

    if yscale == 'log':
        ax0.set_ylim(ymin - y_text_shift * 15, ymax)
    if yscale == 'linear':
        # ax0.set_ylim(ymin - y_text_shift * 15, ymax)
        ax0.set_ylim(0, ymax)
    ymin, ymax = ax0.get_ylim()
    if yscale == 'log':
        y_text_shift = np.abs((np.log10(ymax) - np.log10(ymin))) / 100 * 2
        y_text_pos = ymin  # 10**(np.log10(ymin) + y_text_shift)
        ax0.set_ylim(10**(np.log10(ymin) - y_text_shift), ymax)
    elif yscale == 'linear':
        y_text_pos = ymin + 0.1

    if plot_vertical_lod:
        fwhm = instrument.fwhm[0]
        xposition = (np.array([1, 2, 3, 5, 10]) * fwhm).value
        vert_labels = [
            "$1 \lambda/D$", "$2 \lambda/D$", "$3 \lambda/D$",
            "$5 \lambda/D$", "$10 \lambda/D$"]

        text_y = ymin

        for idx, xc in enumerate(xposition):
            ax0.axvline(x=xc, color='k', linestyle='--', alpha=0.3)  # , label=vert_labels[idx])
            plt.text(
                xc + x_text_shift, y_text_pos,
                vert_labels[idx], rotation=90, verticalalignment='bottom')

    ax0.set_xlabel("Separation (pixel)")
    ax0.set_ylabel("Factor gained in contrast")

    # set ticks visible, if using sharex = True. Not needed otherwise
    ax0.set_xlim(xmin, xmax + x_text_shift)

    if plot_iwa is not False:
        if plot_iwa > xmin:
            if set_ylim is not None:
                ymin, ymax = set_ylim
            else:
                ymin, ymax = ax0.get_ylim()
            ax0.axvspan(xmin, plot_iwa, alpha=0.3, color='black')
            ax0.text(
                plot_iwa, ymax - (ymax - ymin) * 0.05,
                r"IWA",
                rotation=90,
                horizontalalignment='right',
                verticalalignment='center',
                color='black',
                family='monospace',
                fontsize=12)

    # ax0.set_ylim(np.min(contrast_table['contrast_min']), np.max(contrast_table['contrast_99.85']))
    # for tick in axes[0].get_xticklabels():
    ax0.tick_params(right=0, top=0, which='both', direction='out')  # bottom='on')
    # ax0.ticklabel_format(which='both', style='sci')
    # ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    ax2 = ax0.twiny()  # .twinx()
    # ax2.set_ylabel("Contrast (mag)")
    ax2.tick_params(which='both', direction='out')
    ax2.tick_params(left=0, right=0, bottom=0, which='both', direction='out')

    # get left axis limits
    # xmin, xmax = ax0.get_xlim()
    if mirror_axis == 'lod':
        ax2.set_xlabel("Separation ($\lambda / D$)")
        ax2.set_xlim((
            sep_pix_to_lod(xmin * u.pixel, wavelengths[0], instrument),
            sep_pix_to_lod(xmax * u.pixel, wavelengths[0], instrument)))
        ax2.plot([], [])
    elif mirror_axis == 'mas':
        ax2.set_xlim((sep_pix_to_mas(xmin, instrument), sep_pix_to_mas(xmax, instrument)))
        ax2.set_xlabel("Separation (mas)")
        ax2.plot([], [])
    else:
        raise ValueError("Only 'lod' and 'mas' possible")
    ax0.axhline(y=1., color='k', linestyle='-', alpha=0.5)

    # ax3 = ax0.twinx()
    # ax3.set_ylim((contrast_to_magnitude(ymin), contrast_to_magnitude(ymax)))
    # ax3.set_ylabel("magnitude")
    # ax3.plot([], [])
    ax0.minorticks_on()
    ax2.minorticks_on()
    # ax3.minorticks_on()
    # ymin, ymax = ax0.get_ylim()
    # ax2.set_xlim((contrast_to_magnitude(ymin), contrast_to_magnitude(ymax)))
    # apply function and set transformed values to right axis limits
    # ax2.set_xlim(contrast_table['sep (mas)'][0], contrast_table['sep (mas)'][-1])
    # set an invisible artist to twin axes
    # to prevent falling back to initial values on rescale events

    if set_ylim is not None:
        ax0.set_ylim(set_ylim)

    # ax0.yaxis.label.set_size(40)
    # ax0.xaxis.label.set_size(40)

    ax0.legend(loc=1)
    fig.tight_layout()
    if convert_to_mag:
        plt.gca().invert_yaxis()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
        try:
            pdf.savefig(bbox_inches='tight')
        except RuntimeError:
            print("Could not output pdf-version of contrast curve (this may be a Mac issue).")
        pdf.close()

    if show is True:
        plt.show()

    return fig


def prepare_andromeda_output(andromeda_contrast, andromeda_norm_stddev, andromeda_snr):
    from scipy.ndimage.interpolation import shift
    andromeda_std = andromeda_snr / andromeda_contrast
    andromeda_std = 1. / andromeda_std
    andromeda_stack = np.array([andromeda_contrast, andromeda_std, andromeda_snr])

    # andromeda_stack[np.isnan(andromeda_stack)] = 0.
    for i, image in enumerate(andromeda_stack):
        andromeda_stack[i] = shift(image, (0.5, 0.5), order=1, prefilter=False)

    # Mask inner most annulus of non-zero values (corrupted by shift)
    radial_bounds_test = (1, 25)
    _, andro_radial_test = make_radial_profile(
        andromeda_stack[2], radial_bounds=radial_bounds_test, bin_width=1,
        operation='mad_std',
        yx_center=None, known_companion_mask=None)
    corrupt_separation = radial_bounds_test[0] + \
        np.max(np.argwhere(np.isnan(andro_radial_test))) + 1
    assert corrupt_separation != radial_bounds_test[-1], "Innermost non-zero andromeda result at edge of image"

    xy_center = (andromeda_stack.shape[-1] // 2, andromeda_stack.shape[-2] // 2)
    annulus_aperture = CircularAnnulus(
        xy_center,
        r_in=corrupt_separation - 0.5,
        r_out=corrupt_separation + 0.5)
    annulus_mask = annulus_aperture.to_mask(method='center')
    mask = annulus_mask.to_image(andromeda_stack[0].shape) > 0
    andromeda_stack[:, mask] = 0.

    return andromeda_stack


def plot_distribution(detection_image, radial_bounds=None,
                      plot_type='qqplot', sigma=5,
                      companion_mask_radius=10,
                      pixel_scale=12.25, yx_known_companion_position=None):

    yx_dim = (detection_image.shape[-2], detection_image.shape[-1])
    if yx_known_companion_position is not None:
        detected_signal_mask = regressor_selection.make_signal_mask(
            yx_dim, yx_known_companion_position, companion_mask_radius, relative_pos=True, yx_center=None)
    else:
        detected_signal_mask = np.zeros(detection_image[0].shape, dtype='bool')

    annulus_mask = regressor_selection.make_annulus_mask(
        radial_bounds[0], radial_bounds[1],
        yx_dim=(detection_image.shape[-2], detection_image.shape[-1]),
        yx_center=None)
    mask = np.logical_and(annulus_mask, ~detected_signal_mask)
    if plot_type == 'qqplot':
        res = stats.probplot(detection_image[mask], dist="norm", plot=plt)
    elif plot_type == 'distplot':
        sns.distplot(detection_image[mask], label='TRAP')
    plt.legend()
    plt.show()


def fit_2d_gaussian(
        image, yx_position=None, yx_center=None, x_stddev=1.43, y_stddev=2.63,
        box_size=7, mask_deviating=False, deviation_threshold=0.1,
        fix_width=True, fix_orientation=True, plot=False):

    if yx_center is None:
        yx_center = (image.shape[0] // 2., image.shape[1] // 2)
    # spot fitting
    if yx_position is None:
        cy, cx = np.unravel_index(np.nanargmax(image), image.shape)
    else:
        cy, cx = yx_position
    cutout = Cutout2D(image, (cx, cy), box_size)
    # stamp = image[cy - box_size:cy + box_size, cx - box_size:cx + box_size].copy()
    xx, yy = np.meshgrid(np.arange(box_size), np.arange(box_size))
    yx_position_cutout = np.unravel_index(np.nanargmax(cutout.data), cutout.shape)
    gbounds = {
        'amplitude': (1e-9, None),
        'x_mean': (-2.0, box_size+2),
        'y_mean': (-2.0, box_size+2),
        'x_stddev': (0.5, box_size),
        'y_stddev': (0.5, box_size)
    }
    relative_yx = absolute_yx_to_relative_yx(yx_position, yx_center)
    rhophi = relative_yx_to_rhophi(relative_yx)
    phi = rhophi[1] * np.pi / 180.

    g_init = models.Gaussian2D(
        amplitude=np.nanmax(cutout.data),
        x_mean=yx_position_cutout[1],
        y_mean=yx_position_cutout[0],
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=phi,
        bounds=gbounds)  # + models.Const2D(amplitude=stamp.min())

    if fix_width:
        g_init.x_stddev.fixed = True
        g_init.y_stddev.fixed = True
    if fix_orientation:
        g_init.theta.fixed = True

    finite_mask = np.logical_and(np.isfinite(cutout.data), cutout.data != 0.)
    fitter = fitting.LevMarLSQFitter()
    par = fitter(g_init, xx[finite_mask], yy[finite_mask], cutout.data[finite_mask])
    model = par(xx, yy)
    if plot:
        plt.imshow(cutout.data, origin='lower')
        plt.show()
        plt.imshow(model, origin='lower')
        plt.show()

    # Check if model fit is close to data (this is not perfect in case the detection is on the edge)
    # May require special treatment or warning
    mask = abs(cutout.data - model) / np.mean(
        np.vstack([cutout.data[None, :], model[None, :]]), axis=0) < deviation_threshold  # Filter out
    if mask_deviating:
        g_init = models.Gaussian2D(
            amplitude=par.amplitude.value,
            x_mean=par.x_mean.value,
            y_mean=par.y_mean.value,
            x_stddev=par.x_stddev.value,
            y_stddev=par.y_stddev.value,
            theta=par.theta.value)
        par = fitter(g_init, xx[mask], yy[mask], cutout.data[mask])
        model = par(xx, yy)

    # Cutout works with x first and y second
    xy_fit_position_orig = cutout.to_original_position((par.x_mean.value, par.y_mean.value))
    yx_fit_position_orig = xy_fit_position_orig[::-1]
    yx_fit_relative = (xy_fit_position_orig[1] - yx_center[0],
                       xy_fit_position_orig[0] - yx_center[1])

    fwhm_area = par.x_stddev.value * par.y_stddev.value * 2.355**2 * np.pi

    parameters = {
        'parameters': par, 'model': model, 'cutout': cutout.data,
        'yx_fit_position_orig': yx_fit_position_orig,
        'yx_fit_relative': yx_fit_relative,
        'mask': mask,
        'fwhm_area': fwhm_area
    }

    return parameters


def plot_model_and_data(model, stamp):
    plt.close()
    plt.figure(200)
    plt.imshow(model, origin='lower', interpolation='nearest')
    plt.figure(300)
    plt.imshow(stamp, origin='lower', interpolation='nearest')
    plt.show()


def fit_planet_parameters(
        detection_image, normalized_detection_image,
        contrast_table, yx_position, x_stddev=1.43, y_stddev=2.63,
        box_size=7, iterate=False, mask_deviating=False, deviation_threshold=0.1,
        fix_width=True, fix_orientation=True, plot=False):

    # yx_dim = (detection_image.shape[-2], detection_image.shape[-1])
    # yx_center = (yx_dim[0] // 2, yx_dim[1] // 2)

    # yx_known_companion_position = [-35.95, -8.43]
    # companion_mask_radius = 15
    # if yx_known_companion_position is not None:
    #     detected_signal_mask = make_signal_mask(
    #         yx_dim, yx_known_companion_position, companion_mask_radius,
    #         relative_pos=True, yx_center=None)
    #
    # profile, values = make_radial_profile(
    #     detection_image[0], bin_width=1,
    #     radial_bounds=radial_bounds,
    #     known_companion_mask=detected_signal_mask,
    #     operation='median')
    # detection_image[0] = detection_image[0] - profile
    # fit_snr, model_snr, stamp_snr, yx_pos_absolute_snr, yx_pos_relative_snr = fit_2d_gaussian(
    # normalized_detection_image, yx_position=yx_position, box_size=box_size)

    # if iterate:
    #     yx_position = np.round(np.array(yx_pos_absolute_snr)).astype('int')
    #     fit_snr, model_snr, stamp_snr, yx_pos_absolute_snr, yx_pos_relative_snr = fit_2d_gaussian(
    #         normalized_detection_image, yx_position=yx_position, box_size=box_size)

    # xy_in_stamp = (fit_snr.x_mean.value, fit_snr.y_mean.value)

    contrast_image_parameters = fit_2d_gaussian(
        detection_image[0], yx_position=yx_position,
        x_stddev=x_stddev, y_stddev=y_stddev, box_size=box_size,
        mask_deviating=mask_deviating, deviation_threshold=deviation_threshold,
        fix_width=fix_width, fix_orientation=fix_orientation)

    snr_image_parameters = fit_2d_gaussian(
        # normalized_detection_image,
        detection_image[2],
        yx_position=yx_position,
        x_stddev=x_stddev, y_stddev=y_stddev, box_size=box_size,
        mask_deviating=mask_deviating, deviation_threshold=deviation_threshold,
        fix_width=fix_width, fix_orientation=fix_orientation)

    norm_snr_image_parameters = fit_2d_gaussian(
        normalized_detection_image,
        yx_position=yx_position,
        x_stddev=x_stddev, y_stddev=y_stddev, box_size=box_size,
        mask_deviating=mask_deviating, deviation_threshold=deviation_threshold,
        fix_width=fix_width, fix_orientation=fix_orientation)

    contrast_image_results = summarize_2d_gauss_fit_result(contrast_image_parameters)
    snr_image_results = summarize_2d_gauss_fit_result(snr_image_parameters)
    norm_snr_image_results = summarize_2d_gauss_fit_result(norm_snr_image_parameters)

    # uncertainty_nearest = normalized_noise_map[int(yx_pos_absolute_snr[0]), int(yx_pos_absolute_snr[1])]
    # snr_nearest = contrast / (uncertainty_nearest / 5)

    # contrast = contrast_image_parameters['parameters'].amplitude.value  # fit_contrast(*xy_in_stamp)

    if plot:
        plot_model_and_data(snr_image_parameters['model'], snr_image_parameters['cutout'])
        plot_model_and_data(contrast_image_parameters['model'], contrast_image_parameters['cutout'])

    return contrast_image_results, snr_image_results, norm_snr_image_results


def summarize_2d_gauss_fit_result(result_dictionary):

    fitted_parameters = {'x': [], 'y': [], 'x_relative': [], 'y_relative': [],
                         'separation': [], 'position_angle': [], 'amplitude': [],
                         'x_fwhm': [], 'y_fwhm': [], 'theta': [],
                         'good_pixels': [], 'fwhm_area': [],
                         'good_fraction': []}

    fitted_parameters['x'].append(result_dictionary['yx_fit_position_orig'][1])
    fitted_parameters['y'].append(result_dictionary['yx_fit_position_orig'][0])
    fitted_parameters['x_relative'].append(result_dictionary['yx_fit_relative'][1])
    fitted_parameters['y_relative'].append(result_dictionary['yx_fit_relative'][0])
    rhophi = image_coordinates.relative_yx_to_rhophi(
        result_dictionary['yx_fit_relative'])
    fitted_parameters['separation'].append(rhophi[0])
    fitted_parameters['position_angle'].append(rhophi[1])
    fitted_parameters['amplitude'].append(result_dictionary['parameters'].amplitude.value)
    fitted_parameters['x_fwhm'].append(result_dictionary['parameters'].x_stddev.value * 2.355)
    fitted_parameters['y_fwhm'].append(result_dictionary['parameters'].y_stddev.value * 2.355)
    fitted_parameters['theta'].append(
        (result_dictionary['parameters'].theta.value * u.radian).to(u.degree).value)
    fitted_parameters['good_pixels'].append(
        np.sum(result_dictionary['mask']))
    fitted_parameters['fwhm_area'].append(result_dictionary['fwhm_area'])
    fitted_parameters['good_fraction'].append(
        np.sum(result_dictionary['mask']) / result_dictionary['fwhm_area'])
    fitted_parameters = pd.DataFrame(fitted_parameters)
    return fitted_parameters


class DetectionAnalysis(object):
    """ Class for """

    def __init__(
            self,
            result_folder=None,
            detection_images=None,
            wavelength_indices=None,
            instrument=None,
            reduction_parameters=None):
        """

        """

        self.detection_images = detection_images
        self.wavelength_indices = wavelength_indices
        self.instrument = instrument
        self.reduction_parameters = reduction_parameters
        self.detected_signal_mask = None
        self.templates = OrderedDict()

        if instrument is not None:
            self.instrument.compute_fwhm()

    def read_output(
            self, component_fraction, result_folder=None,
            reduction_type='temporal', correlated_residuals=False,
            read_parameters=True, reduction_parameters=None,
            instrument=None):

        if result_folder is None:
            self.result_folder = self.reduction_parameters.result_folder
        else:
            self.result_folder = result_folder

        self.component_fraction = component_fraction
        self.correlated_residuals = correlated_residuals
        self.reduction_type = reduction_type

        if read_parameters:
            with open(os.path.join(result_folder, "reduction_parameters.obj"), 'rb') as handle:
                self.reduction_parameters = pickle.load(handle)

            with open(os.path.join(result_folder, "instrument.obj"), 'rb') as handle:
                self.instrument = pickle.load(handle)
        else:
            if reduction_parameters is not None and instrument is not None:
                self.reduction_parameters = reduction_parameters
                self.instrument = instrument
        self.instrument.compute_fwhm()

        if correlated_residuals:
            detection_image_name = "detection_corr_lam"
        else:
            detection_image_name = "detection_lam"

        glob_pattern = os.path.join(
            self.result_folder,
            detection_image_name + "*frac{:.2f}_{}.fits".format(component_fraction, reduction_type))
        detection_file_paths = natsorted(glob(glob_pattern))

        assert len(detection_file_paths) > 0, "No output files found for:\n{}.".format(glob_pattern)

        # Read in data
        detection_cube = []
        for file in detection_file_paths:
            detection_cube.append(fits.getdata(file))
        self.detection_cube = np.array(detection_cube)

        # Determine indices reduced
        filenames = [os.path.basename(file_path) for file_path in detection_file_paths]
        character_index = filenames[0].find('lam')
        self.wavelength_indices = np.array(
            [int(file[character_index+3:character_index+5]) for file in filenames])

        # Remove wavelength from name
        # Add ouput paths to class
        self.file_paths = {}
        self.basename = filenames[0].replace('_lam{:02d}'.format(self.wavelength_indices[0]), '')
        self.file_paths['detection_image_path'] = os.path.join(
            self.result_folder, self.basename)
        self.file_paths['norm_detection_image_path'] = os.path.join(
            self.result_folder, self.basename.replace('detection', 'norm_detection'))
        self.file_paths['contrast_table_path'] = os.path.join(
            self.result_folder, self.basename.replace('detection', 'contrast_table'))
        self.file_paths['uncertainty_image_path'] = os.path.join(
            self.result_folder, self.basename.replace('detection', 'uncertainty_image'))
        self.file_paths['median_uncertainty_image_path'] = os.path.join(
            self.result_folder, self.basename.replace('detection', 'median_uncertainty_image'))
        self.file_paths['contrast_plot_path'] = os.path.join(
            self.result_folder, os.path.splitext(self.basename)[0].replace('detection', 'contrast_plot') + '.jpg')

        if not os.path.exists(self.file_paths['detection_image_path']):
            fits.writeto(
                self.file_paths['detection_image_path'], self.detection_cube, overwrite=True)

    def contrast_table_and_normalization(
            self, detection_cube=None,
            cube_indices=None,
            yx_known_companion_position=None,
            mask_above_sigma=None,
            save=False,
            file_paths=None,
            overwrite=True,
            inplace=True):
        """ detection_cube contains the detection_image for all wavelengths """

        if detection_cube is None:
            detection_cube_used = self.detection_cube
        else:
            detection_cube_used = detection_cube

        if yx_known_companion_position is None:
            yx_known_companion_position = self.reduction_parameters.yx_known_companion_position

        detection_products = {}
        normalized_detection_cube = []
        contrast_tables = []
        uncertainty_cube = []
        median_uncertainty_cube = []

        self.pixel_scale_mas = (1 * u.pixel).to(u.mas, self.instrument.pixel_scale).value

        if cube_indices is None:
            cube_indices = list(range(len(detection_cube_used)))

        for cube_index in cube_indices:
            # for detection_image in detection_cube_used:
            normalized_detection_image, contrast_table, uncertainty_image, median_uncertainty_image = make_contrast_curve(
                detection_cube_used[cube_index],
                radial_bounds=None,
                bin_width=self.reduction_parameters.normalization_width,
                companion_mask_radius=self.reduction_parameters.companion_mask_radius,
                pixel_scale=self.pixel_scale_mas,
                mask_above_sigma=mask_above_sigma,
                yx_known_companion_position=yx_known_companion_position)
            normalized_detection_cube.append(normalized_detection_image)
            contrast_tables.append(contrast_table)
            uncertainty_cube.append(uncertainty_image)
            median_uncertainty_cube.append(median_uncertainty_image)

        detection_products['normalized_detection_cube'] = np.array(normalized_detection_cube)
        detection_products['uncertainty_cube'] = np.array(uncertainty_cube)
        detection_products['median_uncertainty_cube'] = np.array(median_uncertainty_cube)
        detection_products['contrast_tables'] = contrast_tables

        # Add real wavelength to contrast tables and concatenate into one table
        contrast_table = detection_products['contrast_tables'].copy()
        for idx, wavelength_index in enumerate(self.wavelength_indices[cube_indices]):
            wavelength_index_column = np.ones(
                len(contrast_table[idx])) * wavelength_index
            contrast_table[idx] = contrast_table[idx].to_pandas()
            contrast_table[idx].insert(
                loc=0, column='wavelength_index',
                value=wavelength_index_column.astype('int'))
        contrast_table = pd.concat(contrast_table)
        detection_products['contrast_table'] = contrast_table

        if save:
            if file_paths is None:
                contrast_table_file_name = os.path.splitext(
                    file_paths['contrast_table_path'])[0]+'.csv'
                file_paths = self.file_paths
            else:
                contrast_table_file_name = file_paths['contrast_table_path']

            fits.writeto(
                file_paths['norm_detection_image_path'],
                detection_products['normalized_detection_cube'],
                overwrite=overwrite)
            fits.writeto(
                file_paths['uncertainty_image_path'],
                detection_products['uncertainty_cube'],
                overwrite=overwrite)
            fits.writeto(
                file_paths['median_uncertainty_image_path'],
                detection_products['median_uncertainty_cube'],
                overwrite=overwrite)
            contrast_table.to_csv(
                contrast_table_file_name,
                index=False)

            with open(os.path.splitext(
                    file_paths['contrast_table_path'])[0]+'.obj', 'wb') as handle:
                pickle.dump(
                    detection_products['contrast_tables'], handle,
                    protocol=4)

        if inplace:
            self.detection_products = detection_products
        else:
            return detection_products

    def contrast_plot(self, detection_products=None,
                      wavelengths=None, add_wavelength_label=True,
                      companion_table=None,
                      template_fitted=False,
                      plot_companions=True, curvelabels=None,
                      linestyles=None, colors=None,
                      plot_vertical_lod=True,
                      file_paths=None,
                      savefig=True, show=False):

        if detection_products is None:
            detection_products = self.detection_products

        if wavelengths is None:
            wavelengths = self.instrument.wavelengths[self.wavelength_indices]

        if companion_table is None and plot_companions:
            companion_table = self.validated_companion_table
        if not plot_companions:
            companion_table = None

        if curvelabels is None:
            curvelabels = np.array([None]).repeat(len(detection_products))
        if linestyles is None:
            linestyles = np.array(['-']).repeat(len(detection_products))

        if file_paths is None:
            file_paths = self.file_paths
        # colors = plt.cm.viridis(np.linspace(0, 1, len(self.wavelength_indices)))
        if savefig:
            figure_path = file_paths['contrast_plot_path']
        else:
            figure_path = None

        if companion_table is not None:
            mask = companion_table['wavelength_index'].isin(self.wavelength_indices)
            companion_table_used = companion_table[mask]
        else:
            companion_table_used = None

        fig = plot_contrast_curve(
            detection_products['contrast_tables'],
            instrument=self.instrument,
            companion_table=companion_table_used,
            template_fitted=template_fitted,
            # [wavelength_index:wavelength_index + 1],
            wavelengths=wavelengths,
            add_wavelength_label=add_wavelength_label,
            curvelabels=curvelabels,
            linestyles=linestyles,
            colors=colors,  # ['#1b1cd5'],  # '#de650a', '#ba174e'],
            plot_vertical_lod=plot_vertical_lod,
            mirror_axis='mas',
            convert_to_mag=False, yscale='log',
            savefig=figure_path,  # contrast_plot_path[key],
            show=show)

        return fig

    def mask_companions_in_detection(self, yx_known_companion_position=None,
                                     mask_radius=None):

        yx_dim = (self.detection_cube.shape[-2], self.detection_cube.shape[-1])

        if yx_known_companion_position is None:
            yx_known_companion_position = self.reduction_parameters.yx_known_companion_position

        if mask_radius is None:
            mask_radius = self.reduction_parameters.companion_mask_radius

        if yx_known_companion_position is not None:
            yx_known_companion_position = np.array(yx_known_companion_position)
            if yx_known_companion_position.ndim == 1:
                self.detected_signal_mask = regressor_selection.make_signal_mask(
                    yx_dim, yx_known_companion_position,
                    mask_radius,
                    relative_pos=True, yx_center=None)
            elif yx_known_companion_position.ndim == 2:
                detected_signal_masks = []
                for yx_pos in yx_known_companion_position:
                    detected_signal_masks.append(
                        regressor_selection.make_signal_mask(
                            yx_dim, yx_pos, mask_radius,
                            relative_pos=True, yx_center=None))
                self.detected_signal_mask = np.logical_or.reduce(detected_signal_masks)
            else:
                raise ValueError(
                    "Dimensionality of known companion positions for contrast curve too large.")
        else:
            self.detected_signal_mask = np.zeros([yx_dim[0], yx_dim[1]]).astype('bool')

    def make_spectral_correlation_matrices(self, radial_bounds=None, bin_width=3,
                                           yx_center=None, detected_signal_mask=None):

        yx_dim = [self.detection_cube.shape[-2], self.detection_cube.shape[-1]]
        separations_used = []
        empirical_correlation_matrices = []
        empirical_correlation = {}

        self.detection_cube[self.detection_cube == 0.] = np.nan

        if detected_signal_mask is None:
            self.mask_companions_in_detection()
            detected_signal_mask = self.detected_signal_mask

        if radial_bounds is None:
            separation_max = self.detection_cube.shape[-1] // 2 * np.sqrt(2)
            radial_bounds = [1, int(separation_max)]

        if yx_center is None:
            yx_center = (self.detection_cube.shape[-2] // 2., self.detection_cube.shape[-1] // 2.)
        xy_center = yx_center[::-1]

        # Determine first non-zero separation, to prevent results below IWA
        inner_bound_index = int(yx_center[0] + radial_bounds[0])
        try:
            non_zero_separation = radial_bounds[0] + np.max(
                np.argwhere(np.isnan(self.detection_cube[0, 0][inner_bound_index:inner_bound_index + 15,
                                                               int(yx_center[1])]))) + 1
        except ValueError:
            non_zero_separation = 0
        if non_zero_separation > radial_bounds[0] + 13:
            non_zero_separation = 0

        separations = np.arange(radial_bounds[0], radial_bounds[1])

        for idx, separation in enumerate(separations):
            # annulus_data = annulus_mask[0].multiply(data)
            # mask = annulus_mask[0].to_image(data.shape) > 0
            r_in = separation - bin_width / 2.
            r_out = separation + bin_width / 2.
            if r_in < 0.5:
                r_in = 0.5
            annulus_aperture = CircularAnnulus(
                xy_center, r_in=r_in, r_out=r_out)
            annulus_mask = annulus_aperture.to_mask(method='center')
            # Make sure only pixels are used for which data exists
            mask = annulus_mask.to_image(yx_dim) > 0
            mask[int(xy_center[1]), int(xy_center[0])] = False

            if detected_signal_mask is None:
                mask_computation_annulus = mask
            else:
                mask_computation_annulus = np.logical_and(mask, ~detected_signal_mask)

            annulus_data_1d = self.detection_cube[:, 0, mask_computation_annulus]

            if np.all(np.isfinite(annulus_data_1d)):
                psi_ij = compute_empirical_correlation_matrix(annulus_data_1d)
                empirical_correlation_matrices.append(psi_ij)
                separations_used.append(separation)

        empirical_correlation['separation'] = np.array(separations_used)
        empirical_correlation['matrices'] = np.array(empirical_correlation_matrices)

        self.empirical_correlation = empirical_correlation

    def find_approximate_candidate_positions(self, snr_image, candidate_threshold=4.75, mask_radius=6.):
        snr_image = np.ma.masked_array(snr_image)

        yx_dim = snr_image.shape
        yx_center = (yx_dim[0] // 2., yx_dim[1] // 2.)

        significant_pixel_mask = snr_image.data > candidate_threshold
        snr_image.mask = ~significant_pixel_mask

        candidates = {'x': [], 'y': [], 'x_relative': [], 'y_relative': [],
                      'separation': [], 'position_angle': [], 'snr': []}

        candidate_index = 0
        while not np.all(snr_image.mask):
            # candidates['candidate_index'].append(candidate_index)
            candidates['snr'].append(np.max(snr_image))

            highest_value_position = np.unravel_index(
                snr_image.argmax(), snr_image.shape)
            candidates['x'].append(highest_value_position[1])
            candidates['y'].append(highest_value_position[0])
            relative_yx = image_coordinates.absolute_yx_to_relative_yx(
                highest_value_position,
                image_center_yx=yx_center)
            candidates['x_relative'].append(relative_yx[1])
            candidates['y_relative'].append(relative_yx[0])
            rhophi = image_coordinates.relative_yx_to_rhophi(
                relative_yx)
            candidates['separation'].append(rhophi[0])
            candidates['position_angle'].append(rhophi[1])

            candidate_mask = regressor_selection.make_signal_mask(
                snr_image.shape, highest_value_position, mask_radius,
                relative_pos=False, yx_center=None)
            snr_image.mask[candidate_mask] = True
            candidate_index += 1

        candidates = pd.DataFrame(candidates)
        self.candidates = candidates
        return candidates

    def find_candidates(self, detection_product_index=0, detection_threshold=5., candidate_threshold=3.5,
                        mask_radius=6, detection_products=None):

        if detection_products is None:
            detection_products = self.detection_products

        wavelength_index = self.wavelength_indices[detection_product_index]

        smallest_non_nan_separation_idx = np.min(
            np.argwhere(
                np.isfinite(detection_products['contrast_tables'][detection_product_index]['snr_normalization'])))

        smallest_separation_in_pixel = detection_products[
            'contrast_tables'][detection_product_index]['sep (pix)'][smallest_non_nan_separation_idx]

        candidates = self.find_approximate_candidate_positions(
            detection_products['normalized_detection_cube'][detection_product_index],
            candidate_threshold=candidate_threshold, mask_radius=mask_radius)

        # Exclude "detections" very close to the IWA of the reduction
        mask_too_close = candidates['separation'] < smallest_separation_in_pixel + 2
        candidates = candidates[~mask_too_close]

        # if len(candidates) > 0:
        #     yx_known_companion_position = candidates[['y_relative', 'x_relative']].values
        #     self.reduction_parameters.yx_known_companion_position = yx_known_companion_position
        #     detection_products = self.contrast_table_and_normalization(save=False, inplace=False)
        #
        #     candidates = self.find_approximate_candidate_positions(
        #         detection_products['normalized_detection_cube'][detection_product_index],
        #         detection_threshold=detection_threshold, mask_radius=mask_radius)
        #     mask_too_close = candidates['separation'] < smallest_separation_in_pixel + 2
        #     candidates = candidates[~mask_too_close]
        #     yx_known_companion_position2 = candidates[['y_relative', 'x_relative']].values
        #     if len(candidates) > 0 and not np.all(yx_known_companion_position == yx_known_companion_position2):
        #         self.reduction_parameters.yx_known_companion_position = yx_known_companion_position2
        #         # self.mask_companions_in_detection()
        #         detection_products = self.contrast_table_and_normalization(
        #             save=False, inplace=False)
        #     self.contrast_table_and_normalization(save=True)
        #
        #         candidates = self.find_approximate_candidate_positions(
        #             detection_products['normalized_detection_cube'][detection_product_index],
        #             detection_threshold=candidate_threshold, mask_radius=mask_radius)
        #         mask_too_close = candidates['separation'] < smallest_separation_in_pixel + 2
        candidates = candidates[~mask_too_close].sort_values('separation', ignore_index=False)

        number_of_candidates = len(candidates)
        wavelength_index_column = np.ones(number_of_candidates) * wavelength_index
        wavelength_index_column = wavelength_index_column.astype('int')

        candidates.insert(
            loc=0, column='wavelength_index',
            value=wavelength_index_column)

        self.candidates = candidates

        return candidates

    def fit_candidates(self, candidates=None, detection_cube=None, detection_products=None,
                       x_stddev=1.43, y_stddev=2.63, box_size=11,
                       deviation_threshold=0.1, plot=False):
        """ Fits 2D Gaussians to contrast, snr, and normalized snr images for candidate position.
            The normalization excludes positions marked in `self.reduction_parameters.yx_known_companion_position`,
            which is set by `find_candidates` for candidates above the `detection_threshold` parameter.

            The resulting tables include both fit results from unconstrained fits and fits with fixed `x_stddev`, `y_stddev`,
            and `theta` of the 2D Gaussian, where the `theta` parameter is given by the position angle.

        """

        if candidates is None:
            candidates = self.candidates

        # Allow to run this on wavelength combined cubes for example
        if detection_cube is None:
            detection_cube = self.detection_cube

        if detection_products is None:
            detection_products = self.detection_products

        contrast_image_results = []
        snr_image_results = []
        norm_snr_image_results = []

        # Only consider candidates from one channel for normalization to prevent
        # accumulating too many false positives in the normalization

        for candidate_idx in tqdm(range(len(candidates))):
            # Temporarily remove candidate position from normalization
            wavelength_index = candidates['wavelength_index'].values[candidate_idx]
            detection_product_index = np.argwhere(self.wavelength_indices == wavelength_index)[0][0]
            yx_position_relative = candidates[['y_relative', 'x_relative']].values[candidate_idx]

            if self.reduction_parameters.yx_known_companion_position is not None:
                self.reduction_parameters.yx_known_companion_position = np.vstack(
                    [self.reduction_parameters.yx_known_companion_position,
                     yx_position_relative])
            else:
                self.reduction_parameters.yx_known_companion_position = np.expand_dims(
                    yx_position_relative, axis=0)

            detection_products = self.contrast_table_and_normalization(
                detection_cube=detection_cube, cube_indices=[detection_product_index],
                mask_above_sigma=5., save=False, inplace=False)
            self.reduction_parameters.yx_known_companion_position = np.delete(
                self.reduction_parameters.yx_known_companion_position,
                -1, axis=0)

            contrast_image_result, snr_image_result, norm_snr_image_result = fit_planet_parameters(
                detection_image=detection_cube[detection_product_index],
                # uncertainty_image=detection_products['uncertainty_cube'][0],
                normalized_detection_image=detection_products['normalized_detection_cube'][0],
                contrast_table=detection_products['contrast_tables'][0],
                yx_position=candidates[['y', 'x']].values[candidate_idx],
                x_stddev=x_stddev, y_stddev=y_stddev,
                box_size=box_size, mask_deviating=False,
                deviation_threshold=deviation_threshold,
                fix_width=True, fix_orientation=True,
                plot=plot)

            contrast_image_result_free, snr_image_result_free, norm_snr_image_result_free = fit_planet_parameters(
                detection_image=detection_cube[detection_product_index],
                # uncertainty_image=detection_products['uncertainty_cube'][0],
                normalized_detection_image=detection_products['normalized_detection_cube'][0],
                contrast_table=detection_products['contrast_tables'][0],
                yx_position=candidates[['y', 'x']].values[candidate_idx],
                x_stddev=x_stddev, y_stddev=y_stddev,
                box_size=box_size, mask_deviating=False,
                deviation_threshold=deviation_threshold,
                fix_width=False, fix_orientation=False,
                plot=plot)

            contrast_image_result = pd.merge(
                contrast_image_result, contrast_image_result_free[[
                    'amplitude', 'x_fwhm', 'y_fwhm', 'theta', 'good_pixels', 'fwhm_area', 'good_fraction']],
                left_index=True, right_index=True, how='left', suffixes=[None, '_free'])
            snr_image_result = pd.merge(
                snr_image_result, snr_image_result_free[[
                    'amplitude', 'x_fwhm', 'y_fwhm', 'theta', 'good_pixels', 'fwhm_area', 'good_fraction']],
                left_index=True, right_index=True, how='left', suffixes=[None, '_free'])
            norm_snr_image_result = pd.merge(
                norm_snr_image_result, norm_snr_image_result_free[[
                    'amplitude', 'x_fwhm', 'y_fwhm', 'theta', 'good_pixels', 'fwhm_area', 'good_fraction']],
                left_index=True, right_index=True, how='left', suffixes=[None, '_free'])

            contrast_image_result.insert(loc=0, column='candidate_index',
                                         value=np.array([candidate_idx]))
            snr_image_result.insert(loc=0, column='candidate_index',
                                    value=np.array([candidate_idx]))
            norm_snr_image_result.insert(loc=0, column='candidate_index',
                                         value=np.array([candidate_idx]))
            contrast_image_result.insert(
                loc=1, column='wavelength_index',
                value=[wavelength_index])
            snr_image_result.insert(
                loc=1, column='wavelength_index',
                value=[wavelength_index])
            norm_snr_image_result.insert(
                loc=1, column='wavelength_index',
                value=[wavelength_index])

            contrast_image_results.append(contrast_image_result)
            snr_image_results.append(snr_image_result)
            norm_snr_image_results.append(norm_snr_image_result)

        candidates_fit = {}
        contrast_image_results = pd.concat(contrast_image_results, axis=0, ignore_index=True)
        snr_image_results = pd.concat(snr_image_results, axis=0, ignore_index=True)
        norm_snr_image_results = pd.concat(norm_snr_image_results, axis=0, ignore_index=True)

        candidates_fit['contrast_image'] = contrast_image_results
        candidates_fit['snr_image'] = snr_image_results
        candidates_fit['norm_snr_image'] = norm_snr_image_results

        self.candidates_fit = candidates_fit

        return candidates_fit

    def find_candidates_all_wavelengths(self, detection_cube=None, detection_products=None,
                                        wavelength_indices=None,
                                        candidate_threshold=4.):

        if detection_cube is None:
            detection_cube = self.detection_cube
        if detection_products is None:
            detection_products = self.detection_products
        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices

        candidates = []
        for detection_product_index in tqdm(range(len(wavelength_indices))):
            candidates.append(self.find_candidates(
                detection_product_index=detection_product_index, detection_products=detection_products,
                candidate_threshold=candidate_threshold))

        candidates = pd.concat(candidates, axis=0, ignore_index=True)
        candidates = candidates.sort_values('separation')

        return candidates

    def complete_candidate_table(self, candidates=None,
                                 detection_cube=None, detection_products=None,
                                 wavelength_indices=None, detection_threshold=5.,
                                 candidate_threshold=4., search_radius=5):

        if detection_cube is None:
            detection_cube = self.detection_cube
        if detection_products is None:
            detection_products = self.detection_products
        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices

        if candidates is None:
            candidates = self.find_candidates_all_wavelengths(
                detection_cube=detection_cube,
                detection_products=detection_products,
                wavelength_indices=wavelength_indices,
                candidate_threshold=candidate_threshold)

        if len(candidates) == 0:
            return None, None

        # NOTE: This fits all signals above threshold. Can be a lot for multiple cadidates
        # detected in multiple wavelengths.
        candidates_fit = self.fit_candidates(
            candidates=candidates, detection_cube=detection_cube,
            detection_products=detection_products, plot=False)

        unique_candidate_indices = []
        rejected = []
        final_position_table = []
        weighted_average_key_list = [
            'x', 'y', 'x_relative', 'y_relative', 'separation', 'position_angle',
            'x_fwhm', 'y_fwhm', 'theta', 'good_pixels', 'fwhm_area',
            'good_fraction', 'x_fwhm_free', 'y_fwhm_free', 'theta_free',
            'good_pixels_free', 'fwhm_area_free', 'good_fraction_free']
        for idx in range(len(candidates)):
            pos1 = candidates_fit['snr_image'].iloc[idx][['x_relative', 'y_relative']].values
            pos2 = candidates_fit['snr_image'][['x_relative', 'y_relative']].values
            dist = linalg.norm(pos1 - pos2, axis=1)

            mask = dist < search_radius
            if np.sum(mask) == 1:
                unique_candidate_indices.append(idx)
                # If candidate is only detected in one channel, uncertainty is NaN
                entry = candidates_fit['snr_image'].iloc[idx][weighted_average_key_list].to_frame().T
                entry.insert(loc=5, column='separation_sigma', value=np.nan)
                entry.insert(loc=7, column='position_angle_sigma', value=np.nan)
                entry.insert(loc=8, column='channels_above_threshold', value=np.array([1]))
                final_position_table.append(entry)
            else:
                # Find wavelength with highest SNR
                if idx not in unique_candidate_indices and idx not in rejected:
                    # Perform weighted average of columns
                    df = candidates_fit['snr_image'][mask]
                    snr = candidates_fit['norm_snr_image']['amplitude'][mask]
                    df.insert(loc=0, column='snr', value=snr)
                    df.insert(loc=1, column='group', value=np.ones(len(df)))

                    # Sigma clipping for position
                    # from astropy.stats import sigma_clip, mad_std
                    # filtered_data1 = sigma_clip(df['separation'], sigma=3.5, maxiters=3,
                    #                             cenfunc=np.median, stdfunc=mad_std)
                    # filtered_data2 = sigma_clip(df['position_angle'], sigma=3.5,
                    #                             maxiters=3, cenfunc=np.median, stdfunc=mad_std)

                    # Make new data frame of average weighted by SNR
                    def weighted_average(grp, weight_column='snr'):
                        return grp._get_numeric_data().multiply(grp[weight_column], axis=0).sum()/grp[weight_column].sum()
                    weighted_agg = df.groupby('group').apply(weighted_average)
                    weighted_agg = weighted_agg[weighted_average_key_list]

                    # Compute uncertainty based on standard deviation
                    std_dev_df = df.groupby('group').apply(np.std)
                    weighted_agg.insert(loc=5, column='separation_sigma',
                                        value=std_dev_df['separation'].values)
                    weighted_agg.insert(loc=7, column='position_angle_sigma',
                                        value=std_dev_df['position_angle'].values)
                    weighted_agg.insert(loc=8, column='channels_above_threshold',
                                        value=np.array([len(df)]))

                    # Get candidate id of channel with highest SNR
                    temp_idx = np.argmax(candidates_fit['norm_snr_image'][mask]['amplitude'])
                    candidate_index = int(
                        candidates_fit['snr_image'][mask].iloc[temp_idx]['candidate_index'])
                    # Set in mask, such that it won't be used in next iteration
                    mask[candidate_index] = False
                    rejected = rejected + list(np.argwhere(mask)[:, 0])

                    # Add to our final position table
                    final_position_table_key_list = [
                        'x', 'y', 'x_relative', 'y_relative', 'separation', 'separation_sigma',
                        'position_angle', 'position_angle_sigma', 'channels_above_threshold',
                        'x_fwhm', 'y_fwhm', 'theta', 'good_pixels', 'fwhm_area',
                        'good_fraction', 'x_fwhm_free', 'y_fwhm_free', 'theta_free',
                        'good_pixels_free', 'fwhm_area_free', 'good_fraction_free']
                    if candidate_index not in unique_candidate_indices and candidate_index not in rejected:
                        unique_candidate_indices.append(candidate_index)
                        final_position_table.append(
                            weighted_agg[final_position_table_key_list])

        final_position_table = pd.concat(final_position_table, ignore_index=True)

        # Filter out duplicates from candidate table, only retain one row entry per candidate
        candidates = candidates.iloc[unique_candidate_indices].sort_values('separation')
        candidates_fit['contrast_image'] = candidates_fit['contrast_image'].iloc[unique_candidate_indices].sort_values(
            'separation', ignore_index=True)
        candidates_fit['snr_image'] = candidates_fit['snr_image'].iloc[unique_candidate_indices].reset_index()
        candidates_fit['norm_snr_image'] = candidates_fit['norm_snr_image'].iloc[unique_candidate_indices].sort_values(
            'separation', ignore_index=True)

        candidates_fit['snr_image'][weighted_average_key_list] = final_position_table[weighted_average_key_list]
        candidates_fit['snr_image'].insert(loc=8, column='separation_sigma',
                                           value=final_position_table['separation_sigma'].values)
        candidates_fit['snr_image'].insert(loc=10, column='position_angle_sigma',
                                           value=final_position_table['position_angle_sigma'].values)
        candidates_fit['snr_image'].insert(loc=11, column='channels_above_threshold',
                                           value=final_position_table['channels_above_threshold'].values)

        candidates_fit['snr_image'].sort_values('separation', ignore_index=True, inplace=True)

        self.candidates = candidates
        self.candidates_fit = candidates_fit

        return candidates, candidates_fit

    def rereduce_single_position(
            self, candidate_index, candidate_position, data_full, flux_psf_full, pa, wavelength_indices,
            temporal_components_fraction, variance_full=None, instrument=None,
            bad_frames=None, bad_pixel_mask_full=None, xy_image_centers=None,
            amplitude_modulation_full=None,
            return_table=False,
            verbose=False):

        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices

        re_reduction_parameters = copy.copy(self.reduction_parameters)

        re_reduction_parameters.guess_position = candidate_position
        re_reduction_parameters.reduce_single_position = True
        re_reduction_parameters.data_auto_crop = True
        re_reduction_parameters.yx_known_companion_position = None
        # To avoid manipulating data in place (multiple injections)
        # Set remove_known_companions to False
        re_reduction_parameters.remove_known_companions = False

        all_results = run_complete_reduction(
            data_full=data_full.copy(),
            flux_psf_full=flux_psf_full.copy(),
            pa=pa,
            instrument=self.instrument,
            reduction_parameters=re_reduction_parameters,
            temporal_components_fraction=temporal_components_fraction,
            wavelength_indices=wavelength_indices,
            variance_full=variance_full,
            bad_frames=bad_frames,
            bad_pixel_mask_full=bad_pixel_mask_full,
            xy_image_centers=xy_image_centers,
            amplitude_modulation_full=amplitude_modulation_full,
            verbose=verbose)

        # AUTOMATICALLY COLLECT ALL WAVELENGTHS FOR REDUCTION
        # NOTE: This should be generalized to allow automatically collect results from
        # using various component fractions
        contrast = []
        uncertainty = []
        component_key = str(temporal_components_fraction[0])
        for key in all_results[component_key]:
            contrast.append(all_results[component_key][key][self.reduction_type].measured_contrast)
            uncertainty.append(all_results[component_key][key]
                               [self.reduction_type].contrast_uncertainty)

        contrast = np.array(contrast)
        uncertainty = np.array(uncertainty)

        # REDO NORMALIZATION
        self.reduction_parameters.yx_known_companion_position = np.vstack(
            [self.reduction_parameters.yx_known_companion_position,
             self.candidates[['y_relative', 'x_relative']].values])
        self.contrast_table_and_normalization(save=False, inplace=True)
        self.reduction_parameters.yx_known_companion_position = np.delete(
            self.reduction_parameters.yx_known_companion_position, -1, axis=0)

        normalization_factors = []
        for contrast_table_index in range(len(wavelength_indices)):
            mask = np.isfinite(
                self.detection_products['contrast_tables'][contrast_table_index]['snr_normalization'])
            separation = self.detection_products['contrast_tables'][contrast_table_index]['sep (pix)'][
                mask]
            norm_factors = self.detection_products['contrast_tables'][contrast_table_index]['snr_normalization'][mask]

            norm_factor_function = interp1d(separation, norm_factors, fill_value='extrapolate')
            normalization_factors.append(
                norm_factor_function(self.candidates_fit['snr_image']['separation'][candidate_index]))

        normalization_factors = np.array(normalization_factors)
        normalized_uncertainty = uncertainty * normalization_factors
        snr = contrast / normalized_uncertainty

        # Table entries
        id = np.ones(len(self.instrument.wavelengths)) * candidate_index
        contrast_for_table = np.empty_like(id)
        contrast_for_table[:] = np.nan
        normalized_uncertainty_for_table = np.empty_like(id)
        normalized_uncertainty_for_table[:] = np.nan
        snr_for_table = np.empty_like(id)
        snr_for_table[:] = np.nan
        uncertainty_for_table = np.empty_like(id)
        uncertainty_for_table[:] = np.nan
        norm_factor_for_table = np.empty_like(id)
        norm_factor_for_table[:] = np.nan

        contrast_for_table[wavelength_indices] = contrast
        normalized_uncertainty_for_table[wavelength_indices] = normalized_uncertainty
        snr_for_table[wavelength_indices] = snr
        uncertainty_for_table[wavelength_indices] = uncertainty
        norm_factor_for_table[wavelength_indices] = normalization_factors

        candidate_spectrum = {
            'candidate_id': id.astype('int'),
            'wavelength_index': np.arange(len(self.instrument.wavelengths)),
            'wavelength': self.instrument.wavelengths.value,
            'contrast': contrast_for_table,
            'uncertainty': normalized_uncertainty_for_table,
            'snr': snr_for_table,
            'original_unc': uncertainty_for_table,
            'norm_factor': norm_factor_for_table}

        candidate_spectrum = pd.DataFrame(candidate_spectrum)

        return candidate_spectrum

    def extract_candidate_spectra(
            self,
            temporal_components_fraction, data_full,
            flux_psf_full, pa,
            candidate_positions=None,
            wavelength_indices=None,
            variance_full=None, instrument=None,
            bad_frames=None, bad_pixel_mask_full=None,
            xy_image_centers=None,
            amplitude_modulation_full=None,
            return_spectra=False):

        candidate_spectra = []

        if candidate_positions is None:
            candidate_positions = self.candidates_fit['snr_image'][[
                'y_relative', 'x_relative']].values
        # detection1.reduction_parameters.reduce_single_position = True
        if len(candidate_positions) == 0 or candidate_positions is None:
            return None

        for candidate_index, candidate_position in tqdm(enumerate(candidate_positions)):
            candidate_spectrum = self.rereduce_single_position(
                candidate_index=candidate_index,
                candidate_position=candidate_position, data_full=data_full,
                flux_psf_full=flux_psf_full, pa=pa,
                temporal_components_fraction=temporal_components_fraction,
                wavelength_indices=wavelength_indices,
                variance_full=variance_full, instrument=instrument,
                bad_frames=bad_frames, bad_pixel_mask_full=bad_pixel_mask_full,
                xy_image_centers=xy_image_centers,
                amplitude_modulation_full=amplitude_modulation_full,
                verbose=False)
            candidate_spectra.append(candidate_spectrum)

        candidate_spectra = pd.concat(candidate_spectra, axis=0, ignore_index=False)
        self.candidate_spectra = candidate_spectra

        if return_spectra:
            return candidate_spectra

    def detection_summary(self, candidates, candidates_fit, candidate_spectra=None,
                          use_spectra=True, template_name=None,
                          snr_threshold=4.5,
                          snr_threshold_spectrum=True,
                          good_fraction_threshold=0.05,
                          theta_deviation_threshold=25.,
                          yx_fwhm_ratio_threshold=[1.1, 4.5]):

        if candidates is None:
            candidates = self.candidates

        if candidates_fit is None:
            candidates_fit = self.candidates_fit

        if candidate_spectra is None and use_spectra:
            candidate_spectra = self.candidate_spectra

        companion_table = candidates_fit['snr_image'][
            ['x', 'y', 'x_relative', 'y_relative', 'separation', 'separation_sigma',
             'position_angle', 'position_angle_sigma', 'channels_above_threshold',
             'theta_free', 'x_fwhm', 'y_fwhm', 'fwhm_area', 'x_fwhm_free', 'y_fwhm_free',
             'good_fraction', 'good_fraction_free']]

        companion_table.insert(loc=16, column='theta_deviation',
                               value=subtract_angles(
                                   companion_table['position_angle'], companion_table['theta_free']))
        companion_table.insert(loc=17, column='yx_fwhm_ratio',
                               value=companion_table['y_fwhm_free'] / companion_table['x_fwhm_free'])
        companion_table.insert(loc=18, column='fwhm_area_free',
                               value=np.pi * companion_table['x_fwhm_free'] * companion_table['y_fwhm_free'])
        companion_table.insert(loc=19, column='norm_snr_fit',
                               value=candidates_fit['norm_snr_image']['amplitude'])
        companion_table.insert(loc=20, column='norm_snr_fit_free',
                               value=candidates_fit['norm_snr_image']['amplitude_free'])
        companion_table.insert(loc=21, column='peak_pixel_snr',
                               value=candidates['snr'])

        if template_name is not None:
            companion_table.insert(loc=22, column='template_name',
                                   value=np.array([template_name]).repeat(
                                       len(companion_table)))

        if use_spectra:
            companion_table = pd.merge(
                companion_table, candidate_spectra,
                left_index=True, right_on='candidate_id', how='left')

        if snr_threshold_spectrum and use_spectra:
            snr = companion_table['snr']
        else:
            snr = np.max([companion_table['norm_snr_fit_free'],
                         companion_table['peak_pixel_snr']], axis=0)

        mask = (snr > snr_threshold) \
            & (companion_table['good_fraction_free'] > good_fraction_threshold) \
            & (np.abs(companion_table['theta_deviation']) < theta_deviation_threshold) \
            & (companion_table['yx_fwhm_ratio'] > yx_fwhm_ratio_threshold[0]) \
            & (companion_table['yx_fwhm_ratio'] < yx_fwhm_ratio_threshold[1])

        if use_spectra:
            unique_candidates = np.unique(companion_table[mask]['candidate_id'].values)
            validated_companion_table = companion_table[companion_table['candidate_id'].isin(
                unique_candidates)]
        else:
            validated_companion_table = companion_table[mask]

        self.companion_table = companion_table
        self.validated_companion_table = validated_companion_table

        return companion_table, validated_companion_table

    def add_templates(self, template):
        self.templates[template.name] = template

    def add_default_templates(
            self, stellar_modelbox, species_database_directory,
            stellar_parameters=None,
            instrument=None,
            correct_transmission=False,
            use_spectral_correlation=True):

        os.chdir(species_database_directory)
        try:
            database = species.Database()
        except:
            FileNotFoundError(
                f"No initialized species database found in: {species_database_directory}")

        if instrument is None:
            instrument = self.instrument
        try:
            cool_planet_read_model = species.ReadModel(
                model='petitcode-cool-cloudy', wavel_range=(0.85, 3.6))
        except:
            print("First time running cool planet template: adding 'petit-cool-cloudy' models to database.")
            database.add_model(model='petitcode-cool-cloudy', teff_range=(700., 800.))
            cool_planet_read_model = species.ReadModel(
                model='petitcode-cool-cloudy', wavel_range=(0.85, 3.6))

        cool_planet_model_param = {'teff': 760., 'logg': 4.26,
                                   'feh': 1.0, 'fsed': 1.26, 'radius': 1.1,
                                   'distance': 30.}
        try:
            hot_planet_read_model = species.ReadModel(
                model='drift-phoenix', wavel_range=(0.85, 3.6))
        except:
            print("First time running hot planet template: adding 'drift-phoenix' models to database.")
            database.add_model(model='drift-phoenix', teff_range=(1400., 1600.))
            hot_planet_read_model = species.ReadModel(
                model='drift-phoenix', wavel_range=(0.85, 3.6))

        hot_planet_model_param = {'teff': 1500., 'logg': 4.,
                                  'feh': 0., 'radius': 1.1,
                                  'distance': 30.}

        cool_planet_modelbox = cool_planet_read_model.get_model(
            model_param=cool_planet_model_param)
        hot_planet_modelbox = hot_planet_read_model.get_model(
            model_param=hot_planet_model_param)

        flat_model = copy.deepcopy(cool_planet_modelbox)
        flat_model.flux = np.ones_like(flat_model.wavelength)

        if stellar_modelbox is None:
            if stellar_parameters is None:
                stellar_modelbox = copy.deepcopy(flat_model)
            else:
                try:
                    star_read_model = species.ReadModel(
                        model='bt-nextgen', wavel_range=(0.85, 3.6))
                except:
                    print("First time running stellar template: adding 'bt-nextgen' models to database.")
                    database.add_model(model='bt-nextgen',
                                       teff_range=(3000., 30000.))
                    star_read_model = species.ReadModel(
                        model='bt-nextgen', wavel_range=(0.85, 3.6))

                stellar_modelbox = star_read_model.get_model(
                    model_param=stellar_parameters)

        if instrument.instrument_type == 'photometry' or len(instrument.wavelengths) <= 2:
            t_type_slope_fit = False
        else:
            t_type_slope_fit = True

        self.templates['L-type'] = \
            SpectralTemplate(
                name='L-type',
                instrument=instrument,
                companion_modelbox=hot_planet_modelbox,
                stellar_modelbox=stellar_modelbox,  # star_modelflux,
                wavelength_indices=self.wavelength_indices,
                correct_transmission=correct_transmission,
                fit_offset=False,
                fit_slope=False,
                number_of_pca_regressors=0,
                use_spectral_correlation=use_spectral_correlation)

        # self.templates['L-type + offset'] = \
        #     SpectralTemplate(
        #         name='L-type + offset',
        #         instrument=instrument,
        #         companion_modelbox=hot_planet_modelbox,
        #         stellar_modelbox=stellar_modelbox,  # star_modelflux,
        #         wavelength_indices=self.wavelength_indices,
        #         correct_transmission=correct_transmission,
        #         fit_offset=True,
        #         fit_slope=False,
        #         number_of_pca_regressors=0,
        #         use_spectral_correlation=use_spectral_correlation)
        #
        # self.templates['L-type + offset + slope'] = \
        #     SpectralTemplate(
        #         name='L-type + offset + slope',
        #         instrument=instrument,
        #         companion_modelbox=hot_planet_modelbox,
        #         stellar_modelbox=stellar_modelbox,  # star_modelflux,
        #         wavelength_indices=self.wavelength_indices,
        #         correct_transmission=correct_transmission,
        #         fit_offset=True,
        #         fit_slope=True,
        #         number_of_pca_regressors=0,
        #         use_spectral_correlation=use_spectral_correlation)

        self.templates['T-type'] = \
            SpectralTemplate(
                name='T-type',
                instrument=instrument,
                companion_modelbox=cool_planet_modelbox,
                stellar_modelbox=stellar_modelbox,  # star_modelflux,
                wavelength_indices=self.wavelength_indices,
                correct_transmission=correct_transmission,
                fit_offset=True,
                fit_slope=t_type_slope_fit,
                number_of_pca_regressors=0,
                use_spectral_correlation=use_spectral_correlation)

        self.templates['flat'] = \
            SpectralTemplate(
                name='flat',
                instrument=instrument,
                companion_modelbox=flat_model,
                stellar_modelbox=flat_model,  # star_modelflux,
                wavelength_indices=self.wavelength_indices,
                correct_transmission=correct_transmission,
                fit_offset=False,
                fit_slope=False,
                number_of_pca_regressors=0,
                use_spectral_correlation=use_spectral_correlation)

    def template_matching_detection(
            self, template,
            inner_mask_radius=4.,
            detection_threshold=5.,
            file_paths=None,
            save=True):

        template_name = template.name

        wavelengths = self.instrument.wavelengths[self.wavelength_indices]

        contrast_cube = self.detection_cube[:, 0].astype('float64')
        uncertainty_cube = self.detection_products['uncertainty_cube'].astype(
            'float64')  # / detection1.reduction_parameters.contrast_curve_sigma
        yx_dim = [contrast_cube.shape[-2], contrast_cube.shape[-1]]
        yx_center_output = [yx_dim[0] // 2, yx_dim[1]]

        reduced_positions_mask = np.logical_and(
            np.all(np.isfinite(contrast_cube), axis=0),
            ~np.any(contrast_cube == 0., axis=0))
        self.reduction_parameters.annulus_width = 3

        center_mask = regressor_selection.make_signal_mask(
            yx_dim, (0, 0), inner_mask_radius,
            relative_pos=True, yx_center=None)

        reduced_positions_mask = np.logical_and(reduced_positions_mask, ~center_mask)

        position_indices = np.argwhere(reduced_positions_mask)

        template_matched_image = np.zeros((3, yx_dim[0], yx_dim[1]))

        median_contrast = bn.nanmedian(contrast_cube, axis=0)

        # regressors_centered = contrast_cube - median_contrast

        # position_indices = [[32, 77]]
        # number_of_pca_regressors = int(np.round(38 * 0.1))
        wavelength_indices = self.wavelength_indices
        self.make_spectral_correlation_matrices()

        for idx, yx_pixel in tqdm(enumerate(position_indices)):
            # wavelength indices are not applicable to contrast cube when not all wavelengths are reduced
            # contrasts = contrast_cube[wavelength_indices, yx_pixel[0], yx_pixel[1]]
            contrasts = contrast_cube[:, yx_pixel[0], yx_pixel[1]]
            # contrasts = contrasts[self.good_residual_mask].astype('float64')
            # uncertainties = np.sqrt(self.reduced_result[:, 1])
            uncertainties = uncertainty_cube[:, yx_pixel[0], yx_pixel[1]]
            contrasts_mean = np.mean(contrasts)
            contrasts_norm = contrasts / contrasts_mean
            uncertainties_norm = uncertainties / contrasts_mean

            yx_center_output = (yx_dim[0] // 2, yx_dim[1] // 2)
            relative_coords = image_coordinates.absolute_yx_to_relative_yx(
                yx_pixel, yx_center_output)

            if template.use_spectral_correlation:
                separation = np.sqrt(
                    relative_coords[0]**2 + relative_coords[1]**2)

                correlation_array_index = find_nearest(
                    array=self.empirical_correlation['separation'],
                    value=separation)

                # channel_mask = np.zeros(len(self.instrument.wavelengths)).astype('bool')
                # channel_mask[wavelength_indices] = True
                psi_ij = self.empirical_correlation['matrices'][correlation_array_index]
                # psi_ij = remove_channel_from_correlation_matrix(channel_mask, psi_ij)
                cov_ij = uncertainties[:, None] * psi_ij * uncertainties[None, :]
                cov_ij_norm = uncertainties_norm[:, None] * psi_ij * uncertainties_norm[None, :]
                # plot_scale(cov_ij)
                # plt.show()
                inv_cov_ij = np.linalg.inv(cov_ij)
            else:
                cov_ij = np.identity(len(contrasts)) * uncertainties**2
                inv_cov_ij = np.identity(len(contrasts)) * (1. / uncertainties)
            # if show:
            # plot_scale(np.dot(inverse, cov_ij))
            # plt.show()

            model_components = []

            if template.fit_offset:
                model_components.append(np.ones(contrasts.shape[0]))
            if template.fit_slope:
                model_components.append(wavelengths.value)
            if template.normalized_contrast_modelbox.flux is not None:
                model_components.append(template.normalized_contrast_modelbox.flux[None, :])

            if len(model_components) > 0:
                model_matrix = np.vstack(model_components)
            else:
                raise ValueError(
                    "No model present to fit. Provide `model`, `fit_offset` or `fit_slope`")

            if template.number_of_pca_regressors > 0:
                self.reduction_parameters.target_pix_mask_radius = 11
                regressor_pool_mask_global = regressor_selection.make_regressor_pool_for_pixel(
                    reduction_parameters=self.reduction_parameters,
                    yx_pixel=yx_pixel,
                    yx_dim=yx_dim,
                    yx_center=yx_center_output,
                    known_companion_mask=None)
                regressor_pool_mask_global = np.logical_and(
                    regressor_pool_mask_global,
                    reduced_positions_mask)
                training_matrix = contrast_cube[:][:, regressor_pool_mask_global]
                B_full, lambdas_full, S_full, V_full = pca_regression.compute_SVD(
                    training_matrix, n_components=None, scaling=None)  # 'temp-median')
                B = B_full[:, :number_of_pca_regressors]
                A = np.hstack((B, model_matrix.T))
            # A = np.ones(len(uncertainties))[:, None]
            else:
                A = model_matrix.T

            # ipsh()
            P, P_sigma_squared = pca_regression.solve_linear_equation_simple(
                design_matrix=A.T,
                data=contrasts,
                inverse_covariance_matrix=inv_cov_ij)

            # fit_parameters, err_fit_parameters, sigma_hat_sqr = pca_regression.ols(
            #     design_matrix=A, data=contrasts_norm, covariance=cov_ij_norm)

            # P, P_sigma, _ = pca_regression.ols(
            #     design_matrix=A, data=contrasts, covariance=cov_ij)

            # P = P * contrasts_mean
            # P_sigma_squared = P_sigma**2  # * np.abs(contrasts_mean))**2
            # fit_parameters[-1] * mean_data
            # reconstructed_lightcurve = np.dot(A, P)

            template_matched_image[0, yx_pixel[0], yx_pixel[1]] = P[-1]
            template_matched_image[1, yx_pixel[0], yx_pixel[1]] = np.sqrt(P_sigma_squared[-1])
            template_matched_image[2, yx_pixel[0], yx_pixel[1]
                                   ] = P[-1] / np.sqrt(P_sigma_squared[-1])

        if file_paths is None:
            file_paths = {}
            output_dir_matching = os.path.join(
                reduction_parameters.result_folder, 'template_matching/')
            if not os.path.exists(output_dir_matching):
                os.makedirs(output_dir_matching)
            file_paths['norm_detection_image_path'] = os.path.join(
                output_dir_matching, f'normalized_detection_image_{template_name}.fits')
            file_paths['uncertainty_image_path'] = os.path.join(
                output_dir_matching, f'uncertainty_image_{template_name}.fits')
            file_paths['median_uncertainty_image_path'] = os.path.join(
                output_dir_matching, f'median_uncertainty_image_{template_name}.fits')
            file_paths['contrast_table_path'] = os.path.join(
                output_dir_matching, f'contrast_table_{template_name}.csv')
            file_paths['contrast_plot_path'] = os.path.join(
                output_dir_matching, f'contrast_plot_{template_name}')

        self.reduction_parameters.yx_known_companion_position = None
        detection_products_matched = self.contrast_table_and_normalization(
            detection_cube=[template_matched_image], cube_indices=[0],
            yx_known_companion_position=None, inplace=False,
            save=save, file_paths=file_paths,
            mask_above_sigma=detection_threshold)

        detection_cube = np.expand_dims(template_matched_image, axis=0)
        detection_products = detection_products_matched

        return detection_cube, detection_products

    def run_template_matching(self,
                              template,
                              detection_threshold=5.,
                              candidate_threshold=4.75,
                              inner_mask_radius=4,
                              search_radius=5,
                              good_fraction_threshold=0.05,
                              theta_deviation_threshold=25,
                              yx_fwhm_ratio_threshold=[1.1, 4.5],
                              data_full=None,
                              flux_psf_full=None,
                              pa=None,
                              instrument=None,
                              temporal_components_fraction=None,
                              wavelength_indices=None,
                              variance_full=None,
                              bad_frames=None,
                              bad_pixel_mask_full=None,
                              xy_image_centers=None,
                              amplitude_modulation_full=None,
                              file_paths=None,
                              save=True):

        if file_paths is None:
            template_name = template.name
            file_paths = {}
            output_dir_matching = os.path.join(
                self.reduction_parameters.result_folder, 'template_matching/')
            if not os.path.exists(output_dir_matching):
                os.makedirs(output_dir_matching)
            file_paths['norm_detection_image_path'] = os.path.join(
                output_dir_matching, f'normalized_detection_image_{template_name}.fits')
            file_paths['uncertainty_image_path'] = os.path.join(
                output_dir_matching, f'uncertainty_image_{template_name}.fits')
            file_paths['median_uncertainty_image_path'] = os.path.join(
                output_dir_matching, f'median_uncertainty_image_{template_name}.fits')
            file_paths['contrast_table_path'] = os.path.join(
                output_dir_matching, f'contrast_table_{template_name}.csv')
            file_paths['contrast_plot_path'] = os.path.join(
                output_dir_matching, f'contrast_plot_{template_name}')
        wavelengths = self.instrument.wavelengths[self.wavelength_indices]

        detection_cube, detection_products = self.template_matching_detection(
            template,
            inner_mask_radius=inner_mask_radius,
            detection_threshold=detection_threshold,
            file_paths=file_paths,
            save=save)

        candidates = self.find_candidates_all_wavelengths(
            detection_cube=detection_cube, detection_products=detection_products,
            wavelength_indices=[0],
            candidate_threshold=candidate_threshold)

        # if self.reduction_parameters.yx_known_companion_position is not None:
        #     self.reduction_parameters.yx_known_companion_position = np.vstack(
        #         [self.reduction_parameters.yx_known_companion_position,
        #          yx_position_relative])
        # else:
        #     self.reduction_parameters.yx_known_companion_position = np.expand_dims(
        #         yx_position_relative, axis=0)

        # _, candidates_fit_all_indices = analysis.complete_candidate_table(
        #     candidates=candidates,
        #     detection_cube=None, detection_products=None,
        #     wavelength_indices=None, detection_threshold=detection_threshold,
        #     candidate_threshold=candidate_threshold, search_radius=5)

        _, candidates_fit_template = self.complete_candidate_table(
            candidates=candidates,
            detection_cube=detection_cube, detection_products=detection_products,
            wavelength_indices=[0], detection_threshold=detection_threshold,
            candidate_threshold=candidate_threshold,
            search_radius=search_radius)

        # fig = analysis.contrast_plot(detection_products=detection_products_matched,
        #                              companion_table=None,
        #                              plot_companions=False, savefig=False, show=True)

        if candidates is None or len(candidates) == 0:
            template.companion_table = None
            template.validated_companion_table = None
            template.validated_companion_table_short = None
        else:
            # companion_table, validated_companion_table = analysis.detection_summary(
            #     candidates=candidates, candidates_fit=candidates_fit_template, candidate_spectra=None, use_spectra=False,
            #     template_name=template_name, snr_threshold_spectrum=False,
            #     snr_threshold=5., good_fraction_threshold=0.25,
            #     theta_deviation_threshold=25.,
            #     yx_fwhm_ratio_threshold=[1.1, 4.5])

            # mask = candidates['snr'] > detection_threshold
            yx_known_companion_position = candidates_fit_template['snr_image'][[
                'y_relative', 'x_relative']].values  # [mask]
            # yx_known_companion_position = np.unique(
            #     validated_companion_table[['y_relative', 'x_relative']].values, axis=0)

            self.reduction_parameters.companion_mask_radius = 11.

            # Masking out detections
            detection_products_matched = self.contrast_table_and_normalization(
                detection_cube=detection_cube, cube_indices=[0],
                yx_known_companion_position=yx_known_companion_position,
                inplace=False, save=save, file_paths=file_paths, mask_above_sigma=None)

            candidates = self.find_candidates_all_wavelengths(
                detection_cube=detection_cube, detection_products=detection_products_matched,
                wavelength_indices=[0],
                candidate_threshold=candidate_threshold)

            _, candidates_fit_template = self.complete_candidate_table(
                candidates=candidates,
                detection_cube=detection_cube, detection_products=detection_products_matched,
                wavelength_indices=[0], detection_threshold=detection_threshold,
                candidate_threshold=candidate_threshold,
                search_radius=search_radius)

            print('Extracting candidate spectra.')
            candidate_spectra = self.extract_candidate_spectra(
                candidate_positions=candidates_fit_template['snr_image'][[
                    'y_relative', 'x_relative']].values,
                temporal_components_fraction=temporal_components_fraction,
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                wavelength_indices=None,
                variance_full=variance_full,
                instrument=None,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full,
                xy_image_centers=xy_image_centers,
                amplitude_modulation_full=amplitude_modulation_full,
                return_spectra=True)

            companion_table, validated_companion_table = self.detection_summary(
                candidates=candidates, candidates_fit=candidates_fit_template, candidate_spectra=candidate_spectra, use_spectra=True,
                template_name=template_name, snr_threshold_spectrum=False,
                snr_threshold=detection_threshold, good_fraction_threshold=good_fraction_threshold,
                theta_deviation_threshold=theta_deviation_threshold,
                yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold)

            # companion_table, validated_companion_table = analysis.detection_summary(
            #     candidates_fit_template, candidate_spectra,
            #     snr_threshold=detection_threshold, good_fraction_threshold=0.3,
            #     theta_deviation_threshold=25.,
            #     yx_fwhm_ratio_threshold=[1.1, 3.5])

            # yx_known_companion_position = np.unique(
            #     validated_companion_table[['y_relative', 'x_relative']].values, axis=0)

            self.reduction_parameters.yx_known_companion_position = yx_known_companion_position

            companion_table.to_csv(
                os.path.join(output_dir_matching, f'companion_table_{template_name}.csv'),
                index=False)

            validated_companion_table.to_csv(
                os.path.join(output_dir_matching,
                             f'validated_companion_table_{template_name}.csv'),
                index=False)

            validated_companion_table_short = validated_companion_table[
                ['candidate_id', 'x', 'y', 'x_relative',
                 'y_relative', 'separation', 'separation_sigma',
                 'position_angle', 'position_angle_sigma',
                 # 'channels_above_threshold',
                 'template_name',
                 'norm_snr_fit_free',
                 'peak_pixel_snr',
                 'wavelength_index', 'wavelength',
                 'contrast', 'uncertainty']]

            validated_companion_table_short.to_csv(
                os.path.join(output_dir_matching,
                             f'validated_companion_table_short_{template_name}.csv'),
                index=False)

            plt.close()
            candidate_indices = np.unique(validated_companion_table['candidate_id'])
            for candidate_index in candidate_indices:
                temp_table = validated_companion_table[validated_companion_table['candidate_id']
                                                       == candidate_index]
                plt.errorbar(
                    x=temp_table['wavelength'],
                    y=temp_table['contrast'],
                    yerr=temp_table['uncertainty'],
                    fmt='o',
                    label='candidate {}'.format(candidate_index))
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('wavelength')
            plt.ylabel('contrast')
            plt.legend()
            plt.savefig(os.path.join(output_dir_matching,
                        f'companion_spectra_{template_name}.pdf'))
            plt.close()

            # analysis.contrast_table_and_normalization(save=False)
            # _ = analysis.contrast_plot(
            #     savefig=False, plot_companions=plot_companions,
            #     companion_table=validated_companion_table, show=True)

            _ = self.contrast_plot(detection_products=detection_products,
                                   companion_table=validated_companion_table,
                                   wavelengths=np.median(wavelengths).repeat(2)[:1],
                                   add_wavelength_label=False,
                                   curvelabels=[f'{template_name}'],
                                   linestyles=['-', '--'],
                                   colors=['blue'],
                                   plot_companions=True,
                                   template_fitted=True,
                                   savefig=True,
                                   file_paths=file_paths,
                                   show=False)

            template.companion_table = companion_table
            template.validated_companion_table = validated_companion_table
            template.validated_companion_table_short = validated_companion_table_short
            # return companion_table, validated_companion_table, validated_companion_table_short, detection_products

        template.detection_products = detection_products
        # return None, None, None, detection_products

    def match_all_templates(self,
                            detection_threshold=5.,
                            candidate_threshold=4.75,
                            inner_mask_radius=4,
                            search_radius=5,
                            good_fraction_threshold=0.05,
                            theta_deviation_threshold=25,
                            yx_fwhm_ratio_threshold=[1.1, 4.5],
                            data_full=None,
                            flux_psf_full=None,
                            pa=None,
                            instrument=None,
                            temporal_components_fraction=None,
                            wavelength_indices=None,
                            variance_full=None,
                            bad_frames=None,
                            bad_pixel_mask_full=None,
                            xy_image_centers=None,
                            amplitude_modulation_full=None,
                            file_paths=None,
                            save=True):

        if self.templates:
            for key in self.templates:
                self.run_template_matching(
                    template=self.templates[key],
                    detection_threshold=detection_threshold,
                    candidate_threshold=candidate_threshold,
                    inner_mask_radius=inner_mask_radius,
                    search_radius=search_radius,
                    good_fraction_threshold=good_fraction_threshold,
                    theta_deviation_threshold=theta_deviation_threshold,
                    yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold,
                    data_full=data_full,
                    flux_psf_full=flux_psf_full,
                    pa=pa,
                    instrument=instrument,
                    temporal_components_fraction=temporal_components_fraction,
                    wavelength_indices=wavelength_indices,
                    variance_full=variance_full,
                    bad_frames=bad_frames,
                    bad_pixel_mask_full=bad_pixel_mask_full,
                    xy_image_centers=xy_image_centers,
                    amplitude_modulation_full=amplitude_modulation_full,
                    file_paths=file_paths,
                    save=save)

    # def run_characterization(self, ):
