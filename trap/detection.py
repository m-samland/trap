"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.stats import mad_std
from astropy.table import Table
# import seaborn as sns
from matplotlib import rc, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from photutils import CircularAnnulus
from scipy import stats

from trap import image_coordinates, regressor_selection
from trap.embed_shell import ipsh

# plt.style.use("paper")

# rcParams['font.size'] = 12
rc('font', **{'family': "DejaVu Sans", 'size': "12"})
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
                azimuthal_quantity = np.nanmedian(annulus_data_1d)
            elif operation == 'mean':
                azimuthal_quantity = np.nanmean(annulus_data_1d)
            elif operation == 'min':
                azimuthal_quantity = np.nanmin(annulus_data_1d)
            elif operation == 'std':
                azimuthal_quantity = np.nanstd(annulus_data_1d)
            elif operation == 'percentiles':
                azimuthal_quantity = np.nanpercentile(
                    annulus_data_1d, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
            else:
                raise ValueError('Unknown operation: use mad_std, median, or mean')
            values.append(azimuthal_quantity)

            if operation != 'percentiles':
                profile[mask] = azimuthal_quantity
            else:
                profile[mask] = azimuthal_quantity[3]  # Median

    return profile, np.array(values)


def make_contrast_curve(detection_image, radial_bounds=None,
                        bin_width=3., sigma=5.,
                        companion_mask_radius=11,
                        pixel_scale=12.25,
                        yx_known_companion_position=None):
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
            raise ValueError("Dimensionality of known companion positions for contrast curve too large.")
    else:
        detected_signal_mask = None
        # detected_signal_mask = np.zeros(detection_image[0].shape, dtype='bool')

    snr_norm_profile, snr_norm = make_radial_profile(
        detection_image[2], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='mad_std', known_companion_mask=detected_signal_mask)
    median_flux_profile, percentile_flux_values = make_radial_profile(
        detection_image[0], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='percentiles', known_companion_mask=detected_signal_mask)
    stddev_flux_profile, stddev_flux_values = make_radial_profile(
        detection_image[0], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='mad_std', known_companion_mask=detected_signal_mask)
    median_uncertainty_profile, percentile_uncertainty_values = make_radial_profile(
        detection_image[1], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='percentiles', known_companion_mask=detected_signal_mask)
    min_uncertainty_profile, min_uncertainty_values = make_radial_profile(
        detection_image[1], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='min', known_companion_mask=detected_signal_mask)
    contrast_norm_profile, contrast_norm_values = make_radial_profile(
        detection_image[1], (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation='std', known_companion_mask=detected_signal_mask)

    # contrast_curve_table = np.vstack([np.arange(radial_bounds[0], radial_bounds[1])])
    contrast_image = detection_image[1] * sigma * snr_norm_profile
    median_contrast_image = median_uncertainty_profile * sigma * snr_norm_profile
    percentile_contrast_curve = percentile_uncertainty_values * sigma * snr_norm[:, None]
    min_contrast_curve = min_uncertainty_values * sigma * snr_norm
    normalized_detection_image = detection_image[2] / snr_norm_profile
    separation_pix = np.arange(radial_bounds[0], radial_bounds[1])
    separation_mas = np.arange(radial_bounds[0], radial_bounds[1]) * pixel_scale

    cols = [separation_pix,
            separation_mas,
            min_contrast_curve]
    for column in percentile_contrast_curve.T:
        cols.append(column)

    col_names = [
        'sep (pix)', 'sep (mas)',
        'contrast_min', 'contrast_0.15', 'contrast_2.5',
        'contrast_16', 'contrast_50', 'contrast_84',
        'contrast_97.5', 'contrast_99.85']
    contrast_table = Table(cols, names=col_names)

    return normalized_detection_image, contrast_table, contrast_image, median_contrast_image


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


def sep_pix_to_lod(separation, wavelength, instrument):
    fwhm = image_coordinates.compute_fwhm(
        wavelength, instrument.telescope_diameter, instrument.pixel_scale)
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
        ax0, contrast_table, color='#1b1cd5',
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
             convert_flux(contrast_table['contrast_50'], convert=convert_to_mag).data,
             color=color, linestyle=linestyle, label=label.format(curvelabel))  # plotting y versus x
    if plot_percentiles:
        ax0.fill_between(
            contrast_table['sep (pix)'],
            convert_flux(contrast_table['contrast_16'], convert=convert_to_mag),
            convert_flux(contrast_table['contrast_84'], convert=convert_to_mag),
            alpha=percentile_1sigma_alpha, color=color)  # shade the area between +- 1 sigma
        if plot_dashed_outline:
            ax0.plot(contrast_table['sep (pix)'],
                     convert_flux(contrast_table['contrast_16'].data, convert=convert_to_mag).data,
                     color=color, alpha=percentile_1sigma_alpha, linestyle='--', label=None)  # plotting y versus x
            ax0.plot(contrast_table['sep (pix)'],
                     convert_flux(contrast_table['contrast_84'], convert=convert_to_mag).data,
                     color=color, alpha=percentile_1sigma_alpha, linestyle='--', label=None)  # plotting y versus x

        if percentile_2sigma_alpha > 0:
            ax0.fill_between(
                contrast_table['sep (pix)'],
                convert_flux(contrast_table['contrast_84'], convert=convert_to_mag),
                convert_flux(contrast_table['contrast_97.5'], convert=convert_to_mag),
                alpha=percentile_2sigma_alpha, color=color)
        if percentile_3sigma_alpha > 0:
            ax0.fill_between(
                contrast_table['sep (pix)'],
                convert_flux(contrast_table['contrast_2.5'], convert=convert_to_mag),
                convert_flux(contrast_table['contrast_16'], convert=convert_to_mag),
                alpha=percentile_3sigma_alpha, color=color)
    return ax0


def plot_contrast_curve(
        contrast_table, instrument=None, wavelengths=[None],
        colors=['#1b1cd5'],
        linestyles=['-'],
        curvelabels=[None],
        add_wavelength_label=False,
        plot_vertical_lod=False,
        mirror_axis='mas', convert_to_mag=False,
        yscale='linear', savefig=None,
        sigma_label=5,
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
            curvelabel = 'median'
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = curvelabel + " ({:.2f} ".format(wavelengths[idx].value) \
                + wavelengths.unit.to_string() + ")"

        ax0 = add_contrast_curve_to_ax(
            ax0, contrast_curve, color=colors[idx],
            linestyle=linestyles[idx], curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            plot_dashed_outline=plot_dashed_outline,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha
        )

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
        min_wavelength = np.min(wavelengths)
        fwhm = image_coordinates.compute_fwhm(
            min_wavelength, instrument.telescope_diameter, instrument.pixel_scale)
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
        ax0.set_ylabel("{}$\sigma$ contrast (mag)".format(sigma_label))
    else:
        ax0.set_ylabel("{}$\sigma$ contrast".format(sigma_label))

    ax0.set_xlim(xmin, xmax + x_text_shift)  # set ticks visible, if using sharex = True. Not needed otherwise

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

    ax0.legend(loc=1)
    fig.tight_layout()
    if convert_to_mag:
        plt.gca().invert_yaxis()
    if savefig is not None:
        plt.savefig(savefig, dpi=300)
        pdf.savefig(bbox_inches='tight')
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
        sigma_label=5,
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
            curvelabel = 'median'
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = curvelabel + " ({:.2f} ".format(wavelengths[idx].value) \
                + wavelengths.unit.to_string() + ")"
        ax0 = add_contrast_curve_to_ax(
            ax0, contrast_curve, color=colors[idx],
            linestyle=linestyles[idx], curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha
        )

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
        # try:
        min_wavelength = np.min(wavelengths)
        # except TypeError:
        #     min_wavelength = wavelengths[]
        fwhm = image_coordinates.compute_fwhm(
            min_wavelength, instrument.telescope_diameter, instrument.pixel_scale)
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

    ax0.set_xlim(xmin, xmax + x_text_shift)  # set ticks visible, if using sharex = True. Not needed otherwise

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
        pdf.savefig(bbox_inches='tight')
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
    corrupt_separation = radial_bounds_test[0] + np.max(np.argwhere(np.isnan(andro_radial_test))) + 1
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


def fit_planet_detection(
        image, yx_position=None, yx_center=None, x_stddev=2., y_stddev=2.,
        box_size=7, mask_deviating=False, deviation_threshold=0.2):
    # spot fitting
    if yx_position is None:
        cy, cx = np.unravel_index(np.argmax(image), image.shape)
    else:
        cy, cx = yx_position
    cutout = Cutout2D(image, (cx, cy), box_size)
    # stamp = image[cy - box_size:cy + box_size, cx - box_size:cx + box_size].copy()
    xx, yy = np.meshgrid(np.arange(box_size), np.arange(box_size))
    yx_position_cutout = np.unravel_index(np.argmax(cutout.data), cutout.shape)
    g_init = models.Gaussian2D(
        amplitude=cutout.data.max(),
        x_mean=yx_position_cutout[1],
        y_mean=yx_position_cutout[0],
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=0.785)  # + models.Const2D(amplitude=stamp.min())
    fitter = fitting.LevMarLSQFitter()
    par = fitter(g_init, xx, yy, cutout.data)
    model = par(xx, yy)

    if mask_deviating:
        mask = abs((cutout.data - model) / cutout.data) < deviation_threshold  # Filter out
        g_init = models.Gaussian2D(
            amplitude=par.amplitude.value,
            x_mean=par.x_mean.value,
            y_mean=par.y_mean.value,
            x_stddev=par.x_stddev.value,
            y_stddev=par.y_stddev.value,
            theta=par.theta.value)
        par = fitter(g_init, xx[mask], yy[mask], cutout.data[mask])
        model = par(xx, yy)

    if yx_center is None:
        yx_center = (image.shape[0] // 2., image.shape[1] // 2)
    # Cutout works with x first and y second
    xy_fit_position_orig = cutout.to_original_position((par.x_mean.value, par.y_mean.value))
    yx_fit_position_orig = xy_fit_position_orig[::-1]
    yx_fit_relative = (xy_fit_position_orig[1] - yx_center[0], xy_fit_position_orig[0] - yx_center[1])
    return par, model, cutout.data, yx_fit_position_orig, yx_fit_relative


def plot_planet_detection(model, stamp):
    plt.close()
    plt.figure(200)
    plt.imshow(model, origin='bottom', interpolation='nearest')
    plt.figure(300)
    plt.imshow(stamp, origin='bottom', interpolation='nearest')
    plt.show()
