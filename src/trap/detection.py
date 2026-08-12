"""
Routines used in TRAP

@author: Matthias Samland
         MPIA Heidelberg
"""

import copy
import logging
import os
import warnings
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
from matplotlib.backends.backend_pdf import PdfPages
from natsort import natsorted
from numpy import interp
from photutils.aperture import CircularAnnulus
from scipy import linalg, ndimage, stats
from species import SpeciesInit
from species.data.database import Database
from species.read.read_model import ReadModel
from tqdm.auto import tqdm

from trap import image_coordinates, pca_regression, regressor_selection
from trap.image_coordinates import absolute_yx_to_relative_yx, relative_yx_to_rhophi
from trap.parameters import _to_reduction_config
from trap.reduction_wrapper import run_complete_reduction
from trap.template import SpectralTemplate
from trap.utils import (
    compute_empirical_correlation_matrix,
    find_nearest,
    load_object,
    save_object,
    subtract_angles,
)

logger = logging.getLogger(__name__)

# Mirrors `DetectionParameters.minimum_candidate_separation`; duplicated as a
# module constant so the candidate finders keep a sane floor when called
# directly (tests, notebooks) rather than through the configured entry points.
DEFAULT_MINIMUM_CANDIDATE_SEPARATION = 5.0

# Mirrors `DetectionParameters.max_candidates`. Every candidate costs a full
# contrast-table renormalization, so an unbounded list is a runtime hazard as
# much as a scientific one.
DEFAULT_MAX_CANDIDATES = 50

# Cap on the SNR-scaled candidate exclusion radius, as a multiple of the base
# radius. 2.5 takes IRDIS's 11 px base to ~28 px, which is where a 100-sigma
# binary's blob swarm stops on the HD_140408 test case; more starts eating real
# search area.
DEFAULT_MAX_EXCLUSION_RADIUS_FACTOR = 2.5


def _resolve_exclusion_radius(candidate_exclusion_radius, search_radius):
    """Fall back to `search_radius` when no separate exclusion radius is set."""
    if candidate_exclusion_radius is None:
        return search_radius
    return candidate_exclusion_radius


def _scaled_exclusion_radius(
    snr, candidate_threshold, mask_radius, max_factor, enabled=True
):
    """Grow a peak's exclusion radius with its significance.

    The contaminated area around a source scales with how far its wings stay
    above threshold, which grows with SNR; a radius tuned for a marginal
    detection leaves a bright binary's wings to re-enter the search as dozens of
    spurious candidates. `sqrt` keeps the growth gentle and the cap keeps a very
    bright source from blanking the search region.
    """
    if not enabled or not np.isfinite(snr) or snr <= candidate_threshold:
        return mask_radius
    factor = float(np.sqrt(snr / candidate_threshold))
    return mask_radius * float(np.clip(factor, 1.0, max_factor))

# plt.style.use("paper")

# rcParams['font.size'] = 12
# rc('font', **{'family': "DejaVu Sans", 'size': "12"})
# rc('legend', **{'fontsize': "11"})

# rc('text', usetex=True)
# rc('font', **{'family': "sans-serif"})
# params = {'text.latex.preamble': [r'\usepackage{siunitx}',
#                                   r'\usepackage{sfmath}', r'\sisetup{detect-family = true}', r'\usepackage{amsmath}']}
# plt.rcParams.update(params)


def make_radial_profile(
    data,
    radial_bounds=None,
    bin_width=3.0,
    operation="mad_std",
    yx_center=None,
    known_companion_mask=None,
    minimum_annulus_pixels=10,
):
    """
    Compute the radial profile of a 2D data array.

    Parameters:
    - data (ndarray): The 2D data array.
    - radial_bounds (tuple, optional): The radial bounds of the profile. If not provided, the bounds are determined based on the size of the data array.
    - bin_width (float, optional): The width of each radial bin.
    - operation (str, optional): The operation to be applied to the data within each bin. Options are "mad_std", "median", "mean", "min", "std", and "percentiles".
    - yx_center (tuple, optional): The center coordinates of the data array. If not provided, the center is assumed to be the center of the data array.
    - known_companion_mask (ndarray, optional): A mask indicating the positions of known companions to be excluded from the profile.
    - minimum_annulus_pixels (int, optional): Fall back to the un-masked annulus when
      excluding known companions leaves fewer than this many finite pixels. Without
      the fallback a companion mask that covers a whole annulus makes the profile —
      and therefore the normalized detection image — NaN across that annulus, which
      no downstream fit can recover from. Set to 0 to restore the un-guarded
      behaviour.

    Returns:
    - profile (ndarray): The computed radial profile.
    - values (ndarray): The values used to compute the profile for each radial bin.
    """
    
    profile = np.empty_like(data)
    profile[:] = np.nan
    values = []

    data[data == 0.0] = np.nan

    if radial_bounds is None:
        separation_max = data.shape[-1] // 2 * np.sqrt(2)
        radial_bounds = (1, int(separation_max))

    if yx_center is None:
        yx_center = (data.shape[-2] // 2.0, data.shape[-1] // 2.0)
    xy_center = yx_center[::-1]

    # Determine first non-zero separation, to prevent results below IWA
    inner_bound_index = int(yx_center[0] + radial_bounds[0])
    try:
        non_zero_separation = (
            radial_bounds[0]
            + np.max(
                np.argwhere(
                    np.isnan(
                        data[
                            inner_bound_index : inner_bound_index + 15,
                            int(yx_center[1]),
                        ]
                    )
                )
            )
            + 1
        )
    except ValueError:
        non_zero_separation = 0
    if non_zero_separation > radial_bounds[0] + 13:
        non_zero_separation = 0

    for separation in range(radial_bounds[0], radial_bounds[1]):
        if separation < non_zero_separation:
            if operation == "percentiles":
                values.append(np.ones(7) * np.nan)
            else:
                values.append(np.nan)
        else:
            # annulus_data = annulus_mask[0].multiply(data)
            # mask = annulus_mask[0].to_image(data.shape) > 0
            r_in = separation - bin_width / 2.0
            r_out = separation + bin_width / 2.0
            if r_in < 0.5:
                r_in = 0.5
            annulus_aperture = CircularAnnulus(xy_center, r_in=r_in, r_out=r_out)
            annulus_mask = annulus_aperture.to_mask(method="center")
            # Make sure only pixels are used for which data exists
            mask = annulus_mask.to_image(data.shape) > 0
            mask[int(xy_center[1]), int(xy_center[0])] = False

            if known_companion_mask is None:
                mask_wo_companion = mask
            else:
                mask_wo_companion = np.logical_and(mask, ~known_companion_mask)
                if (
                    np.count_nonzero(np.isfinite(data[mask_wo_companion]))
                    < minimum_annulus_pixels
                ):
                    # Keeping the companion in is a biased noise estimate; a NaN
                    # annulus is an unusable one. Bias high (which suppresses the
                    # source's own SNR) rather than lose the annulus entirely.
                    logger.debug(
                        "Annulus at separation %d retains too few unmasked pixels; "
                        "falling back to the un-masked statistic.", separation,
                    )
                    mask_wo_companion = mask

            # Data on which statistic is applied
            annulus_data_1d = data[mask_wo_companion]
            all_nan = False
            if bn.allnan(annulus_data_1d):
                if operation == "percentiles":
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
                        xy_center, r_in=r_in, r_out=r_out
                    )
                    annulus_mask = annulus_aperture.to_mask(method="center")
                    # Make sure only pixels are used for which data exists
                    mask = annulus_mask.to_image(data.shape) > 0
                    mask[int(xy_center[1]), int(xy_center[0])] = False

                if operation == "mad_std":
                    azimuthal_quantity = mad_std(annulus_data_1d, ignore_nan=True)
                elif operation == "median":
                    azimuthal_quantity = bn.nanmedian(annulus_data_1d)
                elif operation == "mean":
                    azimuthal_quantity = bn.nanmean(annulus_data_1d)
                elif operation == "min":
                    azimuthal_quantity = bn.nanmin(annulus_data_1d)
                elif operation == "std":
                    azimuthal_quantity = bn.nanstd(annulus_data_1d)
                elif operation == "percentiles":
                    azimuthal_quantity = np.nanpercentile(
                        annulus_data_1d, [0.15, 2.5, 16, 50, 84, 97.5, 99.85]
                    )
                else:
                    raise ValueError("Unknown operation: use mad_std, median, or mean")
            if operation != "percentiles":
                profile[mask] = azimuthal_quantity
            else:
                if all_nan:
                    profile[mask] = np.nan
                else:
                    profile[mask] = azimuthal_quantity[3]

            values.append(azimuthal_quantity)

    return profile, np.array(values)


def _adaptive_companion_mask_radius(
    yx_relative_position, companion_mask_radius, minimum_companion_mask_radius
):
    """Shrink a companion's exclusion radius so it cannot swallow its own annuli.

    A fixed radius applied to a source at small separation covers every annulus
    inside it, leaving `make_radial_profile` nothing to work with. Capping the
    radius at `separation - 1` keeps unmasked pixels at every separation the
    source could be measured at; the floor keeps the source's PSF core out of
    the noise estimate.
    """
    separation = float(np.hypot(yx_relative_position[0], yx_relative_position[1]))
    return float(
        np.clip(
            separation - 1.0, minimum_companion_mask_radius, companion_mask_radius
        )
    )


def make_contrast_curve(
    detection_image,
    radial_bounds=None,
    bin_width=3.0,
    companion_mask_radius=11,
    pixel_scale=12.25,
    yx_known_companion_position=None,
    mask_above_sigma=None,
    minimum_companion_mask_radius=3.0,
    minimum_annulus_pixels=10,
):
    yx_dim = (detection_image.shape[-2], detection_image.shape[-1])

    if radial_bounds is None:
        separation_max = detection_image.shape[-1] // 2 * np.sqrt(2)
        radial_bounds = (1, int(separation_max))

    if yx_known_companion_position is not None:
        yx_known_companion_position = np.array(yx_known_companion_position)
        if yx_known_companion_position.ndim == 1:
            yx_known_companion_position = yx_known_companion_position[None, :]
        elif yx_known_companion_position.ndim != 2:
            raise ValueError(
                "Dimensionality of known companion positions for contrast curve too large."
            )
        detected_signal_masks = []
        for yx_pos in yx_known_companion_position:
            detected_signal_masks.append(
                regressor_selection.make_signal_mask(
                    yx_dim,
                    yx_pos,
                    _adaptive_companion_mask_radius(
                        yx_pos, companion_mask_radius, minimum_companion_mask_radius
                    ),
                    relative_pos=True,
                    yx_center=None,
                )
            )
        detected_signal_mask = np.logical_or.reduce(detected_signal_masks)
    else:
        detected_signal_mask = None
        # detected_signal_mask = np.zeros(detection_image[0].shape, dtype='bool')

    snr_norm_profile, snr_norm = make_radial_profile(
        detection_image[2],
        (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation="mad_std",
        known_companion_mask=detected_signal_mask,
        minimum_annulus_pixels=minimum_annulus_pixels,
    )
    normalized_detection_image = detection_image[2] / snr_norm_profile

    # Repeat normalization without higher than 'mask_above_sigma' values
    # Removes most contamination by clear companion signals and binaries
    if mask_above_sigma is not None:
        local_detection_image = detection_image.copy()
        mask_high_values = normalized_detection_image > mask_above_sigma
        local_detection_image[:, mask_high_values] = np.nan

        snr_norm_profile, snr_norm = make_radial_profile(
            local_detection_image[2],
            (radial_bounds[0], radial_bounds[1]),
            bin_width=bin_width,
            operation="mad_std",
            known_companion_mask=detected_signal_mask,
            minimum_annulus_pixels=minimum_annulus_pixels,
        )
        normalized_detection_image = detection_image[2] / snr_norm_profile
    else:
        local_detection_image = detection_image

    # median_flux_profile, percentile_flux_values = make_radial_profile(
    #     local_detection_image[0],
    #     (radial_bounds[0], radial_bounds[1]),
    #     bin_width=bin_width,
    #     operation="percentiles",
    #     known_companion_mask=detected_signal_mask,
    # )
    # stddev_flux_profile, stddev_flux_values = make_radial_profile(
    #     local_detection_image[0],
    #     (radial_bounds[0], radial_bounds[1]),
    #     bin_width=bin_width,
    #     operation="mad_std",
    #     known_companion_mask=detected_signal_mask,
    # )
    median_uncertainty_profile, percentile_uncertainty_values = make_radial_profile(
        local_detection_image[1],
        (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation="percentiles",
        known_companion_mask=detected_signal_mask,
        minimum_annulus_pixels=minimum_annulus_pixels,
    )
    _, min_uncertainty_values = make_radial_profile(
        local_detection_image[1],
        (radial_bounds[0], radial_bounds[1]),
        bin_width=bin_width,
        operation="min",
        known_companion_mask=detected_signal_mask,
        minimum_annulus_pixels=minimum_annulus_pixels,
    )

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

    cols = [separation_pix, separation_mas, min_contrast_curve]
    for column in percentile_contrast_curve.T:
        cols.append(column)
    cols.append(snr_norm)

    col_names = [
        "sep (pix)",
        "sep (mas)",
        "contrast_min",
        "contrast_0.15",
        "contrast_2.5",
        "contrast_16",
        "contrast_50",
        "contrast_84",
        "contrast_97.5",
        "contrast_99.85",
        "snr_normalization",
    ]
    contrast_table = Table(cols, names=col_names)

    return (
        normalized_detection_image,
        contrast_table,
        uncertainty_image,
        median_uncertainty_image,
    )


def ratio_contrast_tables(table1, table2):
    ratio_table = table1.copy()
    ratio_table["contrast_50"] = table1["contrast_50"] / table2["contrast_50"]
    ratio_table["contrast_16"] = table1["contrast_16"] / table2["contrast_16"]
    ratio_table["contrast_84"] = table1["contrast_84"] / table2["contrast_84"]
    ratio_table["contrast_2.5"] = table1["contrast_2.5"] / table2["contrast_2.5"]
    ratio_table["contrast_97.5"] = table1["contrast_97.5"] / table2["contrast_97.5"]
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
    ax0,
    contrast_table,
    sigma=5,
    color="#1b1cd5",
    linestyle="-",
    curvelabel=None,
    convert_to_mag=False,
    plot_percentiles=True,
    plot_dashed_outline=True,
    percentile_1sigma_alpha=0.6,
    percentile_2sigma_alpha=0.3,
    percentile_3sigma_alpha=0.0,
):
    if curvelabel is not None:
        label = curvelabel
    else:
        label = None

    if linestyle is None:
        linestyle = "-"

    ax0.plot(
        contrast_table["sep (pix)"],
        convert_flux(
            contrast_table["contrast_50"] * sigma, convert=convert_to_mag
        ).data,
        color=color,
        linestyle=linestyle,
        label=label.format(curvelabel),
    )  # plotting y versus x
    if plot_percentiles:
        ax0.fill_between(
            contrast_table["sep (pix)"],
            convert_flux(contrast_table["contrast_16"] * sigma, convert=convert_to_mag),
            convert_flux(contrast_table["contrast_84"] * sigma, convert=convert_to_mag),
            alpha=percentile_1sigma_alpha,
            color=color,
        )  # shade the area between +- 1 sigma
        if plot_dashed_outline:
            ax0.plot(
                contrast_table["sep (pix)"],
                convert_flux(
                    contrast_table["contrast_16"] * sigma, convert=convert_to_mag
                ).data,
                color=color,
                alpha=percentile_1sigma_alpha,
                linestyle="--",
                label=None,
            )  # plotting y versus x
            ax0.plot(
                contrast_table["sep (pix)"],
                convert_flux(
                    contrast_table["contrast_84"] * sigma, convert=convert_to_mag
                ).data,
                color=color,
                alpha=percentile_1sigma_alpha,
                linestyle="--",
                label=None,
            )  # plotting y versus x

        if percentile_2sigma_alpha > 0:
            ax0.fill_between(
                contrast_table["sep (pix)"],
                convert_flux(
                    contrast_table["contrast_84"] * sigma, convert=convert_to_mag
                ),
                convert_flux(
                    contrast_table["contrast_97.5"] * sigma, convert=convert_to_mag
                ),
                alpha=percentile_2sigma_alpha,
                color=color,
            )
        if percentile_3sigma_alpha > 0:
            ax0.fill_between(
                contrast_table["sep (pix)"],
                convert_flux(
                    contrast_table["contrast_2.5"] * sigma, convert=convert_to_mag
                ),
                convert_flux(
                    contrast_table["contrast_16"] * sigma, convert=convert_to_mag
                ),
                alpha=percentile_3sigma_alpha,
                color=color,
            )
    return ax0


def plot_contrast_curve(
    contrast_table,
    instrument=None,
    wavelengths=[None],
    companion_table=None,
    template_fitted=False,
    colors=["#1b1cd5"],
    linestyles=["-"],
    curvelabels=[None],
    add_wavelength_label=False,
    plot_vertical_lod=False,
    mirror_axis="mas",
    convert_to_mag=False,
    yscale="linear",
    savefig=None,
    sigma=5,
    radial_bound=None,
    plot_percentiles=True,
    plot_dashed_outline=True,
    percentile_1sigma_alpha=0.6,
    percentile_2sigma_alpha=0.3,
    percentile_3sigma_alpha=0.0,
    set_xlim=None,
    set_ylim=None,
    plot_iwa=False,
    title=None,
    cmap=plt.cm.viridis,
    figsize=(8, 6),
    show=False,
):
    try:
        wavelengths = wavelengths.to(u.micron)
    except:
        warnings.warn("Provided wavelength could not be converted with astropy units.")

    if savefig is not None:
        result_folder = os.path.dirname(savefig)
        contrast_plot_path_pdf = os.path.join(
            result_folder, os.path.splitext(os.path.basename(savefig))[0] + ".pdf"
        )
        pdf = PdfPages(contrast_plot_path_pdf)

    plt.close()
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(1, 1)
    ax0 = fig.add_subplot(grid[0, 0])

    if colors is None:
        colors = colors = cmap(np.linspace(0, 1, len(contrast_table)))

        if wavelengths[0] is not None:
            wavelength_range = np.nanmax(wavelengths - np.nanmin(wavelengths))
            scaled_values = (wavelengths - np.nanmin(wavelengths)) / wavelength_range
            colors = cmap(scaled_values)
        else:
            colors = cmap(np.linspace(0, 1, len(contrast_table)))

    # Add contrast curve(s) to axis
    for idx, contrast_curve in enumerate(contrast_table):
        if curvelabels[idx] is None:
            curvelabel = ""
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = (
                curvelabel
                + "{:.2f} ".format(wavelengths[idx].value)
                + wavelengths.unit.to_string()
                + ""
            )

        ax0 = add_contrast_curve_to_ax(
            ax0,
            contrast_curve,
            sigma=sigma,
            color=colors[idx],
            linestyle=linestyles[idx],
            curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            plot_dashed_outline=plot_dashed_outline,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha,
        )

        if companion_table is not None:
            if len(companion_table) > 0:
                wavelength_indices = np.unique(companion_table["wavelength_index"])
                temp_table = companion_table[
                    companion_table["wavelength_index"] == wavelength_indices[idx]
                ]
                if template_fitted:
                    contrast_indices_closest_to_candidate = []
                    contrast_closest_to_candidate = []
                    for separation in temp_table["separation"].values:
                        contrast_index_closest_to_candidate = find_nearest(
                            separation, contrast_curve["sep (pix)"]
                        )
                        contrast_indices_closest_to_candidate.append(
                            contrast_index_closest_to_candidate
                        )
                        contrast_closest_to_candidate.append(
                            contrast_curve["contrast_50"][
                                contrast_index_closest_to_candidate
                            ]
                        )

                    contrast = (
                        temp_table["norm_snr_fit"].values
                        * contrast_closest_to_candidate
                    )
                    contrast_uncertainty = None
                else:
                    contrast = temp_table["contrast"].values
                    contrast_uncertainty = temp_table["uncertainty"].values
                if np.all(np.isfinite(contrast)):
                    ax0.errorbar(
                        x=temp_table["separation"].values,
                        y=contrast,
                        yerr=contrast_uncertainty,
                        fmt="o",
                        color=colors[idx],
                        markeredgecolor="k",
                        markeredgewidth=1,
                        capsize=3,
                    )

            for separation in np.unique(companion_table["separation"]):
                ax0.axvline(x=separation, color="k", linestyle=":", alpha=0.3)

    ax0.set_yscale(yscale)

    if set_xlim is not None:
        ax0.set_xlim(set_xlim)
   
    if set_ylim is not None:
        ax0.set_ylim(set_ylim)

    ymin, ymax = ax0.get_ylim()
    xmin, xmax = ax0.get_xlim()

    x_text_shift = np.abs((xmax - xmin) / 100.0) * 1.1
    y_text_shift = np.abs((ymax - ymin) / 100.0)

    # 10% padding at bottom for text
    # if convert_to_mag:
    #     ax0.set_ylim(ymin, ymax + y_text_shift * 10)
    # else:
    #     ax0.set_ylim(ymin - y_text_shift * 15, ymax)
    # if yscale == 'log':
    #     ax0.set_ylim(ymin - y_text_shift * 20, ymax)
    if yscale == "linear":
        ax0.set_ylim(ymin - y_text_shift, ymax)
    ymin, ymax = ax0.get_ylim()
    if yscale == "log":
        y_text_shift = np.abs((np.log10(ymax) - np.log10(ymin))) / 100 * 6
        # y_text_pos = ymin  # 10**(np.log10(ymin) + y_text_shift)
        ax0.set_ylim(10 ** (np.log10(ymin) - y_text_shift), ymax)
        y_text_pos = 10 ** (np.log10(ymin) - y_text_shift / 2)
    elif yscale == "linear":
        y_text_pos = ymin

    if plot_vertical_lod:
        fwhm = np.mean(instrument.fwhm)
        xposition = (np.array([1, 2, 3, 5, 10]) * fwhm).value
        mask = np.logical_and(xposition > xmin, xposition < xmax)
        xposition = xposition[mask]
        vert_labels = np.array(
            [
                r"$1 \lambda/D$",
                r"$2 \lambda/D$",
                r"$3 \lambda/D$",
                r"$5 \lambda/D$",
                r"$10 \lambda/D$",
            ]
        )[mask]

        if np.sum(mask) > 0:
            if convert_to_mag:
                # text_y = ymax
                y_text_shift *= -1.0
            # else:
            #     text_y = ymin

            for idx, xc in enumerate(xposition):
                ax0.axvline(
                    x=xc, color="k", linestyle="--", alpha=0.3
                )  # , label=vert_labels[idx])
                plt.text(
                    xc + x_text_shift,
                    y_text_pos,
                    vert_labels[idx],
                    rotation=90,
                    verticalalignment="bottom",
                )

    ax0.set_xlabel("Separation (pixel)")
    if convert_to_mag:
        ax0.set_ylabel("{}$\\sigma$ contrast (mag)".format(sigma))
    else:
        ax0.set_ylabel("{}$\\sigma$ contrast".format(sigma))

    # set ticks visible, if using sharex = True. Not needed otherwise
    ax0.set_xlim(xmin, xmax + x_text_shift)

    if plot_iwa is not False:
        if plot_iwa > xmin:
            if set_ylim is not None:
                ymin, ymax = set_ylim
            else:
                ymin, ymax = ax0.get_ylim()

            ax0.axvspan(xmin, plot_iwa, alpha=0.3, color="black")
            if yscale == "log":
                ytext_iwa = 10 ** (
                    np.log10(ymax) + (np.log10(ymin) - np.log10(ymax)) / 2.0
                )
            else:
                ytext_iwa = ymax - (ymax - ymin) * 0.05
            ax0.text(
                plot_iwa,
                ytext_iwa,
                r"IWA",
                rotation=90,
                horizontalalignment="right",
                verticalalignment="center",
                color="black",
                family="monospace",
                fontsize=12,
            )

    # ax0.set_ylim(np.min(contrast_table['contrast_min']), np.max(contrast_table['contrast_99.85']))
    # for tick in axes[0].get_xticklabels():
    ax0.tick_params(right=0, top=0, which="both", direction="out")  # bottom='on')
    # ax0.ticklabel_format(which='both', style='sci')
    # ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    ax2 = ax0.twiny()  # .twinx()
    # ax2.set_ylabel("Contrast (mag)")
    ax2.tick_params(which="both", direction="out")
    ax2.tick_params(left=0, right=0, bottom=0, which="both", direction="out")

    # get left axis limits
    # xmin, xmax = ax0.get_xlim()
    if mirror_axis == "lod":
        ax2.set_xlabel(r"Separation ($\lambda / D$)")
        ax2.set_xlim(
            (
                sep_pix_to_lod(xmin * u.pixel, wavelengths[0], instrument),
                sep_pix_to_lod(xmax * u.pixel, wavelengths[0], instrument),
            )
        )
        ax2.plot([], [])
    elif mirror_axis == "mas":
        ax2.set_xlim(
            (sep_pix_to_mas(xmin, instrument), sep_pix_to_mas(xmax, instrument))
        )
        ax2.set_xlabel("Separation (mas)")
        ax2.plot([], [])
    else:
        raise ValueError("Only 'lod' and 'mas' possible")

    ax3 = ax0.twinx()
    ax3.set_ylim((contrast_to_magnitude(ymin), contrast_to_magnitude(ymax)))
    ax3.set_ylabel(r"$\Delta \,$magnitude")
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
                cmap=cmap,
                norm=plt.Normalize(
                    vmin=wavelengths.value[0], vmax=wavelengths.value[-1]
                ),
            )
            cb = plt.colorbar(
                sm, ax=ax3, pad=0.13, use_gridspec=True, fraction=0.045
            )  # , format='%.2f')
            cb.set_label("wavelength (micron)", rotation=90, labelpad=10)

    # fig.tight_layout()
    if convert_to_mag:
        plt.gca().invert_yaxis()
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        try:
            pdf.savefig(bbox_inches="tight")
        except RuntimeError:
            logger.warning("Could not output pdf-version of contrast curve (this may be a Mac issue).")
        pdf.close()

    if show is True:
        plt.show()

    return fig


def plot_contrast_curve_ratio(
    contrast_table,
    instrument=None,
    wavelengths=[None],
    colors=["#1b1cd5"],
    linestyles=["-"],
    curvelabels=[None],
    add_wavelength_label=False,
    plot_vertical_lod=False,
    mirror_axis="mas",
    convert_to_mag=False,
    yscale="linear",
    savefig=None,
    radial_bound=None,
    plot_percentiles=True,
    percentile_1sigma_alpha=0.6,
    percentile_2sigma_alpha=0.3,
    percentile_3sigma_alpha=0.0,
    set_xlim=None,
    set_ylim=None,
    plot_iwa=False,
    show=False,
):
    try:
        wavelengths = wavelengths.to(u.micron)
    except:
        raise TypeError("Wavelengths must be a quantity array.")

    if savefig is not None:
        result_folder = os.path.dirname(savefig)
        contrast_plot_path_pdf = os.path.join(
            result_folder, os.path.splitext(os.path.basename(savefig))[0] + ".pdf"
        )
        pdf = PdfPages(contrast_plot_path_pdf)

    plt.close()
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(1, 1)
    ax0 = fig.add_subplot(grid[0, 0])

    # Add contrast curve(s) to axis
    for idx, contrast_curve in enumerate(contrast_table):
        if curvelabels[idx] is None:
            curvelabel = ""
        else:
            curvelabel = curvelabels[idx]

        if wavelengths[idx] is not None and add_wavelength_label:
            curvelabel = (
                "{:.2f}".format(wavelengths[idx].value)
                + wavelengths.unit.to_string()
                + ""
            )
        ax0 = add_contrast_curve_to_ax(
            ax0,
            contrast_curve,
            color=colors[idx],
            linestyle=linestyles[idx],
            curvelabel=curvelabel,
            convert_to_mag=convert_to_mag,
            plot_percentiles=plot_percentiles,
            percentile_1sigma_alpha=percentile_1sigma_alpha,
            percentile_2sigma_alpha=percentile_2sigma_alpha,
        )

    ax0.set_yscale(yscale)

    if set_xlim is not None:
        ax0.set_xlim(set_xlim)

    ymin, ymax = ax0.get_ylim()
    xmin, xmax = ax0.get_xlim()

    x_text_shift = np.abs((xmax - xmin) / 100.0) * 1.1
    y_text_shift = np.abs((ymax - ymin) / 100.0)

    if yscale == "log":
        ax0.set_ylim(ymin - y_text_shift * 15, ymax)
    if yscale == "linear":
        # ax0.set_ylim(ymin - y_text_shift * 15, ymax)
        ax0.set_ylim(0, ymax)
    ymin, ymax = ax0.get_ylim()
    if yscale == "log":
        y_text_shift = np.abs((np.log10(ymax) - np.log10(ymin))) / 100 * 2
        y_text_pos = ymin  # 10**(np.log10(ymin) + y_text_shift)
        ax0.set_ylim(10 ** (np.log10(ymin) - y_text_shift), ymax)
    elif yscale == "linear":
        y_text_pos = ymin + 0.1

    if plot_vertical_lod:
        fwhm = instrument.fwhm[0]
        xposition = (np.array([1, 2, 3, 5, 10]) * fwhm).value
        vert_labels = [
            r"$1 \lambda/D$",
            r"$2 \lambda/D$",
            r"$3 \lambda/D$",
            r"$5 \lambda/D$",
            r"$10 \lambda/D$",
        ]

        # text_y = ymin

        for idx, xc in enumerate(xposition):
            ax0.axvline(
                x=xc, color="k", linestyle="--", alpha=0.3
            )  # , label=vert_labels[idx])
            plt.text(
                xc + x_text_shift,
                y_text_pos,
                vert_labels[idx],
                rotation=90,
                verticalalignment="bottom",
            )

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
            ax0.axvspan(xmin, plot_iwa, alpha=0.3, color="black")
            ax0.text(
                plot_iwa,
                ymax - (ymax - ymin) * 0.05,
                r"IWA",
                rotation=90,
                horizontalalignment="right",
                verticalalignment="center",
                color="black",
                family="monospace",
                fontsize=12,
            )

    # ax0.set_ylim(np.min(contrast_table['contrast_min']), np.max(contrast_table['contrast_99.85']))
    # for tick in axes[0].get_xticklabels():
    ax0.tick_params(right=0, top=0, which="both", direction="out")  # bottom='on')
    # ax0.ticklabel_format(which='both', style='sci')
    # ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    ax2 = ax0.twiny()  # .twinx()
    # ax2.set_ylabel("Contrast (mag)")
    ax2.tick_params(which="both", direction="out")
    ax2.tick_params(left=0, right=0, bottom=0, which="both", direction="out")

    # get left axis limits
    # xmin, xmax = ax0.get_xlim()
    if mirror_axis == "lod":
        ax2.set_xlabel(r"Separation ($\lambda / D$)")
        ax2.set_xlim(
            (
                sep_pix_to_lod(xmin * u.pixel, wavelengths[0], instrument),
                sep_pix_to_lod(xmax * u.pixel, wavelengths[0], instrument),
            )
        )
        ax2.plot([], [])
    elif mirror_axis == "mas":
        ax2.set_xlim(
            (sep_pix_to_mas(xmin, instrument), sep_pix_to_mas(xmax, instrument))
        )
        ax2.set_xlabel("Separation (mas)")
        ax2.plot([], [])
    else:
        raise ValueError("Only 'lod' and 'mas' possible")
    ax0.axhline(y=1.0, color="k", linestyle="-", alpha=0.5)

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
            pdf.savefig(bbox_inches="tight")
        except RuntimeError:
            logger.warning("Could not output pdf-version of contrast curve (this may be a Mac issue).")
        pdf.close()

    if show is True:
        plt.show()

    return fig


def prepare_andromeda_output(andromeda_contrast, andromeda_norm_stddev, andromeda_snr):
    from scipy.ndimage.interpolation import shift

    andromeda_std = andromeda_snr / andromeda_contrast
    andromeda_std = 1.0 / andromeda_std
    andromeda_stack = np.array([andromeda_contrast, andromeda_std, andromeda_snr])

    # andromeda_stack[np.isnan(andromeda_stack)] = 0.
    for i, image in enumerate(andromeda_stack):
        andromeda_stack[i] = shift(image, (0.5, 0.5), order=1, prefilter=False)

    # Mask inner most annulus of non-zero values (corrupted by shift)
    radial_bounds_test = (1, 25)
    _, andro_radial_test = make_radial_profile(
        andromeda_stack[2],
        radial_bounds=radial_bounds_test,
        bin_width=1,
        operation="mad_std",
        yx_center=None,
        known_companion_mask=None,
    )
    corrupt_separation = (
        radial_bounds_test[0] + np.nanmax(np.argwhere(np.isnan(andro_radial_test))) + 1
    )
    assert (
        corrupt_separation != radial_bounds_test[-1]
    ), "Innermost non-zero andromeda result at edge of image"

    xy_center = (andromeda_stack.shape[-1] // 2, andromeda_stack.shape[-2] // 2)
    annulus_aperture = CircularAnnulus(
        xy_center, r_in=corrupt_separation - 0.5, r_out=corrupt_separation + 0.5
    )
    annulus_mask = annulus_aperture.to_mask(method="center")
    mask = annulus_mask.to_image(andromeda_stack[0].shape) > 0
    andromeda_stack[:, mask] = 0.0

    return andromeda_stack


def plot_distribution(
    detection_image,
    radial_bounds=None,
    plot_type="qqplot",
    sigma=5,
    companion_mask_radius=11,
    pixel_scale=12.25,
    yx_known_companion_position=None,
):
    yx_dim = (detection_image.shape[-2], detection_image.shape[-1])
    if yx_known_companion_position is not None:
        detected_signal_mask = regressor_selection.make_signal_mask(
            yx_dim,
            yx_known_companion_position,
            companion_mask_radius,
            relative_pos=True,
            yx_center=None,
        )
    else:
        detected_signal_mask = np.zeros(detection_image[0].shape, dtype="bool")

    annulus_mask = regressor_selection.make_annulus_mask(
        radial_bounds[0],
        radial_bounds[1],
        yx_dim=(detection_image.shape[-2], detection_image.shape[-1]),
        yx_center=None,
    )
    mask = np.logical_and(annulus_mask, ~detected_signal_mask)
    if plot_type == "qqplot":
        _ = stats.probplot(detection_image[mask], dist="norm", plot=plt)
        # QQ plot doesn't create labeled artists, so no legend needed
    elif plot_type == "distplot":
        import seaborn as sns
        sns.histplot(detection_image[mask], label="TRAP", kde=True, stat="density", linewidth=0)
        plt.legend()
    plt.show()


def _failed_gaussian_fit_result(yx_position, yx_center, cutout_shape):
    """Build a `fit_2d_gaussian` result whose parameters are all NaN.

    Returned instead of raising when a candidate cannot be fitted, so one
    pathological position costs that candidate's row rather than the whole
    target. `fit_ok=False` propagates into the candidate tables, where the
    existing shape validation drops the row.

    Parameters
    ----------
    yx_position : tuple of float
        Candidate position in original image coordinates, used verbatim as the
        reported (unfitted) position.
    yx_center : tuple of float
        Image center, for the relative position.
    cutout_shape : tuple of int
        Shape of the attempted cutout, so `mask` matches a successful result.

    Returns
    -------
    dict
        Same keys as a successful `fit_2d_gaussian` result.
    """
    nan_model = models.Gaussian2D(
        amplitude=np.nan, x_mean=np.nan, y_mean=np.nan,
        x_stddev=np.nan, y_stddev=np.nan, theta=np.nan,
    )
    return {
        "parameters": nan_model,
        "model": np.full(cutout_shape, np.nan),
        "cutout": np.full(cutout_shape, np.nan),
        "yx_fit_position_orig": (float(yx_position[0]), float(yx_position[1])),
        "yx_fit_relative": (
            float(yx_position[0]) - yx_center[0],
            float(yx_position[1]) - yx_center[1],
        ),
        "mask": np.zeros(cutout_shape, dtype=bool),
        "fwhm_area": np.nan,
        "param_cov_xy": None,
        "param_names": [],
        "fit_ok": False,
    }


def fit_2d_gaussian(
    image,
    yx_position=None,
    yx_center=None,
    x_stddev=1.43,
    y_stddev=2.63,
    box_size=7,
    mask_deviating=False,
    deviation_threshold=0.1,
    fix_width=True,
    fix_orientation=True,
    plot=False,
    fixed_position=None,
):
    if yx_center is None:
        yx_center = (image.shape[0] // 2.0, image.shape[1] // 2)
    # spot fitting
    if yx_position is None:
        cy, cx = np.unravel_index(np.nanargmax(image), image.shape)
    else:
        cy, cx = yx_position
    cutout = Cutout2D(image, (cx, cy), box_size)

    finite_mask = np.logical_and(np.isfinite(cutout.data), cutout.data != 0.0)

    # Too few finite pixels to constrain even the 3-parameter fallback below.
    # Historically an all-NaN cutout raised, which aborted the whole target: it
    # is produced by make_radial_profile blanking every annulus a candidate's
    # own companion mask covers, so a single candidate near the inner working
    # angle could destroy an otherwise complete reduction.
    if np.count_nonzero(finite_mask) < 6:
        logger.warning(
            "Gaussian fit at (y, x) = (%s, %s) has only %d usable pixels in its "
            "%dx%d cutout; reporting an unfitted candidate. A fully masked cutout "
            "usually means a candidate sits inside its own companion mask.",
            cy, cx, int(np.count_nonzero(finite_mask)), box_size, box_size,
        )
        return _failed_gaussian_fit_result((cy, cx), yx_center, cutout.shape)

    # Cutout2D trims at the frame edge, so the model grid has to follow the
    # cutout rather than box_size or the boolean indexing below misaligns.
    xx, yy = np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0]))
    yx_position_cutout = np.unravel_index(np.nanargmax(cutout.data), cutout.shape)
    gbounds = {
        "amplitude": (1e-9, None),
        "x_mean": (-2.0, box_size + 2),
        "y_mean": (-2.0, box_size + 2),
        "x_stddev": (0.5, box_size),
        "y_stddev": (0.5, box_size),
    }
    relative_yx = absolute_yx_to_relative_yx(yx_position, yx_center)
    rhophi = relative_yx_to_rhophi(relative_yx)
    phi = rhophi[1] * np.pi / 180.0

    g_init = models.Gaussian2D(
        amplitude=np.max(cutout.data[finite_mask]),
        x_mean=yx_position_cutout[1],
        y_mean=yx_position_cutout[0],
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=phi,
        bounds=gbounds,
    )  # + models.Const2D(amplitude=stamp.min())

    if fix_width:
        g_init.x_stddev.fixed = True
        g_init.y_stddev.fixed = True
    if fix_orientation:
        g_init.theta.fixed = True
    if fixed_position is not None:
        # Clamp the centroid to a caller-supplied sub-pixel (y, x) position in
        # original image coordinates (Fits B/C pin position to Fit A's centroid
        # so their amplitudes are measured at the radially-unbiased location).
        fx_cut, fy_cut = cutout.to_cutout_position(
            (fixed_position[1], fixed_position[0])
        )
        g_init.x_mean = fx_cut
        g_init.y_mean = fy_cut
        g_init.x_mean.fixed = True
        g_init.y_mean.fixed = True

    def _extract_param_cov_xy(par):
        param_cov_full = fitter.fit_info.get("param_cov", None)
        # Free-parameter names in the order LevMar exposes them:
        names = [n for n in par.param_names if not par.fixed[n]]
        if (
            param_cov_full is not None
            and "x_mean" in names
            and "y_mean" in names
        ):
            i_x = names.index("x_mean")
            i_y = names.index("y_mean")
            cov_xy = np.array(
                [
                    [param_cov_full[i_x, i_x], param_cov_full[i_x, i_y]],
                    [param_cov_full[i_y, i_x], param_cov_full[i_y, i_y]],
                ]
            )
        else:
            cov_xy = None
        return cov_xy, names

    # LevMarLSQFitter enforces bounds by clipping inside the objective, so a
    # parameter pushed past a bound lands in a flat region whose numerical
    # Jacobian column is identically zero; MINPACK then returns NaN parameters
    # and astropy raises NonFiniteValueError. TRF is genuinely bounded, which
    # matters because Fit A leaves both widths and the orientation free and
    # routinely drives x_stddev into its lower bound on speckle structure.
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    try:
        par = fitter(g_init, xx[finite_mask], yy[finite_mask], cutout.data[finite_mask])
    except Exception as error:
        # Speckles are not Gaussian. When the free fit still fails, retry with
        # the shape locked to the instrument PSF so only position and amplitude
        # stay free; the outcome is flagged either way, never raised.
        logger.warning(
            "Free 2D Gaussian fit at (y, x) = (%s, %s) failed (%s); retrying with "
            "the PSF shape held fixed.", cy, cx, type(error).__name__,
        )
        g_retry = g_init.copy()
        g_retry.x_stddev = x_stddev
        g_retry.y_stddev = y_stddev
        g_retry.x_stddev.fixed = True
        g_retry.y_stddev.fixed = True
        g_retry.theta.fixed = True
        fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
        try:
            par = fitter(
                g_retry, xx[finite_mask], yy[finite_mask], cutout.data[finite_mask]
            )
        except Exception as retry_error:
            logger.warning(
                "Constrained 2D Gaussian fit at (y, x) = (%s, %s) failed as well "
                "(%s); reporting an unfitted candidate.",
                cy, cx, type(retry_error).__name__,
            )
            return _failed_gaussian_fit_result((cy, cx), yx_center, cutout.shape)
    model = par(xx, yy)
    param_cov_xy, param_names = _extract_param_cov_xy(par)

    if plot:
        plt.imshow(cutout.data, origin="lower")
        plt.show()
        plt.imshow(model, origin="lower")
        plt.show()
    # Check if model fit is close to data (this is not perfect in case the detection is on the edge)
    # May require special treatment or warning
    mask = (
        abs(cutout.data - model)
        / np.nanmean(np.vstack([cutout.data[None, :], model[None, :]]), axis=0)
        < deviation_threshold
    )  # Filter out
    if mask_deviating:
        g_init = models.Gaussian2D(
            amplitude=par.amplitude.value,
            x_mean=par.x_mean.value,
            y_mean=par.y_mean.value,
            x_stddev=par.x_stddev.value,
            y_stddev=par.y_stddev.value,
            theta=par.theta.value,
        )
        par = fitter(g_init, xx[mask], yy[mask], cutout.data[mask])
        model = par(xx, yy)
        param_cov_xy, param_names = _extract_param_cov_xy(par)

    # Cutout works with x first and y second
    xy_fit_position_orig = cutout.to_original_position(
        (par.x_mean.value, par.y_mean.value)
    )
    yx_fit_position_orig = xy_fit_position_orig[::-1]
    yx_fit_relative = (
        xy_fit_position_orig[1] - yx_center[0],
        xy_fit_position_orig[0] - yx_center[1],
    )

    fwhm_area = par.x_stddev.value * par.y_stddev.value * 2.355**2 * np.pi

    parameters = {
        "parameters": par,
        "model": model,
        "cutout": cutout.data,
        "yx_fit_position_orig": yx_fit_position_orig,
        "yx_fit_relative": yx_fit_relative,
        "mask": mask,
        "fwhm_area": fwhm_area,
        "param_cov_xy": param_cov_xy,
        "param_names": param_names,
        "fit_ok": True,
    }

    return parameters


def plot_model_and_data(model, stamp):
    plt.close()
    plt.figure(200)
    plt.imshow(model, origin="lower", interpolation="nearest")
    plt.figure(300)
    plt.imshow(stamp, origin="lower", interpolation="nearest")
    plt.show()


def fit_planet_parameters(
    detection_image,
    normalized_detection_image,
    contrast_table,
    yx_position,
    x_stddev=1.43,
    y_stddev=2.63,
    box_size=7,
    iterate=False,
    mask_deviating=False,
    deviation_threshold=0.1,
    fix_width=True,
    fix_orientation=True,
    plot=False,
    phi_source=None,
):
    """Fit A/B/C on the three detection images per the astrometry-uncertainty
    spec:

    - A: raw SNR image (``detection_image[2]``), fully free → authoritative
      position, empirical widths ``x_fwhm_free``/``y_fwhm_free``, orientation
      ``theta_free``, and LevMar centroid covariance.
    - B: contrast image, position clamped to A's sub-pixel centroid → physical
      contrast amplitude at the radially-unbiased location.
    - C: norm-SNR image, position clamped to A → calibrated ``SNR_local`` at
      A's centroid (used for the Cramér-Rao floor and validation).

    Fixing B and C to A's centroid removes the radial-normalisation bias that a
    free norm-SNR fit would suffer while keeping every downstream column. The
    ``fix_width`` / ``fix_orientation`` arguments are retained for signature
    compatibility but are ignored: Fit A is always free (that is the point of
    the three-role setup).
    """
    # Fit A: raw SNR image, fully free.
    snr_image_parameters = fit_2d_gaussian(
        detection_image[2],
        yx_position=yx_position,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        box_size=box_size,
        mask_deviating=mask_deviating,
        deviation_threshold=deviation_threshold,
        fix_width=False,
        fix_orientation=False,
    )
    fit_a_yx_orig = snr_image_parameters["yx_fit_position_orig"]
    if not snr_image_parameters["fit_ok"]:
        # Fits B and C clamp their centroid to Fit A's, so without it there is
        # nothing to measure; report the candidate as unfitted across all three.
        failed = summarize_2d_gauss_fit_result(snr_image_parameters)
        return failed.copy(), failed.copy(), failed.copy()

    fit_a_start = (
        int(round(fit_a_yx_orig[0])),
        int(round(fit_a_yx_orig[1])),
    )

    # Fit C: norm-SNR image, position clamped to Fit A, amplitude + widths free.
    norm_snr_image_parameters = fit_2d_gaussian(
        normalized_detection_image,
        yx_position=fit_a_start,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        box_size=box_size,
        mask_deviating=mask_deviating,
        deviation_threshold=deviation_threshold,
        fix_width=False,
        fix_orientation=False,
        fixed_position=fit_a_yx_orig,
    )

    # Fit B: contrast image, position clamped to Fit A, amplitude + widths free.
    contrast_image_parameters = fit_2d_gaussian(
        detection_image[0],
        yx_position=fit_a_start,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        box_size=box_size,
        mask_deviating=mask_deviating,
        deviation_threshold=deviation_threshold,
        fix_width=False,
        fix_orientation=False,
        fixed_position=fit_a_yx_orig,
    )

    snr_local_normalized = float(
        norm_snr_image_parameters["parameters"].amplitude.value
    )

    if phi_source is None:
        # Math-angle of the source vector (radial = +x at α=0), matching the
        # (r, t) rotation convention in `_compute_rt_frame_sigmas` — not the
        # astronomical position angle.
        y_rel, x_rel = snr_image_parameters["yx_fit_relative"]
        phi_source_local = float(np.arctan2(y_rel, x_rel))
    else:
        phi_source_local = float(phi_source)

    contrast_image_results = summarize_2d_gauss_fit_result(
        contrast_image_parameters,
        phi_source=phi_source_local,
        snr_local_normalized=snr_local_normalized,
    )
    snr_image_results = summarize_2d_gauss_fit_result(
        snr_image_parameters,
        phi_source=phi_source_local,
        snr_local_normalized=snr_local_normalized,
    )
    norm_snr_image_results = summarize_2d_gauss_fit_result(
        norm_snr_image_parameters,
        phi_source=phi_source_local,
        snr_local_normalized=snr_local_normalized,
    )

    if plot:
        plot_model_and_data(
            snr_image_parameters["model"], snr_image_parameters["cutout"]
        )
        plot_model_and_data(
            contrast_image_parameters["model"], contrast_image_parameters["cutout"]
        )

    return contrast_image_results, snr_image_results, norm_snr_image_results


def _compute_rt_frame_sigmas(
    x_fwhm_free,
    y_fwhm_free,
    theta_free,
    phi_source,
    param_cov_xy,
    snr_local_normalized,
    eps_snr=1.0,
):
    """Compute source-aligned (radial, tangential) statistical σ per Section 3
    of the astrometry-uncertainty spec.

    Parameters
    ----------
    x_fwhm_free, y_fwhm_free : float
        Empirical FWHMs from a free 2D Gaussian fit on the SNR image, in pixels.
    theta_free : float
        Fitted orientation, radians (astropy Gaussian2D convention).
    phi_source : float
        Source-star direction in the fit frame, radians (as computed at fit
        init in `fit_2d_gaussian`).
    param_cov_xy : ndarray or None
        2×2 LevMar covariance of the centroid (x_mean, y_mean), or None if the
        fit's covariance was unavailable/singular.
    snr_local_normalized : float
        Calibrated SNR at the source position (amplitude of a norm-SNR fit).
    eps_snr : float
        Floor on SNR to avoid division by zero.

    Returns
    -------
    dict with keys sigma_r_stat, sigma_t_stat, rho_rt, sigma_r_fit,
    sigma_t_fit, sigma_r_cr, sigma_t_cr. All in pixels.
    """
    sigma_x_free = x_fwhm_free / 2.355
    sigma_y_free = y_fwhm_free / 2.355
    delta = theta_free - phi_source
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    sigma_psf_r = np.sqrt((sigma_x_free * cos_d) ** 2 + (sigma_y_free * sin_d) ** 2)
    sigma_psf_t = np.sqrt((sigma_x_free * sin_d) ** 2 + (sigma_y_free * cos_d) ** 2)

    snr_eff = max(snr_local_normalized, eps_snr)
    sigma_r_cr = sigma_psf_r / snr_eff
    sigma_t_cr = sigma_psf_t / snr_eff

    if param_cov_xy is not None and np.all(np.isfinite(param_cov_xy)):
        cos_p, sin_p = np.cos(phi_source), np.sin(phi_source)
        R = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
        c_rt = R.T @ param_cov_xy @ R
        var_r, var_t = c_rt[0, 0], c_rt[1, 1]
        if var_r > 0 and var_t > 0:
            sigma_r_fit = np.sqrt(var_r)
            sigma_t_fit = np.sqrt(var_t)
            rho_rt_fit = c_rt[0, 1] / (sigma_r_fit * sigma_t_fit)
        else:
            sigma_r_fit = np.nan
            sigma_t_fit = np.nan
            rho_rt_fit = np.nan
    else:
        sigma_r_fit = np.nan
        sigma_t_fit = np.nan
        rho_rt_fit = np.nan

    sigma_r_stat = sigma_r_cr if np.isnan(sigma_r_fit) else max(sigma_r_fit, sigma_r_cr)
    sigma_t_stat = sigma_t_cr if np.isnan(sigma_t_fit) else max(sigma_t_fit, sigma_t_cr)

    from_fit_r = (not np.isnan(sigma_r_fit)) and sigma_r_fit >= sigma_r_cr
    from_fit_t = (not np.isnan(sigma_t_fit)) and sigma_t_fit >= sigma_t_cr
    rho_rt = rho_rt_fit if (from_fit_r and from_fit_t) else 0.0

    return {
        "sigma_r_stat": sigma_r_stat,
        "sigma_t_stat": sigma_t_stat,
        "rho_rt": rho_rt,
        "sigma_r_fit": sigma_r_fit,
        "sigma_t_fit": sigma_t_fit,
        "sigma_r_cr": sigma_r_cr,
        "sigma_t_cr": sigma_t_cr,
    }


def summarize_2d_gauss_fit_result(
    result_dictionary, phi_source=None, snr_local_normalized=None
):
    fitted_parameters = {
        "x": [],
        "y": [],
        "x_relative": [],
        "y_relative": [],
        "separation": [],
        "position_angle": [],
        "amplitude": [],
        "x_fwhm": [],
        "y_fwhm": [],
        "theta": [],
        "good_pixels": [],
        "fwhm_area": [],
        "good_fraction": [],
    }

    fitted_parameters["x"].append(result_dictionary["yx_fit_position_orig"][1])
    fitted_parameters["y"].append(result_dictionary["yx_fit_position_orig"][0])
    fitted_parameters["x_relative"].append(result_dictionary["yx_fit_relative"][1])
    fitted_parameters["y_relative"].append(result_dictionary["yx_fit_relative"][0])
    rhophi = image_coordinates.relative_yx_to_rhophi(
        result_dictionary["yx_fit_relative"]
    )
    fitted_parameters["separation"].append(rhophi[0])
    fitted_parameters["position_angle"].append(rhophi[1])
    fitted_parameters["amplitude"].append(
        result_dictionary["parameters"].amplitude.value
    )
    fitted_parameters["x_fwhm"].append(
        result_dictionary["parameters"].x_stddev.value * 2.355
    )
    fitted_parameters["y_fwhm"].append(
        result_dictionary["parameters"].y_stddev.value * 2.355
    )
    fitted_parameters["theta"].append(
        (result_dictionary["parameters"].theta.value * u.radian).to(u.degree).value
    )
    fitted_parameters["good_pixels"].append(np.sum(result_dictionary["mask"]))
    fitted_parameters["fwhm_area"].append(result_dictionary["fwhm_area"])
    fitted_parameters["good_fraction"].append(
        np.sum(result_dictionary["mask"]) / result_dictionary["fwhm_area"]
    )

    rt = _compute_rt_frame_sigmas(
        x_fwhm_free=result_dictionary["parameters"].x_stddev.value * 2.355,
        y_fwhm_free=result_dictionary["parameters"].y_stddev.value * 2.355,
        theta_free=result_dictionary["parameters"].theta.value,
        phi_source=(0.0 if phi_source is None else phi_source),
        param_cov_xy=result_dictionary.get("param_cov_xy", None),
        snr_local_normalized=(
            np.nan if snr_local_normalized is None else snr_local_normalized
        ),
    )
    fitted_parameters["radial_sigma_stat"] = [rt["sigma_r_stat"]]
    fitted_parameters["tangential_sigma_stat"] = [rt["sigma_t_stat"]]
    fitted_parameters["rt_corr_stat"] = [rt["rho_rt"]]
    fitted_parameters["radial_sigma_fit"] = [rt["sigma_r_fit"]]
    fitted_parameters["tangential_sigma_fit"] = [rt["sigma_t_fit"]]
    fitted_parameters["radial_sigma_cr"] = [rt["sigma_r_cr"]]
    fitted_parameters["tangential_sigma_cr"] = [rt["sigma_t_cr"]]
    fitted_parameters["fit_ok"] = [bool(result_dictionary.get("fit_ok", True))]

    fitted_parameters = pd.DataFrame(fitted_parameters)
    return fitted_parameters


def _combine_channels_rt_frame(
    group_rows, search_radius, snr_values=None, independent_channels=False
):
    """Inverse-variance combination of per-channel (r, t) σ with a PDG scale
    factor, per Section 4 of the astrometry-uncertainty spec.

    ``group_rows`` is a DataFrame with one row per channel that detected this
    candidate (already grouped upstream). Position and σ columns are combined;
    all other columns are copied from the highest-SNR channel so the resulting
    headline row is internally consistent (shape params, good-fraction, etc.).
    ``snr_values`` optionally supplies the per-row ranking SNR (calibrated
    norm-SNR amplitude); when omitted the raw-SNR fit ``amplitude`` is used.

    ``independent_channels`` controls whether the formal ``1/Σ(1/σ²)`` shrinkage
    is trusted. Speckle residuals are strongly correlated between neighbouring
    wavelength channels — on 51 Eri IFS the 37-channel template collapse reaches
    a *lower* normalised SNR than its single best channel, i.e. the multiplex
    gain is ~0 — so combining channels as if independent understates σ by up to
    √n. With the default ``False`` the combined σ is floored at the best
    contributing channel's σ, which is the most that can be claimed without
    demonstrating independence. χ²_red ≪ 1 in the returned diagnostics is the
    signature of correlated inputs.

    Returns a single-row DataFrame with the combined position, σ columns, and
    χ²_red diagnostics.
    """
    group_rows = group_rows.reset_index(drop=True)
    n = len(group_rows)

    x_rel = group_rows["x_relative"].values.astype(float)
    y_rel = group_rows["y_relative"].values.astype(float)
    sigma_r = group_rows["radial_sigma_stat"].values.astype(float)
    sigma_t = group_rows["tangential_sigma_stat"].values.astype(float)
    # Rotation uses the math-angle of the source vector (radial = +x at α=0),
    # matching the (r, t) convention in `_compute_rt_frame_sigmas`. The
    # astronomical position_angle is emitted separately for output.
    alpha = np.arctan2(y_rel, x_rel)

    if snr_values is None:
        snr_values = group_rows["amplitude"].values.astype(float)
    else:
        snr_values = np.asarray(snr_values, dtype=float)
    donor = int(np.nanargmax(snr_values))

    # Weighted mean position in (x, y): weights use the tighter (tangential)
    # axis, combining per-channel frames that differ only slightly in angle.
    w = 1.0 / np.maximum(sigma_t**2, 1e-12)
    x_bar = float(np.sum(w * x_rel) / np.sum(w))
    y_bar = float(np.sum(w * y_rel) / np.sum(w))
    sep_bar = float(np.hypot(x_bar, y_bar))
    alpha_bar = float(np.arctan2(y_bar, x_bar))
    pa_bar = float(np.degrees(np.arctan2(-x_bar, y_bar)) % 360.0)

    cos_p, sin_p = np.cos(alpha_bar), np.sin(alpha_bar)
    R = np.array([[cos_p, -sin_p], [sin_p, cos_p]])

    # Rotate each channel's (r, t) covariance into the combined (r, t) frame.
    var_r_list, var_t_list = [], []
    for k in range(n):
        c_rt_k = np.diag([sigma_r[k] ** 2, sigma_t[k] ** 2])
        cos_k, sin_k = np.cos(alpha[k]), np.sin(alpha[k])
        R_k = np.array([[cos_k, -sin_k], [sin_k, cos_k]])
        c_xy_k = R_k @ c_rt_k @ R_k.T
        c_rt_bar_k = R.T @ c_xy_k @ R
        var_r_list.append(c_rt_bar_k[0, 0])
        var_t_list.append(c_rt_bar_k[1, 1])
    var_r_arr = np.array(var_r_list)
    var_t_arr = np.array(var_t_list)

    w_r = 1.0 / np.maximum(var_r_arr, 1e-12)
    w_t = 1.0 / np.maximum(var_t_arr, 1e-12)
    var_r_formal = 1.0 / np.sum(w_r)
    var_t_formal = 1.0 / np.sum(w_t)

    dx = x_rel - x_bar
    dy = y_rel - y_bar
    dr_k = dx * cos_p + dy * sin_p
    dt_k = -dx * sin_p + dy * cos_p

    if n > 1:
        chi2_red_r = float(np.sum(w_r * dr_k**2) / (n - 1))
        chi2_red_t = float(np.sum(w_t * dt_k**2) / (n - 1))
        scale_r = max(1.0, np.sqrt(chi2_red_r))
        scale_t = max(1.0, np.sqrt(chi2_red_t))
    else:
        chi2_red_r = np.nan
        chi2_red_t = np.nan
        scale_r = 1.0
        scale_t = 1.0

    sigma_r_combined = np.sqrt(var_r_formal) * scale_r
    sigma_t_combined = np.sqrt(var_t_formal) * scale_t

    if not independent_channels:
        sigma_r_combined = max(sigma_r_combined, float(np.sqrt(np.min(var_r_arr))))
        sigma_t_combined = max(sigma_t_combined, float(np.sqrt(np.min(var_t_arr))))

    c_rt_combined = np.array(
        [[sigma_r_combined**2, 0.0], [0.0, sigma_t_combined**2]]
    )
    c_xy_combined = R @ c_rt_combined @ R.T
    sigma_x_combined = float(np.sqrt(c_xy_combined[0, 0]))
    sigma_y_combined = float(np.sqrt(c_xy_combined[1, 1]))
    rho_xy_combined = float(
        c_xy_combined[0, 1] / (sigma_x_combined * sigma_y_combined)
    )

    row = group_rows.iloc[[donor]].copy()
    # Recompute the absolute position from the donor's own image center offset.
    x_center_abs = float(group_rows["x"].values[donor] - x_rel[donor])
    y_center_abs = float(group_rows["y"].values[donor] - y_rel[donor])
    row["x"] = x_center_abs + x_bar
    row["y"] = y_center_abs + y_bar
    row["x_relative"] = x_bar
    row["y_relative"] = y_bar
    row["separation"] = sep_bar
    row["position_angle"] = pa_bar
    row["radial_sigma_stat"] = sigma_r_combined
    row["tangential_sigma_stat"] = sigma_t_combined
    row["separation_sigma"] = sigma_r_combined
    row["position_angle_sigma"] = float(
        np.degrees(sigma_t_combined / max(sep_bar, 1e-6))
    )
    row["x_relative_sigma"] = sigma_x_combined
    row["y_relative_sigma"] = sigma_y_combined
    row["xy_relative_corr"] = rho_xy_combined
    row["chi2_red_radial"] = chi2_red_r
    row["chi2_red_tangential"] = chi2_red_t
    row["channels_above_threshold"] = n
    return row


def _template_output_filenames(template_name):
    """Per-template products written only when a template yields a candidate."""
    return [
        f"companion_table_{template_name}.csv",
        f"validated_companion_table_{template_name}.csv",
        f"validated_companion_table_short_{template_name}.csv",
        f"companion_spectra_{template_name}.pdf",
        f"contrast_plot_{template_name}.pdf",
        f"contrast_plot_{template_name}.png",
    ]


def _overall_output_filenames(prefix):
    """Cross-template products written only when at least one template detects."""
    return [
        f"overall_{prefix}companion_detections.csv",
        f"overall_{prefix}companion_detections_spectra.csv",
    ]


def _remove_stale_outputs(output_dir, filenames):
    """Delete `filenames` so a run can only leave behind its own products.

    The writes for these files sit on the success path alone: a template that
    finds no candidate takes an early exit, and a crash in the spectra extraction
    aborts before them. Either way the previous run's copies survive next to this
    run's freshly written detection maps and are indistinguishable from current
    results. Removing up front rather than on the failure branches also covers
    exceptions. Absence is the unambiguous signal for "this run found nothing";
    an empty table would not separate that from "never ran".
    """
    for filename in filenames:
        try:
            os.remove(os.path.join(output_dir, filename))
        except FileNotFoundError:
            pass


def _combine_templates_best_snr(per_template_tables, search_radius):
    """Cross-template combination per Section 5 of the astrometry-uncertainty
    spec. Each element of ``per_template_tables`` is one template's companion
    table (one row per candidate *per wavelength*). Sources are grouped across
    templates within ``search_radius`` in (x_relative, y_relative); the headline
    per source is the highest-SNR template's row set.

    Templates share the same pixel-noise realisation, so σ is *not* combined in
    quadrature. Instead the winning template's every-wavelength rows are kept
    verbatim (preserving the spectrum) and the across-template scatter is
    reported in separate ``*_sigma_template_scatter`` columns, with a boolean
    ``astrometry_template_disagreement`` flag.
    """
    tables = [t for t in per_template_tables if t is not None and not t.empty]
    if not tables:
        return pd.DataFrame()

    # One representative row per (template, candidate) drives grouping and the
    # best-SNR choice; the full wavelength rows are re-fetched for the winner.
    reps = []
    for table_index, table in enumerate(tables):
        if "candidate_id" in table.columns:
            for _, grp in table.groupby("candidate_id"):
                rep = grp.sort_values("wavelength_index").iloc[[0]].copy()
                rep["_table_index"] = table_index
                reps.append(rep)
        else:
            rep = table.iloc[[0]].copy()
            rep["_table_index"] = table_index
            reps.append(rep)
    reps = pd.concat(reps, ignore_index=True)

    positions = reps[["x_relative", "y_relative"]].values.astype(float)
    used = np.zeros(len(reps), dtype=bool)
    output_rows = []
    for i in range(len(reps)):
        if used[i]:
            continue
        dist = np.linalg.norm(positions - positions[i], axis=1)
        group_mask = (dist < search_radius) & (~used)
        used[group_mask] = True
        group = reps[group_mask]

        best_pos = int(np.nanargmax(group["norm_snr_fit_free"].values))
        best_rep = group.iloc[best_pos]
        best_table = tables[int(best_rep["_table_index"])]
        best_template_name = best_rep["template_name"]

        if "candidate_id" in best_rep.index:
            best_candidate_id = best_rep["candidate_id"]
            winner_rows = best_table[
                best_table["candidate_id"] == best_candidate_id
            ].copy()
        else:
            winner_rows = best_table.copy()

        n = len(group)
        if n >= 2:
            scatter_x = float(group["x_relative"].std(ddof=1))
            scatter_y = float(group["y_relative"].std(ddof=1))
            scatter_sep = float(group["separation"].std(ddof=1))
            scatter_pa = float(group["position_angle"].std(ddof=1))
        else:
            scatter_x = np.nan
            scatter_y = np.nan
            scatter_sep = np.nan
            scatter_pa = np.nan

        winner_rows["best_template"] = best_template_name
        winner_rows["n_templates_above_threshold"] = int(n)
        winner_rows["x_relative_sigma_template_scatter"] = scatter_x
        winner_rows["y_relative_sigma_template_scatter"] = scatter_y
        winner_rows["separation_sigma_template_scatter"] = scatter_sep
        winner_rows["position_angle_sigma_template_scatter"] = scatter_pa

        # Disagreement: template scatter exceeds 2× the headline σ on either
        # axis (only defined for n ≥ 2).
        if n >= 2:
            headline_x = float(winner_rows["x_relative_sigma"].values[0])
            headline_y = float(winner_rows["y_relative_sigma"].values[0])
            disagree = (scatter_x > 2.0 * headline_x) or (scatter_y > 2.0 * headline_y)
        else:
            disagree = False
        winner_rows["astrometry_template_disagreement"] = bool(disagree)
        output_rows.append(winner_rows)

    combined = pd.concat(output_rows, ignore_index=True)
    combined = combined.drop(columns=["_table_index"], errors="ignore")
    return combined.sort_values("separation", ignore_index=True)


_PER_CHANNEL_OVERRIDE_COLS = [
    "x",
    "y",
    "x_relative",
    "y_relative",
    "separation",
    "position_angle",
    "x_relative_sigma",
    "y_relative_sigma",
    "xy_relative_corr",
    "separation_sigma",
    "position_angle_sigma",
    "radial_sigma_stat",
    "tangential_sigma_stat",
    "channels_above_threshold",
]


def _override_astrometry_from_per_channel(
    overall_table,
    per_channel_table,
    search_radius,
    n_channels_total=None,
    min_channel_fraction=0.5,
):
    """Make per-channel astrometry the reported position/σ of each source, when
    it rests on enough of the data to be an improvement.

    The spectral collapse maximises detection SNR, not astrometric accuracy: it
    folds in channels with no signal at the source, whose speckle structure
    biases the centroid. On 51 Eri **IRDIS** (2 channels) that is a real defect —
    the collapse sits ~9 mas / 2.6σ off GRAVITY because the signal-free K2
    channel carries a large weight, while the single detected channel (K1) is
    within 0.7 mas.

    That reasoning does **not** extend to a many-channel IFS cube, and applying
    it there makes the astrometry worse. On 51 Eri IFS only 2 of 37 channels
    clear ``candidate_threshold``, so the "cleaner" position discards ~95% of the
    signal (including the entire J-band peak) and is selected by the very noise
    that promoted those channels above threshold; the collapse, where signal-free
    channels get low template weight and their speckle contributions average
    down, lands 4.4 mas closer to GRAVITY.

    The override therefore requires the contributing channels to be at least
    ``min_channel_fraction`` of ``n_channels_total``. A 2-channel DBI detection
    in one channel (0.5) passes; 2 of 37 IFS channels (0.054) does not, and the
    collapse is kept (``astrometry_source == "collapse"``). Passing
    ``n_channels_total=None`` disables the gate. Detection significance, spectrum
    and template diagnostics are left untouched either way, and
    ``per_channel_astrometry.csv`` is still written for inspection.
    """
    out = overall_table.copy()
    out["astrometry_source"] = "collapse"
    if per_channel_table is None or len(per_channel_table) == 0:
        return out

    pc = per_channel_table.reset_index(drop=True)
    pc_pos = pc[["x_relative", "y_relative"]].values.astype(float)
    cols = [c for c in _PER_CHANNEL_OVERRIDE_COLS if c in pc.columns and c in out.columns]

    min_channels = None
    if n_channels_total is not None and "channels_above_threshold" in pc.columns:
        min_channels = min_channel_fraction * float(n_channels_total)

    if "candidate_id" in out.columns:
        groups = list(out.groupby("candidate_id").groups.values())
    else:
        groups = [out.index]

    for idx in groups:
        rep = out.loc[idx[0]]
        dist = np.hypot(
            pc_pos[:, 0] - float(rep["x_relative"]),
            pc_pos[:, 1] - float(rep["y_relative"]),
        )
        j = int(np.argmin(dist))
        if dist[j] > search_radius:
            continue
        if min_channels is not None:
            n_used = float(pc.iloc[j]["channels_above_threshold"])
            if n_used < min_channels:
                logger.info(
                    "Per-channel astrometry uses %g of %g channels (< %.0f%% of "
                    "them); keeping the template-collapse position. The "
                    "per-channel measurement is still written to "
                    "per_channel_astrometry.csv.",
                    n_used,
                    float(n_channels_total),
                    100 * min_channel_fraction,
                )
                continue
        for c in cols:
            out.loc[idx, c] = pc.iloc[j][c]
        out.loc[idx, "astrometry_source"] = "per_channel"
    return out


class DetectionAnalysis(object):
    """Class for analyzing TRAP detection results and candidate characterization.
    
    This class provides methods for reading TRAP reduction outputs, generating
    contrast curves, finding and fitting candidates, and performing template 
    matching analysis.
    """

    def __init__(
        self,
        result_folder=None,
        detection_images=None,
        wavelength_indices=None,
        instrument=None,
        reduction_parameters=None,
    ):
        """Initialize DetectionAnalysis object.
        
        Parameters
        ----------
        result_folder : str, optional
            Path to folder containing TRAP reduction outputs.
        detection_images : array_like, optional
            Pre-loaded detection images.
        wavelength_indices : array_like, optional
            Wavelength indices for the analysis.
        instrument : Instrument, optional
            Instrument object containing observational parameters.
        reduction_parameters : TrapReductionConfig or TrapConfig, optional
            Reduction configuration; a TrapConfig is reduced to its
            ``reduction`` sub-config.
        """

        self.detection_images = detection_images
        self.wavelength_indices = wavelength_indices
        self.instrument = instrument
        
        if reduction_parameters is not None:
            self.reduction_parameters = _to_reduction_config(reduction_parameters)
        else:
            self.reduction_parameters = None
            
        self.detected_signal_mask = None
        self.templates = OrderedDict()
        self.empirical_correlation = None

        if instrument is not None:
            self.instrument.compute_fwhm()

    def read_output(
        self,
        component_fraction,
        result_folder=None,
        reduction_type="temporal",
        correlated_residuals=False,
        read_parameters=True,
        read_instrument=True,
        reduction_parameters=None,
        instrument=None,
    ):
        """Read TRAP reduction output files and set up detection analysis.
        
        Parameters
        ----------
        component_fraction : float
            Component fraction used in the reduction.
        result_folder : str, optional
            Path to folder containing reduction outputs. If None, uses
            self.reduction_parameters.result_folder.
        reduction_type : str, optional
            Type of reduction ("temporal", "spatial", "temporal_plus_spatial").
            Default is "temporal".
        correlated_residuals : bool, optional
            Whether to read correlated residual outputs. Default is False.
        read_parameters : bool, optional
            Whether to read parameters from saved files. Default is True.
        reduction_parameters : TrapReductionConfig or TrapConfig, optional
            Reduction configuration. Only used if read_parameters=False.
        instrument : Instrument, optional
            Instrument object. Only used if read_parameters=False.
        """
        if result_folder is None:
            self.result_folder = self.reduction_parameters.result_folder
        else:
            self.result_folder = result_folder

        self.component_fraction = component_fraction
        self.correlated_residuals = correlated_residuals
        self.reduction_type = reduction_type
        if read_instrument:
            self.instrument = load_object(os.path.join(result_folder, "instrument.obj"))

        if read_parameters:
            config_path = os.path.join(result_folder, "reduction_config.obj")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"No 'reduction_config.obj' in {result_folder}. Reductions produced by "
                    "TRAP < 2.0 only wrote 'reduction_parameters.obj', holding the removed "
                    "Reduction_parameters object. Re-run the reduction, or pass the config "
                    "explicitly with read_parameters=False."
                )
            self.reduction_parameters = load_object(config_path)
        else:
            if reduction_parameters is not None and instrument is not None:
                self.reduction_parameters = _to_reduction_config(reduction_parameters)
                self.instrument = instrument
        self.instrument.compute_fwhm()

        if correlated_residuals:
            detection_image_name = "detection_corr_lam"
        else:
            detection_image_name = "detection_lam"

        glob_pattern = os.path.join(
            self.result_folder,
            detection_image_name
            + f"*frac{component_fraction:.2f}_{reduction_type}.fits")

        detection_file_paths = natsorted(glob(glob_pattern))

        assert len(detection_file_paths) > 0, "No output files found for:\n{}.".format(
            glob_pattern
        )

        # Read in data
        detection_cube = []
        for file in detection_file_paths:
            detection_cube.append(fits.getdata(file))
        self.detection_cube = np.array(detection_cube)

        self.detection_cube[self.detection_cube == 0.] = np.nan

        # Determine indices reduced
        filenames = [os.path.basename(file_path) for file_path in detection_file_paths]
        character_index = filenames[0].find("lam")
        self.wavelength_indices = np.array(
            [int(file[character_index + 3 : character_index + 5]) for file in filenames]
        )

        # Remove wavelength from name
        # Add ouput paths to class
        self.file_paths = {}
        self.basename = filenames[0].replace(
            "_lam{:02d}".format(self.wavelength_indices[0]), ""
        )
        self.file_paths["detection_image_path"] = os.path.join(
            self.result_folder, self.basename
        )
        self.file_paths["norm_detection_image_path"] = os.path.join(
            self.result_folder, self.basename.replace("detection", "norm_detection")
        )
        self.file_paths["contrast_table_path"] = os.path.join(
            self.result_folder, self.basename.replace("detection", "contrast_table")
        )
        self.file_paths["uncertainty_image_path"] = os.path.join(
            self.result_folder, self.basename.replace("detection", "uncertainty_image")
        )
        self.file_paths["median_uncertainty_image_path"] = os.path.join(
            self.result_folder,
            self.basename.replace("detection", "median_uncertainty_image"),
        )
        self.file_paths["contrast_plot_path"] = os.path.join(
            self.result_folder,
            os.path.splitext(self.basename)[0].replace("detection", "contrast_plot")
            + ".jpg",
        )

        # Derived from the per-wavelength files just read, so it has to be
        # rewritten every time: guarding on existence froze it at whatever the
        # first run produced, and no overwrite/force path reached it afterwards.
        fits.writeto(
            self.file_paths["detection_image_path"],
            self.detection_cube,
            overwrite=True,
        )

    def contrast_table_and_normalization(
        self,
        detection_cube=None,
        cube_indices=None,
        yx_known_companion_position=None,
        mask_above_sigma=None,
        save=False,
        file_paths=None,
        overwrite=True,
        inplace=True,
    ):
        """detection_cube contains the detection_image for all wavelengths"""

        if detection_cube is None:
            detection_cube_used = self.detection_cube
        else:
            detection_cube_used = detection_cube

        if yx_known_companion_position is None:
            yx_known_companion_position = (
                self.reduction_parameters.yx_known_companion_position
            )

        if file_paths is None:
            file_paths = self.file_paths

        detection_products = {}
        normalized_detection_cube = []
        contrast_tables = []
        uncertainty_cube = []
        median_uncertainty_cube = []

        self.pixel_scale_mas = (
            (1 * u.pixel).to(u.mas, self.instrument.pixel_scale).value
        )

        if cube_indices is None:
            cube_indices = list(range(len(detection_cube_used)))

        for cube_index in cube_indices:
            # for detection_image in detection_cube_used:
            (
                normalized_detection_image,
                contrast_table,
                uncertainty_image,
                median_uncertainty_image,
            ) = make_contrast_curve(
                detection_cube_used[cube_index],
                radial_bounds=None,
                bin_width=self.reduction_parameters.normalization_width,
                companion_mask_radius=self.reduction_parameters.companion_mask_radius,
                pixel_scale=self.pixel_scale_mas,
                mask_above_sigma=mask_above_sigma,
                yx_known_companion_position=yx_known_companion_position,
            )
            normalized_detection_cube.append(normalized_detection_image)
            contrast_tables.append(contrast_table)
            uncertainty_cube.append(uncertainty_image)
            median_uncertainty_cube.append(median_uncertainty_image)

        detection_products["normalized_detection_cube"] = np.array(
            normalized_detection_cube
        )
        detection_products["uncertainty_cube"] = np.array(uncertainty_cube)
        detection_products["median_uncertainty_cube"] = np.array(
            median_uncertainty_cube
        )
        detection_products["contrast_tables"] = contrast_tables

        # Add real wavelength to contrast tables and concatenate into one table
        contrast_table = detection_products["contrast_tables"].copy()
        for idx, wavelength_index in enumerate(self.wavelength_indices[cube_indices]):
            wavelength_index_column = (
                np.ones(len(contrast_table[idx])) * wavelength_index
            )
            contrast_table[idx] = contrast_table[idx].to_pandas()
            contrast_table[idx].insert(
                loc=0,
                column="wavelength_index",
                value=wavelength_index_column.astype("int"),
            )
        contrast_table = pd.concat(contrast_table)
        detection_products["contrast_table"] = contrast_table

        if save:
            if file_paths is None:
                file_paths = self.file_paths

            fits.writeto(
                file_paths["norm_detection_image_path"],
                detection_products["normalized_detection_cube"],
                overwrite=overwrite,
            )
            fits.writeto(
                file_paths["uncertainty_image_path"],
                detection_products["uncertainty_cube"],
                overwrite=overwrite,
            )
            fits.writeto(
                file_paths["median_uncertainty_image_path"],
                detection_products["median_uncertainty_cube"],
                overwrite=overwrite,
            )
            contrast_table.to_csv(os.path.splitext(file_paths["contrast_table_path"])[0] + ".csv", index=False)


            save_object(detection_products["contrast_tables"], os.path.splitext(file_paths["contrast_table_path"])[0] + ".obj")

        if inplace:
            self.detection_products = detection_products
        else:
            return detection_products

    def contrast_plot(
        self,
        detection_products=None,
        wavelengths=None,
        add_wavelength_label=True,
        companion_table=None,
        template_fitted=False,
        plot_companions=True,
        curvelabels=None,
        linestyles=None,
        colors=None,
        plot_vertical_lod=True,
        file_paths=None,
        savefig=True,
        figsize=(8, 6),
        show=False,
    ):
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
            linestyles = np.array(["-"]).repeat(len(detection_products))

        if file_paths is None:
            file_paths = self.file_paths
        # colors = plt.cm.viridis(np.linspace(0, 1, len(self.wavelength_indices)))
        if savefig:
            figure_path = file_paths["contrast_plot_path"]
        else:
            figure_path = None

        if companion_table is not None:
            mask = companion_table["wavelength_index"].isin(self.wavelength_indices)
            companion_table_used = companion_table[mask]
        else:
            companion_table_used = None

        fig = plot_contrast_curve(
            detection_products["contrast_tables"],
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
            mirror_axis="mas",
            convert_to_mag=False,
            yscale="log",
            savefig=figure_path,
            figsize=figsize,  # contrast_plot_path[key],
            show=show,
        )

        return fig

    def mask_companions_in_detection(
        self, yx_known_companion_position=None, companion_mask_radius=None
    ):
        yx_dim = (self.detection_cube.shape[-2], self.detection_cube.shape[-1])

        if yx_known_companion_position is None:
            yx_known_companion_position = (
                self.reduction_parameters.yx_known_companion_position
            )

        if companion_mask_radius is None:
            companion_mask_radius = self.reduction_parameters.companion_mask_radius

        if yx_known_companion_position is not None:
            yx_known_companion_position = np.array(yx_known_companion_position)
            if yx_known_companion_position.ndim == 1:
                self.detected_signal_mask = regressor_selection.make_signal_mask(
                    yx_dim,
                    yx_known_companion_position,
                    companion_mask_radius,
                    relative_pos=True,
                    yx_center=None,
                )
            elif yx_known_companion_position.ndim == 2:
                detected_signal_masks = []
                for yx_pos in yx_known_companion_position:
                    detected_signal_masks.append(
                        regressor_selection.make_signal_mask(
                            yx_dim,
                            yx_pos,
                            companion_mask_radius,
                            relative_pos=True,
                            yx_center=None,
                        )
                    )
                self.detected_signal_mask = np.logical_or.reduce(detected_signal_masks)
            else:
                raise ValueError(
                    "Dimensionality of known companion positions for contrast curve too large."
                )
        else:
            self.detected_signal_mask = np.zeros([yx_dim[0], yx_dim[1]]).astype("bool")

    def make_spectral_correlation_matrices(
        self, radial_bounds=None, bin_width=3, yx_center=None, detected_signal_mask=None
    ):
        yx_dim = [self.detection_cube.shape[-2], self.detection_cube.shape[-1]]
        separations_used = []
        empirical_correlation_matrices = []
        empirical_correlation = {}

        self.detection_cube[self.detection_cube == 0.0] = np.nan

        if detected_signal_mask is None:
            self.mask_companions_in_detection()
            detected_signal_mask = self.detected_signal_mask

        if radial_bounds is None:
            separation_max = self.detection_cube.shape[-1] // 2 * np.sqrt(2)
            radial_bounds = [1, int(separation_max)]

        if yx_center is None:
            yx_center = (
                self.detection_cube.shape[-2] // 2.0,
                self.detection_cube.shape[-1] // 2.0,
            )
        xy_center = yx_center[::-1]

        # Determine first non-zero separation, to prevent results below IWA
        # inner_bound_index = int(yx_center[0] + radial_bounds[0])
        # try:
        #     non_zero_separation = (
        #         radial_bounds[0]
        #         + np.nanmax(
        #             np.argwhere(
        #                 np.isnan(
        #                     self.detection_cube[0, 0][
        #                         inner_bound_index : inner_bound_index + 15,
        #                         int(yx_center[1]),
        #                     ]
        #                 )
        #             )
        #         )
        #         + 1
        #     )
        # except ValueError:
        #     non_zero_separation = 0
        # if non_zero_separation > radial_bounds[0] + 13:
        #     non_zero_separation = 0

        separations = np.arange(radial_bounds[0], radial_bounds[1])

        for _, separation in enumerate(separations):
            # annulus_data = annulus_mask[0].multiply(data)
            # mask = annulus_mask[0].to_image(data.shape) > 0
            r_in = separation - bin_width / 2.0
            r_out = separation + bin_width / 2.0
            if r_in < 0.5:
                r_in = 0.5
            annulus_aperture = CircularAnnulus(xy_center, r_in=r_in, r_out=r_out)
            annulus_mask = annulus_aperture.to_mask(method="center")
            # Make sure only pixels are used for which data exists
            mask = annulus_mask.to_image(yx_dim) > 0
            mask[int(xy_center[1]), int(xy_center[0])] = False

            if detected_signal_mask is None:
                mask_computation_annulus = mask
            else:
                mask_computation_annulus = np.logical_and(mask, ~detected_signal_mask)

            annulus_data_1d = self.detection_cube[:, 0, mask_computation_annulus]
           
            # Check that residuals are present and not everything is masked
            if np.all(np.isfinite(annulus_data_1d)) and (0 not in annulus_data_1d.shape):
                psi_ij = compute_empirical_correlation_matrix(annulus_data_1d)
                empirical_correlation_matrices.append(psi_ij)
                separations_used.append(separation)

        empirical_correlation["separation"] = np.array(separations_used)
        empirical_correlation["matrices"] = np.array(empirical_correlation_matrices)

        self.empirical_correlation = empirical_correlation

    def find_approximate_candidate_positions(
        self,
        snr_image,
        candidate_threshold=4.75,
        mask_radius=15,
        max_candidates=DEFAULT_MAX_CANDIDATES,
        mask_connected_region=True,
        exclusion_radius_snr_scaling=True,
        max_exclusion_radius_factor=DEFAULT_MAX_EXCLUSION_RADIUS_FACTOR,
    ):
        """Iteratively pick significant peaks, masking each source before the next.

        Parameters
        ----------
        snr_image : ndarray
            Normalized detection map.
        candidate_threshold : float
            Significance a pixel must exceed to be considered.
        mask_radius : float
            Base radius of the disk blanked around an accepted peak; the radius
            actually used scales with the peak's SNR unless disabled below.
        max_candidates : int
            Upper bound on the number of peaks returned, highest-SNR first. A
            saturated frame (a bright binary, a badly centred reduction) can
            otherwise yield hundreds, and every one of them costs a full
            re-normalization downstream.
        mask_connected_region : bool
            Also blank the contiguous above-threshold region the peak belongs to.
            Cheap, but on real data a bright source's contamination is a swarm of
            separate blobs with sub-threshold gaps, so this alone is not enough —
            the SNR scaling below is what reaches them.
        exclusion_radius_snr_scaling : bool
            Scale the exclusion radius as ``sqrt(snr / candidate_threshold)``. A
            fixed radius is tuned for a marginal detection, but a 100σ binary
            contaminates a far larger area, and each leftover blob re-enters the
            loop as a spurious candidate. Marginal peaks keep the base radius, so
            genuine close pairs are not merged.
        max_exclusion_radius_factor : float
            Cap on the scaling, as a multiple of ``mask_radius``. Without it a
            very bright source would blank most of the search region.
        """
        snr_image = np.ma.masked_array(snr_image)

        yx_dim = snr_image.shape
        yx_center = (yx_dim[0] // 2.0, yx_dim[1] // 2.0)

        significant_pixel_mask = np.logical_and(
            snr_image.data > candidate_threshold,
            np.isfinite(snr_image.data))

        snr_image.mask = ~significant_pixel_mask

        if mask_connected_region:
            connected_labels, _ = ndimage.label(significant_pixel_mask)
        else:
            connected_labels = None

        candidates = {
            "x": [],
            "y": [],
            "x_relative": [],
            "y_relative": [],
            "separation": [],
            "position_angle": [],
            "snr": [],
        }

        truncated = False
        while not np.all(snr_image.mask):
            if len(candidates["snr"]) >= max_candidates:
                truncated = True
                break

            candidates["snr"].append(snr_image.max())

            highest_value_position = np.unravel_index(
                snr_image.argmax(), snr_image.shape
            )
            candidates["x"].append(highest_value_position[1])
            candidates["y"].append(highest_value_position[0])
            relative_yx = image_coordinates.absolute_yx_to_relative_yx(
                highest_value_position, image_center_yx=yx_center
            )
            candidates["x_relative"].append(relative_yx[1])
            candidates["y_relative"].append(relative_yx[0])
            rhophi = image_coordinates.relative_yx_to_rhophi(relative_yx)
            candidates["separation"].append(rhophi[0])
            candidates["position_angle"].append(rhophi[1])

            candidate_mask = regressor_selection.make_signal_mask(
                snr_image.shape,
                highest_value_position,
                _scaled_exclusion_radius(
                    candidates["snr"][-1],
                    candidate_threshold,
                    mask_radius,
                    max_exclusion_radius_factor,
                    enabled=exclusion_radius_snr_scaling,
                ),
                relative_pos=False,
                yx_center=None,
            )
            if connected_labels is not None:
                peak_label = connected_labels[highest_value_position]
                if peak_label > 0:
                    candidate_mask = np.logical_or(
                        candidate_mask, connected_labels == peak_label
                    )
            snr_image.mask[candidate_mask] = True

        if truncated:
            logger.warning(
                "Candidate search stopped at the %d-candidate limit with "
                "significant pixels remaining. The detection map is likely "
                "dominated by a bright source or a reduction artefact.",
                max_candidates,
            )

        candidates = pd.DataFrame(candidates)
        self.candidates = candidates
        return candidates

    def find_candidates(
        self,
        detection_product_index=0,
        candidate_threshold=3.5,
        iterative_search_exclusion_radius=15,
        detection_products=None,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        if detection_products is None:
            detection_products = self.detection_products

        wavelength_index = self.wavelength_indices[detection_product_index]

        smallest_non_nan_separation_idx = np.nanmin(
            np.argwhere(
                np.isfinite(
                    detection_products["contrast_tables"][detection_product_index][
                        "snr_normalization"
                    ]
                )
            )
        )

        smallest_separation_in_pixel = detection_products["contrast_tables"][
            detection_product_index
        ]["sep (pix)"][smallest_non_nan_separation_idx]

        candidates = self.find_approximate_candidate_positions(
            detection_products["normalized_detection_cube"][detection_product_index],
            candidate_threshold=candidate_threshold,
            mask_radius=iterative_search_exclusion_radius,
            max_candidates=max_candidates,
        )

        # The reduction's own inner bound is not a usable floor on its own: with
        # `search_region_inner_bound=1` the normalization is finite from 1 px, so
        # this guard admitted the coronagraph centre residual as a candidate.
        separation_floor = max(
            float(smallest_separation_in_pixel), float(minimum_candidate_separation)
        )
        mask_too_close = candidates["separation"] < separation_floor
        if mask_too_close.any():
            logger.info(
                "Dropping %d candidate(s) inside %.1f px of the star; the stellar "
                "PSF core is not a detection region.",
                int(mask_too_close.sum()), separation_floor,
            )
        candidates = candidates[~mask_too_close].sort_values(
            "separation", ignore_index=False
        )

        number_of_candidates = len(candidates)
        wavelength_index_column = np.ones(number_of_candidates) * wavelength_index
        wavelength_index_column = wavelength_index_column.astype("int")

        candidates.insert(
            loc=0, column="wavelength_index", value=wavelength_index_column
        )

        self.candidates = candidates

        return candidates

    def fit_candidates(
        self,
        candidates=None,
        detection_cube=None,
        detection_products=None,
        x_stddev=1.43,
        y_stddev=2.63,
        box_size=11,
        mask_deviating=False,
        deviation_threshold=0.1,
        plot=False,
    ):
        """Fits 2D Gaussians to contrast, snr, and normalized snr images for candidate position.
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
            wavelength_index = candidates["wavelength_index"].values[candidate_idx]
            detection_product_index = np.argwhere(
                self.wavelength_indices == wavelength_index
            )[0][0]
            yx_position_relative = candidates[["y_relative", "x_relative"]].values[
                candidate_idx
            ]

            base_positions = self.reduction_parameters.yx_known_companion_position
            if base_positions is not None:
                combined_positions = np.vstack([base_positions, yx_position_relative])
            else:
                combined_positions = np.expand_dims(yx_position_relative, axis=0)

            detection_products = self.contrast_table_and_normalization(
                detection_cube=detection_cube,
                cube_indices=[detection_product_index],
                yx_known_companion_position=combined_positions,
                mask_above_sigma=5.0,
                save=False,
                inplace=False,
            )

            # Seed the fit widths from the instrument PSF at this wavelength
            # instead of the hardcoded IRDIS-H defaults; Fit A is free, so this
            # only sets the initial guess (spec "Files touched", fit_2d_gaussian).
            instrument = getattr(self, "instrument", None)
            if instrument is not None and getattr(instrument, "fwhm", None) is not None:
                fwhm_seed = instrument.fwhm[wavelength_index]
                fwhm_seed = float(getattr(fwhm_seed, "value", fwhm_seed))
                x_stddev_seed = fwhm_seed / 2.355
                y_stddev_seed = fwhm_seed / 2.355
            else:
                x_stddev_seed = x_stddev
                y_stddev_seed = y_stddev

            (
                contrast_image_result,
                snr_image_result,
                norm_snr_image_result,
            ) = fit_planet_parameters(
                detection_image=detection_cube[detection_product_index],
                # uncertainty_image=detection_products['uncertainty_cube'][0],
                normalized_detection_image=detection_products[
                    "normalized_detection_cube"
                ][0],
                contrast_table=detection_products["contrast_tables"][0],
                yx_position=candidates[["y", "x"]].values[candidate_idx],
                x_stddev=x_stddev_seed,
                y_stddev=y_stddev_seed,
                box_size=box_size,
                mask_deviating=mask_deviating,
                deviation_threshold=deviation_threshold,
                plot=plot,
            )

            # Fit A is already a free fit, so the "_free" columns downstream
            # code reads are just the same quantities under the new setup.
            for image_result in (
                contrast_image_result,
                snr_image_result,
                norm_snr_image_result,
            ):
                for col in (
                    "amplitude",
                    "x_fwhm",
                    "y_fwhm",
                    "theta",
                    "good_pixels",
                    "fwhm_area",
                    "good_fraction",
                ):
                    image_result[f"{col}_free"] = image_result[col]

            contrast_image_result.insert(
                loc=0, column="candidate_index", value=np.array([candidate_idx])
            )
            snr_image_result.insert(
                loc=0, column="candidate_index", value=np.array([candidate_idx])
            )
            norm_snr_image_result.insert(
                loc=0, column="candidate_index", value=np.array([candidate_idx])
            )
            contrast_image_result.insert(
                loc=1, column="wavelength_index", value=[wavelength_index]
            )
            snr_image_result.insert(
                loc=1, column="wavelength_index", value=[wavelength_index]
            )
            norm_snr_image_result.insert(
                loc=1, column="wavelength_index", value=[wavelength_index]
            )

            contrast_image_results.append(contrast_image_result)
            snr_image_results.append(snr_image_result)
            norm_snr_image_results.append(norm_snr_image_result)

        candidates_fit = {}
        contrast_image_results = pd.concat(
            contrast_image_results, axis=0, ignore_index=True
        )
        snr_image_results = pd.concat(snr_image_results, axis=0, ignore_index=True)
        norm_snr_image_results = pd.concat(
            norm_snr_image_results, axis=0, ignore_index=True
        )

        candidates_fit["contrast_image"] = contrast_image_results
        candidates_fit["snr_image"] = snr_image_results
        candidates_fit["norm_snr_image"] = norm_snr_image_results

        self.candidates_fit = candidates_fit

        return candidates_fit

    def find_candidates_all_wavelengths(
        self,
        detection_cube=None,
        detection_products=None,
        wavelength_indices=None,
        candidate_threshold=4.0,
        iterative_search_exclusion_radius=15,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        if detection_cube is None:
            detection_cube = self.detection_cube
        if detection_products is None:
            detection_products = self.detection_products
        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices

        candidates = []
        for detection_product_index in tqdm(range(len(wavelength_indices))):
            candidates.append(
                self.find_candidates(
                    detection_product_index=detection_product_index,
                    detection_products=detection_products,
                    candidate_threshold=candidate_threshold,
                    iterative_search_exclusion_radius=iterative_search_exclusion_radius,
                    minimum_candidate_separation=minimum_candidate_separation,
                    max_candidates=max_candidates,
                )
            )

        candidates = pd.concat(candidates, axis=0, ignore_index=True)
        candidates = candidates.sort_values("separation")

        return candidates

    def complete_candidate_table(
        self,
        candidates=None,
        detection_cube=None,
        detection_products=None,
        wavelength_indices=None,
        candidate_threshold=4.75,
        search_radius=15,
        mask_deviating=False,
        independent_channels=False,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        candidate_exclusion_radius=None,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        """
        Consolidate candidate detections with 2D Gaussian fitting and duplicate removal.

        This function takes a list of candidate detections (potentially with duplicates 
        across wavelengths) and consolidates them into a single table with precise 
        fitted parameters. It performs 2D Gaussian fitting on contrast, SNR, and 
        normalized SNR images, removes duplicates within a search radius, and provides 
        position uncertainty estimates through weighted averaging.

        Parameters
        ----------
        candidates : pandas.DataFrame, optional
            Table of candidate detections with columns including 'x', 'y', 
            'x_relative', 'y_relative', 'separation', 'position_angle', 'snr'.
            If None, candidates are automatically found using 
            find_candidates_all_wavelengths().
        detection_cube : ndarray, optional
            Detection cube with shape (n_wavelengths, n_images, n_y, n_x) where 
            n_images typically includes [contrast, uncertainty, snr] maps.
            If None, uses self.detection_cube.
        detection_products : dict, optional
            Dictionary containing detection products including 'contrast_tables' 
            and 'normalized_detection_cube'. If None, uses self.detection_products.
        wavelength_indices : array_like, optional
            Indices of wavelengths to process. If None, uses self.wavelength_indices.
        candidate_threshold : float, optional
            Signal-to-noise ratio threshold for candidate identification.
            Default is 4.0.
        search_radius : float, optional
            Radius in pixels for grouping multiple detections of the same source.
            Candidates within this radius are considered duplicates and consolidated.
            Default is 11.0.
        mask_deviating : bool, optional
            Whether to mask candidates with deviating PSF parameters during 
            2D Gaussian fitting. Default is False.

        Returns
        -------
        candidates : pandas.DataFrame or None
            Consolidated candidate table with duplicate entries removed, sorted by 
            separation. Contains columns from original candidates list.
            Returns None if no candidates found.
        candidates_fit : dict or None
            Dictionary containing fitting results with keys:
            - 'contrast_image' : pandas.DataFrame
                2D Gaussian fit results on contrast images
            - 'snr_image' : pandas.DataFrame  
                2D Gaussian fit results on SNR images (primary position source)
            - 'norm_snr_image' : pandas.DataFrame
                2D Gaussian fit results on normalized SNR images
            Returns None if no candidates found.

        Notes
        -----
        The function performs the following key operations:

        1. **2D Gaussian Fitting**: Fits 2D Gaussians to three different images:
           - Contrast image: Provides actual contrast measurements
           - SNR image: Used for primary position determination (most reliable)
           - Normalized SNR image: Used for detection significance assessment

        2. **Duplicate Removal**: Candidates within search_radius are grouped and 
           consolidated using SNR-weighted averaging of fit parameters.

        3. **Position Refinement**: Initial candidate positions from peak-finding 
           are refined through 2D Gaussian centroiding for sub-pixel accuracy.

        4. **Uncertainty Estimation**: For candidates detected in multiple 
           wavelengths, standard deviations provide position uncertainty estimates.

        The SNR image fit results are used as the primary source for positional 
        information because they provide the best signal-to-noise for accurate 
        centroiding while being less affected by calibration uncertainties.

        Position coordinates are in image pixels with origin at (0,0). Relative 
        coordinates are measured from the image center. Position angles are 
        measured east of north in degrees.

        Examples
        --------
        >>> # Basic usage with automatic candidate finding
        >>> candidates, candidates_fit = analysis.complete_candidate_table(
        ...     candidate_threshold=4.5,
        ...     search_radius=8.0
        ... )
        >>> 
        >>> # Check results
        >>> if candidates is not None:
        ...     print(f"Found {len(candidates)} unique candidates")
        ...     print(f"Position precision: {candidates_fit['snr_image']['separation_sigma'].mean():.2f} pixels")
        """
 
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
                candidate_threshold=candidate_threshold,
                iterative_search_exclusion_radius=_resolve_exclusion_radius(
                    candidate_exclusion_radius, search_radius
                ),
                minimum_candidate_separation=minimum_candidate_separation,
                max_candidates=max_candidates,
            )

        if len(candidates) == 0:
            return None, None

        # NOTE: This fits all signals above threshold. Can be a lot for multiple cadidates
        # detected in multiple wavelengths.
        candidates_fit = self.fit_candidates(
            candidates=candidates,
            detection_cube=detection_cube,
            detection_products=detection_products,
            plot=False,
        )
        
        # Group per-wavelength detections of the same source (within
        # search_radius) and combine them in the source-aligned (r, t) frame
        # per Section 4 of the astrometry-uncertainty spec. Positional indexing
        # throughout so a non-range candidate index never aliases a position.
        snr_table = candidates_fit["snr_image"].reset_index(drop=True)
        norm_amp = (
            candidates_fit["norm_snr_image"]["amplitude"]
            .reset_index(drop=True)
            .values.astype(float)
        )
        positions = snr_table[["x_relative", "y_relative"]].values.astype(float)

        assigned = np.zeros(len(snr_table), dtype=bool)
        unique_candidate_indices = []
        combined_rows = []
        for idx in range(len(snr_table)):
            if assigned[idx]:
                continue
            dist = linalg.norm(positions[idx] - positions, axis=1)
            group_mask = (dist < search_radius) & (~assigned)
            group_positions = np.where(group_mask)[0]
            group_rows = snr_table.iloc[group_positions]
            group_snr = norm_amp[group_positions]

            combined = _combine_channels_rt_frame(
                group_rows,
                search_radius=search_radius,
                snr_values=group_snr,
                independent_channels=independent_channels,
            )
            # Keep the highest-SNR channel as the surviving row in the sibling
            # (contrast / norm-SNR) tables.
            keeper = int(group_positions[int(np.nanargmax(group_snr))])
            unique_candidate_indices.append(keeper)
            assigned[group_positions] = True
            combined_rows.append(combined)

        final_position_table = pd.concat(combined_rows, ignore_index=True)

        # Order all tables consistently by the combined separation.
        order = np.argsort(final_position_table["separation"].values, kind="stable")
        final_position_table = final_position_table.iloc[order].reset_index(drop=True)
        kept = [unique_candidate_indices[i] for i in order]

        candidates = candidates.iloc[kept].reset_index(drop=True)
        candidates_fit["contrast_image"] = (
            candidates_fit["contrast_image"].iloc[kept].reset_index(drop=True)
        )
        candidates_fit["norm_snr_image"] = (
            candidates_fit["norm_snr_image"].iloc[kept].reset_index(drop=True)
        )
        candidates_fit["snr_image"] = final_position_table

        self.candidates = candidates
        self.candidates_fit = candidates_fit

        return candidates, candidates_fit

    def rereduce_single_position(
        self,
        candidate_index,
        yx_candidate_position,
        data_full,
        flux_psf_full,
        pa,
        wavelength_indices,
        temporal_components_fraction,
        inverse_variance_full=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        return_all_results=False,
        verbose=False,
    ):
        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices

        re_reduction_parameters = self.reduction_parameters.merge(
            guess_position=yx_candidate_position,
            use_multiprocess=False,
            reduce_single_position=True,
            data_auto_crop=True,
            yx_known_companion_position=None,
            remove_known_companions=False,
        )
        detection_products_orig = copy.deepcopy(self.detection_products)
        all_results = run_complete_reduction(
            data_full=data_full.copy(),
            flux_psf_full=flux_psf_full.copy(),
            pa=pa,
            instrument=self.instrument,
            reduction_parameters=re_reduction_parameters,
            temporal_components_fraction=temporal_components_fraction,
            wavelength_indices=wavelength_indices,
            inverse_variance_full=inverse_variance_full,
            bad_frames=bad_frames,
            bad_pixel_mask_full=bad_pixel_mask_full,
            xy_image_centers=xy_image_centers,
            amplitude_modulation_full=amplitude_modulation_full,
            verbose=verbose,
        )
        
        if return_all_results:
            return all_results

        # AUTOMATICALLY COLLECT ALL WAVELENGTHS FOR REDUCTION
        # NOTE: This should be generalized to allow automatically collect results from
        # using various component fractions
        contrast = []
        uncertainty = []
        try:
            _ = iter(temporal_components_fraction)
        except TypeError:
            temporal_components_fraction = [temporal_components_fraction]

        component_key = str(temporal_components_fraction[0])
        for key in all_results[component_key]:
            contrast.append(
                all_results[component_key][key][self.reduction_type].measured_contrast
            )
            uncertainty.append(
                all_results[component_key][key][
                    self.reduction_type
                ].contrast_uncertainty
            )

        contrast = np.array(contrast)
        uncertainty = np.array(uncertainty)

        # REDO NORMALIZATION WITHOUT CANDIDATES
        if hasattr(self, 'candidates'):
            base_positions = self.reduction_parameters.yx_known_companion_position
            if base_positions is not None:
                combined_positions = np.vstack([
                    base_positions,
                    self.candidates[["y_relative", "x_relative"]].values,
                ])
            else:
                combined_positions = self.candidates[["y_relative", "x_relative"]].values
            self.contrast_table_and_normalization(
                save=False, inplace=True, yx_known_companion_position=combined_positions
            )
        else:
            self.contrast_table_and_normalization(save=False, inplace=True)

        normalization_factors = []
        for contrast_table_index in range(len(wavelength_indices)):
            mask = np.isfinite(
                self.detection_products["contrast_tables"][contrast_table_index][
                    "snr_normalization"
                ]
            )
            separation = self.detection_products["contrast_tables"][
                contrast_table_index
            ]["sep (pix)"][mask]
            norm_factors = self.detection_products["contrast_tables"][
                contrast_table_index
            ]["snr_normalization"][mask]

            if hasattr(self, 'candidates_fit'):
                normalization_factors.append(
                    interp(x=self.candidates_fit["snr_image"]["separation"][candidate_index],
                           xp=separation,
                           fp=norm_factors)
                )
            else:
                separation_candidate = np.sqrt(yx_candidate_position[0] ** 2 + yx_candidate_position[1] ** 2)
                normalization_factors.append(
                    interp(x=separation_candidate,
                           xp=separation,
                           fp=norm_factors)
                )

        normalization_factors = np.array(normalization_factors)
        normalized_uncertainty = uncertainty * normalization_factors
        snr = contrast / normalized_uncertainty

        # Table entries
        candidate_id = np.ones(len(self.instrument.wavelengths)) * candidate_index
        contrast_for_table = np.empty_like(candidate_id)
        contrast_for_table[:] = np.nan
        normalized_uncertainty_for_table = np.empty_like(candidate_id)
        normalized_uncertainty_for_table[:] = np.nan
        snr_for_table = np.empty_like(candidate_id)
        snr_for_table[:] = np.nan
        uncertainty_for_table = np.empty_like(candidate_id)
        uncertainty_for_table[:] = np.nan
        norm_factor_for_table = np.empty_like(candidate_id)
        norm_factor_for_table[:] = np.nan

        contrast_for_table[wavelength_indices] = contrast
        normalized_uncertainty_for_table[wavelength_indices] = normalized_uncertainty
        snr_for_table[wavelength_indices] = snr
        uncertainty_for_table[wavelength_indices] = uncertainty
        norm_factor_for_table[wavelength_indices] = normalization_factors

        wavelengths = np.zeros(self.instrument.wavelengths.shape)
        wavelengths[:] = self.instrument.wavelengths.value

        candidate_spectrum = {
            "candidate_id": candidate_id.astype("int"),
            "wavelength_index": np.arange(len(self.instrument.wavelengths)),
            "wavelength": wavelengths,
            "contrast": contrast_for_table,
            "uncertainty": normalized_uncertainty_for_table,
            "snr": snr_for_table,
            "original_unc": uncertainty_for_table,
            "norm_factor": norm_factor_for_table,
        }

        candidate_spectrum = pd.DataFrame(candidate_spectrum).sort_values("wavelength")

        self.detection_products = detection_products_orig

        return candidate_spectrum

    def extract_candidate_spectra(
        self,
        temporal_components_fraction,
        data_full,
        flux_psf_full,
        pa,
        yx_candidate_positions=None,
        wavelength_indices=None,
        inverse_variance_full=None,
        instrument=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        return_spectra=False,
    ):
        """
        Extract high-fidelity spectra for candidate companions using full reduction.

        This function performs dedicated spectral extraction for each candidate
        by running the complete TRAP reduction pipeline at the candidate positions.
        This provides the most accurate contrast measurements and uncertainties
        for spectral characterization, free from the approximations used in the
        initial detection phase.

        Parameters
        ----------
        temporal_components_fraction : float
            Fraction of temporal components to retain during PCA reduction.
            Typical values range from 0.1 to 0.5, with smaller values providing
            more aggressive speckle suppression but potentially removing real signals.
        data_full : ndarray
            Full data cube with shape (n_frames, n_wavelengths, n_y, n_x) containing
            the raw observations for spectral extraction.
        flux_psf_full : ndarray
            Full PSF flux cube with shape matching data_full, containing the
            normalized PSF template for each frame and wavelength.
        pa : array_like
            Position angles in degrees for each frame, with shape (n_frames,).
            Used for field rotation tracking during reduction.
        yx_candidate_positions : array_like, optional
            Candidate positions in image coordinates with shape (n_candidates, 2).
            Each row contains [y_relative, x_relative] coordinates from image center.
            If None, uses positions from self.candidates_fit['snr_image'].
        wavelength_indices : array_like, optional
            Indices of wavelengths to process. If None, processes all wavelengths
            in the instrument specification.
        inverse_variance_full : ndarray, optional
            Inverse variance weights with shape matching data_full for optimal
            extraction. If None, uses uniform weighting.
        instrument : Instrument, optional
            Instrument specification containing wavelength and detector parameters.
            If None, uses self.instrument.
        bad_frames : array_like, optional
            Boolean array or indices of frames to exclude from extraction.
        bad_pixel_mask_full : ndarray, optional
            Bad pixel mask with shape matching data_full.
        xy_image_centers : ndarray, optional
            Image center coordinates for each frame with shape (n_frames, 2).
            Used for coordinate transformations.
        amplitude_modulation_full : ndarray, optional
            Amplitude modulation factors accounting for instrumental effects.
        return_spectra : bool, optional
            Whether to return the extracted spectra. Default is False.

        Returns
        -------
        candidate_spectra : pandas.DataFrame or None
            Combined spectral table for all candidates with columns:
            - 'candidate_id' : int, identifier for each candidate
            - 'wavelength_index' : int, index in instrument wavelength array
            - 'wavelength' : float, wavelength in micrometers
            - 'contrast' : float, companion-to-star contrast ratio
            - 'uncertainty' : float, normalized uncertainty (contrast units)
            - 'snr' : float, signal-to-noise ratio (contrast/uncertainty)
            - 'original_unc' : float, original uncertainty before normalization
            - 'norm_factor' : float, normalization factor applied
            Returns None if no candidate positions provided or if return_spectra=False.

        Notes
        -----
        This function performs the following operations for each candidate:

        1. **Individual Reduction**: Runs the complete TRAP reduction pipeline
           at each candidate position with optimized parameters for single-source
           extraction.

        2. **Contrast Measurement**: Measures the companion contrast at each
           wavelength using the same algorithms as the main reduction.

        3. **Uncertainty Estimation**: Calculates realistic uncertainties
           accounting for noise correlation and systematic effects.

        4. **Normalization**: Applies the same normalization factors used in
           the detection phase for consistent SNR calculations.

        The extraction uses a dedicated reduction configuration that:
        - Disables multiprocessing for deterministic results
        - Enables automatic data cropping around the candidate position
        - Excludes other known companions to avoid contamination
        - Uses the candidate position as an initial guess for optimization

        The resulting spectra provide the most accurate characterization possible
        with the given data and should be used for scientific analysis rather
        than the approximate values from the detection phase.

        Position coordinates are in image pixels with origin at (0,0). Relative
        coordinates are measured from the image center.

        Examples
        --------
        >>> # Extract spectra for validated candidates
        >>> spectra = analysis.extract_candidate_spectra(
        ...     temporal_components_fraction=0.2,
        ...     data_full=data_cube,
        ...     flux_psf_full=psf_cube,
        ...     pa=position_angles,
        ...     return_spectra=True
        ... )
        >>> 
        >>> # Analyze spectral properties
        >>> if spectra is not None:
        ...     for cand_id in spectra['candidate_id'].unique():
        ...         cand_spec = spectra[spectra['candidate_id'] == cand_id]
        ...         median_snr = cand_spec['snr'].median()
        ...         print(f"Candidate {cand_id}: median SNR = {median_snr:.1f}")
        """
        """
        Extracts candidate spectra from the given data.

        Parameters:
        - temporal_components_fraction (float): Fraction of temporal components to use.
        - data_full (array-like): Full data array.
        - flux_psf_full (array-like): Full flux PSF array.
        - pa (float): Position angle.
        - yx_candidate_positions (array-like, optional): Positions of the candidates. If not provided, uses the positions from self.candidates_fit.
        - wavelength_indices (array-like, optional): Indices of the wavelengths to consider.
        - inverse_variance_full (array-like, optional): Full inverse variance array.
        - instrument (str, optional): Instrument name.
        - bad_frames (array-like, optional): Indices of bad frames.
        - bad_pixel_mask_full (array-like, optional): Full bad pixel mask array.
        - xy_image_centers (array-like, optional): XY image centers.
        - amplitude_modulation_full (array-like, optional): Full amplitude modulation array.
        - return_spectra (bool, optional): Whether to return the extracted spectra.

        Returns:
        - candidate_spectra (DataFrame): Extracted candidate spectra.

        """
        candidate_spectra = []

        if yx_candidate_positions is None:
            yx_candidate_positions = self.candidates_fit["snr_image"][
                ["y_relative", "x_relative"]
            ].values
        else:
            yx_candidate_positions = np.array(yx_candidate_positions)

        if len(yx_candidate_positions) == 0 or yx_candidate_positions is None:
            return None

        for candidate_index, yx_candidate_position in tqdm(enumerate(yx_candidate_positions)):
            logger.info("Running TRAP at candidate position: %s", yx_candidate_position)
            candidate_spectrum = self.rereduce_single_position(
                candidate_index=candidate_index,
                yx_candidate_position=yx_candidate_position,
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                temporal_components_fraction=temporal_components_fraction,
                wavelength_indices=wavelength_indices,
                inverse_variance_full=inverse_variance_full,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full,
                xy_image_centers=xy_image_centers,
                amplitude_modulation_full=amplitude_modulation_full,
                verbose=False,
            )
            candidate_spectra.append(candidate_spectrum)

        candidate_spectra = pd.concat(candidate_spectra, axis=0, ignore_index=False)
        self.candidate_spectra = candidate_spectra

        if return_spectra:
            return candidate_spectra

    def detection_summary(
        self,
        candidates,
        candidates_fit,
        candidate_spectra=None,
        use_spectra=True,
        template_name=None,
        snr_threshold=4.5,
        snr_threshold_spectrum=True,
        good_fraction_threshold=0.05,
        theta_deviation_threshold=25.0,
        yx_fwhm_ratio_threshold=[1.1, 4.5],
    ):
        """
        Create companion tables with validation criteria and spectral information.

        This function consolidates candidate detection results into comprehensive 
        tables, applies validation criteria to identify reliable detections, and 
        optionally incorporates spectral characterization data. It produces both 
        a complete companion table and a validated subset meeting quality criteria.

        Parameters
        ----------
        candidates : pandas.DataFrame
            Table of candidate detections with columns including 'snr' (peak pixel 
            signal-to-noise ratio) and positional information.
        candidates_fit : dict
            Dictionary containing 2D Gaussian fit results with keys:
            - 'snr_image' : pandas.DataFrame with fitted positions and PSF parameters
            - 'norm_snr_image' : pandas.DataFrame with normalized SNR fit amplitudes
        candidate_spectra : pandas.DataFrame, optional
            Spectral characterization results for each candidate with columns 
            including 'candidate_id', 'wavelength', 'contrast', 'uncertainty', 'snr'.
            If None and use_spectra=True, uses self.candidate_spectra.
        use_spectra : bool, optional
            Whether to incorporate spectral information into the analysis.
            Default is True.
        template_name : str, optional
            Name of the template used for detection, added as a column to the 
            companion table. Default is None.
        snr_threshold : float, optional
            Minimum signal-to-noise ratio required for validation. Default is 4.5.
        snr_threshold_spectrum : bool, optional
            If True and use_spectra=True, uses spectral SNR for thresholding.
            If False, uses the maximum of fitted SNR and peak pixel SNR.
            Default is True.
        good_fraction_threshold : float, optional
            Minimum fraction of good pixels required in 2D Gaussian fit.
            Default is 0.05.
        theta_deviation_threshold : float, optional
            Maximum allowed deviation in degrees between expected position angle
            (from separation vector) and fitted PSF position angle. Default is 25.0.
        yx_fwhm_ratio_threshold : list of float, optional
            Allowed range [min, max] for the ratio of y-axis to x-axis FWHM
            in unconstrained 2D Gaussian fits. Default is [1.1, 4.5].

        Returns
        -------
        companion_table : pandas.DataFrame
            Complete table of all candidates with columns including:
            - Position: 'x', 'y', 'x_relative', 'y_relative', 'separation', 'position_angle'
            - Uncertainties: 'x_relative_sigma', 'y_relative_sigma', 'separation_sigma', 'position_angle_sigma'
            - PSF parameters: 'x_fwhm', 'y_fwhm', 'theta_free', 'yx_fwhm_ratio'
            - Detection metrics: 'norm_snr_fit', 'norm_snr_fit_free', 'peak_pixel_snr'
            - Validation metrics: 'theta_deviation', 'good_fraction_free'
            - Spectral data: 'wavelength', 'contrast', 'uncertainty', 'snr' (if use_spectra=True)
        validated_companion_table : pandas.DataFrame
            Subset of companion_table containing only candidates that pass all
            validation criteria defined by the threshold parameters.

        Notes
        -----
        The function performs the following validation steps:

        1. **SNR Validation**: Candidates must exceed snr_threshold using either
           spectral SNR (if available) or the maximum of fitted and peak pixel SNR.

        2. **PSF Quality**: The good_fraction_free must exceed the threshold,
           indicating sufficient good pixels for reliable 2D Gaussian fitting.

        3. **Position Angle Consistency**: The deviation between expected position
           angle (from separation vector) and fitted PSF angle must be within limits.

        4. **PSF Shape Validation**: The ratio of y-axis to x-axis FWHM must be
           within physical limits to exclude artifacts and badly fitted sources.

        The companion table includes multiple SNR measurements:
        - 'peak_pixel_snr': Direct pixel value from initial detection
        - 'norm_snr_fit': Amplitude from constrained 2D Gaussian fit on normalized SNR
        - 'norm_snr_fit_free': Amplitude from unconstrained fit (primary validation metric)

        Position coordinates are in image pixels with origin at (0,0). Relative
        coordinates are measured from the image center. Position angles are 
        measured east of north in degrees.

        Examples
        --------
        >>> # Basic validation with default criteria
        >>> companion_table, validated_table = analysis.detection_summary(
        ...     candidates=candidates,
        ...     candidates_fit=candidates_fit,
        ...     candidate_spectra=spectra,
        ...     snr_threshold=5.0,
        ...     template_name='T-type'
        ... )
        >>> 
        >>> # Check validation results
        >>> print(f"Total candidates: {len(companion_table)}")
        >>> print(f"Validated candidates: {len(validated_table)}")
        >>> print(f"Validation rate: {len(validated_table)/len(companion_table)*100:.1f}%")
        """
        if candidates is None:
            candidates = self.candidates

        if candidates_fit is None:
            candidates_fit = self.candidates_fit

        if candidate_spectra is None and use_spectra:
            candidate_spectra = self.candidate_spectra

        companion_table = candidates_fit["snr_image"][
            [
                "x",
                "y",
                "x_relative",
                "x_relative_sigma",
                "y_relative",
                "y_relative_sigma",
                "separation",
                "separation_sigma",
                "position_angle",
                "position_angle_sigma",
                "xy_relative_corr",
                "radial_sigma_stat",
                "tangential_sigma_stat",
                "channels_above_threshold",
                "theta_free",
                "x_fwhm",
                "y_fwhm",
                "fwhm_area",
                "x_fwhm_free",
                "y_fwhm_free",
                "good_fraction",
                "good_fraction_free",
            ]
        ]

        companion_table.insert(
            loc=16,
            column="theta_deviation",
            value=subtract_angles(
                companion_table["position_angle"], companion_table["theta_free"]
            ),
        )
        companion_table.insert(
            loc=17,
            column="yx_fwhm_ratio",
            value=companion_table["y_fwhm_free"] / companion_table["x_fwhm_free"],
        )
        companion_table.insert(
            loc=18,
            column="fwhm_area_free",
            value=np.pi
            * companion_table["x_fwhm_free"]
            * companion_table["y_fwhm_free"],
        )
        companion_table.insert(
            loc=19,
            column="norm_snr_fit",
            value=candidates_fit["norm_snr_image"]["amplitude"],
        )
        companion_table.insert(
            loc=20,
            column="norm_snr_fit_free",
            value=candidates_fit["norm_snr_image"]["amplitude_free"],
        )
        companion_table.insert(loc=21, column="peak_pixel_snr", value=candidates["snr"])

        if template_name is not None:
            companion_table.insert(
                loc=22,
                column="template_name",
                value=np.array([template_name]).repeat(len(companion_table)),
            )

        if use_spectra:
            companion_table = pd.merge(
                companion_table,
                candidate_spectra,
                left_index=True,
                right_on="candidate_id",
                how="left",
            )

        if snr_threshold_spectrum and use_spectra:
            snr = companion_table["snr"]
        else:
            snr = np.nanmax(
                [
                    companion_table["norm_snr_fit_free"],
                    companion_table["peak_pixel_snr"],
                ],
                axis=0,
            )

        mask = (
            (snr > snr_threshold)
            & (companion_table["good_fraction_free"] > good_fraction_threshold)
            & (np.abs(companion_table["theta_deviation"]) < theta_deviation_threshold)
            & (companion_table["yx_fwhm_ratio"] > yx_fwhm_ratio_threshold[0])
            & (companion_table["yx_fwhm_ratio"] < yx_fwhm_ratio_threshold[1])
        )

        if use_spectra:
            unique_candidates = np.unique(companion_table[mask]["candidate_id"].values)
            validated_companion_table = companion_table[
                companion_table["candidate_id"].isin(unique_candidates)
            ]
        else:
            validated_companion_table = companion_table[mask]

        self.companion_table = companion_table
        self.validated_companion_table = validated_companion_table

        return companion_table, validated_companion_table

    def add_templates(self, template):
        """
        Add a SpectralTemplate object to the templates dictionary.

        Args:
            template (SpectralTemplate): The SpectralTemplate object to be added.

        Returns:
            None
        """
        self.templates[template.name] = template

    def add_default_templates(
        self,
        stellar_modelbox,
        species_database_directory,
        stellar_parameters=None,
        instrument=None,
        correct_transmission=False,
        use_spectral_correlation=True,
    ):
        """
        Add default templates to the template collection.

        Parameters:
            stellar_modelbox (ModelBox): The stellar model box.
            species_database_directory (str): The directory path for the species database.
            stellar_parameters (dict, optional): The stellar parameters. Defaults to None.
            instrument (Instrument, optional): The instrument. Defaults to None.
            correct_transmission (bool, optional): Flag indicating whether to correct for transmission. Defaults to False.
            use_spectral_correlation (bool, optional): Flag indicating whether to use spectral correlation. Defaults to True.
        """
        
        if species_database_directory is None:
            ValueError("Need to specify species database directory.")
        
        if not os.path.exists(species_database_directory):
            os.makedirs(species_database_directory)
            os.chdir(species_database_directory)
            SpeciesInit()

        os.chdir(species_database_directory)
        
        try:
            database = Database()
        except:
            logger.warning("No initialized species database found in: %s", species_database_directory)
            SpeciesInit()
            database = Database()

        if instrument is None:
            instrument = self.instrument

        database.add_model(model="petitcode-cool-cloudy", teff_range=(700.0, 800.0))
        cool_planet_read_model = ReadModel(
            model="petitcode-cool-cloudy", wavel_range=(0.85, 3.6)
        )
        cool_planet_model_param = {
            "teff": 760.0,
            "logg": 4.26,
            "feh": 1.0,
            "fsed": 1.26,
            "radius": 1.1,
            "distance": 30.0,
        }

        database.add_model(model="drift-phoenix", teff_range=(1400.0, 1600.0))
        hot_planet_read_model = ReadModel(
            model="drift-phoenix", wavel_range=(0.85, 3.6)
        )
        hot_planet_model_param = {
            "teff": 1500.0,
            "logg": 4.0,
            "feh": 0.0,
            "radius": 1.1,
            "distance": 30.0,
        }

        cool_planet_modelbox = cool_planet_read_model.get_model(
            model_param=cool_planet_model_param
        )
        hot_planet_modelbox = hot_planet_read_model.get_model(
            model_param=hot_planet_model_param
        )

        flat_model = copy.deepcopy(cool_planet_modelbox)
        flat_model.flux = np.ones_like(flat_model.wavelength)

        if stellar_modelbox is None:
            if stellar_parameters is None:
                stellar_modelbox = copy.deepcopy(flat_model)
            else:
                database.add_model(model="bt-nextgen", teff_range=(3000.0, 30000.0))
                star_read_model = ReadModel(
                    model="bt-nextgen", wavel_range=(0.85, 3.6)
                )

                # Snap stellar parameters to the model-grid boundaries so values
                # outside the grid (e.g. sub-solar [Fe/H] on this solar-only grid,
                # or a log g past the grid edge) clamp to the nearest boundary
                # instead of raising in species' get_model.
                stellar_parameters = dict(stellar_parameters)
                for param, (low, high) in star_read_model.get_bounds().items():
                    value = stellar_parameters.get(param)
                    if value is None:
                        continue
                    clamped = min(max(value, low), high)
                    if clamped != value:
                        warnings.warn(
                            f"Stellar parameter '{param}'={value:.4g} is outside the "
                            f"bt-nextgen grid [{low:.4g}, {high:.4g}]; clamping to "
                            f"{clamped:.4g} for template matching."
                        )
                        stellar_parameters[param] = clamped

                stellar_modelbox = star_read_model.get_model(
                    model_param=stellar_parameters
                )

        if (
            instrument.instrument_type == "photometry"
            or len(instrument.wavelengths) <= 2
        ):
            t_type_slope_fit = False
        else:
            t_type_slope_fit = True

        self.templates["L-type"] = SpectralTemplate(
            name="L-type",
            instrument=instrument,
            companion_modelbox=hot_planet_modelbox,
            stellar_modelbox=stellar_modelbox,
            wavelength_indices=self.wavelength_indices,
            correct_transmission=correct_transmission,
            fit_offset=False,
            fit_slope=False,
            number_of_pca_regressors=0,
            use_spectral_correlation=use_spectral_correlation,
            species_database_directory=species_database_directory,
        )

        self.templates["T-type"] = SpectralTemplate(
            name="T-type",
            instrument=instrument,
            companion_modelbox=cool_planet_modelbox,
            stellar_modelbox=stellar_modelbox,
            wavelength_indices=self.wavelength_indices,
            correct_transmission=correct_transmission,
            fit_offset=True,
            fit_slope=t_type_slope_fit,
            number_of_pca_regressors=0,
            use_spectral_correlation=use_spectral_correlation,
            species_database_directory=species_database_directory,
        )

        self.templates["flat"] = SpectralTemplate(
            name="flat",
            instrument=instrument,
            companion_modelbox=flat_model,
            stellar_modelbox=flat_model,
            wavelength_indices=self.wavelength_indices,
            correct_transmission=correct_transmission,
            fit_offset=False,
            fit_slope=False,
            number_of_pca_regressors=0,
            use_spectral_correlation=use_spectral_correlation,
            species_database_directory=species_database_directory,
        )

    def template_matching_detection(
        self,
        template,
        inner_mask_radius=1.0,
        detection_threshold=5.0,
        file_paths=None,
        save=True,
    ):
        """
        Perform pixel-by-pixel spectral template matching for companion detection.

        This function implements the core template matching algorithm that fits 
        spectral templates to each pixel in the detection cube using generalized 
        least squares with optional spectral correlation. The method creates a 
        template-matched detection map with contrast, uncertainty, and SNR images.

        Parameters
        ----------
        template : SpectralTemplate
            Spectral template object containing the companion model spectrum,
            stellar model spectrum, and fitting configuration parameters including:
            - fit_offset : bool, whether to fit a constant offset
            - fit_slope : bool, whether to fit a linear slope in wavelength
            - use_spectral_correlation : bool, whether to use correlated noise model
            - number_of_pca_regressors : int, number of PCA components for regression
        inner_mask_radius : float, optional
            Radius in pixels to mask around the central star position during 
            template matching. Default is 1.0.
        detection_threshold : float, optional
            Signal-to-noise ratio threshold used for contrast curve normalization.
            Default is 5.0.
        file_paths : dict, optional
            Dictionary specifying output file paths for detection products.
            Expected keys include 'norm_detection_image_path', 'uncertainty_image_path',
            'contrast_table_path', 'contrast_plot_path'. If None, default paths 
            are generated.
        save : bool, optional
            Whether to save detection products to disk. Default is True.

        Returns
        -------
        detection_cube : ndarray
            Template-matched detection cube with shape (1, 3, n_y, n_x) containing:
            - [0, 0, :, :] : Contrast image (fitted amplitudes)
            - [0, 1, :, :] : Uncertainty image (fit uncertainties)  
            - [0, 2, :, :] : SNR image (contrast/uncertainty)
        detection_products : dict
            Dictionary containing detection analysis products:
            - 'contrast_tables' : list of pandas.DataFrame
                Radial contrast curves and normalization factors
            - 'normalized_detection_cube' : list of ndarray
                Normalized detection maps for candidate identification

        Notes
        -----
        The template matching process performs the following steps:

        1. **Spectral Correlation**: Computes empirical spectral correlation 
           matrices as a function of separation to model correlated noise.

        2. **Model Construction**: For each pixel, builds a design matrix including:
           - Constant offset (if template.fit_offset=True)
           - Linear slope in wavelength (if template.fit_slope=True)  
           - Spectral template (normalized companion model)
           - PCA regressors (if template.number_of_pca_regressors > 0)

        3. **Generalized Least Squares**: Solves the linear system using either:
           - Correlated noise model with empirical covariance matrix
           - Simple diagonal covariance matrix for independent noise

        4. **Output Generation**: Creates template-matched images with fitted 
           contrast amplitudes, uncertainties, and signal-to-noise ratios.

        The algorithm assumes that the detection cube contains contrast measurements
        with associated uncertainty estimates. Zero or NaN values are masked during
        processing to avoid numerical issues.

        Coordinates are in image pixels with origin at (0,0). The central star is
        assumed to be at the image center for masking purposes.

        Examples
        --------
        >>> # Basic template matching
        >>> detection_cube, detection_products = analysis.template_matching_detection(
        ...     template=my_template,
        ...     inner_mask_radius=2.0,
        ...     detection_threshold=5.0
        ... )
        >>> 
        >>> # Check detection map quality
        >>> snr_map = detection_cube[0, 2, :, :]
        >>> max_snr = np.nanmax(snr_map)
        >>> print(f"Peak SNR in template-matched map: {max_snr:.1f}")
        """
        template_name = template.name

        # wavelengths = np.zeros(self.instrument.wavelengths.shape)
        wavelengths = self.instrument.wavelengths[self.wavelength_indices]

        contrast_cube = self.detection_cube[:, 0].astype("float64").copy()
        uncertainty_cube = self.detection_products["uncertainty_cube"].astype(
            "float64"
        ).copy()  # / detection1.reduction_parameters.contrast_curve_sigma
        
        contrast_cube[contrast_cube == 0.0] = np.nan
        uncertainty_cube[uncertainty_cube == 0.0] = np.nan
        
        yx_dim = [contrast_cube.shape[-2], contrast_cube.shape[-1]]
        yx_center_output = [yx_dim[0] // 2, yx_dim[1]]

        reduced_positions_mask = np.logical_and(
            np.all(np.isfinite(contrast_cube), axis=0),
            ~np.any(contrast_cube == 0.0, axis=0),
        )
        # self.reduction_parameters.annulus_width = 3

        center_mask = regressor_selection.make_signal_mask(
            yx_dim, (0, 0), inner_mask_radius, relative_pos=True, yx_center=None
        )

        reduced_positions_mask = np.logical_and(reduced_positions_mask, ~center_mask)

        position_indices = np.argwhere(reduced_positions_mask)

        template_matched_image = np.zeros((3, yx_dim[0], yx_dim[1]))

        # median_contrast = bn.nanmedian(contrast_cube, axis=0)
        # regressors_centered = contrast_cube - median_contrast

        # position_indices = [[32, 77]]
        # number_of_pca_regressors = int(np.round(38 * 0.1))
        # wavelength_indices = self.wavelength_indices
        self.make_spectral_correlation_matrices()

        for _, yx_pixel in tqdm(enumerate(position_indices)):
            # wavelength indices are not applicable to contrast cube when not all wavelengths are reduced
            # contrasts = contrast_cube[wavelength_indices, yx_pixel[0], yx_pixel[1]]
            contrasts = contrast_cube[:, yx_pixel[0], yx_pixel[1]]
            # contrasts = contrasts[self.good_residual_mask].astype('float64')
            uncertainties = uncertainty_cube[:, yx_pixel[0], yx_pixel[1]]

            # contrasts_mean = np.mean(contrasts)
            # contrasts_norm = contrasts / contrasts_mean
            # uncertainties_norm = uncertainties / contrasts_mean

            yx_center_output = (yx_dim[0] // 2, yx_dim[1] // 2)
            relative_coords = image_coordinates.absolute_yx_to_relative_yx(
                yx_pixel, yx_center_output
            )

            if template.use_spectral_correlation:
                separation = np.sqrt(relative_coords[0] ** 2 + relative_coords[1] ** 2)

                correlation_array_index = find_nearest(
                    array=self.empirical_correlation["separation"], value=separation
                )

                # channel_mask = np.zeros(len(self.instrument.wavelengths)).astype('bool')
                # channel_mask[wavelength_indices] = True
                psi_ij = self.empirical_correlation["matrices"][correlation_array_index]
                # psi_ij = remove_channel_from_correlation_matrix(channel_mask, psi_ij)
                cov_ij = uncertainties[:, None] * psi_ij * uncertainties[None, :]
                # cov_ij_norm = (
                #     uncertainties_norm[:, None] * psi_ij * uncertainties_norm[None, :]
                # )
                # plot_scale(cov_ij)
                # plt.show()
                inv_cov_ij = np.linalg.inv(cov_ij)
            else:
                cov_ij = np.identity(len(contrasts)) * uncertainties**2
                inv_cov = 1.0 / uncertainties**2
            # if show:
            # plot_scale(np.dot(inverse, cov_ij))
            # plt.show()

            model_components = []

            if template.fit_offset:
                model_components.append(np.ones(contrasts.shape[0]))
            if template.fit_slope:
                model_components.append(wavelengths.value)
            if template.normalized_contrast_modelbox.flux is not None:
                model_components.append(
                    template.normalized_contrast_modelbox.flux[None, :]
                )

            if len(model_components) > 0:
                model_matrix = np.vstack(model_components)
            else:
                raise ValueError(
                    "No model present to fit. Provide `model`, `fit_offset` or `fit_slope`"
                )

            if template.number_of_pca_regressors > 0:
                local_config = self.reduction_parameters.merge(target_pix_mask_radius=11)
                regressor_pool_mask_global = (
                    regressor_selection.make_regressor_pool_for_pixel(
                        reduction_parameters=local_config,
                        yx_pixel=yx_pixel,
                        yx_dim=yx_dim,
                        yx_center=yx_center_output,
                        known_companion_mask=None,
                    )
                )
                regressor_pool_mask_global = np.logical_and(
                    regressor_pool_mask_global, reduced_positions_mask
                )
                training_matrix = contrast_cube[:][:, regressor_pool_mask_global]
                B_full, _, _, _ = pca_regression.compute_SVD(
                    training_matrix, n_components=None, scaling=None
                )  # 'temp-median')
                B = B_full[:, :template.number_of_pca_regressors]
                A = np.hstack((B, model_matrix.T))
            # A = np.ones(len(uncertainties))[:, None]
            else:
                A = model_matrix.T

            if template.use_spectral_correlation:
                (
                    P,
                    P_sigma_squared,
                ) = pca_regression.solve_linear_equation_with_correlation(
                    design_matrix=A.T,
                    data=contrasts,
                    inverse_covariance_matrix=inv_cov_ij,
                )
            else:
                P, P_sigma_squared = pca_regression.solve_linear_equation_simple(
                    design_matrix=A.T, data=contrasts, inverse_covariance=inv_cov
                )

            # fit_parameters, err_fit_parameters, sigma_hat_sqr = pca_regression.ols(
            #     design_matrix=A, data=contrasts_norm, covariance=cov_ij_norm)

            # P, P_sigma, _ = pca_regression.ols(
            #     design_matrix=A, data=contrasts, covariance=cov_ij)

            # P = P * contrasts_mean
            # P_sigma_squared = P_sigma**2  # * np.abs(contrasts_mean))**2
            # fit_parameters[-1] * mean_data
            # reconstructed_lightcurve = np.dot(A, P)

            template_matched_image[0, yx_pixel[0], yx_pixel[1]] = P[-1]
            template_matched_image[1, yx_pixel[0], yx_pixel[1]] = np.sqrt(
                P_sigma_squared[-1]
            )
            template_matched_image[2, yx_pixel[0], yx_pixel[1]] = P[-1] / np.sqrt(
                P_sigma_squared[-1]
            )

        if file_paths is None:
            file_paths = {}
            output_dir_matching = os.path.join(
                self.reduction_parameters.result_folder, "template_matching/"
            )
            if not os.path.exists(output_dir_matching):
                os.makedirs(output_dir_matching)
            file_paths["norm_detection_image_path"] = os.path.join(
                output_dir_matching, f"normalized_detection_image_{template_name}.fits"
            )
            file_paths["uncertainty_image_path"] = os.path.join(
                output_dir_matching, f"uncertainty_image_{template_name}.fits"
            )
            file_paths["median_uncertainty_image_path"] = os.path.join(
                output_dir_matching, f"median_uncertainty_image_{template_name}.fits"
            )
            file_paths["contrast_table_path"] = os.path.join(
                output_dir_matching, f"contrast_table_{template_name}.csv"
            )
            file_paths["contrast_plot_path"] = os.path.join(
                output_dir_matching, f"contrast_plot_{template_name}"
            )

        detection_products_matched = self.contrast_table_and_normalization(
            detection_cube=[template_matched_image],
            cube_indices=[0],
            yx_known_companion_position=None,
            inplace=False,
            save=save,
            file_paths=file_paths,
            mask_above_sigma=detection_threshold,
        )

        detection_cube = np.expand_dims(template_matched_image, axis=0)
        detection_products = detection_products_matched

        return detection_cube, detection_products

    def run_template_matching(
        self,
        template,
        detection_threshold=5.0,
        candidate_threshold=4.75,
        inner_mask_radius=1,
        search_radius=15,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        candidate_exclusion_radius=None,
        max_candidates=DEFAULT_MAX_CANDIDATES,
        good_fraction_threshold=0.05,
        theta_deviation_threshold=25,
        yx_fwhm_ratio_threshold=[1.1, 4.5],
        mask_deviating=False,
        data_full=None,
        flux_psf_full=None,
        pa=None,
        instrument=None,
        temporal_components_fraction=None,
        wavelength_indices=None,
        inverse_variance_full=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        file_paths=None,
        save=True,
    ):
        """
        Perform robust template matching detection with two-iteration bias correction.

        This function implements a sophisticated detection algorithm that uses spectral 
        template matching followed by a two-iteration detection process to avoid bias 
        in background statistics. The method first detects candidates using potentially 
        biased statistics, then masks these detections to compute unbiased background 
        statistics, and finally re-detects candidates using the corrected statistics.

        Parameters
        ----------
        template : SpectralTemplate
            The spectral template object containing the companion model spectrum,
            stellar model spectrum, and fitting parameters.
        detection_threshold : float, optional
            Signal-to-noise ratio threshold for final detection validation.
            Default is 5.0.
        candidate_threshold : float, optional
            Signal-to-noise ratio threshold for initial candidate identification.
            Default is 4.75.
        inner_mask_radius : float, optional
            Radius in pixels to mask around the central star during template matching.
            Default is 1.0.
        search_radius : float, optional
            Radius in pixels for grouping multiple detections of the same source.
            Default is 15.0.
        good_fraction_threshold : float, optional
            Minimum fraction of good pixels required in 2D Gaussian fit for validation.
            Default is 0.05.
        theta_deviation_threshold : float, optional
            Maximum allowed deviation in degrees between expected and fitted PSF 
            position angle for validation. Default is 25.0.
        yx_fwhm_ratio_threshold : list of float, optional
            Allowed range [min, max] for the ratio of y-axis to x-axis FWHM in 
            2D Gaussian fits for validation. Default is [1.1, 4.5].
        mask_deviating : bool, optional
            Whether to mask candidates with deviating PSF parameters during fitting.
            Default is False.
        data_full : ndarray, optional
            Full data cube for spectral re-extraction. Shape (n_frames, n_wavelengths, 
            n_y, n_x). Required for spectral characterization.
        flux_psf_full : ndarray, optional
            Full PSF flux cube matching data_full dimensions. Required for spectral 
            characterization.
        pa : ndarray, optional
            Position angles in degrees for each frame. Required for spectral 
            characterization.
        instrument : Instrument, optional
            Instrument object containing wavelength and detector information.
            If None, uses self.instrument.
        temporal_components_fraction : float, optional
            Fraction of temporal components to retain during PCA reduction for 
            spectral characterization. Typical values are 0.1-0.5.
        wavelength_indices : array_like, optional
            Indices of wavelengths to process. If None, uses self.wavelength_indices.
        inverse_variance_full : ndarray, optional
            Inverse variance cube matching data_full dimensions for weighted fitting.
        bad_frames : array_like, optional
            Boolean array or indices of frames to exclude from analysis.
        bad_pixel_mask_full : ndarray, optional
            Bad pixel mask cube matching data_full dimensions.
        xy_image_centers : ndarray, optional
            Center coordinates for each frame. Shape (n_frames, 2).
        amplitude_modulation_full : ndarray, optional
            Amplitude modulation factors for each frame and wavelength.
        file_paths : dict, optional
            Dictionary specifying output file paths. If None, default paths are 
            generated based on template name.
        save : bool, optional
            Whether to save intermediate detection products and results to disk.
            Default is True.

        Returns
        -------
        None
            Results are stored in the template object as:
            - template.companion_table : pandas.DataFrame or None
                Complete table of all detected candidates with fitted parameters
            - template.validated_companion_table : pandas.DataFrame or None
                Subset of candidates passing validation criteria
            - template.validated_companion_table_short : pandas.DataFrame or None
                Condensed version of validated candidates with key columns only
            - template.detection_products : dict
                Detection maps and contrast curves from template matching

        Notes
        -----
        The two-iteration detection process prevents bias in background statistics:
        
        1. **First iteration**: Detect candidates using initial (potentially biased) 
           background statistics
        2. **Masking phase**: Mask detected candidates to exclude them from 
           background statistics calculation
        3. **Second iteration**: Re-detect candidates using unbiased background 
           statistics for final validation
        
        This approach is mathematically justified for robust detection in the 
        presence of strong signals that could contaminate background noise estimates.
        
        Position coordinates are in image pixels with origin at (0,0) for absolute 
        coordinates. Relative coordinates are measured from the image center.
        Position angles are measured east of north in degrees.

        Examples
        --------
        >>> # Basic template matching with default parameters
        >>> analysis.run_template_matching(
        ...     template=my_template,
        ...     detection_threshold=5.0,
        ...     data_full=data_cube,
        ...     flux_psf_full=psf_cube,
        ...     pa=position_angles,
        ...     temporal_components_fraction=0.2
        ... )
        >>> 
        >>> # Access results
        >>> if analysis.templates['T-type'].validated_companion_table is not None:
        ...     print(f"Found {len(analysis.templates['T-type'].validated_companion_table)} candidates")
        """
        # Both are used unconditionally further down, so they cannot stay inside
        # the `file_paths is None` branch that used to define them.
        template_name = template.name
        output_dir_matching = os.path.join(
            self.reduction_parameters.result_folder, "template_matching/"
        )
        os.makedirs(output_dir_matching, exist_ok=True)
        if file_paths is None:
            file_paths = {}
            file_paths["norm_detection_image_path"] = os.path.join(
                output_dir_matching, f"normalized_detection_image_{template_name}.fits"
            )
            file_paths["uncertainty_image_path"] = os.path.join(
                output_dir_matching, f"uncertainty_image_{template_name}.fits"
            )
            file_paths["median_uncertainty_image_path"] = os.path.join(
                output_dir_matching, f"median_uncertainty_image_{template_name}.fits"
            )
            file_paths["contrast_table_path"] = os.path.join(
                output_dir_matching, f"contrast_table_{template_name}.csv"
            )
            file_paths["contrast_plot_path"] = os.path.join(
                output_dir_matching, f"contrast_plot_{template_name}"
            )
        _remove_stale_outputs(
            output_dir_matching, _template_output_filenames(template_name)
        )
        wavelengths = self.instrument.wavelengths[self.wavelength_indices]

        detection_cube, detection_products = self.template_matching_detection(
            template,
            inner_mask_radius=inner_mask_radius,
            detection_threshold=detection_threshold,
            file_paths=file_paths,
            save=save,
        )

        candidates_initial = self.find_candidates_all_wavelengths(
            detection_cube=detection_cube,
            detection_products=detection_products,
            wavelength_indices=[0],
            candidate_threshold=candidate_threshold,
            iterative_search_exclusion_radius=_resolve_exclusion_radius(
                candidate_exclusion_radius, search_radius
            ),
            minimum_candidate_separation=minimum_candidate_separation,
            max_candidates=max_candidates,
        )
        
        _, candidates_fit_initial = self.complete_candidate_table(
            candidates=candidates_initial,
            detection_cube=detection_cube,
            detection_products=detection_products,
            wavelength_indices=[0],
            candidate_threshold=candidate_threshold,
            search_radius=search_radius,
            mask_deviating=mask_deviating,
            minimum_candidate_separation=minimum_candidate_separation,
            candidate_exclusion_radius=candidate_exclusion_radius,
            max_candidates=max_candidates,
        )

        if candidates_initial is None or len(candidates_initial) == 0:
            template.companion_table = None
            template.validated_companion_table = None
            template.validated_companion_table_short = None
        else:
            yx_known_companion_position = candidates_fit_initial["snr_image"][
                ["y_relative", "x_relative"]
            ].values  # [mask]

            # Masking out detections
            detection_products_masked = self.contrast_table_and_normalization(
                detection_cube=detection_cube,
                cube_indices=[0], # only collapsed wavelength after template matching
                yx_known_companion_position=yx_known_companion_position,
                inplace=False,
                save=save,
                file_paths=file_paths,
                mask_above_sigma=None,
            )

            candidates_final = self.find_candidates_all_wavelengths(
                detection_cube=detection_cube,
                detection_products=detection_products_masked,
                wavelength_indices=[0], # only collapsed wavelength after template matching
                candidate_threshold=candidate_threshold,
                iterative_search_exclusion_radius=_resolve_exclusion_radius(
                    candidate_exclusion_radius, search_radius
                ),
                minimum_candidate_separation=minimum_candidate_separation,
                max_candidates=max_candidates,
            )

            _, candidates_fit_final = self.complete_candidate_table(
                candidates=candidates_final,
                detection_cube=detection_cube,
                detection_products=detection_products_masked,
                wavelength_indices=[0], # only collapsed wavelength after template matching
                candidate_threshold=candidate_threshold,
                search_radius=search_radius,
                mask_deviating=mask_deviating,
                minimum_candidate_separation=minimum_candidate_separation,
                candidate_exclusion_radius=candidate_exclusion_radius,
                max_candidates=max_candidates,
            )
            
            # Check if candidates survived the second iteration validation
            if candidates_fit_final is None or len(candidates_fit_final) == 0:
                logger.warning(
                    "No candidates survived second iteration validation. This typically occurs "
                    "when initial detections were false positives that did not meet the criteria "
                    "when background statistics were corrected."
                )
                template.companion_table = None
                template.validated_companion_table = None
                template.validated_companion_table_short = None
                template.detection_products = detection_products
                return
            
            logger.info("Extracting candidate spectra.")
            candidate_spectra = self.extract_candidate_spectra(
                yx_candidate_positions=candidates_fit_final["snr_image"][
                    ["y_relative", "x_relative"]
                ].values,
                temporal_components_fraction=temporal_components_fraction,
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                wavelength_indices=None,
                inverse_variance_full=inverse_variance_full,
                instrument=None,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full,
                xy_image_centers=xy_image_centers,
                amplitude_modulation_full=amplitude_modulation_full,
                return_spectra=True,
            )

            companion_table, validated_companion_table = self.detection_summary(
                candidates=candidates_final,
                candidates_fit=candidates_fit_final,
                candidate_spectra=candidate_spectra,
                use_spectra=True,
                template_name=template_name,
                snr_threshold_spectrum=False,
                snr_threshold=detection_threshold,
                good_fraction_threshold=good_fraction_threshold,
                theta_deviation_threshold=theta_deviation_threshold,
                yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold,
            )

            # companion_table, validated_companion_table = analysis.detection_summary(
            #     candidates_fit_template, candidate_spectra,
            #     snr_threshold=detection_threshold, good_fraction_threshold=0.3,
            #     theta_deviation_threshold=25.,
            #     yx_fwhm_ratio_threshold=[1.1, 3.5])

            # yx_known_companion_position = np.unique(
            #     validated_companion_table[['y_relative', 'x_relative']].values, axis=0)

            self.reduction_parameters = self.reduction_parameters.merge(
                yx_known_companion_position=yx_known_companion_position
            )

            companion_table.to_csv(
                os.path.join(
                    output_dir_matching, f"companion_table_{template_name}.csv"
                ),
                index=False,
            )

            validated_companion_table.to_csv(
                os.path.join(
                    output_dir_matching,
                    f"validated_companion_table_{template_name}.csv",
                ),
                index=False,
            )

            validated_companion_table_short = validated_companion_table[
                [
                    "candidate_id",
                    "x",
                    "y",
                    "x_relative",
                    "x_relative_sigma",
                    "y_relative",
                    "y_relative_sigma",
                    "separation",
                    "separation_sigma",
                    "position_angle",
                    "position_angle_sigma",
                    # 'channels_above_threshold',
                    "template_name",
                    "norm_snr_fit_free",
                    "peak_pixel_snr",
                    "wavelength_index",
                    "wavelength",
                    "contrast",
                    "uncertainty",
                ]
            ]

            validated_companion_table_short.to_csv(
                os.path.join(
                    output_dir_matching,
                    f"validated_companion_table_short_{template_name}.csv",
                ),
                index=False,
            )

            plt.close()
            candidate_indices = np.unique(validated_companion_table["candidate_id"])
            for candidate_index in candidate_indices:
                temp_table = validated_companion_table[
                    validated_companion_table["candidate_id"] == candidate_index
                ]
                plt.errorbar(
                    x=temp_table["wavelength"],
                    y=temp_table["contrast"],
                    yerr=temp_table["uncertainty"],
                    fmt="o",
                    label="candidate {}".format(candidate_index),
                )
            plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            plt.xlabel("wavelength")
            plt.ylabel("contrast")
            # Only add legend if there are labeled artists
            if len(candidate_indices) > 0:
                plt.legend()
            plt.savefig(
                os.path.join(
                    output_dir_matching, f"companion_spectra_{template_name}.pdf"
                )
            )
            plt.close()

            _ = self.contrast_plot(
                detection_products=detection_products,
                companion_table=validated_companion_table,
                wavelengths=np.median(wavelengths).repeat(2)[:1],
                add_wavelength_label=False,
                curvelabels=[f"{template_name}"],
                linestyles=["-", "--"],
                colors=["blue"],
                plot_companions=True,
                template_fitted=True,
                savefig=True,
                file_paths=file_paths,
                show=False,
            )

            template.companion_table = companion_table
            template.validated_companion_table = validated_companion_table
            template.validated_companion_table_short = validated_companion_table_short
            # return companion_table, validated_companion_table, validated_companion_table_short, detection_products

        template.detection_products = detection_products
        # return None, None, None, detection_products

    def match_all_templates(
        self,
        detection_threshold=5.0,
        candidate_threshold=4.75,
        inner_mask_radius=1,
        search_radius=15,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        candidate_exclusion_radius=None,
        max_candidates=DEFAULT_MAX_CANDIDATES,
        good_fraction_threshold=0.05,
        theta_deviation_threshold=25,
        yx_fwhm_ratio_threshold=[1.1, 4.5],
        data_full=None,
        flux_psf_full=None,
        pa=None,
        instrument=None,
        temporal_components_fraction=None,
        wavelength_indices=None,
        inverse_variance_full=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        file_paths=None,
        save=True,
    ):
        if self.templates:
            for key in self.templates:
                try:
                    self.run_template_matching(
                        template=self.templates[key],
                        detection_threshold=detection_threshold,
                        candidate_threshold=candidate_threshold,
                        inner_mask_radius=inner_mask_radius,
                        search_radius=search_radius,
                        minimum_candidate_separation=minimum_candidate_separation,
                        candidate_exclusion_radius=candidate_exclusion_radius,
                        max_candidates=max_candidates,
                        good_fraction_threshold=good_fraction_threshold,
                        theta_deviation_threshold=theta_deviation_threshold,
                        yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold,
                        data_full=data_full,
                        flux_psf_full=flux_psf_full,
                        pa=pa,
                        instrument=instrument,
                        temporal_components_fraction=temporal_components_fraction,
                        wavelength_indices=wavelength_indices,
                        inverse_variance_full=inverse_variance_full,
                        bad_frames=bad_frames,
                        bad_pixel_mask_full=bad_pixel_mask_full,
                        xy_image_centers=xy_image_centers,
                        amplitude_modulation_full=amplitude_modulation_full,
                        file_paths=file_paths,
                        save=save,
                    )
                except Exception:
                    # One template failing must not cost the templates after it
                    # (they run in dict order) nor the combined tables built from
                    # whichever templates did succeed.
                    logger.exception(
                        "Template matching failed for the '%s' template; continuing "
                        "with the remaining templates.", key,
                    )
                    template = self.templates[key]
                    template.companion_table = None
                    template.validated_companion_table = None
                    template.validated_companion_table_short = None

    def plot_template_matched_contrasts(self):

        combined_detection_products = {}
        contrast_tables = []
        combined_validated_companion_table = []
        combined_companion_table = []
        for key in self.templates:
            contrast_tables.append(
                self.templates[key].detection_products["contrast_tables"][0]
            )
            combined_validated_companion_table.append(
                self.templates[key].validated_companion_table
            )
            combined_companion_table.append(self.templates[key].companion_table)

        combined_detection_products["contrast_tables"] = contrast_tables

        wavelengths = self.instrument.wavelengths[self.wavelength_indices]
        file_paths = {}
        output_dir_matching = os.path.join(
            self.reduction_parameters.result_folder, "template_matching/"
        )
        if not os.path.exists(output_dir_matching):
            os.makedirs(output_dir_matching)
        file_paths["contrast_plot_path"] = os.path.join(
            output_dir_matching, "contrast_plot_template_matched"
        )

        number_of_detection_products = len(combined_detection_products["contrast_tables"])

        labels = []
        for label in list(self.templates.keys()):
            labels.append(label+" template")

        _ = self.contrast_plot(
            detection_products=combined_detection_products,
            companion_table=None,  # validated_companion_table,
            wavelengths=np.median(wavelengths).repeat(number_of_detection_products)[
                :number_of_detection_products
            ],
            add_wavelength_label=False,
            curvelabels=labels,
            linestyles=["-", "--", "-.", ":"],
            colors=["blue", "red", "gray", "black"],
            plot_companions=False,
            template_fitted=True,
            savefig=True,
            file_paths=file_paths,
            show=False,
        )

    def measure_per_channel_astrometry(
        self,
        wavelength_indices=None,
        candidate_threshold=4.75,
        search_radius=15,
        mask_deviating=False,
        independent_channels=False,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        candidate_exclusion_radius=None,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        """Detect and fit the companion in each wavelength channel separately,
        then combine the channels that individually clear ``candidate_threshold``
        in the source-aligned (r, t) frame.

        Template-independent — it runs on the per-channel detection maps
        (`self.detection_cube`), so it is the astrometrically-clean alternative
        to the spectral collapse used for detection. Returns the combined
        per-source table (one row per source, `_combine_channels_rt_frame`
        applied) or ``None`` when no channel yields a candidate above threshold.
        """
        if wavelength_indices is None:
            wavelength_indices = self.wavelength_indices
        candidates = self.find_candidates_all_wavelengths(
            wavelength_indices=wavelength_indices,
            candidate_threshold=candidate_threshold,
            iterative_search_exclusion_radius=_resolve_exclusion_radius(
                candidate_exclusion_radius, search_radius
            ),
            minimum_candidate_separation=minimum_candidate_separation,
            max_candidates=max_candidates,
        )
        if candidates is None or len(candidates) == 0:
            return None
        _, candidates_fit = self.complete_candidate_table(
            candidates=candidates,
            wavelength_indices=wavelength_indices,
            candidate_threshold=candidate_threshold,
            search_radius=search_radius,
            mask_deviating=mask_deviating,
            independent_channels=independent_channels,
            minimum_candidate_separation=minimum_candidate_separation,
            candidate_exclusion_radius=candidate_exclusion_radius,
            max_candidates=max_candidates,
        )
        if candidates_fit is None:
            return None
        return candidates_fit["snr_image"]

    def combine_template_matched_companion_tables(
        self,
        search_radius=15,
        validated_only=True,
        per_channel_min_channel_fraction=0.5,
    ):
        """
        Combines the template-matched companion tables into a single table.

        Args:
            validated_only (bool, optional): If True, only combines the validated companion tables.
                If False, combines all companion tables. Defaults to True.
            per_channel_min_channel_fraction (float, optional): Minimum share of the
                reduced wavelength channels that must individually clear the detection
                threshold before the per-channel astrometry replaces the template-collapse
                position. See `_override_astrometry_from_per_channel`. Defaults to 0.5.
        """
        
        output_dir_matching = os.path.join(
            self.reduction_parameters.result_folder, "template_matching/"
        )
        if validated_only:
            prefix = "validated_"
        else:
            prefix = ""

        # The "no companion tables found" branch below writes nothing, so without
        # this a run that detects nothing at all keeps the previous run's overall
        # tables — which downstream reads as this run's result.
        os.makedirs(output_dir_matching, exist_ok=True)
        _remove_stale_outputs(output_dir_matching, _overall_output_filenames(prefix))

        # Combine the in-memory per-template tables populated by
        # match_all_templates in this run. Re-reading the per-template CSVs from
        # disk would silently ingest a stale table from a previous run — a
        # template that finds nothing this run keeps its old file — which
        # contaminates the cross-template scatter diagnostics with detections
        # from another run.
        combined_detection_products = []
        for key in self.templates:
            template = self.templates[key]
            companion_table = (
                template.validated_companion_table
                if validated_only
                else template.companion_table
            )
            if companion_table is not None and not companion_table.empty:
                combined_detection_products.append(companion_table)

        if combined_detection_products:  # Check if list is not empty
            best_companion_matches = _combine_templates_best_snr(
                per_template_tables=combined_detection_products,
                search_radius=search_radius,
            )

            # Assign a stable candidate_id per unique source (ordered by separation).
            for idx, separation in enumerate(
                np.unique(best_companion_matches["separation"])
            ):
                mask = best_companion_matches["separation"] == separation
                n = int(np.sum(mask))
                best_companion_matches.loc[mask, "candidate_id"] = np.array([idx]).repeat(n)

            # The template collapse is optimal for detection SNR, not astrometry.
            # Where a source is detected in *enough* individual channels, its
            # per-channel inverse-variance position/σ replaces the collapse position
            # (which a signal-free channel can bias); the collapse remains as fallback.
            wavelength_indices = getattr(self, "wavelength_indices", None)
            n_channels_total = (
                len(wavelength_indices) if wavelength_indices is not None else None
            )
            best_companion_matches = _override_astrometry_from_per_channel(
                best_companion_matches,
                getattr(self, "per_channel_astrometry", None),
                search_radius=search_radius,
                n_channels_total=n_channels_total,
                min_channel_fraction=per_channel_min_channel_fraction,
            )

            spectra_cols = [
                "candidate_id", "x", "y",
                "x_relative", "x_relative_sigma", "y_relative", "y_relative_sigma",
                "xy_relative_corr", "separation", "separation_sigma",
                "position_angle", "position_angle_sigma",
                "radial_sigma_stat", "tangential_sigma_stat", "astrometry_source",
                "template_name", "best_template", "n_templates_above_threshold",
                "astrometry_template_disagreement",
                "x_relative_sigma_template_scatter", "y_relative_sigma_template_scatter",
                "separation_sigma_template_scatter", "position_angle_sigma_template_scatter",
                "norm_snr_fit_free", "peak_pixel_snr",
                "wavelength_index", "wavelength", "contrast", "uncertainty",
            ]
            spectra_present = [
                c for c in spectra_cols if c in best_companion_matches.columns
            ]
            best_companion_matches_spectra = best_companion_matches[
                spectra_present
            ].sort_values(["candidate_id", "wavelength"], ignore_index=True)

            per_wavelength = {"wavelength_index", "wavelength", "contrast", "uncertainty"}
            short_cols = [c for c in spectra_present if c not in per_wavelength]
            best_companion_matches_short = (
                best_companion_matches[best_companion_matches["wavelength_index"] == 0][
                    short_cols
                ].sort_values(["candidate_id"], ignore_index=True)
            )

            best_companion_matches_spectra.to_csv(
                os.path.join(
                    output_dir_matching, f"overall_{prefix}companion_detections_spectra.csv"
                ),
                index=False,
            )
            best_companion_matches_short.to_csv(
                os.path.join(output_dir_matching, f"overall_{prefix}companion_detections.csv"),
                index=False,
            )
        else:
            logger.warning("No companion tables found.")


    def detection_and_characterization(
        self,
        detection_products=None,
        data_full=None,
        flux_psf_full=None,
        pa=None,
        temporal_components_fraction=None,
        inverse_variance_full=None,
        bad_frames=None,
        bad_pixel_mask_full=None,
        xy_image_centers=None,
        amplitude_modulation_full=None,
        candidate_threshold=4.75,
        detection_threshold=5.0,
        search_radius=15,
        good_fraction_threshold=0.05,
        theta_deviation_threshold=25,
        yx_fwhm_ratio_threshold=[1.1, 4.5],
        save_initial_detection_products=False,
    ):

        self.detection_cube[self.detection_cube == 0.] = np.nan

        self.contrast_table_and_normalization(
            save=save_initial_detection_products,
            mask_above_sigma=detection_threshold,
            file_paths=self.file_paths
        )

        logger.info("Identifying and fitting potential candidates.")
        candidates, candidates_fit = self.complete_candidate_table(
            wavelength_indices=None,
            candidate_threshold=candidate_threshold,
            search_radius=search_radius,
            detection_products=detection_products,
        )

        if candidates is None or candidates_fit is None:
            companion_table = None
            validated_companion_table = None
            validated_companion_table_short = None
            plot_companions = False
        else:
            plot_companions = True
            logger.info("Extracting candidate spectra.")
            candidate_spectra = self.extract_candidate_spectra(
                yx_candidate_positions=candidates_fit["snr_image"][
                    ["y_relative", "x_relative"]
                ].values,
                temporal_components_fraction=temporal_components_fraction,
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                wavelength_indices=None,
                inverse_variance_full=inverse_variance_full,
                instrument=None,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full,
                xy_image_centers=xy_image_centers,
                amplitude_modulation_full=amplitude_modulation_full,
                return_spectra=True,
            )

            # companion_table, validated_companion_table = self.detection_summary(
            #     candidates_fit, candidate_spectra,
            #     snr_threshold=detection_threshold,
            #     good_fraction_threshold=good_fraction_threshold,
            #     theta_deviation_threshold=theta_deviation_threshold,
            #     yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold)

            companion_table, validated_companion_table = self.detection_summary(
                candidates=candidates,
                candidates_fit=candidates_fit,
                candidate_spectra=candidate_spectra,
                use_spectra=True,
                template_name=None,
                snr_threshold_spectrum=False,
                snr_threshold=detection_threshold,
                good_fraction_threshold=good_fraction_threshold,
                theta_deviation_threshold=theta_deviation_threshold,
                yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold,
            )

            yx_known_companion_position = np.unique(
                validated_companion_table[["y_relative", "x_relative"]].values, axis=0
            )

            self.reduction_parameters = self.reduction_parameters.merge(
                yx_known_companion_position=yx_known_companion_position
            )

            companion_table.to_csv(
                os.path.join(
                    self.reduction_parameters.result_folder, "companion_table.csv"
                ),
                index=False,
            )

            validated_companion_table.to_csv(
                os.path.join(
                    self.reduction_parameters.result_folder,
                    "validated_companion_table.csv",
                ),
                index=False,
            )

            validated_companion_table_short = validated_companion_table[
                [
                    "candidate_id",
                    "x",
                    "y",
                    "x_relative",
                    "x_relative_sigma",
                    "y_relative",
                    "y_relative_sigma",
                    "separation",
                    "separation_sigma",
                    "position_angle",
                    "position_angle_sigma",
                    # 'channels_above_threshold',
                    "norm_snr_fit_free",
                    "peak_pixel_snr",
                    "wavelength_index",
                    "wavelength",
                    "contrast",
                    "uncertainty",
                ]
            ]

            validated_companion_table_short.to_csv(
                os.path.join(
                    self.reduction_parameters.result_folder,
                    "validated_companion_table_short.csv",
                ),
                index=False,
            )

            plt.close()
            candidate_indices = np.unique(validated_companion_table["candidate_id"])
            for candidate_index in candidate_indices:
                temp_table = validated_companion_table[
                    validated_companion_table["candidate_id"] == candidate_index
                ]
                plt.errorbar(
                    x=temp_table["wavelength"],
                    y=temp_table["contrast"],
                    yerr=temp_table["uncertainty"],
                    fmt="o",
                    label="candidate {}".format(candidate_index),
                )
            plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            plt.xlabel("wavelength")
            plt.ylabel("contrast")
            # Only add legend if there are labeled artists
            if len(candidate_indices) > 0:
                plt.legend()
            plt.savefig(
                os.path.join(
                    self.reduction_parameters.result_folder, "companion_spectra.pdf"
                )
            )
            plt.close()

        self.contrast_table_and_normalization(save=True)
        _ = self.contrast_plot(
            savefig=True,
            plot_companions=plot_companions,
            companion_table=validated_companion_table,
            show=False,
        )

        self.validated_companion_table_short = validated_companion_table_short
        self.validated_companion_table = validated_companion_table
        self.companion_table = companion_table


    def detection_and_characterization_with_template_matching(
        self, reduction_parameters, instrument, species_database_directory, stellar_parameters, data_full, flux_psf_full,
        pa, temporal_components_fraction, wavelength_indices, xy_image_centers=None, 
        inverse_variance_full=None, bad_frames=None, bad_pixel_mask_full=None, 
        amplitude_modulation_full=None, 
        detection_threshold=5., candidate_threshold=4.75,
        use_spectral_correlation=False,
        inner_mask_radius=1, search_radius=15, good_fraction_threshold=0.05,
        minimum_candidate_separation=DEFAULT_MINIMUM_CANDIDATE_SEPARATION,
        candidate_exclusion_radius=None, max_candidates=DEFAULT_MAX_CANDIDATES,
        theta_deviation_threshold=25, yx_fwhm_ratio_threshold=[1.1, 4.5],
        save_initial_detection_products=True,
        per_channel_min_channel_fraction=0.5,
        per_channel_independent_channels=False):

        if reduction_parameters is not None and instrument is not None:
            self.reduction_parameters = _to_reduction_config(reduction_parameters)

        self.reduction_parameters = self.reduction_parameters.merge(
            yx_known_companion_position=None
        )

        self.detection_cube[self.detection_cube == 0.] = np.nan

        self.contrast_table_and_normalization(
            save=save_initial_detection_products,
            mask_above_sigma=detection_threshold,
            file_paths=self.file_paths
        )

        self.add_default_templates(
            stellar_modelbox=None,
            stellar_parameters=stellar_parameters,
            species_database_directory=species_database_directory,
            instrument=instrument,
            correct_transmission=False,
            use_spectral_correlation=use_spectral_correlation,
        )

        self.match_all_templates(
            detection_threshold=detection_threshold,
            candidate_threshold=candidate_threshold,
            inner_mask_radius=inner_mask_radius,
            search_radius=search_radius,
            minimum_candidate_separation=minimum_candidate_separation,
            candidate_exclusion_radius=candidate_exclusion_radius,
            max_candidates=max_candidates,
            good_fraction_threshold=good_fraction_threshold,
            theta_deviation_threshold=theta_deviation_threshold,
            yx_fwhm_ratio_threshold=yx_fwhm_ratio_threshold,
            data_full=data_full,
            flux_psf_full=flux_psf_full,
            pa=pa,
            instrument=None,
            temporal_components_fraction=temporal_components_fraction,
            wavelength_indices=wavelength_indices,
            inverse_variance_full=inverse_variance_full,
            bad_frames=bad_frames,
            bad_pixel_mask_full=bad_pixel_mask_full,
            xy_image_centers=xy_image_centers,
            amplitude_modulation_full=amplitude_modulation_full,
            file_paths=None,
            save=True,
        )
        
        try:
            self.plot_template_matched_contrasts()
        except Exception:
            # Diagnostics only; never worth losing the tables below over.
            logger.exception("Contrast plotting failed; continuing.")

        # Per-channel astrometry (template-independent) drives the reported
        # position/σ; the template collapse is optimal for detection SNR, not
        # for astrometry. Measured once here and merged into both overall tables.
        # It is a refinement of an astrometry the collapse already provides, so
        # a failure here must not cost the overall tables — which is exactly what
        # used to happen, since they are written after this call.
        try:
            self.per_channel_astrometry = self.measure_per_channel_astrometry(
                wavelength_indices=wavelength_indices,
                candidate_threshold=candidate_threshold,
                search_radius=search_radius,
                independent_channels=per_channel_independent_channels,
                minimum_candidate_separation=minimum_candidate_separation,
                candidate_exclusion_radius=candidate_exclusion_radius,
                max_candidates=max_candidates,
            )
        except Exception:
            logger.exception(
                "Per-channel astrometry failed; falling back to the "
                "template-collapse astrometry for the combined tables."
            )
            self.per_channel_astrometry = None
        if self.per_channel_astrometry is not None:
            output_dir_matching = os.path.join(
                self.reduction_parameters.result_folder, "template_matching/"
            )
            os.makedirs(output_dir_matching, exist_ok=True)
            self.per_channel_astrometry.to_csv(
                os.path.join(output_dir_matching, "per_channel_astrometry.csv"),
                index=False,
            )

        self.combine_template_matched_companion_tables(
            search_radius=search_radius,
            validated_only=True,
            per_channel_min_channel_fraction=per_channel_min_channel_fraction,
        )
        self.combine_template_matched_companion_tables(
            search_radius=search_radius,
            validated_only=False,
            per_channel_min_channel_fraction=per_channel_min_channel_fraction,
        )

        # spectrum = self.extract_candidate_spectra(
        #     temporal_components_fraction=temporal_components_fraction,
        #     data_full=data_full,
        #     flux_psf_full=flux_psf_full,
        #     pa=pa,
        #     yx_candidate_positions=[[-48.733759, -23.876381]], #[[5.90801, 50.190368]], #  
        #     wavelength_indices=wavelength_indices,
        #     inverse_variance_full=None,
        #     instrument=None,
        #     bad_frames=bad_frames,
        #     bad_pixel_mask_full=bad_pixel_mask_full,
        #     xy_image_centers=xy_image_centers,
        #     amplitude_modulation_full=amplitude_modulation_full,
        #     return_spectra=True
        # )
        
        # spectrum.to_csv("manual_extraction")
