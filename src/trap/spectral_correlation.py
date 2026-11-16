"""Channel correlation of the contrast maps per separation ring, and its analytic fit.

The empirical matrix of a ring is noisy, because the pixels of a ring hold far fewer
independent resolution elements than there are channel pairs, and it is often not positive
definite. The product is therefore the fit of the form of Greco and Brandt (2016, ApJ 833,
134),

    psi_ij(rho) = a_rho exp(-(rho / sigma_rho * dl_ij)^2 / 2)
                  + a_lambda exp(-(dl_ij / sigma_lambda)^2 / 2) + a_delta delta_ij,

with ``dl_ij = |lambda_i - lambda_j| / lambda_c``, ``rho`` the separation in lambda_c / D and
``a_delta = 1 - a_rho - a_lambda`` kept at or above 0.01. The lengths are shared by all rings and the amplitudes
fitted per ring. The empirical stack is kept for checks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from trap.utils import compute_empirical_correlation_matrix

logger = logging.getLogger(__name__)

MINIMUM_FIT_CHANNELS = 4
# Floor of the uncorrelated amplitude. The Gaussian kernels are nearly singular over many
# channels, so the smallest eigenvalue of the matrix is about a_delta; the floor keeps the
# matrix and the covariance built from it invertible.
MINIMUM_A_DELTA = 0.01
SIGMA_RHO_START = 1.0
SIGMA_LAMBDA_START = 0.05
SIGMA_RHO_BOUNDS = (0.01, 50.0)
SIGMA_LAMBDA_BOUNDS = (0.001, 2.0)
CORRELATED_START = 0.6
SPECKLE_FRACTION_START = 0.5

FIT_COLUMNS = [
    "a_rho",
    "a_lambda",
    "a_delta",
    "sigma_rho",
    "sigma_lambda",
    "rms",
    "free_a_rho",
    "free_a_lambda",
    "free_sigma_rho",
    "free_sigma_lambda",
    "free_rms",
]


def _wavelength_values(wavelengths):
    if isinstance(wavelengths, u.Quantity):
        return wavelengths.to_value(u.micron)
    return np.asarray(wavelengths, dtype=float)


def spectral_correlation_settings(config):
    """Spectral correlation settings of a reduction config, with defaults for configs saved before the fields existed."""
    return {
        "width": getattr(config, "spectral_correlation_width", 9.0),
        "fit": getattr(config, "spectral_correlation_fit", True),
        "flag_rms": getattr(config, "spectral_correlation_flag_rms", 0.2),
    }


def relative_wavelength_difference(wavelengths):
    """Return ``|lambda_i - lambda_j| / lambda_c`` for all channel pairs, lambda_c the mean wavelength."""
    wavelengths = _wavelength_values(wavelengths)
    return np.abs(wavelengths[:, None] - wavelengths[None, :]) / np.mean(wavelengths)


def empirical_stack(cube, mask, separations, width, min_pixels=40, yx_center=None):
    """Compute the empirical channel correlation matrix of each separation ring.

    A pixel enters a ring when its centre lies within ``width / 2`` of the separation and it
    is finite and non-zero in every channel and outside ``mask``. Rings with fewer than
    ``min_pixels`` such pixels are dropped, with a warning naming the reason when no ring is left.

    Args:
        cube: Maps of all channels, shape (n_ch, ny, nx). Not modified.
        mask: Boolean map of excluded pixels, or None.
        separations: Ring centres in pixels.
        width: Ring width in pixels.
        min_pixels: Fewest pixels a ring needs to be kept.
        yx_center: Centre pixel; default (ny // 2, nx // 2).

    Returns:
        ``(psi, separations_used, n_pixels)``: the matrices, shape (n_sep, n_ch, n_ch), the
        separations of the kept rings and the number of pixels each used.
    """
    cube = np.asarray(cube)
    n_channels, ny, nx = cube.shape
    yy, xx = np.indices((ny, nx))
    if yx_center is None:
        yx_center = (ny // 2, nx // 2)
    distance = np.hypot(yy - yx_center[0], xx - yx_center[1])
    usable = np.all(np.isfinite(cube) & (cube != 0), axis=0)
    usable[int(yx_center[0]), int(yx_center[1])] = False
    if mask is not None:
        usable &= ~np.asarray(mask, dtype=bool)

    matrices, separations_used, n_pixels = [], [], []
    for separation in np.asarray(separations):
        r_in = max(separation - width / 2.0, 0.5)
        ring = usable & (distance >= r_in) & (distance <= separation + width / 2.0)
        count = int(np.count_nonzero(ring))
        if count < min_pixels:
            continue
        matrices.append(compute_empirical_correlation_matrix(cube[:, ring]))
        separations_used.append(separation)
        n_pixels.append(count)

    if not matrices:
        if not np.all(np.isfinite(cube) & (cube != 0), axis=0).any():
            reason = "no pixel is finite and non-zero in every channel"
        elif not usable.any():
            reason = "the mask covers every pixel that is finite and non-zero in every channel"
        else:
            reason = f"no ring keeps {min_pixels} pixels that are finite and non-zero in every channel and unmasked"
        logger.warning("No ring for the channel correlation: %s; template matching uses the identity.", reason)
    psi = np.array(matrices, dtype=float).reshape(-1, n_channels, n_channels)
    return psi, np.array(separations_used), np.array(n_pixels, dtype=int)


def greco_brandt(rho, dl, a_rho, a_lambda, sigma_rho, sigma_lambda):
    """Return the correlation matrix of the Greco and Brandt (2016) form.

    Args:
        rho: Separation in lambda_c / D.
        dl: Matrix of ``|lambda_i - lambda_j| / lambda_c``, shape (n_ch, n_ch).
        a_rho: Amplitude of the speckle term.
        a_lambda: Amplitude of the spectral term.
        sigma_rho: Radial correlation length of the speckles in lambda_c / D.
        sigma_lambda: Correlation length of the spectral term in units of ``dl``.

    Returns:
        The matrix, with ``1 - a_rho - a_lambda`` on the diagonal added to the two kernels
        so that the diagonal is one. Amplitudes whose sum exceeds ``1 - MINIMUM_A_DELTA``
        are scaled down to that sum, so the matrix stays positive definite.
    """
    dl = np.asarray(dl, dtype=float)
    a_rho, a_lambda = floor_a_delta(a_rho, a_lambda)
    speckle = np.exp(-0.5 * (rho / sigma_rho * dl) ** 2)
    spectral = np.exp(-0.5 * (dl / sigma_lambda) ** 2)
    return a_rho * speckle + a_lambda * spectral + (1.0 - a_rho - a_lambda) * np.identity(dl.shape[0])


def _off_diagonal(dl):
    rows, columns = np.triu_indices(dl.shape[0], k=1)
    return rows, columns, dl[rows, columns]


def floor_a_delta(a_rho, a_lambda):
    """Scale both amplitudes down where their sum exceeds ``1 - MINIMUM_A_DELTA``."""
    total = a_rho + a_lambda
    scale = np.where(total > 1.0 - MINIMUM_A_DELTA, (1.0 - MINIMUM_A_DELTA) / np.where(total > 0, total, 1.0), 1.0)
    return a_rho * scale, a_lambda * scale


def _amplitudes(correlated, speckle_fraction):
    return correlated * speckle_fraction, correlated * (1.0 - speckle_fraction)


def _ring_residuals(psi_pairs, rho, dl_pairs, a_rho, a_lambda, sigma_rho, sigma_lambda):
    model = a_rho * np.exp(-0.5 * (rho / sigma_rho * dl_pairs) ** 2) + a_lambda * np.exp(-0.5 * (dl_pairs / sigma_lambda) ** 2)
    # Non-finite elements of an empirical matrix carry no information about the fit.
    return np.where(np.isfinite(psi_pairs), psi_pairs - model, 0.0)


def _rms(residuals):
    return float(np.sqrt(np.mean(residuals**2)))


def _global_fit(psi_pairs, rho, dl_pairs):
    n_rings, n_pairs = psi_pairs.shape

    def residuals(parameters):
        sigma_rho, sigma_lambda = parameters[:2]
        a_rho, a_lambda = _amplitudes(parameters[2 : 2 + n_rings], parameters[2 + n_rings :])
        ring_residuals = _ring_residuals(
            psi_pairs, rho[:, None], dl_pairs[None, :], a_rho[:, None], a_lambda[:, None], sigma_rho, sigma_lambda
        )
        return ring_residuals.ravel()

    # Each ring's elements depend only on the two lengths and its own amplitudes.
    sparsity = lil_matrix((n_rings * n_pairs, 2 + 2 * n_rings), dtype=int)
    for ring in range(n_rings):
        rows = slice(ring * n_pairs, (ring + 1) * n_pairs)
        sparsity[rows, 0:2] = 1
        sparsity[rows, 2 + ring] = 1
        sparsity[rows, 2 + n_rings + ring] = 1

    start = np.concatenate(
        [
            [SIGMA_RHO_START, SIGMA_LAMBDA_START],
            np.full(n_rings, CORRELATED_START),
            np.full(n_rings, SPECKLE_FRACTION_START),
        ]
    )
    lower = np.concatenate([[SIGMA_RHO_BOUNDS[0], SIGMA_LAMBDA_BOUNDS[0]], np.zeros(2 * n_rings)])
    upper = np.concatenate(
        [[SIGMA_RHO_BOUNDS[1], SIGMA_LAMBDA_BOUNDS[1]], np.full(n_rings, 1.0 - MINIMUM_A_DELTA), np.ones(n_rings)]
    )
    result = least_squares(residuals, start, bounds=(lower, upper), jac_sparsity=sparsity, method="trf")
    sigma_rho, sigma_lambda = result.x[:2]
    a_rho, a_lambda = _amplitudes(result.x[2 : 2 + n_rings], result.x[2 + n_rings :])
    return sigma_rho, sigma_lambda, a_rho, a_lambda


def _free_fit(psi_pairs, rho, dl_pairs, a_rho, a_lambda, sigma_rho, sigma_lambda):
    def residuals(parameters):
        correlated, speckle_fraction, sigma_rho, sigma_lambda = parameters
        return _ring_residuals(psi_pairs, rho, dl_pairs, *_amplitudes(correlated, speckle_fraction), sigma_rho, sigma_lambda)

    correlated = a_rho + a_lambda
    lower = [0.0, 0.0, SIGMA_RHO_BOUNDS[0], SIGMA_LAMBDA_BOUNDS[0]]
    upper = [1.0 - MINIMUM_A_DELTA, 1.0, SIGMA_RHO_BOUNDS[1], SIGMA_LAMBDA_BOUNDS[1]]
    start = [correlated, a_rho / correlated if correlated > 0 else SPECKLE_FRACTION_START, sigma_rho, sigma_lambda]
    start = np.clip(start, lower, upper)
    correlated, speckle_fraction, sigma_rho, sigma_lambda = least_squares(
        residuals, start, bounds=(lower, upper), method="trf"
    ).x
    return (*_amplitudes(correlated, speckle_fraction), sigma_rho, sigma_lambda)


def fit_stack(psi, rho, wavelengths, separations=None, n_pixels=None, flag_rms=0.2, fit=True):
    """Fit the Greco and Brandt (2016) form to a stack of empirical correlation matrices.

    One least-squares fit over all rings gives the shared lengths ``sigma_rho`` and
    ``sigma_lambda`` and the amplitudes ``a_rho`` and ``a_lambda`` of each ring. Both
    amplitudes are fitted as their sum, bounded to [0, 0.99] so that ``a_delta`` stays at or
    above 0.01 and the matrix positive definite, and the speckle fraction of that sum,
    bounded to [0, 1]. A free fit of all parameters per ring, started from the global solution, is
    stored in the ``free_*`` columns as a diagnostic. Only the off-diagonal elements enter.

    Args:
        psi: Empirical matrices, shape (n_sep, n_ch, n_ch).
        rho: Separation of each ring in lambda_c / D.
        wavelengths: Channel wavelengths, a Quantity or values in micron.
        separations: Ring separations in pixels; default ``rho``.
        n_pixels: Pixels used per ring; default -1.
        flag_rms: A ring whose rms residual of the global fit exceeds this is flagged.
        fit: Fit the form. When False, or with fewer than four channels, the fit columns are
            NaN and ``fitted`` is False.

    Returns:
        One row per ring with columns ``separation``, ``rho``, ``n_pixels``, ``a_rho``,
        ``a_lambda``, ``a_delta``, ``sigma_rho``, ``sigma_lambda``, ``rms``, ``free_a_rho``,
        ``free_a_lambda``, ``free_sigma_rho``, ``free_sigma_lambda``, ``free_rms``,
        ``flagged`` and ``fitted``.
    """
    psi = np.asarray(psi, dtype=float)
    rho = np.asarray(rho, dtype=float)
    n_rings, n_channels = len(rho), psi.shape[-1]
    table = pd.DataFrame(
        {
            "separation": rho.copy() if separations is None else np.asarray(separations, dtype=float),
            "rho": rho,
            "n_pixels": np.full(n_rings, -1) if n_pixels is None else np.asarray(n_pixels, dtype=int),
        }
    )
    for column in FIT_COLUMNS:
        table[column] = np.nan
    table["flagged"] = False
    table["fitted"] = False
    if not fit or n_channels < MINIMUM_FIT_CHANNELS or n_rings == 0:
        return table

    dl = relative_wavelength_difference(wavelengths)
    rows, columns, dl_pairs = _off_diagonal(dl)
    psi_pairs = psi[:, rows, columns]

    sigma_rho, sigma_lambda, a_rho, a_lambda = _global_fit(psi_pairs, rho, dl_pairs)
    table["a_rho"] = a_rho
    table["a_lambda"] = a_lambda
    table["a_delta"] = 1.0 - a_rho - a_lambda
    table["sigma_rho"] = sigma_rho
    table["sigma_lambda"] = sigma_lambda
    for ring in range(n_rings):
        table.loc[ring, "rms"] = _rms(
            _ring_residuals(psi_pairs[ring], rho[ring], dl_pairs, a_rho[ring], a_lambda[ring], sigma_rho, sigma_lambda)
        )
        free = _free_fit(psi_pairs[ring], rho[ring], dl_pairs, a_rho[ring], a_lambda[ring], sigma_rho, sigma_lambda)
        table.loc[ring, ["free_a_rho", "free_a_lambda", "free_sigma_rho", "free_sigma_lambda"]] = free
        table.loc[ring, "free_rms"] = _rms(_ring_residuals(psi_pairs[ring], rho[ring], dl_pairs, *free))
    table["flagged"] = table["rms"] > flag_rms
    table["fitted"] = True
    return table


def fitted_stack(table, wavelengths):
    """Return the fitted matrices of all rings, shape (n_sep, n_ch, n_ch); NaN for rings not fitted."""
    dl = relative_wavelength_difference(wavelengths)
    return np.array(
        [greco_brandt(row.rho, dl, row.a_rho, row.a_lambda, row.sigma_rho, row.sigma_lambda) for row in table.itertuples()]
    ).reshape(-1, dl.shape[0], dl.shape[0])


def correlation_at(separation, psi, table, wavelengths):
    """Return the channel correlation matrix at a separation.

    When the nearest ring is fitted, the matrix is the fitted form at ``separation`` with
    the shared lengths and the amplitudes interpolated linearly between the two neighbouring
    rings that are not flagged (all rings when every ring is flagged), held constant beyond
    the outermost rings. ``a_delta`` is kept at or above 0.01 as in `greco_brandt`. Otherwise it is the empirical matrix of
    the nearest ring. Without any ring it is the identity.

    Args:
        separation: Separation in pixels.
        psi: Empirical matrices, shape (n_sep, n_ch, n_ch).
        table: Fit table of `fit_stack`.
        wavelengths: Channel wavelengths, a Quantity or values in micron.
    """
    separations = table["separation"].to_numpy(dtype=float)
    if len(separations) == 0:
        return np.identity(len(_wavelength_values(wavelengths)))
    nearest = int(np.argmin(np.abs(separations - separation)))
    if not bool(table["fitted"].iloc[nearest]):
        return psi[nearest]

    usable = ~table["flagged"].to_numpy(dtype=bool)
    if not usable.any():
        usable[:] = True
    order = np.argsort(separations[usable])
    ring_separations = separations[usable][order]
    a_rho = np.interp(separation, ring_separations, table["a_rho"].to_numpy(dtype=float)[usable][order])
    a_lambda = np.interp(separation, ring_separations, table["a_lambda"].to_numpy(dtype=float)[usable][order])
    rho_per_pixel = np.median(table["rho"].to_numpy(dtype=float) / separations)
    return greco_brandt(
        separation * rho_per_pixel,
        relative_wavelength_difference(wavelengths),
        a_rho,
        a_lambda,
        float(table["sigma_rho"].iloc[nearest]),
        float(table["sigma_lambda"].iloc[nearest]),
    )


def write_stack(path, psi, table, wavelengths, pixel_scale_mas, width=None, flag_rms=None, overwrite=True):
    """Write the empirical stack and its fit to a FITS file.

    Extensions are ``PSI`` (the empirical matrices), ``FIT`` (the fit table with the ring
    separations in pixels and mas and the pixels used) and ``WAVELENGTH`` (micron). The
    primary header holds ``PSIWIDTH`` (ring width in pixels), ``PSIFIT`` (whether the rings
    are fitted) and ``PSIFLAG`` (the rms threshold of ``flagged``); ``PSIWIDTH`` and
    ``PSIFLAG`` are left out when not given.
    """
    table = table.copy()
    table.insert(table.columns.get_loc("separation") + 1, "separation_mas", table["separation"] * pixel_scale_mas)
    primary = fits.PrimaryHDU()
    if width is not None:
        primary.header["PSIWIDTH"] = (float(width), "ring width of the stack in pixels")
    primary.header["PSIFIT"] = (bool(table["fitted"].any()), "Greco and Brandt (2016) form fitted")
    if flag_rms is not None:
        primary.header["PSIFLAG"] = (float(flag_rms), "rms above which a ring is flagged")
    psi_hdu = fits.ImageHDU(np.asarray(psi, dtype=float), name="PSI")
    fit_hdu = fits.BinTableHDU(Table.from_pandas(table), name="FIT")
    wavelength_hdu = fits.ImageHDU(_wavelength_values(wavelengths), name="WAVELENGTH")
    wavelength_hdu.header["BUNIT"] = "micron"
    fits.HDUList([primary, psi_hdu, fit_hdu, wavelength_hdu]).writeto(path, overwrite=overwrite)


def read_stack(path):
    """Read a file of `write_stack` and return ``(psi, table, wavelengths)``, wavelengths in micron."""
    with fits.open(path) as hdul:
        # A stack without rings is stored as an empty image, which reads back as None.
        psi = np.array([] if hdul["PSI"].data is None else hdul["PSI"].data, dtype=float)
        table = Table(hdul["FIT"].data).to_pandas()
        wavelengths = np.array(hdul["WAVELENGTH"].data, dtype=float)
    n_channels = len(wavelengths)
    return psi.reshape(-1, n_channels, n_channels), table, wavelengths
