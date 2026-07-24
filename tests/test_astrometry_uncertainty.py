"""Unit tests for the astrometric-uncertainty additions to
trap.detection. See
docs/superpowers/specs/2026-07-23-trap-astrometry-uncertainty-design.md
for the mathematical justification of every assertion in this file.
"""

import numpy as np
import pandas as pd
import pytest

from trap import detection


def test_module_imports():
    assert hasattr(detection, "fit_2d_gaussian")


def _make_isotropic_gaussian(size=21, amplitude=50.0, center=None, sigma=1.5, noise=0.5):
    """Noisy 2D Gaussian on a square image. Position at (cy, cx)."""
    if center is None:
        center = (size / 2 - 0.5, size / 2 - 0.5)
    yy, xx = np.mgrid[:size, :size]
    image = amplitude * np.exp(
        -((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2 * sigma**2)
    )
    rng = np.random.default_rng(0)
    image = image + rng.normal(0.0, noise, size=image.shape)
    return image, center


def test_fit_2d_gaussian_returns_param_cov_on_clean_source():
    size = 21
    image, center = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    yx_center = (size // 2, size // 2)
    result = detection.fit_2d_gaussian(
        image,
        yx_position=(round(center[0]), round(center[1])),
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
    )
    assert "param_cov_xy" in result
    assert "param_names" in result
    cov = result["param_cov_xy"]
    assert cov is not None
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    assert cov[0, 0] > 0 and cov[1, 1] > 0
    # For a symmetric input, ρ_xy should be small
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(rho) < 0.3


def test_fit_2d_gaussian_handles_singular_fit():
    """All-NaN cutout region should raise, unchanged by the covariance additions."""
    size = 21
    image, _ = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    image[9:12, 9:12] = np.nan
    yx_center = (size // 2, size // 2)
    with pytest.raises(RuntimeError):
        detection.fit_2d_gaussian(
            image,
            yx_position=(10, 10),
            yx_center=yx_center,
            x_stddev=1.5,
            y_stddev=1.5,
            box_size=3,
        )


def test_fit_2d_gaussian_fixed_position_clamps_centroid():
    """With fixed_position set, the fit centroid must equal the requested
    sub-pixel position exactly (used by Fits B/C to clamp to Fit A)."""
    size = 21
    image, center = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    yx_center = (size // 2, size // 2)
    clamp_yx = (center[0] + 0.3, center[1] - 0.2)
    result = detection.fit_2d_gaussian(
        image,
        yx_position=(round(center[0]), round(center[1])),
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
        fixed_position=clamp_yx,
    )
    assert np.isclose(result["yx_fit_position_orig"][0], clamp_yx[0])
    assert np.isclose(result["yx_fit_position_orig"][1], clamp_yx[1])
    # Position was held fixed, so it is not a free parameter and has no cov row
    assert "x_mean" not in result["param_names"]
    assert "y_mean" not in result["param_names"]


def test_rt_sigmas_axis_aligned_recovers_input_widths():
    """phi_source = 0 → (r, t) frame == (y, x) frame; verify identity."""
    x_fwhm_free = 3.0 * 2.355  # sigma_x = 3.0
    y_fwhm_free = 5.0 * 2.355  # sigma_y = 5.0
    theta_free = 0.0
    phi_source = 0.0
    param_cov_xy = np.array([[0.04, 0.0], [0.0, 0.09]])  # sigma_x_fit=0.2, sigma_y_fit=0.3
    snr = 100.0
    result = detection._compute_rt_frame_sigmas(
        x_fwhm_free=x_fwhm_free,
        y_fwhm_free=y_fwhm_free,
        theta_free=theta_free,
        phi_source=phi_source,
        param_cov_xy=param_cov_xy,
        snr_local_normalized=snr,
        eps_snr=1.0,
    )
    # PSF widths in (r, t) at Δ=0: σ_PSF_r = σx_free = 3.0, σ_PSF_t = σy_free = 5.0
    # CR floor at SNR=100: σ_r_CR = 3.0/100, σ_t_CR = 5.0/100
    assert np.isclose(result["sigma_r_cr"], 3.0 / 100.0)
    assert np.isclose(result["sigma_t_cr"], 5.0 / 100.0)
    # Fit cov is diagonal → rotation by 0 leaves it diagonal
    assert np.isclose(result["sigma_r_fit"], np.sqrt(0.04))
    assert np.isclose(result["sigma_t_fit"], np.sqrt(0.09))
    # Both fit values exceed the CR floor (0.2 > 0.03, 0.3 > 0.05)
    assert np.isclose(result["sigma_r_stat"], np.sqrt(0.04))
    assert np.isclose(result["sigma_t_stat"], np.sqrt(0.09))
    # Correlation coefficient survives (both axes from fit; ρ was 0)
    assert np.isclose(result["rho_rt"], 0.0)


def test_rt_sigmas_45deg_rotated_correlates_xy():
    """Source-star vector at 45° → isotropic PSF gives σ_r=σ_t and an
    isotropic centroid cov stays diagonal in the (r, t) frame."""
    x_fwhm_free = 4.0 * 2.355
    y_fwhm_free = 4.0 * 2.355  # isotropic PSF → σ_PSF_r = σ_PSF_t regardless of Δ
    theta_free = 0.0
    phi_source = np.pi / 4
    # Isotropic centroid cov: any rotation leaves diag identical, off-diag 0
    param_cov_xy = np.diag([0.05, 0.05])
    snr = 20.0
    result = detection._compute_rt_frame_sigmas(
        x_fwhm_free=x_fwhm_free,
        y_fwhm_free=y_fwhm_free,
        theta_free=theta_free,
        phi_source=phi_source,
        param_cov_xy=param_cov_xy,
        snr_local_normalized=snr,
    )
    assert np.isclose(result["sigma_r_stat"], np.sqrt(0.05))
    assert np.isclose(result["sigma_t_stat"], np.sqrt(0.05))
    assert np.isclose(result["rho_rt"], 0.0, atol=1e-9)


def test_rt_sigmas_falls_back_to_cr_when_fit_cov_none():
    result = detection._compute_rt_frame_sigmas(
        x_fwhm_free=3.0 * 2.355,
        y_fwhm_free=5.0 * 2.355,
        theta_free=0.0,
        phi_source=0.0,
        param_cov_xy=None,
        snr_local_normalized=10.0,
    )
    assert np.isnan(result["sigma_r_fit"])
    assert np.isnan(result["sigma_t_fit"])
    assert np.isclose(result["sigma_r_stat"], 3.0 / 10.0)
    assert np.isclose(result["sigma_t_stat"], 5.0 / 10.0)
    assert result["rho_rt"] == 0.0


def test_summarize_2d_gauss_fit_result_populates_rt_sigmas():
    size = 31
    image, center = _make_isotropic_gaussian(size=size, amplitude=100.0, sigma=1.5, noise=0.3)
    yx_center = (size // 2, size // 2)
    # Place candidate off-center so PA is nonzero
    yx_position = (size // 2 + 5, size // 2)
    fit_result = detection.fit_2d_gaussian(
        image,
        yx_position=yx_position,
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
    )
    summary = detection.summarize_2d_gauss_fit_result(
        fit_result, phi_source=0.0, snr_local_normalized=50.0,
    )
    for col in (
        "radial_sigma_stat",
        "tangential_sigma_stat",
        "rt_corr_stat",
        "radial_sigma_fit",
        "tangential_sigma_fit",
        "radial_sigma_cr",
        "tangential_sigma_cr",
    ):
        assert col in summary.columns
    assert np.isfinite(summary["radial_sigma_stat"].values[0])
    assert np.isfinite(summary["tangential_sigma_stat"].values[0])


def test_fit_planet_parameters_uses_raw_snr_position_for_all_three():
    """Fit A on the raw SNR image provides the centroid; Fits B (contrast)
    and C (norm-SNR) share that centroid exactly (clamped)."""
    size = 31
    contrast, center = _make_isotropic_gaussian(
        size=size, amplitude=1.0e-4, sigma=1.5, noise=1.0e-5
    )
    uncertainty = np.ones_like(contrast) * 1.0e-5
    snr = contrast / uncertainty
    detection_image = np.stack([contrast, uncertainty, snr], axis=0)

    yy, xx = np.mgrid[:size, :size]
    yx_center = (size // 2, size // 2)
    r = np.hypot(yy - yx_center[0], xx - yx_center[1])
    normalized_detection_image = snr / (1.0 + 0.05 * r)

    contrast_table = pd.DataFrame(
        {"sep (pix)": np.arange(size), "snr_normalization": np.ones(size)}
    )

    contrast_res, snr_res, norm_snr_res = detection.fit_planet_parameters(
        detection_image=detection_image,
        normalized_detection_image=normalized_detection_image,
        contrast_table=contrast_table,
        yx_position=(round(center[0]) + 4, round(center[1])),  # off-center → PA ≠ 0
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=15,
        fix_width=False,
        fix_orientation=False,
        phi_source=None,
    )
    assert np.isclose(contrast_res["x"].values[0], snr_res["x"].values[0])
    assert np.isclose(contrast_res["y"].values[0], snr_res["y"].values[0])
    assert np.isclose(norm_snr_res["x"].values[0], snr_res["x"].values[0])
    assert np.isclose(norm_snr_res["y"].values[0], snr_res["y"].values[0])
    # Fit A carries a finite LevMar-derived σ; the clamped fits fall back to the
    # CR floor (no position cov) with the finite SNR_local from Fit C.
    assert np.isfinite(snr_res["radial_sigma_stat"].values[0])
    assert np.isfinite(norm_snr_res["radial_sigma_stat"].values[0])


def _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1, amplitude=10.0):
    """Emulate one row of candidates_fit['snr_image'] after Task 5."""
    sep = np.hypot(x_rel, y_rel)
    pa = float(np.degrees(np.arctan2(-x_rel, y_rel)) % 360)
    return pd.DataFrame(
        [
            {
                "candidate_index": 0,
                "wavelength_index": 0,
                "x": 32.0 + x_rel,
                "y": 32.0 + y_rel,
                "x_relative": x_rel,
                "y_relative": y_rel,
                "separation": sep,
                "position_angle": pa,
                "amplitude": amplitude,
                "x_fwhm": 3.4,
                "y_fwhm": 3.4,
                "theta": np.pi / 2,
                "good_pixels": 40,
                "fwhm_area": 40.0,
                "good_fraction": 1.0,
                "amplitude_free": amplitude,
                "x_fwhm_free": 3.4,
                "y_fwhm_free": 3.4,
                "theta_free": np.pi / 2,
                "good_pixels_free": 40,
                "fwhm_area_free": 40.0,
                "good_fraction_free": 1.0,
                "radial_sigma_stat": radial_sigma,
                "tangential_sigma_stat": tangential_sigma,
                "rt_corr_stat": 0.0,
                "radial_sigma_fit": radial_sigma,
                "tangential_sigma_fit": tangential_sigma,
                "radial_sigma_cr": radial_sigma * 0.5,
                "tangential_sigma_cr": tangential_sigma * 0.5,
            }
        ]
    )


def test_combine_channels_single_channel_gets_finite_sigma():
    fit_row = _one_channel_snr_row(x_rel=8.0, y_rel=0.0)
    combined = detection._combine_channels_rt_frame(
        group_rows=fit_row,
        search_radius=15.0,
    )
    assert np.isfinite(combined["separation_sigma"].values[0])
    assert np.isfinite(combined["position_angle_sigma"].values[0])
    assert np.isnan(combined["chi2_red_radial"].values[0])
    assert combined["channels_above_threshold"].values[0] == 1


def test_combine_channels_two_agree_shrinks_sigma():
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
    )
    assert combined["radial_sigma_stat"].values[0] < 0.2
    assert combined["tangential_sigma_stat"].values[0] < 0.1
    assert combined["chi2_red_radial"].values[0] < 1.5
    assert combined["channels_above_threshold"].values[0] == 2


def test_combine_channels_two_disagree_inflates_sigma():
    """Two channels 1 px apart in x (radial at PA=90°): scale factor > 1 and
    combined radial σ inflated accordingly."""
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=8.0 + 1.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
    )
    assert combined["chi2_red_radial"].values[0] > 4.0
    assert combined["radial_sigma_stat"].values[0] > 0.2


def test_combine_channels_donor_is_highest_snr_row():
    """Non-position columns come from the highest-SNR channel, not row 0."""
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, amplitude=5.0)
    r1["theta_free"] = 1.0
    r2 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, amplitude=20.0)
    r2["theta_free"] = 2.0
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
        snr_values=np.array([5.0, 20.0]),
    )
    assert np.isclose(combined["theta_free"].values[0], 2.0)


def _fake_template_row(template_name, x_rel, y_rel, snr, sigma_r=0.2, sigma_t=0.1,
                       candidate_id=0, wavelength_index=0, wavelength=1.6):
    sep = float(np.hypot(x_rel, y_rel))
    pa = float(np.degrees(np.arctan2(-x_rel, y_rel)) % 360)
    return {
        "candidate_id": candidate_id,
        "template_name": template_name,
        "x": 32.0 + x_rel,
        "y": 32.0 + y_rel,
        "x_relative": x_rel,
        "y_relative": y_rel,
        "separation": sep,
        "position_angle": pa,
        "x_relative_sigma": sigma_r,  # accepting axis-aligned simplification
        "y_relative_sigma": sigma_t,
        "xy_relative_corr": 0.0,
        "separation_sigma": sigma_r,
        "position_angle_sigma": float(np.degrees(sigma_t / max(sep, 1e-6))),
        "radial_sigma_stat": sigma_r,
        "tangential_sigma_stat": sigma_t,
        "norm_snr_fit_free": snr,
        "peak_pixel_snr": snr - 0.5,
        "wavelength_index": wavelength_index,
        "wavelength": wavelength,
        "contrast": 1.0e-4,
        "uncertainty": 1.0e-5,
    }


def test_combine_templates_picks_highest_snr_and_reports_scatter():
    """Three templates detect the same source; best_template = highest SNR;
    template-scatter columns are finite; single-template groups get NaN
    scatter and disagreement=False."""
    frames = [
        pd.DataFrame([_fake_template_row("Tdwarf", 8.0, 0.0, snr=10.0)]),
        pd.DataFrame([_fake_template_row("Ldwarf", 8.05, 0.02, snr=8.0)]),
        pd.DataFrame([_fake_template_row("flat", 8.1, -0.03, snr=6.0)]),
    ]
    combined = detection._combine_templates_best_snr(
        per_template_tables=frames,
        search_radius=1.0,
    )
    assert combined["best_template"].values[0] == "Tdwarf"
    assert combined["n_templates_above_threshold"].values[0] == 3
    assert np.isfinite(combined["x_relative_sigma_template_scatter"].values[0])
    # Single-template case:
    combined_single = detection._combine_templates_best_snr(
        per_template_tables=[frames[0]], search_radius=1.0,
    )
    assert np.isnan(combined_single["x_relative_sigma_template_scatter"].values[0])
    assert bool(combined_single["astrometry_template_disagreement"].values[0]) is False


def test_combine_templates_preserves_all_wavelength_rows_of_winner():
    """The spectra output must retain every wavelength row of the winning
    template, not collapse to a single row per source (C3)."""
    tdwarf = pd.DataFrame(
        [
            _fake_template_row("Tdwarf", 8.0, 0.0, snr=10.0, wavelength_index=w, wavelength=1.0 + 0.1 * w)
            for w in range(5)
        ]
    )
    ldwarf = pd.DataFrame(
        [
            _fake_template_row("Ldwarf", 8.05, 0.02, snr=8.0, wavelength_index=w, wavelength=1.0 + 0.1 * w)
            for w in range(5)
        ]
    )
    combined = detection._combine_templates_best_snr(
        per_template_tables=[tdwarf, ldwarf], search_radius=1.0,
    )
    # Winner is Tdwarf; all 5 of its wavelength rows survive.
    assert len(combined) == 5
    assert set(combined["best_template"].unique()) == {"Tdwarf"}
    assert sorted(combined["wavelength_index"].tolist()) == [0, 1, 2, 3, 4]
    assert combined["n_templates_above_threshold"].nunique() == 1
    assert combined["n_templates_above_threshold"].values[0] == 2
