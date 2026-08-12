"""Unit tests for the astrometric-uncertainty additions to
trap.detection. See
docs/superpowers/specs/2026-07-23-trap-astrometry-uncertainty-design.md
for the mathematical justification of every assertion in this file.
"""

import numpy as np
import pandas as pd

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
    """An unfittable cutout reports `fit_ok=False` rather than raising.

    It used to raise, which aborted the whole target for one bad candidate.
    """
    size = 21
    image, _ = _make_isotropic_gaussian(size=size, amplitude=50.0, sigma=1.5, noise=0.5)
    image[9:12, 9:12] = np.nan
    yx_center = (size // 2, size // 2)
    result = detection.fit_2d_gaussian(
        image,
        yx_position=(10, 10),
        yx_center=yx_center,
        x_stddev=1.5,
        y_stddev=1.5,
        box_size=3,
    )
    assert result["fit_ok"] is False
    assert np.isnan(result["parameters"].amplitude.value)
    # The candidate position is echoed back so the row stays traceable.
    assert result["yx_fit_position_orig"] == (10.0, 10.0)


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
    """With demonstrated independence the formal 1/sum(1/sigma^2) shrinkage applies."""
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
        independent_channels=True,
    )
    assert combined["radial_sigma_stat"].values[0] < 0.2
    assert combined["tangential_sigma_stat"].values[0] < 0.1
    assert combined["chi2_red_radial"].values[0] < 1.5
    assert combined["channels_above_threshold"].values[0] == 2


def test_combine_channels_sigma_floored_at_best_channel_by_default():
    """Speckle-correlated channels must not buy a sqrt(n) sigma reduction.

    Two identical channels would formally shrink sigma by sqrt(2); by default the
    combined sigma is floored at the best contributing channel's sigma.
    """
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
    )
    assert np.isclose(combined["radial_sigma_stat"].values[0], 0.2)
    assert np.isclose(combined["tangential_sigma_stat"].values[0], 0.1)
    assert np.isclose(combined["separation_sigma"].values[0], 0.2)
    # position stays the inverse-variance mean; only sigma is floored
    assert np.isclose(combined["x_relative"].values[0], 8.0)


def test_combine_channels_floor_uses_best_not_worst_channel():
    """A precise channel combined with a poor one keeps the precise channel's sigma."""
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=1.0, tangential_sigma=0.5)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
    )
    assert np.isclose(combined["radial_sigma_stat"].values[0], 0.2)


def test_combine_channels_floor_does_not_suppress_disagreement_inflation():
    """The floor is a lower bound only: a chi2 > 1 scale-up must still survive."""
    r1 = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2 = _one_channel_snr_row(x_rel=9.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    r2["wavelength_index"] = 1
    r2["candidate_index"] = 1
    combined = detection._combine_channels_rt_frame(
        pd.concat([r1, r2], ignore_index=True),
        search_radius=15.0,
    )
    assert combined["radial_sigma_stat"].values[0] > 0.2


def test_combine_channels_single_channel_unaffected_by_floor():
    """n=1: the floor is the channel itself, so nothing changes."""
    fit_row = _one_channel_snr_row(x_rel=8.0, y_rel=0.0, radial_sigma=0.2, tangential_sigma=0.1)
    floored = detection._combine_channels_rt_frame(fit_row, search_radius=15.0)
    free = detection._combine_channels_rt_frame(
        fit_row, search_radius=15.0, independent_channels=True
    )
    assert np.isclose(
        floored["radial_sigma_stat"].values[0], free["radial_sigma_stat"].values[0]
    )


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


def _collapse_overall_row(candidate_id, x_rel, y_rel, wavelength_index, sr=0.34, st=0.28):
    sep = float(np.hypot(x_rel, y_rel))
    pa = float(np.degrees(np.arctan2(-x_rel, y_rel)) % 360)
    return {
        "candidate_id": candidate_id,
        "x": 80.0 + x_rel,
        "y": 80.0 + y_rel,
        "x_relative": x_rel,
        "y_relative": y_rel,
        "separation": sep,
        "position_angle": pa,
        "x_relative_sigma": sr,
        "y_relative_sigma": st,
        "xy_relative_corr": 0.1,
        "separation_sigma": sr,
        "position_angle_sigma": float(np.degrees(st / max(sep, 1e-6))),
        "radial_sigma_stat": sr,
        "tangential_sigma_stat": st,
        "channels_above_threshold": 1,
        "wavelength_index": wavelength_index,
        "norm_snr_fit_free": 6.4,
    }


def _per_channel_row(x_rel, y_rel, sr=0.58, st=0.20, channels=1):
    sep = float(np.hypot(x_rel, y_rel))
    pa = float(np.degrees(np.arctan2(-x_rel, y_rel)) % 360)
    return {
        "x": 80.0 + x_rel,
        "y": 80.0 + y_rel,
        "x_relative": x_rel,
        "y_relative": y_rel,
        "separation": sep,
        "position_angle": pa,
        "x_relative_sigma": sr,
        "y_relative_sigma": st,
        "xy_relative_corr": 0.0,
        "separation_sigma": sr,
        "position_angle_sigma": float(np.degrees(st / max(sep, 1e-6))),
        "radial_sigma_stat": sr,
        "tangential_sigma_stat": st,
        "channels_above_threshold": channels,
    }


def test_override_uses_per_channel_position_when_within_radius():
    """A per-channel detection near a source replaces the collapse position/σ
    on all of that source's wavelength rows, and flags astrometry_source."""
    overall = pd.DataFrame(
        [
            _collapse_overall_row(0, -9.17, -36.76, wavelength_index=0),
            _collapse_overall_row(0, -9.17, -36.76, wavelength_index=1),
        ]
    )
    per_channel = pd.DataFrame([_per_channel_row(-8.55, -36.12, channels=1)])
    out = detection._override_astrometry_from_per_channel(
        overall, per_channel, search_radius=3.0
    )
    assert (out["astrometry_source"] == "per_channel").all()
    assert np.allclose(out["x_relative"], -8.55)
    assert np.allclose(out["y_relative"], -36.12)
    assert np.allclose(out["radial_sigma_stat"], 0.58)
    # detection significance is left untouched
    assert np.allclose(out["norm_snr_fit_free"], 6.4)


def test_override_falls_back_to_collapse_when_no_per_channel():
    overall = pd.DataFrame([_collapse_overall_row(0, -9.17, -36.76, wavelength_index=0)])
    out = detection._override_astrometry_from_per_channel(
        overall, None, search_radius=3.0
    )
    assert (out["astrometry_source"] == "collapse").all()
    assert np.allclose(out["x_relative"], -9.17)


def test_override_ignores_per_channel_source_outside_radius():
    overall = pd.DataFrame([_collapse_overall_row(0, -9.17, -36.76, wavelength_index=0)])
    far = pd.DataFrame([_per_channel_row(20.0, 20.0)])
    out = detection._override_astrometry_from_per_channel(
        overall, far, search_radius=3.0
    )
    assert (out["astrometry_source"] == "collapse").all()
    assert np.allclose(out["x_relative"], -9.17)


def test_override_rejected_when_too_few_channels_contribute():
    """51 Eri IFS: 2 of 37 channels clear threshold, so the collapse is kept.

    The per-channel position there is a noise-selected subset that discards ~95%
    of the signal and lands further from the interferometric truth than the
    collapse; the coverage gate is what stops it being reported.
    """
    overall = pd.DataFrame([_collapse_overall_row(0, -14.40, -58.86, wavelength_index=0)])
    per_channel = pd.DataFrame([_per_channel_row(-13.24, -58.53, channels=2)])
    out = detection._override_astrometry_from_per_channel(
        overall, per_channel, search_radius=3.0, n_channels_total=37
    )
    assert (out["astrometry_source"] == "collapse").all()
    assert np.allclose(out["x_relative"], -14.40)


def test_override_accepted_for_dbi_single_good_channel():
    """51 Eri IRDIS: 1 of 2 channels is half the data and clears the gate."""
    overall = pd.DataFrame([_collapse_overall_row(0, -9.17, -36.76, wavelength_index=0)])
    per_channel = pd.DataFrame([_per_channel_row(-8.55, -36.12, channels=1)])
    out = detection._override_astrometry_from_per_channel(
        overall, per_channel, search_radius=3.0, n_channels_total=2
    )
    assert (out["astrometry_source"] == "per_channel").all()
    assert np.allclose(out["x_relative"], -8.55)


def test_override_gate_disabled_without_n_channels_total():
    """Callers that don't say how many channels were reduced keep the old behaviour."""
    overall = pd.DataFrame([_collapse_overall_row(0, -14.40, -58.86, wavelength_index=0)])
    per_channel = pd.DataFrame([_per_channel_row(-13.24, -58.53, channels=2)])
    out = detection._override_astrometry_from_per_channel(
        overall, per_channel, search_radius=3.0
    )
    assert (out["astrometry_source"] == "per_channel").all()


def test_override_gate_fraction_is_configurable():
    overall = pd.DataFrame([_collapse_overall_row(0, -14.40, -58.86, wavelength_index=0)])
    per_channel = pd.DataFrame([_per_channel_row(-13.24, -58.53, channels=2)])
    out = detection._override_astrometry_from_per_channel(
        overall, per_channel, search_radius=3.0,
        n_channels_total=37, min_channel_fraction=0.05,
    )
    assert (out["astrometry_source"] == "per_channel").all()


class _StubTemplate:
    """Stand-in for a SpectralTemplate carrying only the two attributes the
    cross-template combination reads."""

    def __init__(self, companion_table, validated_companion_table):
        self.companion_table = companion_table
        self.validated_companion_table = validated_companion_table


def _fake_template_table(template_name, x_rel, y_rel, snr, candidate_id=0, n_wave=2):
    rows = [
        _fake_template_row(
            template_name,
            x_rel,
            y_rel,
            snr,
            candidate_id=candidate_id,
            wavelength_index=w,
            wavelength=2.11 + 0.14 * w,
        )
        for w in range(n_wave)
    ]
    return pd.DataFrame(rows)


def test_combine_template_tables_ignores_stale_on_disk_files(tmp_path):
    """The cross-template combination must use the in-memory per-template tables
    from the current run, never re-read per-template CSVs from disk. A stale
    file left by a previous run (a template that found nothing this run keeps
    its old file) would otherwise be ingested and contaminate the combination.
    """
    from collections import OrderedDict
    from types import SimpleNamespace

    analysis = detection.DetectionAnalysis.__new__(detection.DetectionAnalysis)
    analysis.reduction_parameters = SimpleNamespace(result_folder=str(tmp_path))
    # This run: only T-type detected; flat found nothing (table is None).
    tt = _fake_template_table("T-type", 8.0, 0.0, snr=10.0)
    analysis.templates = OrderedDict()
    analysis.templates["T-type"] = _StubTemplate(tt, tt)
    analysis.templates["flat"] = _StubTemplate(None, None)

    matching_dir = tmp_path / "template_matching"
    matching_dir.mkdir()
    # A STALE flat detection from a previous run, at a different position and a
    # deceptively high SNR. It must NOT enter the combination.
    stale = _fake_template_table("flat", 20.0, 20.0, snr=99.0)
    stale.to_csv(matching_dir / "validated_companion_table_flat.csv", index=False)

    analysis.combine_template_matched_companion_tables(
        search_radius=3.0, validated_only=True
    )

    overall = pd.read_csv(matching_dir / "overall_validated_companion_detections.csv")
    assert set(overall["best_template"].unique()) == {"T-type"}
    assert int(overall["n_templates_above_threshold"].max()) == 1
    assert not overall["astrometry_template_disagreement"].any()


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
