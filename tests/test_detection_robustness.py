"""Regression tests for the detection failure modes seen on SPHERE/IRDIS survey runs.

Two production crashes motivated these: a bounded LevMar fit that returned NaN
parameters on speckle structure, and a candidate at 1 px separation whose own
companion mask blanked every annulus it needed. Both aborted the entire target
and, with it, the combined companion tables.
"""

import numpy as np
import pandas as pd
import pytest
from astropy.modeling import models

from trap import detection


def _speckle_cutout_image(size=41, seed=3):
    """Bipolar speckle-like structure with no Gaussian core to converge on."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    image = 12.0 * np.sin(x * 0.9) * np.cos(y * 0.7)
    image += 6.0 * np.sin((x + y) * 0.35)
    image += rng.normal(0.0, 1.0, (size, size))
    return image


class TestFitFailureIsNonFatal:
    def test_all_nan_cutout_returns_failed_result(self):
        image = np.full((31, 31), np.nan)
        result = detection.fit_2d_gaussian(
            image, yx_position=(15, 15), box_size=11, fix_width=False,
            fix_orientation=False,
        )
        assert result["fit_ok"] is False
        assert result["param_cov_xy"] is None
        assert not np.any(result["mask"])

    def test_speckle_structure_never_raises(self):
        """Fit A is fully free, which is what drove x_stddev into its bound."""
        image = _speckle_cutout_image()
        for yx in [(y, x) for y in range(8, 33, 3) for x in range(8, 33, 3)]:
            result = detection.fit_2d_gaussian(
                image, yx_position=yx, box_size=11,
                fix_width=False, fix_orientation=False,
            )
            assert "fit_ok" in result

    def test_clean_source_still_fits_accurately(self):
        """The fitter swap must not cost accuracy on a well-behaved source."""
        y, x = np.mgrid[:41, :41]
        image = models.Gaussian2D(50.0, 22.4, 18.6, 1.5, 2.4, 0.4)(x, y)
        result = detection.fit_2d_gaussian(
            image, yx_position=(19, 22), box_size=11,
            fix_width=False, fix_orientation=False,
        )
        assert result["fit_ok"] is True
        assert result["yx_fit_position_orig"][0] == pytest.approx(18.6, abs=0.05)
        assert result["yx_fit_position_orig"][1] == pytest.approx(22.4, abs=0.05)
        assert result["parameters"].amplitude.value == pytest.approx(50.0, rel=0.02)

    def test_edge_cutout_does_not_raise(self):
        """Cutout2D trims at the frame edge, so the model grid must follow it."""
        y, x = np.mgrid[:41, :41]
        image = models.Gaussian2D(50.0, 2.0, 2.0, 1.5, 1.5, 0.0)(x, y)
        result = detection.fit_2d_gaussian(
            image, yx_position=(2, 2), box_size=11,
            fix_width=False, fix_orientation=False,
        )
        assert "fit_ok" in result

    def test_failed_fit_summarizes_into_a_row(self):
        failed = detection._failed_gaussian_fit_result((30.0, 40.0), (25.0, 25.0), (11, 11))
        row = detection.summarize_2d_gauss_fit_result(failed)
        assert len(row) == 1
        assert row["fit_ok"].iloc[0] is np.False_ or row["fit_ok"].iloc[0] is False
        assert np.isnan(row["amplitude"].iloc[0])
        assert row["x"].iloc[0] == 40.0


class TestAnnulusFallback:
    def _fully_masked_inner_annuli(self, minimum_annulus_pixels):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (61, 61))
        companion_mask = np.hypot(*(np.mgrid[:61, :61] - 30)) < 11
        profile, _ = detection.make_radial_profile(
            data, (1, 20), bin_width=3.0, operation="mad_std",
            known_companion_mask=companion_mask,
            minimum_annulus_pixels=minimum_annulus_pixels,
        )
        radius = np.hypot(*(np.mgrid[:61, :61] - 30))
        inner = (radius >= 2.5) & (radius < 3.5)
        return profile[inner]

    def test_fallback_keeps_the_annulus_finite(self):
        assert np.all(np.isfinite(self._fully_masked_inner_annuli(10)))

    def test_disabling_the_fallback_restores_nan(self):
        assert np.all(np.isnan(self._fully_masked_inner_annuli(0)))


class TestAdaptiveCompanionMask:
    @pytest.mark.parametrize(
        "separation,expected",
        [(1.0, 3.0), (5.0, 4.0), (8.0, 7.0), (12.0, 11.0), (40.0, 11.0)],
    )
    def test_radius_is_capped_by_separation_and_floored_by_the_psf(
        self, separation, expected
    ):
        radius = detection._adaptive_companion_mask_radius(
            (separation, 0.0), companion_mask_radius=11, minimum_companion_mask_radius=3.0
        )
        assert radius == pytest.approx(expected)

    def test_close_companion_no_longer_blanks_its_own_annuli(self):
        rng = np.random.default_rng(1)
        detection_image = np.stack(
            [rng.normal(0, 1, (61, 61)) for _ in range(3)]
        )
        normalized, _, _, _ = detection.make_contrast_curve(
            detection_image, bin_width=3.0, companion_mask_radius=11,
            yx_known_companion_position=np.array([[-5.0, 0.0]]),
        )
        radius = np.hypot(*(np.mgrid[:61, :61] - 30))
        inner = (radius >= 2.5) & (radius < 8.5)
        assert np.any(np.isfinite(normalized[inner]))


class _PeakFinder(detection.DetectionAnalysis):
    def __init__(self):
        pass


class TestExclusionRadiusScaling:
    def _bright_source_with_wing_blobs(self):
        """A 100σ source plus detached blobs, as a real binary produces."""
        y, x = np.mgrid[:121, :121]
        image = np.zeros((121, 121))
        image += models.Gaussian2D(100.0, 60.0, 40.0, 1.5, 1.5, 0.0)(x, y)
        for dy, dx in [(9, 6), (-8, 11), (14, -5), (-13, -9), (7, 17)]:
            image += models.Gaussian2D(7.0, 60.0 + dx, 40.0 + dy, 1.2, 1.2, 0.0)(x, y)
        # A faint, genuinely separate source far away must survive.
        image += models.Gaussian2D(6.0, 20.0, 100.0, 1.4, 1.4, 0.0)(x, y)
        return image

    def test_scaling_absorbs_the_wing_blobs(self):
        finder = _PeakFinder()
        unscaled = finder.find_approximate_candidate_positions(
            self._bright_source_with_wing_blobs(), candidate_threshold=4.75,
            mask_radius=11, exclusion_radius_snr_scaling=False,
            mask_connected_region=False,
        )
        scaled = finder.find_approximate_candidate_positions(
            self._bright_source_with_wing_blobs(), candidate_threshold=4.75,
            mask_radius=11, exclusion_radius_snr_scaling=True,
        )
        assert len(scaled) < len(unscaled)
        # The distant faint source is not swallowed by the bright one's mask.
        assert np.any(scaled["snr"] < 10.0)

    def test_marginal_peaks_keep_the_base_radius(self):
        radius = detection._scaled_exclusion_radius(
            5.0, candidate_threshold=4.75, mask_radius=11, max_factor=2.5
        )
        assert radius == pytest.approx(11.0, rel=0.05)

    def test_scaling_is_capped(self):
        radius = detection._scaled_exclusion_radius(
            10000.0, candidate_threshold=4.75, mask_radius=11, max_factor=2.5
        )
        assert radius == pytest.approx(27.5)

    def test_scaling_can_be_disabled(self):
        radius = detection._scaled_exclusion_radius(
            10000.0, candidate_threshold=4.75, mask_radius=11, max_factor=2.5,
            enabled=False,
        )
        assert radius == 11


class TestCandidateCap:
    def test_saturated_map_is_truncated(self):
        rng = np.random.default_rng(5)
        image = rng.uniform(6.0, 9.0, (201, 201))
        finder = _PeakFinder()
        candidates = finder.find_approximate_candidate_positions(
            image, candidate_threshold=4.75, mask_radius=3, max_candidates=7,
            exclusion_radius_snr_scaling=False, mask_connected_region=False,
        )
        assert len(candidates) == 7


class TestExclusionRadiusIsDecoupled:
    def test_explicit_exclusion_radius_overrides_search_radius(self):
        assert detection._resolve_exclusion_radius(25, 11) == 25

    def test_none_falls_back_to_search_radius(self):
        assert detection._resolve_exclusion_radius(None, 11) == 11


class _CandidateFinder(detection.DetectionAnalysis):
    def __init__(self, normalized_image):
        self.wavelength_indices = np.array([0])
        self.detection_products = {
            "normalized_detection_cube": np.array([normalized_image]),
            "contrast_tables": [
                pd.DataFrame(
                    {
                        "sep (pix)": np.arange(1, 40, dtype=float),
                        "snr_normalization": np.ones(39),
                    }
                )
            ],
        }


class TestMinimumCandidateSeparation:
    def _image_with_a_central_and_an_outer_peak(self):
        image = np.zeros((81, 81))
        y, x = np.mgrid[:81, :81]
        # One residual 2 px from the star, one genuine source at 25 px.
        image += models.Gaussian2D(9.0, 42.0, 40.0, 1.4, 1.4, 0.0)(x, y)
        image += models.Gaussian2D(9.0, 65.0, 40.0, 1.4, 1.4, 0.0)(x, y)
        return image

    def test_central_residual_is_dropped(self):
        finder = _CandidateFinder(self._image_with_a_central_and_an_outer_peak())
        candidates = finder.find_candidates(
            candidate_threshold=4.75, iterative_search_exclusion_radius=11,
            minimum_candidate_separation=5.0,
        )
        assert len(candidates) == 1
        assert candidates["separation"].iloc[0] == pytest.approx(25.0, abs=1.0)

    def test_zero_floor_keeps_it(self):
        """Guards against the test passing for an unrelated reason."""
        finder = _CandidateFinder(self._image_with_a_central_and_an_outer_peak())
        candidates = finder.find_candidates(
            candidate_threshold=4.75, iterative_search_exclusion_radius=11,
            minimum_candidate_separation=0.0,
        )
        assert len(candidates) == 2


class _FailingTemplates(detection.DetectionAnalysis):
    """Two templates where the first raises, mimicking the T-type failure."""

    def __init__(self):
        self.templates = {"bad": _Template("bad"), "good": _Template("good")}
        self.visited = []

    def run_template_matching(self, template=None, **kwargs):
        self.visited.append(template.name)
        if template.name == "bad":
            raise RuntimeError("template matching blew up")
        template.companion_table = "result"


class _Template:
    def __init__(self, name):
        self.name = name
        self.companion_table = None
        self.validated_companion_table = None
        self.validated_companion_table_short = None


class TestTemplateErrorIsolation:
    def test_a_failing_template_does_not_stop_the_others(self):
        analysis = _FailingTemplates()
        analysis.match_all_templates()
        assert analysis.visited == ["bad", "good"]
        assert analysis.templates["good"].companion_table == "result"
        assert analysis.templates["bad"].companion_table is None
