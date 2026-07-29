"""Regression tests for outputs that must not outlive the run that wrote them.

Two flavours of the same defect. The stacked detection cube is derived from the
per-wavelength ``detection_lam*`` files, so it must follow them across
re-reductions rather than staying frozen at the first run. The companion tables
and their plots are written on the success path only, so a re-run in which a
template finds nothing must remove the previous run's copies instead of leaving
them beside freshly written detection maps.
"""

import numpy as np
import pytest
from astropy.io import fits

from trap.detection import DetectionAnalysis
from trap.parameters import TrapReductionConfig

COMPONENT_FRACTION = 0.15
STACKED_NAME = f"detection_ncomp038_frac{COMPONENT_FRACTION:.2f}_temporal.fits"

TEMPLATE_NAME = "L-type"
STALE_TEMPLATE_FILES = [
    f"companion_table_{TEMPLATE_NAME}.csv",
    f"validated_companion_table_{TEMPLATE_NAME}.csv",
    f"validated_companion_table_short_{TEMPLATE_NAME}.csv",
    f"companion_spectra_{TEMPLATE_NAME}.pdf",
    f"contrast_plot_{TEMPLATE_NAME}.pdf",
    f"contrast_plot_{TEMPLATE_NAME}.png",
]
# Written before the candidate search, so they are this run's products even when
# the template finds nothing, and must survive the purge.
SURVIVING_TEMPLATE_FILES = [
    f"contrast_table_{TEMPLATE_NAME}.csv",
    f"normalized_detection_image_{TEMPLATE_NAME}.fits",
    f"uncertainty_image_{TEMPLATE_NAME}.fits",
]
OVERALL_FILES = [
    "overall_companion_detections.csv",
    "overall_companion_detections_spectra.csv",
    "overall_validated_companion_detections.csv",
    "overall_validated_companion_detections_spectra.csv",
]


class _StubInstrument:
    """read_output only ever calls compute_fwhm() on the instrument."""

    wavelengths = np.array([1.0, 1.1])

    def compute_fwhm(self):
        pass


class _StubTemplate:
    def __init__(self, name):
        self.name = name


def _write_per_wavelength_images(folder, values):
    for index, value in enumerate(values):
        fits.writeto(
            folder / f"detection_lam{index:02d}_ncomp038_frac{COMPONENT_FRACTION:.2f}_temporal.fits",
            np.full((2, 4, 4), value, dtype="float32"),
            overwrite=True,
        )


def _read_output(folder):
    analysis = DetectionAnalysis()
    analysis.read_output(
        COMPONENT_FRACTION,
        result_folder=str(folder),
        read_parameters=False,
        read_instrument=False,
        reduction_parameters=TrapReductionConfig(),
        instrument=_StubInstrument(),
    )
    return analysis


def test_read_output_writes_stacked_detection_image(tmp_path):
    _write_per_wavelength_images(tmp_path, [1.0, 2.0])

    analysis = _read_output(tmp_path)

    stacked = tmp_path / STACKED_NAME
    assert stacked.exists()
    assert np.allclose(fits.getdata(stacked), analysis.detection_cube)


def test_read_output_refreshes_stale_stacked_detection_image(tmp_path):
    _write_per_wavelength_images(tmp_path, [1.0, 2.0])
    _read_output(tmp_path)

    _write_per_wavelength_images(tmp_path, [7.0, 8.0])
    analysis = _read_output(tmp_path)

    stacked_data = fits.getdata(tmp_path / STACKED_NAME)
    assert np.allclose(stacked_data, analysis.detection_cube)
    assert np.allclose(np.unique(stacked_data), [7.0, 8.0])


def _seed_previous_run(folder, filenames):
    folder.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (folder / filename).write_text("stale,from,a,previous,run\n")


def _analysis_for(tmp_path):
    analysis = DetectionAnalysis()
    analysis.reduction_parameters = TrapReductionConfig(result_folder=str(tmp_path))
    return analysis


def test_run_template_matching_removes_stale_tables_when_nothing_is_found(
    tmp_path, monkeypatch
):
    matching_dir = tmp_path / "template_matching"
    _seed_previous_run(matching_dir, STALE_TEMPLATE_FILES + SURVIVING_TEMPLATE_FILES)

    analysis = _analysis_for(tmp_path)
    analysis.instrument = _StubInstrument()
    analysis.wavelength_indices = np.array([0])
    monkeypatch.setattr(
        DetectionAnalysis,
        "template_matching_detection",
        lambda self, *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        DetectionAnalysis,
        "find_candidates_all_wavelengths",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DetectionAnalysis,
        "complete_candidate_table",
        lambda self, *args, **kwargs: (None, None),
    )
    template = _StubTemplate(TEMPLATE_NAME)

    analysis.run_template_matching(template=template)

    assert template.companion_table is None
    for filename in STALE_TEMPLATE_FILES:
        assert not (matching_dir / filename).exists(), filename
    for filename in SURVIVING_TEMPLATE_FILES:
        assert (matching_dir / filename).exists(), filename


@pytest.mark.parametrize(
    "validated_only, prefix", [(True, "validated_"), (False, "")]
)
def test_combine_removes_stale_overall_tables_when_nothing_is_found(
    tmp_path, validated_only, prefix
):
    matching_dir = tmp_path / "template_matching"
    _seed_previous_run(matching_dir, OVERALL_FILES)

    analysis = _analysis_for(tmp_path)
    analysis.templates = {}

    analysis.combine_template_matched_companion_tables(validated_only=validated_only)

    purged = {
        f"overall_{prefix}companion_detections.csv",
        f"overall_{prefix}companion_detections_spectra.csv",
    }
    for filename in purged:
        assert not (matching_dir / filename).exists(), filename
    # The other prefix belongs to the sibling call, not this one.
    for filename in set(OVERALL_FILES) - purged:
        assert (matching_dir / filename).exists(), filename
