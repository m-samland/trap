import pandas as pd

from trap.detection import validate_candidates

FWHM = 4.0


def _table(**row):
    base = dict(
        snr=22.0,
        good_fraction_free=0.9,
        theta_deviation=5.0,
        yx_fwhm_ratio=2.0,
        separation=48.0,
        y_relative=30.0,
        x_relative=37.0,
    )
    base.update(row)
    return pd.DataFrame([base])


def _validate(table, known=None, rotation=1.0):
    return validate_candidates(table, 4.5, 0.05, 25.0, [1.1, 4.5], 50.0, known, rotation, FWHM)[0]


def test_bright_round_source_passes():
    assert _validate(_table(snr=837.0, yx_fwhm_ratio=0.72)) == "shape_skipped_bright"


def test_known_position_passes():
    table = _table(yx_fwhm_ratio=0.99)
    assert _validate(table, known=[[31.0, 38.0]]) == "shape_skipped_known"


def test_short_arc_ignores_orientation():
    table = _table(theta_deviation=42.0, yx_fwhm_ratio=1.46)
    assert _validate(table, rotation=0.004) == "orientation_skipped_short_arc"


def test_elongated_noise_blob_still_fails_ratio():
    assert _validate(_table(snr=7.0, yx_fwhm_ratio=5.0)) == "yx_fwhm_ratio"


def test_bright_source_still_needs_good_fraction():
    assert _validate(_table(snr=837.0, good_fraction_free=0.0)) == "good_fraction"


def test_first_failing_test_is_named():
    assert _validate(_table(snr=1.0, theta_deviation=90.0, yx_fwhm_ratio=9.0)) == "snr"


def test_without_fwhm_nothing_is_skipped():
    out = validate_candidates(_table(snr=837.0, yx_fwhm_ratio=0.72), 4.5, 0.05, 25.0, [1.1, 4.5])
    assert out[0] == "yx_fwhm_ratio"
