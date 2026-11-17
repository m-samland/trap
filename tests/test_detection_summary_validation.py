import numpy as np
import pandas as pd

from trap.detection import find_bright_sources, validate_candidates

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


def _pair(y, x, snr=341.0):
    # A 7700 sigma companion and a second row, positions taken from CD-35 2722 b in K12.
    return pd.concat(
        [_table(snr=7728.0, y_relative=-114.8, x_relative=214.8, separation=243.6),
         _table(snr=snr, y_relative=y, x_relative=x, separation=float((y**2 + x**2) ** 0.5))],
        ignore_index=True,
    )


def _sources(table):
    bright = table[table["snr"] >= 50.0]
    return bright[["y_relative", "x_relative", "snr"]].to_numpy(dtype=float)


def _validate_pair(table, known=None, rotation=0.3, arc_along_fwhm=8.0, sources=None):
    return validate_candidates(
        table, 4.5, 0.05, 25.0, [1.1, 4.5], 50.0, known, rotation, FWHM,
        arc_along_fwhm=arc_along_fwhm, bright_sources=_sources(table) if sources is None else sources,
    )


def test_residual_on_bright_arc_is_rejected():
    out = _validate_pair(_pair(-129.3, 206.9))
    assert list(out) == ["shape_skipped_bright", "on_brighter_arc"]


def test_faint_residual_on_bright_arc_is_rejected():
    # The 12.6 sigma arc candidate of CD-35 2722 b on 2015-11-28 passes the shape tests.
    assert _validate_pair(_pair(-164.7, 177.9, snr=12.6))[1] == "on_brighter_arc"


def test_negative_bright_source_vetoes_its_positive_lobe():
    # In the T-type map CD-35 2722 b is a -900 sigma minimum, so only its 341 sigma lobe is a row.
    lobe = _pair(-129.3, 206.9).iloc[[1]]
    out = _validate_pair(lobe, sources=np.array([[-114.8, 214.8, 900.0]]))
    assert list(out) == ["on_brighter_arc"]


def test_source_at_other_separation_is_not_on_arc():
    assert _validate_pair(_pair(-60.0, 100.0, snr=60.0))[1] == "shape_skipped_bright"


def test_source_beyond_the_arc_is_not_on_arc():
    assert _validate_pair(_pair(114.8, -214.8))[1] == "shape_skipped_bright"


def test_configured_source_on_arc_is_kept():
    assert _validate_pair(_pair(-129.3, 206.9), known=[[-129.3, 206.9]])[1] == "shape_skipped_bright"


def test_arc_test_off_without_margin():
    assert _validate_pair(_pair(-129.3, 206.9), arc_along_fwhm=None)[1] == "shape_skipped_bright"


def test_arc_test_off_without_sources():
    assert _validate_pair(_pair(-129.3, 206.9), sources=np.empty((0, 3)))[1] == "shape_skipped_bright"


def test_candidate_is_not_vetoed_by_its_own_peak():
    table = _pair(-129.3, 206.9, snr=300.0)
    table["candidate_id"] = [0, 0]
    table.loc[0, ["y_relative", "x_relative"]] = [-129.3, 206.9]
    sources = np.array([[-129.3, 206.9, 400.0]])
    assert list(_validate_pair(table, sources=sources)) == ["shape_skipped_bright", "shape_skipped_bright"]


def test_bright_sources_finds_both_signs_above_threshold():
    snr_map = np.zeros((101, 101))
    snr_map[50 + 30, 50 + 10] = -900.0
    snr_map[50 - 20, 50 + 25] = 120.0
    snr_map[50 + 5, 50 - 35] = 12.0
    snr_map[50 + 1, 50 + 1] = 5000.0
    snr_map[0, 0] = np.nan
    found = find_bright_sources(snr_map, 50.0, FWHM, minimum_separation=5.0)
    assert sorted(map(tuple, found.tolist())) == [(-20.0, 25.0, 120.0), (30.0, 10.0, 900.0)]


def test_bright_sources_merges_maps():
    maps = np.zeros((2, 41, 41))
    maps[0, 20, 35] = 80.0
    maps[1, 5, 20] = -60.0
    assert len(find_bright_sources(maps, 50.0, FWHM)) == 2
