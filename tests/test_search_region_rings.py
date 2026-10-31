"""Search region made of several rings (``TrapReductionConfig.search_region_bounds``).

Systems with companions at very different separations, such as HR 8799, only need
the separations around each companion reduced. One annulus spanning all of them
also reduces the empty separations in between, and the outermost ones cost most.
"""

import numpy as np
import pytest

from trap.parameters import TrapReductionConfig, build_runtime_state

DATA_SHAPE = (1, 10, 201, 201)


def _runtime_state(config):
    return build_runtime_state(
        config=config,
        data_shape=DATA_SHAPE,
        stamp_sizes=np.array([19]),
        stamp_sizes_reduction=np.array([19]),
        max_shift=0.0,
    )


def _separation_map(search_region):
    yy, xx = np.indices(search_region.shape)
    cy, cx = search_region.shape[0] // 2, search_region.shape[1] // 2
    return np.hypot(yy - cy, xx - cx)


def test_rings_cover_only_their_separations():
    config = TrapReductionConfig(search_region_bounds=((10, 15), (30, 35)))

    region = _runtime_state(config).search_region
    separation = _separation_map(region)

    assert region[(separation >= 10) & (separation <= 15)].all()
    assert region[(separation >= 30) & (separation <= 35)].all()
    assert not region[(separation > 16) & (separation < 29)].any()
    assert not region[separation > 36].any()
    assert not region[separation < 9].any()


def test_outermost_ring_sets_the_outer_bound_and_crop():
    rings = _runtime_state(TrapReductionConfig(search_region_bounds=((10, 15), (30, 35))))
    annulus = _runtime_state(TrapReductionConfig(search_region_inner_bound=10, search_region_outer_bound=35))

    assert rings.search_region_outer_bound == 35
    assert rings.data_crop_size == annulus.data_crop_size


def test_ring_order_does_not_matter():
    forward = _runtime_state(TrapReductionConfig(search_region_bounds=((10, 15), (30, 35))))
    backward = _runtime_state(TrapReductionConfig(search_region_bounds=((30, 35), (10, 15))))

    np.testing.assert_array_equal(forward.search_region, backward.search_region)


@pytest.mark.parametrize("bounds", [((15, 10),), ((10, 10),), ((-1, 5),), ()])
def test_invalid_rings_are_rejected(bounds):
    with pytest.raises(ValueError, match="search_region_bounds"):
        _runtime_state(TrapReductionConfig(search_region_bounds=bounds))


def test_rings_and_explicit_mask_are_exclusive():
    mask = np.ones(DATA_SHAPE[-2:], dtype=bool)
    config = TrapReductionConfig(search_region=mask, search_region_bounds=((10, 15),), data_auto_crop=False)

    with pytest.raises(ValueError, match="search_region_bounds"):
        _runtime_state(config)


def test_without_rings_the_single_annulus_is_unchanged():
    config = TrapReductionConfig(search_region_inner_bound=10, search_region_outer_bound=35)

    region = _runtime_state(config).search_region
    separation = _separation_map(region)

    assert region[(separation >= 10) & (separation <= 35)].all()
    assert not region[(separation < 9) | (separation > 36)].any()
