import numpy as np
import pytest

from trap.makesource import (
    coronagraph_throughput_factor,
    coronagraph_transmission_to_pixels,
)
from trap.parameters import TrapReductionConfig, build_runtime_state


def test_to_pixels_converts_and_sorts():
    # 10 mas/pixel: 100 mas -> 10 pix, 50 mas -> 5 pix; input unsorted
    table_mas = np.array([[100.0, 1.0], [50.0, 0.5], [0.0, 0.0]])
    out = coronagraph_transmission_to_pixels(table_mas, mas_per_pixel=10.0)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 10.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 0.5, 1.0])


def test_to_pixels_accepts_pair():
    sep_mas = np.array([0.0, 50.0, 100.0])
    throughput = np.array([0.0, 0.5, 1.0])
    out = coronagraph_transmission_to_pixels((sep_mas, throughput), mas_per_pixel=10.0)
    np.testing.assert_allclose(out[:, 0], [0.0, 5.0, 10.0])


def test_to_pixels_warns_when_last_not_one():
    table_mas = np.array([[0.0, 0.0], [100.0, 0.8]])
    with pytest.warns(UserWarning):
        coronagraph_transmission_to_pixels(table_mas, mas_per_pixel=10.0)


def test_factor_interpolates():
    table_pix = np.array([[0.0, 0.0], [5.0, 0.5], [10.0, 1.0]])
    assert coronagraph_throughput_factor(2.5, table_pix) == pytest.approx(0.25)


def test_factor_beyond_table_is_one():
    table_pix = np.array([[0.0, 0.0], [10.0, 1.0]])
    assert coronagraph_throughput_factor(50.0, table_pix) == pytest.approx(1.0)


def test_factor_below_table_clamps_to_first():
    table_pix = np.array([[3.0, 0.2], [10.0, 1.0]])
    assert coronagraph_throughput_factor(0.0, table_pix) == pytest.approx(0.2)


def _minimal_runtime(coronagraph_transmission, mas_per_pixel):
    config = TrapReductionConfig(
        search_region_outer_bound=20,
        data_auto_crop=False,
        data_crop_size=101,
        coronagraph_transmission=coronagraph_transmission,
    )
    return build_runtime_state(
        config=config,
        data_shape=(1, 4, 101, 101),
        stamp_sizes=np.array([21]),
        stamp_sizes_reduction=np.array([19]),
        max_shift=0.0,
        mas_per_pixel=mas_per_pixel,
    )


def test_runtime_builds_pixel_table():
    table_mas = np.array([[0.0, 0.0], [50.0, 0.5], [100.0, 1.0]])
    runtime = _minimal_runtime(table_mas, mas_per_pixel=10.0)
    assert runtime.coronagraph_transmission_pix is not None
    np.testing.assert_allclose(runtime.coronagraph_transmission_pix[:, 0], [0.0, 5.0, 10.0])


def test_runtime_table_none_by_default():
    runtime = _minimal_runtime(None, mas_per_pixel=10.0)
    assert runtime.coronagraph_transmission_pix is None


def test_to_pixels_two_point_pair():
    out = coronagraph_transmission_to_pixels(
        (np.array([0.0, 100.0]), np.array([0.0, 1.0])), mas_per_pixel=10.0
    )
    np.testing.assert_allclose(out[:, 0], [0.0, 10.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0])


def test_to_pixels_clips_and_warns_out_of_range():
    table_mas = np.array([[0.0, 0.0], [100.0, 1.5]])
    with pytest.warns(UserWarning):
        out = coronagraph_transmission_to_pixels(table_mas, mas_per_pixel=10.0)
    assert out[-1, 1] == pytest.approx(1.0)


def test_runtime_requires_mas_per_pixel():
    table_mas = np.array([[0.0, 0.0], [50.0, 0.5], [100.0, 1.0]])
    with pytest.raises(ValueError):
        _minimal_runtime(table_mas, mas_per_pixel=None)


def test_to_reduction_parameters_with_transmission():
    config = TrapReductionConfig(
        coronagraph_transmission=np.array([[0.0, 0.0], [100.0, 1.0]])
    )
    with pytest.warns(DeprecationWarning):
        config.to_reduction_parameters()


def test_amplitude_scaling_does_not_mutate_shared_array():
    # Mirrors the trap_one_position logic: amplitude_modulation comes from
    # ray.put and is shared read-only across positions.
    table_pix = np.array([[0.0, 0.0], [5.0, 0.5], [10.0, 1.0]])
    shared = np.ones(4)
    signal_position = np.array([0.0, 2.5])  # |pos| = 2.5 px -> throughput 0.25

    separation_pix = np.hypot(signal_position[0], signal_position[1])
    factor = coronagraph_throughput_factor(separation_pix, table_pix)
    scaled = shared * factor

    assert factor == pytest.approx(0.25)
    np.testing.assert_allclose(scaled, np.full(4, 0.25))
    np.testing.assert_allclose(shared, np.ones(4))  # shared untouched
