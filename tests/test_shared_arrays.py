import numpy as np
import pytest

from trap.parameters import resolve_scratch_dir
from trap.shared_arrays import SharedArrayStore, resolve


def test_dump_and_resolve_roundtrip(tmp_path):
    array = np.arange(24, dtype="float64").reshape(2, 3, 4)
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        ref = store.dump("data", array)
        loaded = resolve(ref)
        np.testing.assert_array_equal(loaded, array)
        assert isinstance(loaded, np.memmap)


def test_resolved_memmap_is_read_only(tmp_path):
    array = np.zeros((4, 4))
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        loaded = resolve(store.dump("data", array))
        with pytest.raises(ValueError):
            loaded[0, 0] = 1.0


def test_ref_slicing_returns_view(tmp_path):
    array = np.arange(60, dtype="float64").reshape(3, 4, 5)
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        ref = store.dump("data", array)
        sliced = resolve(ref[np.s_[1, :, 1:3]])
        np.testing.assert_array_equal(sliced, array[1, :, 1:3])
        with pytest.raises(ValueError):
            _ = ref[0][0]  # only one index expression allowed


def test_resolve_passes_through_plain_objects():
    array = np.ones(3)
    assert resolve(array) is array
    assert resolve(None) is None


def test_refs_are_picklable(tmp_path):
    import pickle

    array = np.arange(10, dtype="float64")
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        ref = store.dump("data", array)[np.s_[2:5]]
        restored = pickle.loads(pickle.dumps(ref))
        np.testing.assert_array_equal(resolve(restored), array[2:5])


def test_create_fills_incrementally(tmp_path):
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        memmap = store.create("data", shape=(3, 4), dtype="float64")
        for i in range(3):
            memmap[i] = i
        memmap.flush()
        loaded = resolve(store.ref("data"))
        np.testing.assert_array_equal(loaded, np.repeat(np.arange(3.0), 4).reshape(3, 4))


def test_ref_of_missing_array_raises(tmp_path):
    with SharedArrayStore(scratch_dir=tmp_path) as store:
        with pytest.raises(KeyError):
            store.ref("missing")


def test_cleanup_removes_directory(tmp_path):
    store = SharedArrayStore(scratch_dir=tmp_path)
    store.dump("data", np.zeros(3))
    directory = store.directory
    assert directory.exists()
    store.cleanup()
    assert not directory.exists()


def test_resolve_scratch_dir_explicit_wins(tmp_path):
    assert resolve_scratch_dir(scratch_dir=tmp_path) == tmp_path


def test_resolve_scratch_dir_returns_existing_directory():
    resolved = resolve_scratch_dir(required_bytes=1024)
    assert resolved.is_dir()


def test_resolve_scratch_dir_huge_request_avoids_dev_shm():
    # A store larger than any /dev/shm must fall back to the temp dir.
    resolved = resolve_scratch_dir(required_bytes=10**18)
    assert str(resolved) != "/dev/shm"
