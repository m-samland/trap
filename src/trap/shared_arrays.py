"""Shared read-only array store for multiprocessing.

Large input arrays are dumped once as ``.npy`` files to a scratch directory
(``/dev/shm`` on cluster nodes, the system temp dir otherwise, see
`trap.parameters.resolve_scratch_dir`). Worker processes open them with
``np.load(mmap_mode="r")``, so the OS page cache provides a single shared
in-RAM copy per node instead of one copy per process.

Workers never receive the arrays themselves, only lightweight picklable
references (`SharedArrayRef`, optionally sliced via ``ref[key]``). Call
`resolve` on a received argument to obtain the (read-only) array; plain
arrays pass through unchanged, so the same worker code serves both the
serial and the parallel path.

BLAS thread capping in workers is handled by joblib's loky backend
(``parallel_config(inner_max_num_threads=1)`` at the dispatch sites), which
sets the thread-limit environment variables before the worker processes
load their BLAS.

@author: Matthias Samland
         MPIA Heidelberg
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from trap.parameters import resolve_scratch_dir

__all__ = ["SharedArrayRef", "SharedArrayStore", "resolve"]

# Per-process cache of opened memmaps, keyed by file path. Store directories
# are unique per store (mkdtemp), so a path never refers to two different
# arrays within one reduction.
_MEMMAP_CACHE: dict = {}


@dataclass(frozen=True)
class SharedArrayRef:
    """Picklable reference to an array stored in a `SharedArrayStore`.

    Optionally carries an index expression (``ref[key]``) that is applied
    after memmapping, so per-wavelength crops travel as index bounds rather
    than array copies.
    """

    path: str
    key: Optional[Any] = None

    def __getitem__(self, key) -> "SharedArrayRef":
        if self.key is not None:
            raise ValueError("SharedArrayRef already carries an index expression.")
        return SharedArrayRef(self.path, key)

    def load(self) -> np.ndarray:
        """Memmap the backing file (cached per process) and apply the index."""
        try:
            array = _MEMMAP_CACHE[self.path]
        except KeyError:
            array = np.load(self.path, mmap_mode="r")
            _MEMMAP_CACHE[self.path] = array
        if self.key is not None:
            array = array[self.key]
        return array


def resolve(obj):
    """Return the array behind ``obj``.

    `SharedArrayRef` instances are memmapped (read-only) and sliced; any
    other object (including None and plain arrays) is returned unchanged.
    """
    if isinstance(obj, SharedArrayRef):
        return obj.load()
    return obj


class SharedArrayStore:
    """Directory of ``.npy`` files shared read-only with worker processes.

    Use as a context manager; the backing files are deleted on exit.

    Parameters
    ----------
    scratch_dir : str or Path, optional
        Directory in which to create the store. If None, resolved via
        `trap.parameters.resolve_scratch_dir` (using ``required_bytes``
        to check ``/dev/shm`` headroom).
    required_bytes : int, optional
        Estimated total size of the arrays to be stored.
    """

    def __init__(self, scratch_dir=None, required_bytes=None):
        base_dir = resolve_scratch_dir(scratch_dir, required_bytes)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="trap_store_", dir=base_dir))

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.npy"

    def dump(self, name: str, array: np.ndarray) -> SharedArrayRef:
        """Write ``array`` to the store and return a reference to it."""
        path = self._path(name)
        np.save(path, np.ascontiguousarray(array))
        return SharedArrayRef(str(path))

    def create(self, name: str, shape, dtype) -> np.ndarray:
        """Create an empty array in the store and return it as a writable memmap.

        Use this to fill large arrays incrementally (e.g. one wavelength
        slice at a time) without materializing them in memory first. Obtain
        the shareable reference afterwards via `ref`.
        """
        return np.lib.format.open_memmap(
            self._path(name), mode="w+", dtype=dtype, shape=shape
        )

    def ref(self, name: str) -> SharedArrayRef:
        """Return a reference to a previously dumped/created array."""
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"No array named {name!r} in store {self.directory}")
        return SharedArrayRef(str(path))

    def cleanup(self):
        """Delete the store directory and all arrays in it."""
        for path in self.directory.glob("*.npy"):
            _MEMMAP_CACHE.pop(str(path), None)
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> "SharedArrayStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False
