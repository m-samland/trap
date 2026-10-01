"""Every trap module imports with only the declared dependencies installed."""

import importlib
import pkgutil

import pytest

import trap

MODULES = sorted(m.name for m in pkgutil.walk_packages(trap.__path__, "trap."))


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    # Most modules are imported by some other test, but a module nothing else
    # imports can depend on a package that only an optional extra provides.
    importlib.import_module(name)
