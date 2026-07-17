import importlib
import logging
import subprocess
import sys

import pytest


def test_trap_root_logger_has_single_nullhandler():
    import trap  # noqa: F401

    root = logging.getLogger("trap")
    null_handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
    assert len(null_handlers) == 1
    # Library must not force a level on itself.
    assert root.level == logging.NOTSET


def test_import_trap_prints_nothing_to_stdout():
    result = subprocess.run(
        [sys.executable, "-c", "import trap"],
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


@pytest.mark.parametrize(
    "module_name",
    [
        "trap.reduction_wrapper",
        "trap.regression",
        "trap.detection",
        "trap.utils",
        "trap.pca_regression",
        "trap.parameters",
        "trap.template",
    ],
)
def test_module_defines_named_logger(module_name):
    module = importlib.import_module(module_name)
    assert isinstance(module.logger, logging.Logger)
    assert module.logger.name == module_name


def test_derotate_cube_emits_debug_record(caplog):
    import numpy as np

    from trap import utils

    cube = np.zeros((2, 3, 3))
    pa = np.array([10.0, 20.0])
    with caplog.at_level(logging.DEBUG, logger="trap.utils"):
        utils.derotate_cube(cube, pa, right_handed=True, verbose=True)

    debug_records = [
        r for r in caplog.records if r.name == "trap.utils" and r.levelno == logging.DEBUG
    ]
    assert debug_records, "expected a DEBUG record from trap.utils"
    assert "Derotating" in debug_records[0].getMessage()
