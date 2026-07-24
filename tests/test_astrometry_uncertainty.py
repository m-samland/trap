"""Unit tests for the astrometric-uncertainty additions to
trap.detection. See
docs/superpowers/specs/2026-07-23-trap-astrometry-uncertainty-design.md
for the mathematical justification of every assertion in this file.
"""

import numpy as np
import pandas as pd
import pytest

from trap import detection


def test_module_imports():
    assert hasattr(detection, "fit_2d_gaussian")
