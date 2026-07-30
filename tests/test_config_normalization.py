"""Tests for `_to_reduction_config` and the removal of the legacy parameter API."""

from __future__ import annotations

import pytest

import trap.parameters as parameters
from trap.parameters import TrapConfig, TrapReductionConfig, _to_reduction_config


class TestToReductionConfig:
    def test_passes_through_reduction_config(self):
        config = TrapReductionConfig(prefix="run_")
        assert _to_reduction_config(config) is config

    def test_unwraps_trap_config(self):
        config = TrapConfig()
        assert _to_reduction_config(config) is config.reduction

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError, match="TrapReductionConfig or TrapConfig"):
            _to_reduction_config(object())


class TestLegacyApiRemoved:
    def test_reduction_parameters_class_gone(self):
        assert not hasattr(parameters, "Reduction_parameters")

    def test_bridge_methods_gone(self):
        assert not hasattr(TrapReductionConfig, "to_reduction_parameters")
        assert not hasattr(TrapConfig, "get_reduction_parameters")
