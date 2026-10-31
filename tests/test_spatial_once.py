"""The spatial model runs once per reduction, not once per temporal component fraction.

It does not depend on `temporal_components_fraction`, so computing it for every entry
repeated the same reduction and overwrote its output each time.
"""

from trap.parameters import trap_config_for_irdis
from trap.reduction_wrapper import _models_to_run, _reduction_parameters_for_fraction


def test_spatial_model_only_in_the_first_fraction():
    config = trap_config_for_irdis().reduction.merge(spatial_model=True, temporal_plus_spatial_model=True)
    first = _reduction_parameters_for_fraction(config, 0)
    later = _reduction_parameters_for_fraction(config, 1)
    assert first.spatial_model is True
    assert later.spatial_model is False
    assert first.temporal_plus_spatial_model is True and later.temporal_plus_spatial_model is True


def test_existing_outputs_are_checked_only_for_models_that_run():
    config = trap_config_for_irdis().reduction.merge(spatial_model=True, temporal_plus_spatial_model=True)
    assert _models_to_run(_reduction_parameters_for_fraction(config, 0)) == [
        "temporal", "temporal_plus_spatial", "spatial",
    ]
    assert _models_to_run(_reduction_parameters_for_fraction(config, 1)) == ["temporal", "temporal_plus_spatial"]
    assert _models_to_run(trap_config_for_irdis().reduction) == ["temporal"]
