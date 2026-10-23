"""`template_names` selects which default templates are built and fitted.

Single-filter modes (IRDIS broadband) need ``("flat",)``: with two identical
bandpasses L-type duplicates flat and the T-type offset fit is singular.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from trap import detection
from trap.parameters import DetectionParameters


class TestResolveTemplateNames:
    def test_none_selects_all_defaults(self):
        assert detection.resolve_template_names(None) == ("L-type", "T-type", "flat")

    def test_subset_keeps_default_order(self):
        assert detection.resolve_template_names(("flat", "L-type")) == ("L-type", "flat")

    def test_bare_string_is_one_name(self):
        assert detection.resolve_template_names("flat") == ("flat",)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            detection.resolve_template_names(("flat", "Y-type"))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            detection.resolve_template_names(())


class TestDetectionParameters:
    def test_defaults_to_all_templates(self):
        assert DetectionParameters().template_names is None

    def test_merge_sets_subset(self):
        assert DetectionParameters().merge(template_names=("flat",)).template_names == ("flat",)


class _ModelBox:
    def __init__(self, model):
        self.model = model
        self.wavelength = np.linspace(0.9, 2.5, 50)
        self.flux = np.full_like(self.wavelength, 2.0)


class _Template:
    def __init__(self, name, companion_modelbox, stellar_modelbox, **kwargs):
        self.name = name
        self.companion_modelbox = companion_modelbox
        self.stellar_modelbox = stellar_modelbox


@pytest.fixture
def stub_species(monkeypatch):
    """Record which species models get loaded, without a species database."""
    loaded = []

    class _Database:
        def add_model(self, model, teff_range=None):
            loaded.append(model)

    class _ReadModel:
        def __init__(self, model, wavel_range=None):
            self.model = model

        def get_bounds(self):
            return {}

        def get_model(self, model_param):
            return _ModelBox(self.model)

    monkeypatch.setattr(detection, "configure_species", lambda directory: directory)
    monkeypatch.setattr(detection, "Database", _Database)
    monkeypatch.setattr(detection, "ReadModel", _ReadModel)
    monkeypatch.setattr(detection, "SpectralTemplate", _Template)
    return loaded


def _analysis():
    analysis = detection.DetectionAnalysis.__new__(detection.DetectionAnalysis)
    analysis.templates = {}
    analysis.wavelength_indices = None
    analysis.instrument = SimpleNamespace(
        instrument_type="photometry",
        wavelengths=np.array([1.625, 1.625]) * u.micron,
        filters=["Paranal/SPHERE.IRDIS_B_H", "Paranal/SPHERE.IRDIS_B_H"],
    )
    return analysis


class TestAddDefaultTemplates:
    def test_default_builds_all_three(self, stub_species):
        analysis = _analysis()
        analysis.add_default_templates(
            stellar_modelbox=None, species_database_directory="db",
            stellar_parameters={"teff": 6000.0, "logg": 4.0, "feh": 0.0},
        )
        assert list(analysis.templates) == ["L-type", "T-type", "flat"]
        assert "bt-nextgen" in stub_species

    def test_flat_only_skips_the_stellar_and_hot_models(self, stub_species):
        analysis = _analysis()
        analysis.add_default_templates(
            stellar_modelbox=None, species_database_directory="db",
            stellar_parameters={"teff": 6000.0, "logg": 4.0, "feh": 0.0},
            template_names=("flat",),
        )
        assert list(analysis.templates) == ["flat"]
        assert "bt-nextgen" not in stub_species
        assert "drift-phoenix" not in stub_species
        flat = analysis.templates["flat"]
        assert np.all(flat.companion_modelbox.flux == 1.0)
        assert np.all(flat.stellar_modelbox.flux == 1.0)

    def test_l_type_without_stellar_parameters_uses_a_flat_star(self, stub_species):
        analysis = _analysis()
        analysis.add_default_templates(
            stellar_modelbox=None, species_database_directory="db",
            template_names=("L-type",),
        )
        assert list(analysis.templates) == ["L-type"]
        assert np.all(analysis.templates["L-type"].stellar_modelbox.flux == 1.0)


def test_detection_entry_point_forwards_template_names():
    signature = inspect.signature(
        detection.DetectionAnalysis.detection_and_characterization_with_template_matching
    )
    assert signature.parameters["template_names"].default is None
