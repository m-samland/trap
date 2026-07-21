"""Tests for the IRDIS factory (`trap_config_for_irdis`) and the IRDIS
obs-mode dispatch in `InstrumentConfig.to_instrument`.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from trap.parameters import (
    InstrumentConfig,
    TrapConfig,
    trap_config_for_ifs,
    trap_config_for_irdis,
)


class TestTrapConfigForIrdis:
    def test_returns_trapconfig(self):
        cfg = trap_config_for_irdis()
        assert isinstance(cfg, TrapConfig)

    def test_pixel_scale_is_irdis(self):
        cfg = trap_config_for_irdis()
        assert cfg.instrument.pixel_scale_arcsec_per_pixel == pytest.approx(0.01225)

    def test_two_wavelength_channels(self):
        cfg = trap_config_for_irdis()
        assert list(cfg.processing.wavelength_indices) == [0, 1]

    def test_instrument_type_is_imaging(self):
        cfg = trap_config_for_irdis()
        assert cfg.instrument.instrument_type == "imaging"

    def test_instrument_name(self):
        cfg = trap_config_for_irdis()
        assert cfg.instrument.name == "IRDIS"

    def test_ifs_defaults_unchanged(self):
        """Regression: the IFS factory is untouched."""
        cfg = trap_config_for_ifs()
        assert cfg.instrument.name == "IFS"
        assert cfg.instrument.pixel_scale_arcsec_per_pixel == pytest.approx(0.00746)
        assert list(cfg.processing.wavelength_indices) == list(range(1, 38))


class TestInstrumentConfigToInstrumentIrdisModes:
    @pytest.mark.parametrize(
        "obs_mode",
        ["DB_K12", "DB_H23", "DB_H34", "DB_Y23", "DB_J23",
         "BB_H", "BB_K", "BB_J", "BB_Y", "BB_Ks"],
    )
    def test_accepts_irdis_obs_modes(self, obs_mode):
        ic = InstrumentConfig(
            name="IRDIS",
            pixel_scale_arcsec_per_pixel=0.01225,
            instrument_type="imaging",
        )
        inst = ic.to_instrument(obs_mode, wavelengths=None)
        assert inst.name == "IRDIS"
        assert inst.instrument_type == "imaging"
        assert inst.spectral_resolution is None

    def test_ifs_modes_still_work(self):
        ic = InstrumentConfig()
        yj = ic.to_instrument("OBS_YJ", wavelengths=None)
        assert yj.spectral_resolution == ic.spectral_resolution_yj
        h = ic.to_instrument("OBS_H", wavelengths=None)
        assert h.spectral_resolution == ic.spectral_resolution_h

    def test_unknown_mode_still_raises(self):
        ic = InstrumentConfig()
        with pytest.raises(ValueError):
            ic.to_instrument("BOGUS_MODE", wavelengths=None)

    def test_wavelengths_forwarded(self):
        ic = InstrumentConfig(
            name="IRDIS",
            pixel_scale_arcsec_per_pixel=0.01225,
            instrument_type="imaging",
        )
        wl = np.array([2.11, 2.25]) * u.micron
        inst = ic.to_instrument("DB_K12", wavelengths=wl)
        assert inst.wavelengths is not None
        np.testing.assert_allclose(inst.wavelengths.value, [2.11, 2.25])
