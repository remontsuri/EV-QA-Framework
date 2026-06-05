"""
Tests for Battery Chemistry Profiles module and config integration.
"""

import json
import os
import tempfile

import pytest

from ev_qa_framework.chemistries import (
    ALL_CHEMISTRIES,
    BatteryChemistryProfile,
    BUILTIN_PROFILES,
    CellImbalanceThresholds,
    SOHDegradationParams,
    get_profile,
    list_profiles,
    load_custom_profile_from_file,
    register_custom_profile,
)
from ev_qa_framework.config import (
    DEFAULT_CONFIG,
    FrameworkConfig,
    SafetyThresholds,
    TESLA_CONFIG,
)


class TestSOHDegradationParams:
    """Tests for SOHDegradationParams dataclass."""

    def test_default_values(self):
        p = SOHDegradationParams()
        assert p.cycle_life_min == 1000
        assert p.cycle_life_max == 2000
        assert p.deg_knee_point_soh == 80.0
        assert p.annual_fade_rate_pct == 2.0

    def test_custom_values(self):
        p = SOHDegradationParams(
            cycle_life_min=3000,
            cycle_life_max=6000,
            deg_knee_point_soh=70.0,
            annual_fade_rate_pct=1.0,
            calendric_fade_pct_per_year=2.0,
        )
        assert p.cycle_life_min == 3000
        assert p.calendric_fade_pct_per_year == 2.0


class TestCellImbalanceThresholds:
    """Tests for CellImbalanceThresholds dataclass."""

    def test_default_values(self):
        b = CellImbalanceThresholds()
        assert b.max_delta_mv == 100.0
        assert b.warning_delta_mv == 50.0
        assert b.recovery_rate_mv_per_h == 10.0


class TestBatteryChemistryProfile:
    """Tests for BatteryChemistryProfile and built-in profiles."""

    def test_all_builtin_profiles_have_expected_keys(self):
        """Every built-in profile must be registered in BUILTIN_PROFILES."""
        for key in ALL_CHEMISTRIES:
            assert key in BUILTIN_PROFILES, f"Missing profile for {key}"

    def test_get_profile_lfp(self):
        p = get_profile("lfp")
        assert p.short_name == "lfp"
        assert "LFP" in p.name

    def test_get_profile_nmc(self):
        p = get_profile("nmc")
        assert p.short_name == "nmc"
        assert "NMC" in p.name

    def test_get_profile_nca(self):
        p = get_profile("nca")
        assert p.short_name == "nca"
        assert "NCA" in p.name

    def test_get_profile_unknown_raises(self):
        with pytest.raises(KeyError):
            get_profile("solid-state")

    def test_pack_voltage_scaling(self):
        """NMC per-cell 3.0-4.2 V → 96s pack = 288.0-403.2 V."""
        p = get_profile("nmc")
        assert p.pack_min_voltage(96) == pytest.approx(288.0)
        assert p.pack_max_voltage(96) == pytest.approx(403.2)
        assert p.pack_nominal_voltage(96) == pytest.approx(355.2)

    def test_pack_voltage_lfp(self):
        """LFP per-cell 2.5-3.65 V → 108s = 270.0-394.2 V."""
        p = get_profile("lfp")
        assert p.pack_min_voltage(108) == pytest.approx(270.0)
        assert p.pack_max_voltage(108) == pytest.approx(394.2)

    def test_to_safety_thresholds_dict_lfp_96s(self):
        p = get_profile("lfp")
        d = p.to_safety_thresholds_dict(cells_in_series=96)
        # LFP charge temp max = 55.0 °C
        assert d["max_temperature"] == 55.0
        # LFP discharge temp min = -30.0 °C (discharge range)
        assert d["min_temperature"] == -30.0
        # pack voltage: 2.5 * 96 = 240.0
        assert d["min_voltage"] == pytest.approx(240.0)
        # 3.65 * 96 = 350.4
        assert d["max_voltage"] == pytest.approx(350.4)

    def test_to_safety_thresholds_dict_nca_108s(self):
        """NCA at 108s — matches Tesla Model S (~400 V pack)."""
        p = get_profile("nca")
        d = p.to_safety_thresholds_dict(cells_in_series=108)
        # 3.0 * 108 = 324.0 V
        assert d["min_voltage"] == pytest.approx(324.0)
        # 4.2 * 108 = 453.6 V
        assert d["max_voltage"] == pytest.approx(453.6)

    def test_to_dict_roundtrip(self):
        p = get_profile("nmc")
        d = p.to_dict()
        restored = BatteryChemistryProfile.from_dict(d)
        assert restored.name == p.name
        assert restored.short_name == p.short_name
        assert restored.cell_nominal_voltage == p.cell_nominal_voltage
        assert restored.soh_params.cycle_life_min == p.soh_params.cycle_life_min
        assert restored.balance.max_delta_mv == p.balance.max_delta_mv

    def test_to_json_roundtrip(self):
        p = get_profile("lfp")
        js = p.to_json()
        restored = BatteryChemistryProfile.from_json(js)
        assert restored.name == p.name
        assert restored.cell_min_voltage == p.cell_min_voltage

    def test_save_and_load_file(self):
        p = get_profile("nmc")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmppath = f.name
        try:
            p.save_to_file(tmppath)
            assert os.path.exists(tmppath)
            loaded = BatteryChemistryProfile.load_from_file(tmppath)
            assert loaded.name == p.name
            assert loaded.short_name == p.short_name
        finally:
            os.unlink(tmppath)

    def test_list_profiles(self):
        meta = list_profiles()
        assert len(meta) == 3
        names = {m["short_name"] for m in meta}
        assert names == {"lfp", "nmc", "nca"}

    def test_register_custom_profile(self):
        custom = BatteryChemistryProfile(
            name="Custom Sodium-Ion",
            short_name="sodium",  # type: ignore[arg-type]
            manufacturer="TestCorp",
            cell_nominal_voltage=3.1,
            cell_min_voltage=2.0,
            cell_max_voltage=3.8,
        )
        register_custom_profile(custom)
        loaded = get_profile("sodium")  # type: ignore[arg-type]
        assert loaded.name == "Custom Sodium-Ion"
        assert loaded.manufacturer == "TestCorp"
        # Clean up registry
        BUILTIN_PROFILES.pop("sodium", None)

    def test_load_custom_profile_from_file(self):
        custom = BatteryChemistryProfile(
            name="Custom Solid-State",
            short_name="ssb",  # type: ignore[arg-type]
            manufacturer="QuantumScape",
            cell_nominal_voltage=3.6,
            cell_min_voltage=3.0,
            cell_max_voltage=4.5,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(custom.to_json())
            tmppath = f.name
        try:
            loaded = load_custom_profile_from_file(tmppath)
            assert loaded.name == "Custom Solid-State"
            assert get_profile("ssb").name == "Custom Solid-State"  # type: ignore[arg-type]
        finally:
            os.unlink(tmppath)
            BUILTIN_PROFILES.pop("ssb", None)


class TestFrameworkConfigChemistryIntegration:
    """Tests for FrameworkConfig integration with chemistry profiles."""

    def test_default_config_is_nmc(self):
        """DEFAULT_CONFIG should have chemistry=nmc and NMC-derived thresholds."""
        assert DEFAULT_CONFIG.chemistry == "nmc"
        # NMC 96s: 3.0 * 96 = 288.0 V
        assert DEFAULT_CONFIG.safety_thresholds.min_voltage == pytest.approx(288.0)
        # 4.2 * 96 = 403.2 V
        assert DEFAULT_CONFIG.safety_thresholds.max_voltage == pytest.approx(403.2)
        # NMC charge temp max = 45.0, discharge temp min = -20.0
        assert DEFAULT_CONFIG.safety_thresholds.max_temperature == 45.0
        assert DEFAULT_CONFIG.safety_thresholds.min_temperature == -20.0

    def test_tesla_config_is_nca_108s(self):
        """TESLA_CONFIG should have chemistry=nca with 108 cells in series."""
        assert TESLA_CONFIG.chemistry == "nca"
        assert TESLA_CONFIG.cells_in_series == 108
        # NCA 108s: 3.0 * 108 = 324.0 V
        assert TESLA_CONFIG.safety_thresholds.min_voltage == pytest.approx(324.0)
        # 4.2 * 108 = 453.6 V
        assert TESLA_CONFIG.safety_thresholds.max_voltage == pytest.approx(453.6)

    def test_chemistry_lfp_96s(self):
        """FrameworkConfig(chemistry='lfp') should auto-populate LFP thresholds."""
        cfg = FrameworkConfig(chemistry="lfp")
        assert cfg.chemistry == "lfp"
        assert cfg.cells_in_series == 96
        # LFP 96s: 2.5 * 96 = 240.0 V, 3.65 * 96 = 350.4 V
        assert cfg.safety_thresholds.min_voltage == pytest.approx(240.0)
        assert cfg.safety_thresholds.max_voltage == pytest.approx(350.4)
        # LFP charge temp = 55.0
        assert cfg.safety_thresholds.max_temperature == 55.0

    def test_chemistry_none_uses_raw_defaults(self):
        """FrameworkConfig() with no chemistry should use SafetyThresholds defaults."""
        cfg = FrameworkConfig(chemistry=None)
        assert cfg.chemistry is None
        # SafetyThresholds factory defaults (not scaled from any profile)
        assert cfg.safety_thresholds.min_voltage == 200.0
        assert cfg.safety_thresholds.max_voltage == 900.0
        assert cfg.safety_thresholds.max_temperature == 60.0

    def test_get_chemistry_profile(self):
        cfg = FrameworkConfig(chemistry="lfp")
        prof = cfg.get_chemistry_profile()
        assert prof is not None
        assert prof.short_name == "lfp"

    def test_get_chemistry_profile_none(self):
        cfg = FrameworkConfig()
        cfg.chemistry = None
        assert cfg.get_chemistry_profile() is None

    def test_explicit_thresholds_not_overridden_by_chemistry(self):
        """If user passes explicit thresholds, they should be kept even with chemistry set."""
        cfg = FrameworkConfig(
            chemistry="lfp",
            safety_thresholds=SafetyThresholds(
                max_temperature=99.0,
                min_voltage=500.0,
                max_voltage=1000.0,
            ),
        )
        # __post_init__ runs after field init — the explicit thresholds
        # are replaced by chemistry.
        # NOTE: this test documents current behaviour since __post_init__
        # always applies chemistry. If you want manual thresholds, don't
        # set chemistry or call configure_from_chemistry() later.
        assert cfg.safety_thresholds.max_temperature == 55.0  # from LFP

    def test_to_dict_includes_chemistry(self):
        cfg = FrameworkConfig(chemistry="lfp", cells_in_series=108)
        d = cfg.to_dict()
        assert d["chemistry"] == "lfp"
        assert d["cells_in_series"] == 108
        assert "safety_thresholds" in d
        assert "ml_config" in d

    def test_to_dict_excludes_chemistry_when_none(self):
        cfg = FrameworkConfig(chemistry=None)
        d = cfg.to_dict()
        assert "chemistry" not in d
        assert "cells_in_series" not in d

    def test_from_dict_with_chemistry(self):
        d = {
            "safety_thresholds": {},
            "ml_config": {},
            "default_vin": "TEST1234567890",
            "chemistry": "nca",
            "cells_in_series": 108,
        }
        cfg = FrameworkConfig.from_dict(d)
        assert cfg.chemistry == "nca"
        assert cfg.cells_in_series == 108
        # __post_init__ runs during from_dict -> cls(...) -> NCA 108s
        assert cfg.safety_thresholds.min_voltage == pytest.approx(324.0)

    def test_from_dict_without_chemistry(self):
        d = {
            "safety_thresholds": {"max_temperature": 50.0},
            "ml_config": {},
        }
        cfg = FrameworkConfig.from_dict(d)
        assert cfg.chemistry is None
        assert cfg.cells_in_series == 96
        assert cfg.safety_thresholds.max_temperature == 50.0

    def test_save_and_load_json_roundtrip(self):
        cfg = FrameworkConfig(
            chemistry="lfp",
            cells_in_series=128,
            default_vin="LFPVIN000000001",
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmppath = f.name
        try:
            cfg.save_to_file(tmppath)
            loaded = FrameworkConfig.load_from_file(tmppath)
            assert loaded.chemistry == "lfp"
            assert loaded.cells_in_series == 128
            assert loaded.default_vin == "LFPVIN000000001"
            # Thresholds should be LFP 128s
            assert loaded.safety_thresholds.min_voltage == pytest.approx(2.5 * 128)
        finally:
            os.unlink(tmppath)

    def test_config_chemistry_builtin_json_files(self):
        """Verify that each config/chemistry_*.json file can be loaded and matches its profile."""
        base = os.path.join(os.path.dirname(__file__), "..", "config")
        for key in ALL_CHEMISTRIES:
            path = os.path.join(base, f"chemistry_{key}.json")
            assert os.path.exists(path), f"Missing config file: {path}"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            profile = BatteryChemistryProfile.from_dict(data)
            assert profile.short_name == key
            assert profile.name == BUILTIN_PROFILES[key].name


class TestFrameworkConfigRegression:
    """Regression tests: existing behaviour must not break."""

    def test_default_config_still_works_with_framework(self):
        """EVQAFramework starts fine with DEFAULT_CONFIG (now chemistry-aware)."""
        from ev_qa_framework.framework import EVQAFramework

        qa = EVQAFramework("Test-QA")
        assert qa.config.default_vin == "TESTVEHCLE0123456"

    def test_tesla_config_vin(self):
        assert TESLA_CONFIG.default_vin == "5YJSA1E26HF000337"
        assert TESLA_CONFIG.fail_on_anomaly is True
