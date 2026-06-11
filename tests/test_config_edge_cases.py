"""
Edge case and property-based tests for the configuration module.

Covers:
- SafetyThresholds edge cases (boundary values, None current)
- MLConfig edge cases
- FrameworkConfig chemistry integration
- YAML loading edge cases
- from_dict / to_dict round-trips with partial data
- DEFAULT_CONFIG and TESLA_CONFIG constants
"""

import json
import os
import tempfile

import pytest

from ev_qa_framework.config import (
    DEFAULT_CONFIG,
    TESLA_CONFIG,
    FrameworkConfig,
    MLConfig,
    SafetyThresholds,
)


class TestSafetyThresholdsEdgeCases:
    def test_boundary_temperature_values(self):
        t = SafetyThresholds(max_temperature=-40.0)
        assert t.max_temperature == -40.0

    def test_extreme_voltage_values(self):
        t = SafetyThresholds(min_voltage=0.0, max_voltage=1000.0)
        assert t.min_voltage == 0.0
        assert t.max_voltage == 1000.0

    def test_none_max_current(self):
        t = SafetyThresholds(max_current=None)
        assert t.max_current is None

    def test_to_dict_with_none_current(self):
        t = SafetyThresholds(max_current=None)
        d = t.to_dict()
        assert d["max_current"] is None

    def test_from_dict_partial(self):
        """from_dict with only some keys should use defaults for the rest."""
        data = {"max_temperature": 70.0}
        t = SafetyThresholds.from_dict(data)
        assert t.max_temperature == 70.0
        assert t.min_voltage == 200.0  # default
        assert t.max_voltage == 900.0  # default

    def test_from_dict_empty(self):
        """from_dict with empty dict should produce all defaults."""
        t = SafetyThresholds.from_dict({})
        assert t.max_temperature == 60.0
        assert t.min_voltage == 200.0

    def test_roundtrip_serialization(self):
        original = SafetyThresholds(
            max_temperature=55.0,
            min_voltage=250.0,
            max_voltage=450.0,
            max_temperature_jump=3.0,
            min_soc=5.0,
            critical_soh=65.0,
            max_current=400.0,
        )
        d = original.to_dict()
        restored = SafetyThresholds.from_dict(d)
        assert restored.max_temperature == original.max_temperature
        assert restored.min_voltage == original.min_voltage
        assert restored.max_voltage == original.max_voltage
        assert restored.max_temperature_jump == original.max_temperature_jump
        assert restored.min_soc == original.min_soc
        assert restored.critical_soh == original.critical_soh
        assert restored.max_current == original.max_current

    def test_save_load_roundtrip_file(self):
        t = SafetyThresholds(max_temperature=42.0, min_voltage=300.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t.save_to_file(path)
            loaded = SafetyThresholds.load_from_file(path)
            assert loaded.max_temperature == 42.0
            assert loaded.min_voltage == 300.0
        finally:
            os.unlink(path)

    def test_save_load_preserves_none_current(self):
        t = SafetyThresholds(max_current=None)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t.save_to_file(path)
            loaded = SafetyThresholds.load_from_file(path)
            assert loaded.max_current is None
        finally:
            os.unlink(path)


class TestMLConfigEdgeCases:
    def test_zero_contamination(self):
        ml = MLConfig(contamination=0.0)
        assert ml.contamination == 0.0

    def test_one_contamination(self):
        ml = MLConfig(contamination=1.0)
        assert ml.contamination == 1.0

    def test_from_dict_partial(self):
        data = {"contamination": 0.05}
        ml = MLConfig.from_dict(data)
        assert ml.contamination == 0.05
        assert ml.n_estimators == 200  # default

    def test_from_dict_empty(self):
        ml = MLConfig.from_dict({})
        assert ml.contamination == 0.1
        assert ml.n_estimators == 200

    def test_roundtrip(self):
        original = MLConfig(
            contamination=0.07,
            n_estimators=150,
            random_state=123,
            critical_score_threshold=-0.95,
            warning_score_threshold=-0.6,
        )
        d = original.to_dict()
        restored = MLConfig.from_dict(d)
        assert restored.contamination == original.contamination
        assert restored.n_estimators == original.n_estimators
        assert restored.random_state == original.random_state
        assert restored.critical_score_threshold == original.critical_score_threshold
        assert restored.warning_score_threshold == original.warning_score_threshold


class TestFrameworkConfigChemistry:
    def test_nmc_chemistry_auto_thresholds(self):
        cfg = FrameworkConfig(chemistry="nmc")
        assert cfg.chemistry == "nmc"
        # NMC profile should set specific thresholds
        assert cfg.safety_thresholds.max_temperature > 0

    def test_lfp_chemistry_auto_thresholds(self):
        cfg = FrameworkConfig(chemistry="lfp")
        assert cfg.chemistry == "lfp"

    def test_nca_chemistry_auto_thresholds(self):
        cfg = FrameworkConfig(chemistry="nca")
        assert cfg.chemistry == "nca"

    def test_no_chemistry_keeps_defaults(self):
        cfg = FrameworkConfig(chemistry=None)
        assert cfg.chemistry is None
        assert cfg.safety_thresholds.max_temperature == 60.0

    def test_configure_from_chemistry_explicit(self):
        """configure_from_chemistry should re-apply chemistry thresholds."""
        cfg = FrameworkConfig(chemistry="nmc")
        # Manually override thresholds
        cfg.safety_thresholds = SafetyThresholds(max_temperature=99.0)
        assert cfg.safety_thresholds.max_temperature == 99.0
        # Re-apply from chemistry
        cfg.configure_from_chemistry()
        # Should now match NMC profile, not 99.0
        assert cfg.safety_thresholds.max_temperature != 99.0

    def test_get_chemistry_profile_returns_none_when_no_chemistry(self):
        cfg = FrameworkConfig(chemistry=None)
        assert cfg.get_chemistry_profile() is None

    def test_get_chemistry_profile_returns_profile(self):
        cfg = FrameworkConfig(chemistry="nmc")
        profile = cfg.get_chemistry_profile()
        assert profile is not None

    def test_to_dict_includes_chemistry_when_set(self):
        cfg = FrameworkConfig(chemistry="nmc", cells_in_series=96)
        d = cfg.to_dict()
        assert "chemistry" in d
        assert d["chemistry"] == "nmc"
        assert "cells_in_series" in d

    def test_to_dict_omits_chemistry_when_none(self):
        cfg = FrameworkConfig(chemistry=None)
        d = cfg.to_dict()
        assert "chemistry" not in d
        assert "cells_in_series" not in d

    def test_from_dict_with_chemistry(self):
        data = {
            "chemistry": "lfp",
            "cells_in_series": 120,
        }
        cfg = FrameworkConfig.from_dict(data)
        assert cfg.chemistry == "lfp"
        assert cfg.cells_in_series == 120

    def test_fail_on_anomaly_flag(self):
        cfg = FrameworkConfig(fail_on_anomaly=True)
        assert cfg.fail_on_anomaly is True
        d = cfg.to_dict()
        assert d["fail_on_anomaly"] is True
        restored = FrameworkConfig.from_dict(d)
        assert restored.fail_on_anomaly is True


class TestFrameworkConfigSerialization:
    def test_save_load_roundtrip(self):
        cfg = FrameworkConfig()
        cfg.safety_thresholds.max_temperature = 42.0
        cfg.ml_config.contamination = 0.05
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save_to_file(path)
            loaded = FrameworkConfig.load_from_file(path)
            assert loaded.safety_thresholds.max_temperature == 42.0
            assert loaded.ml_config.contamination == 0.05
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_default(self):
        cfg = FrameworkConfig.load_from_file("/nonexistent/path/config.json")
        assert isinstance(cfg, FrameworkConfig)
        assert cfg.safety_thresholds.max_temperature == 60.0

    def test_from_dict_partial(self):
        """from_dict with minimal data should fill in defaults."""
        data = {"default_vin": "CUSTOMVIN12345678"}
        cfg = FrameworkConfig.from_dict(data)
        assert cfg.default_vin == "CUSTOMVIN12345678"
        assert cfg.safety_thresholds.max_temperature == 60.0  # default

    def test_from_dict_empty(self):
        cfg = FrameworkConfig.from_dict({})
        assert cfg.default_vin == "TESTVEHCLE0123456"
        assert cfg.fail_on_anomaly is False


class TestDefaultConfigs:
    def test_default_config_is_nmc(self):
        assert DEFAULT_CONFIG.chemistry == "nmc"

    def test_tesla_config_is_nca(self):
        assert TESLA_CONFIG.chemistry == "nca"
        assert TESLA_CONFIG.cells_in_series == 108
        assert TESLA_CONFIG.fail_on_anomaly is True

    def test_tesla_config_has_valid_vin(self):
        assert len(TESLA_CONFIG.default_vin) == 17


class TestFrameworkConfigYAML:
    def test_load_from_yaml_nonexistent(self):
        cfg = FrameworkConfig.load_from_yaml("/nonexistent/config.yaml")
        assert isinstance(cfg, FrameworkConfig)

    def test_load_from_yaml_with_profiles(self):
        yaml_content = """
profiles:
  default:
    default_vin: YAMLVIN123456789
  strict:
    default_vin: STRICTVIN123456
    fail_on_anomaly: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            cfg_default = FrameworkConfig.load_from_yaml(path)
            assert cfg_default.default_vin == "YAMLVIN123456789"

            cfg_strict = FrameworkConfig.load_from_yaml(path, profile="strict")
            assert cfg_strict.default_vin == "STRICTVIN123456"
            assert cfg_strict.fail_on_anomaly is True
        finally:
            os.unlink(path)

    def test_load_from_yaml_missing_profile_falls_back(self):
        yaml_content = """
profiles:
  default:
    default_vin: DEFAULTVIN123456
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            cfg = FrameworkConfig.load_from_yaml(path, profile="nonexistent")
            assert cfg.default_vin == "DEFAULTVIN123456"
        finally:
            os.unlink(path)

    def test_load_from_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = FrameworkConfig.load_from_yaml(path)
            assert isinstance(cfg, FrameworkConfig)
        finally:
            os.unlink(path)
