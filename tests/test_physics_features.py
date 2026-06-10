from __future__ import annotations

"""Tests for physics-informed feature extraction."""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.physics_features import PhysicsFeatureExtractor


@pytest.fixture
def extractor() -> PhysicsFeatureExtractor:
    return PhysicsFeatureExtractor()


# ─── IC Curve Tests ───────────────────────────────────────────────


class TestExtractIcCurve:
    def test_basic_peaks_detected(self, extractor: PhysicsFeatureExtractor):
        """IC curve should detect peaks in synthetic dQ/dV data."""
        voltage = np.linspace(3.0, 4.2, 200)
        # Simulate a capacity curve with two phase transitions
        capacity = (
            voltage * 10
            + 5.0 * np.exp(-((voltage - 3.5) ** 2) / 0.005)
            + 3.0 * np.exp(-((voltage - 3.9) ** 2) / 0.005)
        )
        result = extractor.extract_ic_curve(voltage, capacity)
        assert result["num_peaks"] >= 1
        assert len(result["peak_voltages"]) == result["num_peaks"]

    def test_output_keys_present(self, extractor: PhysicsFeatureExtractor):
        """Result dict should contain all expected keys."""
        voltage = np.linspace(3.0, 4.2, 100)
        capacity = voltage * 10 + np.sin(voltage) * 2
        result = extractor.extract_ic_curve(voltage, capacity)
        expected_keys = {
            "ic_values",
            "voltage_mid",
            "peaks",
            "valleys",
            "peak_voltages",
            "peak_heights",
            "valley_voltages",
            "valley_depths",
            "num_peaks",
            "num_valleys",
        }
        assert expected_keys.issubset(result.keys())

    def test_empty_input(self, extractor: PhysicsFeatureExtractor):
        """Empty arrays should return empty results without error."""
        result = extractor.extract_ic_curve(np.array([]), np.array([]))
        assert len(result["ic_values"]) == 0
        assert result["num_peaks"] == 0

    def test_short_input(self, extractor: PhysicsFeatureExtractor):
        """Arrays shorter than 3 elements should return empty results."""
        result = extractor.extract_ic_curve(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert len(result["ic_values"]) == 0

    def test_ic_values_length(self, extractor: PhysicsFeatureExtractor):
        """IC values should have length N-1."""
        n = 50
        voltage = np.linspace(3.0, 4.2, n)
        capacity = voltage * 10
        result = extractor.extract_ic_curve(voltage, capacity)
        assert len(result["ic_values"]) == n - 1
        assert len(result["voltage_mid"]) == n - 1


# ─── Delta Q Tests ────────────────────────────────────────────────


class TestComputeDeltaQ:
    def test_linear_fade(self, extractor: PhysicsFeatureExtractor):
        """Linear capacity fade should give accurate fade rate."""
        cycles = np.arange(100, dtype=float)
        capacity = 50.0 - 0.05 * cycles  # 0.05 Ah/cycle fade
        result = extractor.compute_delta_q(capacity, cycles)
        assert pytest.approx(result["fade_rate"], abs=1e-6) == -0.05
        assert pytest.approx(result["total_fade"], abs=0.1) == -4.95

    def test_fade_percentage(self, extractor: PhysicsFeatureExtractor):
        """Fade percentage should be computed correctly."""
        capacity = np.array([50.0, 49.0, 48.0, 47.0, 46.0])
        result = extractor.compute_delta_q(capacity)
        assert pytest.approx(result["fade_percentage"], abs=0.1) == -8.0

    def test_r_squared_perfect_fit(self, extractor: PhysicsFeatureExtractor):
        """Perfect linear fade should give R² ≈ 1.0."""
        cycles = np.arange(50, dtype=float)
        capacity = 50.0 - 0.1 * cycles
        result = extractor.compute_delta_q(capacity, cycles)
        assert pytest.approx(result["r_squared"], abs=1e-6) == 1.0

    def test_single_element(self, extractor: PhysicsFeatureExtractor):
        """Single-element array should return zero fade."""
        result = extractor.compute_delta_q(np.array([50.0]))
        assert result["total_fade"] == 0.0
        assert result["fade_rate"] == 0.0

    def test_initial_final_capacity(self, extractor: PhysicsFeatureExtractor):
        """Initial and final capacity should match input."""
        capacity = np.array([50.0, 49.5, 49.0, 48.5])
        result = extractor.compute_delta_q(capacity)
        assert result["initial_capacity"] == 50.0
        assert result["final_capacity"] == 48.5


# ─── Resistance Tests ─────────────────────────────────────────────


class TestEstimateResistance:
    def test_single_value(self, extractor: PhysicsFeatureExtractor):
        """Single voltage drop and current should give correct R."""
        result = extractor.estimate_resistance(0.5, 10.0)
        assert pytest.approx(result["resistance"], abs=1e-6) == 0.05

    def test_array_input(self, extractor: PhysicsFeatureExtractor):
        """Array input should return statistics."""
        voltage_drop = np.array([0.1, 0.2, 0.3, 0.4])
        current = np.array([10.0, 10.0, 10.0, 10.0])
        result = extractor.estimate_resistance(voltage_drop, current)
        assert pytest.approx(result["mean_resistance"], abs=1e-6) == 0.025
        assert pytest.approx(result["min_resistance"], abs=1e-6) == 0.01
        assert pytest.approx(result["max_resistance"], abs=1e-6) == 0.04

    def test_zero_current(self, extractor: PhysicsFeatureExtractor):
        """Zero current should not cause division by zero."""
        result = extractor.estimate_resistance(0.5, 0.0)
        assert result["resistance"] == 0.0

    def test_std_resistance(self, extractor: PhysicsFeatureExtractor):
        """Std should be computed for array input."""
        voltage_drop = np.array([0.1, 0.2, 0.3])
        current = np.array([10.0, 10.0, 10.0])
        result = extractor.estimate_resistance(voltage_drop, current)
        expected_std = np.std([0.01, 0.02, 0.03])
        assert pytest.approx(result["std_resistance"], abs=1e-6) == expected_std


# ─── Thermal Diffusivity Tests ────────────────────────────────────


class TestComputeThermalDiffusivity:
    def test_heating_detected(self, extractor: PhysicsFeatureExtractor):
        """Rising temperature should produce positive diffusivity values."""
        time_s = np.linspace(0, 100, 50)
        temp = 25.0 + 0.5 * time_s  # Linear heating at 0.5 °C/s
        result = extractor.compute_thermal_diffusivity(temp, time_s)
        assert result["max_temp_rate"] > 0.0

    def test_constant_temperature(self, extractor: PhysicsFeatureExtractor):
        """Constant temperature should give zero diffusivity."""
        time_s = np.linspace(0, 100, 50)
        temp = np.full(50, 25.0)
        result = extractor.compute_thermal_diffusivity(temp, time_s)
        assert result["max_temp_rate"] == 0.0

    def test_output_keys(self, extractor: PhysicsFeatureExtractor):
        """Result should contain all expected keys."""
        time_s = np.linspace(0, 100, 50)
        temp = 25.0 + 0.1 * time_s
        result = extractor.compute_thermal_diffusivity(temp, time_s)
        expected_keys = {
            "thermal_diffusivity",
            "mean_diffusivity",
            "max_temp_rate",
            "temp_gradient",
            "time_gradient",
        }
        assert expected_keys.issubset(result.keys())

    def test_short_input(self, extractor: PhysicsFeatureExtractor):
        """Single-element arrays should return empty results."""
        result = extractor.compute_thermal_diffusivity(np.array([25.0]), np.array([0.0]))
        assert len(result["thermal_diffusivity"]) == 0


# ─── Coulombic Efficiency Tests ───────────────────────────────────


class TestComputeCoulombicEfficiency:
    def test_perfect_efficiency(self, extractor: PhysicsFeatureExtractor):
        """Equal charge and discharge capacity should give CE = 1.0."""
        result = extractor.compute_coulombic_efficiency(50.0, 50.0)
        assert pytest.approx(result["efficiency"], abs=1e-6) == 1.0
        assert pytest.approx(result["mean_efficiency"], abs=1e-6) == 1.0

    def test_less_than_perfect(self, extractor: PhysicsFeatureExtractor):
        """Discharge < charge should give CE < 1.0."""
        result = extractor.compute_coulombic_efficiency(48.0, 50.0)
        assert pytest.approx(result["efficiency"], abs=1e-6) == 0.96

    def test_irreversible_loss(self, extractor: PhysicsFeatureExtractor):
        """Irreversible loss should be 1 - CE."""
        result = extractor.compute_coulombic_efficiency(48.0, 50.0)
        assert pytest.approx(result["irreversible_loss"], abs=1e-6) == 0.04

    def test_array_input(self, extractor: PhysicsFeatureExtractor):
        """Array input should return statistics."""
        discharge = np.array([49.0, 48.5, 48.0, 47.5])
        charge = np.array([50.0, 50.0, 50.0, 50.0])
        result = extractor.compute_coulombic_efficiency(discharge, charge)
        assert pytest.approx(result["mean_efficiency"], abs=1e-6) == 0.965
        assert pytest.approx(result["min_efficiency"], abs=1e-6) == 0.95
        assert pytest.approx(result["max_efficiency"], abs=1e-6) == 0.98

    def test_zero_charge_capacity(self, extractor: PhysicsFeatureExtractor):
        """Zero charge capacity should not cause division by zero."""
        result = extractor.compute_coulombic_efficiency(50.0, 0.0)
        assert result["efficiency"] == 0.0


# ─── Integration with EVBatteryAnalyzer ───────────────────────────


class TestIntegrationWithAnalyzer:
    def test_analyzer_has_physics_extractor(self):
        """EVBatteryAnalyzer should have a physics_extractor attribute."""
        from ev_qa_framework.analysis import EVBatteryAnalyzer

        analyzer = EVBatteryAnalyzer()
        assert hasattr(analyzer, "physics_extractor")
        assert isinstance(analyzer.physics_extractor, PhysicsFeatureExtractor)

    def test_get_physics_features_returns_dict(self):
        """get_physics_features should return a dict with expected keys."""
        from ev_qa_framework.analysis import EVBatteryAnalyzer

        analyzer = EVBatteryAnalyzer()
        df = pd.DataFrame(
            {
                "voltage": np.linspace(3.0, 4.2, 50),
                "current": np.full(50, 10.0),
                "temp": np.linspace(25, 35, 50),
                "soc": np.linspace(100, 20, 50),
                "capacity": np.linspace(50, 45, 50),
                "time": np.linspace(0, 100, 50),
                "charge_capacity": np.full(50, 50.0),
                "discharge_capacity": np.full(50, 48.0),
            }
        )
        result = analyzer.get_physics_features(df)
        assert isinstance(result, dict)
        assert "ic_curve" in result
        assert "delta_q" in result
        assert "resistance" in result
        assert "thermal_diffusivity" in result
        assert "coulombic_efficiency" in result

    def test_get_physics_features_missing_columns(self):
        """get_physics_features should handle missing optional columns."""
        from ev_qa_framework.analysis import EVBatteryAnalyzer

        analyzer = EVBatteryAnalyzer()
        df = pd.DataFrame(
            {
                "voltage": [3.0, 3.5, 4.0],
                "current": [10.0, 10.0, 10.0],
                "temp": [25.0, 26.0, 27.0],
                "soc": [80, 75, 70],
            }
        )
        result = analyzer.get_physics_features(df)
        assert result["ic_curve"] is None
        assert result["delta_q"] is None
        assert result["resistance"] is not None
        assert result["thermal_diffusivity"] is not None
        assert result["coulombic_efficiency"] is None
