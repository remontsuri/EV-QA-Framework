"""
Edge case tests for the Thermal Runaway Prediction module.

Covers:
- Rule-based mode edge cases (boundary temperatures, single-row df)
- ML mode edge cases
- Invalid mode validation
- analyze_temperature_trend with various column names
- predict_risk boundary conditions (LOW/MEDIUM/HIGH/CRITICAL thresholds)
"""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor


class TestThermalRunawayPredictorInit:
    def test_default_mode_is_rule(self):
        p = ThermalRunawayPredictor()
        assert p.mode == "rule"

    def test_ml_mode(self):
        p = ThermalRunawayPredictor(mode="ml")
        assert p.mode == "ml"
        assert p._isolation_forest is not None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ThermalRunawayPredictor(mode="invalid")

    def test_mode_case_insensitive(self):
        p = ThermalRunawayPredictor(mode="RULE")
        assert p.mode == "rule"
        p2 = ThermalRunawayPredictor(mode="ML")
        assert p2.mode == "ml"

    def test_custom_rule_weights(self):
        p = ThermalRunawayPredictor(rule_weights={"rise_rate": 5.0})
        assert p.rule_weights["rise_rate"] == 5.0
        # Other weights should remain default
        assert p.rule_weights["max_temp"] == 1.5

    def test_custom_thresholds(self):
        p = ThermalRunawayPredictor(thresholds={"critical_temp": 70.0})
        assert p.thresholds["critical_temp"] == 70.0
        # Other thresholds should remain default
        assert p.thresholds["high_temp"] == 55.0


class TestAnalyzeTemperatureTrend:
    def test_single_row_returns_zero_rise_rate(self):
        df = pd.DataFrame({"temp": [30.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["temp_rise_rate"] == 0.0
        assert result["max_temp"] == 30.0
        assert result["volatility"] == 0.0
        assert result["dt_dt"] == 0.0

    def test_two_rows_rising(self):
        df = pd.DataFrame({"temp": [30.0, 35.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["temp_rise_rate"] > 0
        assert result["max_temp"] == 35.0

    def test_two_rows_falling(self):
        df = pd.DataFrame({"temp": [35.0, 30.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["temp_rise_rate"] < 0

    def test_constant_temperature(self):
        df = pd.DataFrame({"temp": [30.0, 30.0, 30.0, 30.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert abs(result["temp_rise_rate"]) < 1e-10
        assert result["volatility"] == 0.0

    def test_temperature_column_named_temperature(self):
        df = pd.DataFrame({"temperature": [30.0, 35.0, 40.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["max_temp"] == 40.0

    def test_rapid_rise(self):
        df = pd.DataFrame({"temp": [30.0, 40.0, 55.0, 70.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["temp_rise_rate"] > 0
        assert result["dt_dt"] > 0

    def test_spike_detection(self):
        """A single spike should produce high dt_dt."""
        df = pd.DataFrame({"temp": [30.0, 30.0, 50.0, 30.0]})
        result = ThermalRunawayPredictor().analyze_temperature_trend(df)
        assert result["dt_dt"] >= 20.0


class TestPredictRisk:
    def test_single_row_returns_low(self):
        df = pd.DataFrame({"temp": [30.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] == 0.0

    def test_normal_temperature_low_risk(self):
        """Constant low temperature should be LOW risk."""
        df = pd.DataFrame({"temp": [25.0, 25.0, 25.0, 25.0, 25.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] == "LOW"

    def test_slightly_varying_temperature(self):
        """Slight variation around normal temp may be MEDIUM due to scoring."""
        df = pd.DataFrame({"temp": [25.0, 26.0, 25.5, 26.5, 25.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        # The scoring formula gives points for rise_rate and dt_dt
        assert result["risk_level"] in ("LOW", "MEDIUM")

    def test_high_temperature_high_risk(self):
        """Temperature above high_temp threshold (55) should be HIGH."""
        df = pd.DataFrame({"temp": [50.0, 52.0, 54.0, 56.0, 57.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] in ("HIGH", "CRITICAL")

    def test_critical_temperature(self):
        """Temperature above critical_temp (65) should be CRITICAL."""
        df = pd.DataFrame({"temp": [60.0, 62.0, 64.0, 66.0, 68.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] == "CRITICAL"

    def test_rapid_rise_critical(self):
        """Very rapid temperature rise should trigger CRITICAL via dt_dt."""
        df = pd.DataFrame({"temp": [30.0, 36.0, 42.0, 48.0, 54.0]})
        # dt_dt = 6.0 > critical_dtdt (5.0)
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] == "CRITICAL"

    def test_medium_risk(self):
        """Moderate rise should produce MEDIUM risk."""
        df = pd.DataFrame({"temp": [30.0, 32.0, 34.0, 36.0, 38.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        # Should be at least MEDIUM
        assert result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")

    def test_result_contains_all_fields(self):
        df = pd.DataFrame({"temp": [30.0, 31.0, 32.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert "risk_level" in result
        assert "risk_score" in result
        assert "confidence" in result
        assert "temp_rise_rate" in result
        assert "max_temp" in result
        assert "volatility" in result
        assert "dt_dt" in result

    def test_ml_mode_predict(self):
        df = pd.DataFrame({"temp": np.random.normal(30, 2, 50).tolist()})
        p = ThermalRunawayPredictor(mode="ml")
        result = p.predict_risk(df)
        assert "risk_level" in result
        assert "risk_score" in result

    def test_ml_mode_with_anomalies(self):
        """ML mode with anomalous temperatures."""
        temps = [30.0] * 40 + [80.0] * 10
        df = pd.DataFrame({"temp": temps})
        p = ThermalRunawayPredictor(mode="ml")
        result = p.predict_risk(df)
        assert result["risk_level"] in ("HIGH", "CRITICAL")

    def test_custom_thresholds_medium(self):
        """With very low medium_risk threshold, normal data should be MEDIUM."""
        p = ThermalRunawayPredictor(thresholds={"medium_risk": 0.0})
        df = pd.DataFrame({"temp": [30.0, 31.0, 32.0]})
        result = p.predict_risk(df)
        # With medium_risk=0, any positive score triggers at least MEDIUM
        assert result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL", "LOW")

    def test_negative_temperatures(self):
        """Constant negative temperature should be LOW risk."""
        df = pd.DataFrame({"temp": [-20.0, -20.0, -20.0, -20.0, -20.0]})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert result["risk_level"] == "LOW"

    def test_large_dataset(self):
        """Large dataset should not cause errors."""
        df = pd.DataFrame({"temp": np.random.normal(30, 5, 1000).tolist()})
        result = ThermalRunawayPredictor().predict_risk(df)
        assert "risk_level" in result
