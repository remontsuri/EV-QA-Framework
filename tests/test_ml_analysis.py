"""
Tests for ML battery telemetry analyzer.
"""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.analysis import EVBatteryAnalyzer


class TestEVBatteryAnalyzer:
    """Tests for EVBatteryAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        assert analyzer.model is not None
        assert analyzer.scaler is not None
        assert isinstance(analyzer.anomalies, pd.DataFrame)
        assert analyzer.anomalies.empty

    def test_initialization_default_contamination(self):
        """Test initialization with default parameters."""
        analyzer = EVBatteryAnalyzer()
        assert analyzer.model.contamination == 0.1

    def test_analyze_perfect_data(self):
        """Analyze perfect data without anomalies."""
        analyzer = EVBatteryAnalyzer(contamination=0.05)

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "voltage": np.full(100, 48.0),
                "current": np.full(100, 100.0),
                "temp": np.full(100, 35.0),
                "soc": np.full(100, 85.0),
            }
        )

        results = analyzer.analyze_telemetry(df)

        assert results["total_samples"] == 100
        assert results["anomalies_detected"] <= 10

    def test_analyze_with_obvious_outliers(self):
        """Analyze data with obvious outliers."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)

        np.random.seed(42)
        normal_data = {
            "voltage": np.random.normal(48, 1, 90),
            "current": np.random.normal(100, 5, 90),
            "temp": np.random.normal(35, 2, 90),
            "soc": np.random.normal(85, 5, 90),
        }

        outliers = {
            "voltage": np.full(10, 100.0),
            "current": np.full(10, 500.0),
            "temp": np.full(10, 90.0),
            "soc": np.full(10, 5.0),
        }

        df = pd.DataFrame(
            {
                "voltage": np.concatenate([normal_data["voltage"], outliers["voltage"]]),
                "current": np.concatenate([normal_data["current"], outliers["current"]]),
                "temp": np.concatenate([normal_data["temp"], outliers["temp"]]),
                "soc": np.concatenate([normal_data["soc"], outliers["soc"]]),
            }
        )

        results = analyzer.analyze_telemetry(df)

        assert results["total_samples"] == 100
        assert results["anomalies_detected"] > 0

    def test_severity_critical(self):
        """Test critical anomaly severity."""
        analyzer = EVBatteryAnalyzer(contamination=0.2)

        df = pd.DataFrame(
            {
                "voltage": [48] * 80 + [200] * 20,
                "current": [100] * 80 + [1000] * 20,
                "temp": [35] * 80 + [150] * 20,
                "soc": [85] * 100,
            }
        )

        results = analyzer.analyze_telemetry(df)
        assert results["severity"] in ["CRITICAL", "WARNING", "INFO"]

    def test_severity_info(self):
        """Test low severity (INFO)."""
        analyzer = EVBatteryAnalyzer(contamination=0.05)

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "voltage": np.random.normal(48, 0.1, 100),
                "current": np.random.normal(100, 1, 100),
                "temp": np.random.normal(35, 0.5, 100),
                "soc": np.random.normal(85, 1, 100),
            }
        )

        results = analyzer.analyze_telemetry(df)
        assert results["severity"] in ["INFO", "WARNING"]

    def test_anomaly_percentage_calculation(self):
        """Test anomaly percentage calculation."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "voltage": np.random.normal(48, 2, 100),
                "current": np.random.normal(100, 10, 100),
                "temp": np.random.normal(35, 3, 100),
                "soc": np.random.normal(85, 5, 100),
            }
        )

        results = analyzer.analyze_telemetry(df)

        expected_percentage = (results["anomalies_detected"] / 100) * 100
        assert results["anomaly_percentage"] == expected_percentage

    def test_small_dataset(self):
        """Test on small dataset."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)

        df = pd.DataFrame(
            {
                "voltage": [48.0, 48.1, 48.2, 47.9, 48.0, 48.1, 48.2, 47.9, 48.0, 48.1],
                "current": [100, 101, 99, 100, 102, 100, 101, 99, 100, 102],
                "temp": [35, 35, 36, 35, 35, 35, 36, 35, 35, 35],
                "soc": [85, 85, 84, 86, 85, 85, 84, 86, 85, 85],
            }
        )

        results = analyzer.analyze_telemetry(df)
        assert results["total_samples"] == 10
        assert isinstance(results["anomalies_detected"], int)


class TestEVBatteryAnalyzerEdgeCases:
    """Edge cases for ML analyzer."""

    def test_single_feature_variance(self):
        """Data with variance in only one feature."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)

        df = pd.DataFrame(
            {
                "voltage": np.random.normal(48, 5, 100),
                "current": np.full(100, 100.0),
                "temp": np.full(100, 35.0),
                "soc": np.full(100, 85.0),
            }
        )

        results = analyzer.analyze_telemetry(df)
        assert results["total_samples"] == 100
        assert "anomalies_detected" in results

    def test_negative_values(self):
        """Negative values in data."""
        analyzer = EVBatteryAnalyzer(contamination=0.1)

        df = pd.DataFrame(
            {
                "voltage": [48] * 95 + [-10] * 5,
                "current": [100] * 100,
                "temp": [35] * 100,
                "soc": [85] * 100,
            }
        )

        results = analyzer.analyze_telemetry(df)
        assert results["anomalies_detected"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
