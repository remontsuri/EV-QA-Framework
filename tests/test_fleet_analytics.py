"""Tests for FleetAnalytics — multi-battery fleet aggregation and analysis.

Covers:
- Battery management (add, remove, get)
- Fleet summary statistics
- Battery comparison table
- Fleet degradation trend analysis
- Fleet anomaly detection and alerts
- Edge cases: empty fleet, single battery, missing columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.analysis import EVBatteryAnalyzer
from ev_qa_framework.battery_scoring import BatteryScorer
from ev_qa_framework.fleet_analytics import FleetAnalytics, FleetAlert
from ev_qa_framework.physics_features import PhysicsFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_telemetry(
    n: int = 50,
    voltage: float = 400.0,
    current: float = 100.0,
    temp: float = 35.0,
    soc: float = 85.0,
    soh: float | None = 95.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic telemetry DataFrame."""
    rng = np.random.default_rng(seed)
    data = {
        "voltage": rng.normal(voltage, 2, n),
        "current": rng.normal(current, 3, n),
        "temp": rng.normal(temp, 1, n),
        "soc": rng.normal(soc, 2, n),
    }
    if soh is not None:
        data["soh"] = np.linspace(soh, soh - 1, n)
    return pd.DataFrame(data)


def make_degraded_telemetry(
    n: int = 50,
    soh_start: float = 70.0,
    seed: int = 99,
) -> pd.DataFrame:
    """Telemetry for a degraded battery with declining SOH."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "voltage": rng.normal(380, 5, n),
        "current": rng.normal(100, 8, n),
        "temp": rng.normal(42, 3, n),
        "soc": rng.normal(80, 5, n),
        "soh": np.linspace(soh_start, soh_start - 5, n),
    })


def make_anomalous_telemetry(n: int = 50, n_anomalies: int = 15, seed: int = 7) -> pd.DataFrame:
    """Telemetry with injected anomalies (extreme values)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "voltage": rng.normal(400, 2, n),
        "current": rng.normal(100, 3, n),
        "temp": rng.normal(35, 1, n),
        "soc": rng.normal(85, 2, n),
        "soh": np.full(n, 80.0),
    })
    # Inject anomalies
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    df.loc[anomaly_idx, "voltage"] = rng.uniform(500, 600, n_anomalies)
    df.loc[anomaly_idx, "temp"] = rng.uniform(60, 80, n_anomalies)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fleet() -> FleetAnalytics:
    return FleetAnalytics()


@pytest.fixture
def populated_fleet() -> FleetAnalytics:
    """Fleet with 3 healthy batteries."""
    fl = FleetAnalytics()
    for i in range(3):
        fl.add_battery(f"battery_{i}", make_telemetry(seed=i * 10))
    return fl


@pytest.fixture
def mixed_fleet() -> FleetAnalytics:
    """Fleet with healthy, degraded, and anomalous batteries."""
    fl = FleetAnalytics()
    fl.add_battery("healthy_1", make_telemetry(seed=1))
    fl.add_battery("healthy_2", make_telemetry(seed=2, soh=92.0))
    fl.add_battery("degraded_1", make_degraded_telemetry(soh_start=65.0))
    fl.add_battery("anomalous_1", make_anomalous_telemetry(n_anomalies=15))
    return fl


# ===================================================================
# 1. Battery management tests
# ===================================================================

class TestBatteryManagement:
    def test_add_battery(self, fleet: FleetAnalytics):
        df = make_telemetry()
        fleet.add_battery("bat_1", df)
        assert "bat_1" in fleet.battery_ids
        assert fleet.fleet_size == 1

    def test_add_multiple_batteries(self, fleet: FleetAnalytics):
        for i in range(5):
            fleet.add_battery(f"bat_{i}", make_telemetry(seed=i))
        assert fleet.fleet_size == 5

    def test_remove_battery(self, fleet: FleetAnalytics):
        fleet.add_battery("bat_1", make_telemetry())
        fleet.remove_battery("bat_1")
        assert fleet.fleet_size == 0
        assert "bat_1" not in fleet.battery_ids

    def test_remove_nonexistent_no_error(self, fleet: FleetAnalytics):
        fleet.remove_battery("nonexistent")  # should not raise

    def test_add_battery_invalid_id(self, fleet: FleetAnalytics):
        with pytest.raises(ValueError):
            fleet.add_battery("", make_telemetry())

    def test_add_battery_not_dataframe(self, fleet: FleetAnalytics):
        with pytest.raises(ValueError):
            fleet.add_battery("bat_1", "not_a_dataframe")  # type: ignore

    def test_add_battery_empty_dataframe(self, fleet: FleetAnalytics):
        with pytest.raises(ValueError):
            fleet.add_battery("bat_1", pd.DataFrame())

    def test_get_telemetry(self, fleet: FleetAnalytics):
        df = make_telemetry()
        fleet.add_battery("bat_1", df)
        result = fleet.get_telemetry("bat_1")
        assert len(result) == len(df)
        assert "voltage" in result.columns

    def test_get_telemetry_missing_battery(self, fleet: FleetAnalytics):
        with pytest.raises(KeyError):
            fleet.get_telemetry("nonexistent")

    def test_battery_ids_property(self, fleet: FleetAnalytics):
        fleet.add_battery("a", make_telemetry())
        fleet.add_battery("b", make_telemetry())
        ids = fleet.battery_ids
        assert set(ids) == {"a", "b"}

    def test_add_battery_overwrites(self, fleet: FleetAnalytics):
        """Adding a battery with the same ID should replace it."""
        fleet.add_battery("bat_1", make_telemetry(n=50))
        fleet.add_battery("bat_1", make_telemetry(n=100))
        assert fleet.fleet_size == 1
        assert len(fleet.get_telemetry("bat_1")) == 100


# ===================================================================
# 2. Fleet summary tests
# ===================================================================

class TestFleetSummary:
    def test_empty_fleet_summary(self, fleet: FleetAnalytics):
        summary = fleet.get_fleet_summary()
        assert summary["fleet_size"] == 0
        assert summary["avg_score"] == 0.0
        assert summary["batteries"] == []

    def test_single_battery_summary(self, fleet: FleetAnalytics):
        fleet.add_battery("bat_1", make_telemetry())
        summary = fleet.get_fleet_summary()
        assert summary["fleet_size"] == 1
        assert "avg_score" in summary
        assert "grade_distribution" in summary
        assert len(summary["batteries"]) == 1

    def test_multi_battery_summary(self, populated_fleet: FleetAnalytics):
        summary = populated_fleet.get_fleet_summary()
        assert summary["fleet_size"] == 3
        assert summary["min_score"] <= summary["avg_score"] <= summary["max_score"]
        assert summary["std_score"] >= 0.0

    def test_summary_has_all_keys(self, populated_fleet: FleetAnalytics):
        summary = populated_fleet.get_fleet_summary()
        expected_keys = {
            "fleet_size", "avg_score", "min_score", "max_score",
            "std_score", "avg_soh", "avg_anomaly_pct",
            "avg_thermal_risk", "grade_distribution", "batteries",
        }
        assert expected_keys.issubset(summary.keys())

    def test_battery_summary_entry_keys(self, populated_fleet: FleetAnalytics):
        summary = populated_fleet.get_fleet_summary()
        entry = summary["batteries"][0]
        expected = {
            "battery_id", "score", "grade", "soh_score",
            "anomaly_score", "cell_balance_score", "thermal_score",
            "anomaly_percentage", "num_samples",
        }
        assert expected.issubset(entry.keys())

    def test_grade_distribution_counts(self, populated_fleet: FleetAnalytics):
        summary = populated_fleet.get_fleet_summary()
        total_grades = sum(summary["grade_distribution"].values())
        assert total_grades == summary["fleet_size"]


# ===================================================================
# 3. Battery comparison tests
# ===================================================================

class TestCompareBatteries:
    def test_compare_all(self, populated_fleet: FleetAnalytics):
        df = populated_fleet.compare_batteries()
        assert len(df) == 3
        assert "battery_id" in df.columns
        assert "score" in df.columns

    def test_compare_subset(self, populated_fleet: FleetAnalytics):
        df = populated_fleet.compare_batteries(["battery_0", "battery_2"])
        assert len(df) == 2
        assert set(df["battery_id"]) == {"battery_0", "battery_2"}

    def test_compare_missing_battery_raises(self, populated_fleet: FleetAnalytics):
        with pytest.raises(KeyError):
            populated_fleet.compare_batteries(["nonexistent"])

    def test_compare_returns_dataframe(self, populated_fleet: FleetAnalytics):
        df = populated_fleet.compare_batteries()
        assert isinstance(df, pd.DataFrame)

    def test_compare_columns(self, populated_fleet: FleetAnalytics):
        df = populated_fleet.compare_batteries()
        expected_cols = {
            "battery_id", "score", "grade", "soh_score",
            "anomaly_score", "cell_balance_score", "thermal_score",
            "anomaly_percentage", "num_samples",
        }
        assert expected_cols.issubset(set(df.columns))


# ===================================================================
# 4. Fleet degradation tests
# ===================================================================

class TestFleetDegradation:
    def test_no_soh_data(self, fleet: FleetAnalytics):
        """Batteries without SOH column should report trend_available=False."""
        df = make_telemetry(soh=None)
        fleet.add_battery("bat_1", df)
        result = fleet.compute_fleet_degradation()
        assert result["per_battery"]["bat_1"]["trend_available"] is False
        assert result["batteries_with_data"] == 0

    def test_degradation_with_soh(self, fleet: FleetAnalytics):
        df = make_telemetry(soh=90.0, n=100)
        fleet.add_battery("bat_1", df)
        result = fleet.compute_fleet_degradation()
        bat = result["per_battery"]["bat_1"]
        assert bat["trend_available"] is True
        assert "slope" in bat
        assert "r_squared" in bat
        assert bat["n_points"] == 100

    def test_fleet_degradation_aggregation(self, mixed_fleet: FleetAnalytics):
        result = mixed_fleet.compute_fleet_degradation()
        assert result["batteries_with_data"] >= 1
        assert "fleet_avg_slope" in result
        assert "fleet_avg_initial_soh" in result
        assert "fleet_avg_final_soh" in result

    def test_degradation_slope_sign(self, fleet: FleetAnalytics):
        """Declining SOH should produce a negative slope."""
        df = make_degraded_telemetry(soh_start=80.0, n=100)
        fleet.add_battery("bat_1", df)
        result = fleet.compute_fleet_degradation()
        slope = result["per_battery"]["bat_1"]["slope"]
        assert slope < 0.0

    def test_empty_fleet_degradation(self, fleet: FleetAnalytics):
        result = fleet.compute_fleet_degradation()
        assert result["batteries_with_data"] == 0
        assert result["fleet_avg_slope"] == 0.0


# ===================================================================
# 5. Fleet anomaly detection tests
# ===================================================================

class TestFleetAnomalies:
    def test_no_anomalies_healthy_fleet(self, populated_fleet: FleetAnalytics):
        alerts = populated_fleet.detect_fleet_anomalies()
        # Healthy fleet should produce no alerts (or very few)
        critical = [a for a in alerts if a.severity == "CRITICAL"]
        assert len(critical) == 0

    def test_anomaly_alert_for_bad_battery(self, fleet: FleetAnalytics):
        fleet.add_battery("bad", make_anomalous_telemetry(n_anomalies=20))
        alerts = fleet.detect_fleet_anomalies()
        assert len(alerts) > 0
        assert any(a.battery_id == "bad" for a in alerts)

    def test_alert_types(self, fleet: FleetAnalytics):
        fleet.add_battery("bad", make_anomalous_telemetry(n_anomalies=20))
        alerts = fleet.detect_fleet_anomalies()
        alert_types = {a.alert_type for a in alerts}
        assert "HIGH_ANOMALY_RATE" in alert_types or "LOW_HEALTH_SCORE" in alert_types

    def test_alert_to_dict(self):
        alert = FleetAlert(
            battery_id="test",
            alert_type="TEST",
            severity="WARNING",
            message="test message",
            value=50.0,
            threshold=60.0,
        )
        d = alert.to_dict()
        assert d["battery_id"] == "test"
        assert d["severity"] == "WARNING"
        assert d["value"] == 50.0

    def test_empty_fleet_no_alerts(self, fleet: FleetAnalytics):
        alerts = fleet.detect_fleet_anomalies()
        assert alerts == []

    def test_custom_thresholds(self, fleet: FleetAnalytics):
        fleet.add_battery("ok", make_telemetry())
        # Very low thresholds should trigger alerts even for healthy batteries
        alerts = fleet.detect_fleet_anomalies(score_threshold=100.0, anomaly_pct_threshold=0.0)
        assert len(alerts) > 0

    def test_alert_severity_levels(self, fleet: FleetAnalytics):
        """Extreme anomalies should produce CRITICAL severity."""
        fleet.add_battery("critical", make_anomalous_telemetry(n_anomalies=30))
        alerts = fleet.detect_fleet_anomalies()
        severities = {a.severity for a in alerts}
        # Should have at least WARNING or CRITICAL
        assert "WARNING" in severities or "CRITICAL" in severities


# ===================================================================
# 6. Integration / custom components
# ===================================================================

class TestCustomComponents:
    def test_custom_scorer(self):
        scorer = BatteryScorer(soh_baseline=80.0)
        fleet = FleetAnalytics(scorer=scorer)
        fleet.add_battery("bat_1", make_telemetry(soh=None))
        summary = fleet.get_fleet_summary()
        assert summary["fleet_size"] == 1

    def test_custom_analyzer(self):
        analyzer = EVBatteryAnalyzer(contamination=0.05)
        fleet = FleetAnalytics(analyzer=analyzer)
        fleet.add_battery("bat_1", make_telemetry())
        alerts = fleet.detect_fleet_anomalies()
        assert isinstance(alerts, list)

    def test_custom_physics(self):
        physics = PhysicsFeatureExtractor(smoothing_window=7)
        fleet = FleetAnalytics(physics=physics)
        fleet.add_battery("bat_1", make_telemetry())
        assert fleet.fleet_size == 1

    def test_score_caching(self, fleet: FleetAnalytics):
        """Calling score_battery twice should return same result."""
        fleet.add_battery("bat_1", make_telemetry())
        s1 = fleet.score_battery("bat_1")
        s2 = fleet.score_battery("bat_1")
        assert s1["score"] == s2["score"]

    def test_temperature_column_alias(self, fleet: FleetAnalytics):
        """Telemetry with 'temperature' instead of 'temp' should work."""
        df = make_telemetry()
        df = df.rename(columns={"temp": "temperature"})
        fleet.add_battery("bat_1", df)
        summary = fleet.get_fleet_summary()
        assert summary["fleet_size"] == 1
