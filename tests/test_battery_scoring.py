"""Tests for BatteryScorer — composite battery health scoring system.

Covers:
- Initialization (default and custom weights, invalid weights)
- compute_score with various telemetry profiles
- get_grade boundary values
- get_recommendations for all score ranges
- Component-level behaviour (SOH, anomaly, cell balance, thermal)
- Edge cases: empty data, missing columns, None voltages
"""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.battery_scoring import (
    GRADE_THRESHOLDS,
    BatteryScorer,
)


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


def make_cell_voltages(n_cells: int = 12, base: float = 3.3, spread: float = 0.01) -> list[float]:
    """Generate nearly-balanced cell voltages."""
    rng = np.random.default_rng(0)
    return (base + rng.uniform(0, spread, n_cells)).tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def scorer():
    return BatteryScorer()


@pytest.fixture
def healthy_telemetry():
    return make_telemetry(soh=95.0, temp=30.0)


@pytest.fixture
def degraded_telemetry():
    return make_telemetry(soh=55.0, temp=55.0, n=100)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestBatteryScorerInit:
    """Weight validation and construction."""

    def test_default_weights_sum_to_one(self):
        scorer = BatteryScorer()
        total = (
            scorer.soh_weight
            + scorer.anomaly_weight
            + scorer.cell_balance_weight
            + scorer.thermal_weight
        )
        assert total == pytest.approx(1.0)

    def test_custom_weights(self):
        scorer = BatteryScorer(
            soh_weight=0.25,
            anomaly_weight=0.25,
            cell_balance_weight=0.25,
            thermal_weight=0.25,
        )
        assert scorer.soh_weight == 0.25

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            BatteryScorer(
                soh_weight=0.5, anomaly_weight=0.5, cell_balance_weight=0.5, thermal_weight=0.5
            )

    def test_init_with_cell_voltages(self):
        scorer = BatteryScorer(cell_voltages=[3.3] * 12)
        assert scorer.cell_voltages == [3.3] * 12

    def test_init_with_soh_baseline(self):
        scorer = BatteryScorer(soh_baseline=88.0)
        assert scorer.soh_baseline == 88.0


class TestComputeScore:
    """End-to-end score computation."""

    def test_returns_expected_keys(self, scorer, healthy_telemetry):
        result = scorer.compute_score(healthy_telemetry)
        expected_keys = {
            "score",
            "grade",
            "soh_score",
            "anomaly_score",
            "cell_balance_score",
            "thermal_score",
            "details",
        }
        assert expected_keys.issubset(result.keys())

    def test_score_in_range(self, scorer, healthy_telemetry):
        result = scorer.compute_score(healthy_telemetry)
        assert 0.0 <= result["score"] <= 100.0

    def test_healthy_battery_high_score(self, scorer):
        df = make_telemetry(soh=97.0, temp=28.0, n=100)
        result = scorer.compute_score(df, cell_voltages=make_cell_voltages())
        assert result["score"] >= 70

    def test_degraded_battery_lower_score(self, scorer):
        df = make_telemetry(soh=50.0, temp=58.0, n=100)
        bad_cells = [3.3] * 11 + [2.8]  # one outlier cell
        result = scorer.compute_score(df, cell_voltages=bad_cells)
        assert result["score"] < 70

    def test_with_cell_voltages_override(self, scorer):
        df = make_telemetry(soh=95.0)
        good_cells = make_cell_voltages(spread=0.005)
        bad_cells = [3.3] * 11 + [2.5]
        score_good = scorer.compute_score(df, cell_voltages=good_cells)
        score_bad = scorer.compute_score(df, cell_voltages=bad_cells)
        assert score_good["cell_balance_score"] > score_bad["cell_balance_score"]

    def test_details_contain_recommendations(self, scorer, healthy_telemetry):
        result = scorer.compute_score(healthy_telemetry)
        assert "recommendations" in result["details"]
        assert isinstance(result["details"]["recommendations"], list)
        assert len(result["details"]["recommendations"]) > 0

    def test_details_contain_weights(self, scorer, healthy_telemetry):
        result = scorer.compute_score(healthy_telemetry)
        w = result["details"]["weights"]
        assert w["soh"] == pytest.approx(0.4)
        assert w["anomaly"] == pytest.approx(0.15)  # FIX: was 0.25
        assert w["cell_balance"] == pytest.approx(0.20)
        assert w["thermal"] == pytest.approx(0.25)  # FIX: was 0.15

    def test_temperature_column_alias(self, scorer):
        """'temperature' column should be accepted as alias for 'temp'."""
        df = make_telemetry(soh=95.0)
        df = df.rename(columns={"temp": "temperature"})
        result = scorer.compute_score(df)
        assert 0.0 <= result["score"] <= 100.0

    def test_no_soh_column_uses_baseline(self):
        scorer = BatteryScorer(soh_baseline=80.0)
        df = make_telemetry(soh=None)
        result = scorer.compute_score(df)
        assert result["soh_score"] == 80.0

    def test_no_soh_no_baseline_returns_100(self, scorer):
        df = make_telemetry(soh=None)
        result = scorer.compute_score(df)
        assert result["soh_score"] == 100.0


class TestGetGrade:
    """Letter grade boundaries."""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (100, "A"),
            (95, "A"),
            (90, "A"),
            (89, "B"),
            (75, "B"),
            (74, "C"),
            (60, "C"),
            (59, "D"),
            (40, "D"),
            (39, "F"),
            (0, "F"),
        ],
    )
    def test_grade_boundaries(self, score, expected):
        assert BatteryScorer.get_grade(score) == expected

    def test_grade_a_from_details(self, scorer):
        result = scorer.compute_score(make_telemetry(soh=98.0, temp=25.0, n=100))
        if result["score"] >= 90:
            assert result["grade"] == "A"


class TestGetRecommendations:
    """Recommendation generation."""

    def test_excellent_score_no_critical(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 95,
                "soh_score": 95,
                "anomaly_score": 95,
                "cell_balance_score": 95,
                "thermal_score": 95,
            }
        )
        assert any("excellent" in r.lower() for r in recs)

    def test_low_soh_critical(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 50,
                "soh_score": 45,
                "anomaly_score": 80,
                "cell_balance_score": 90,
                "thermal_score": 90,
            }
        )
        assert any("SOH" in r for r in recs)
        assert any(r.startswith("CRITICAL") for r in recs)

    def test_low_anomaly_warning(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 70,
                "soh_score": 85,
                "anomaly_score": 70,
                "cell_balance_score": 90,
                "thermal_score": 90,
            }
        )
        assert any("anomal" in r.lower() for r in recs)

    def test_low_cell_balance_critical(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 45,
                "soh_score": 80,
                "anomaly_score": 80,
                "cell_balance_score": 30,
                "thermal_score": 90,
            }
        )
        assert any("imbalance" in r.lower() for r in recs)

    def test_low_thermal_critical(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 40,
                "soh_score": 80,
                "anomaly_score": 80,
                "cell_balance_score": 90,
                "thermal_score": 30,
            }
        )
        assert any("thermal" in r.lower() for r in recs)

    def test_all_low_multiple_recommendations(self):
        recs = BatteryScorer.get_recommendations(
            {
                "score": 20,
                "soh_score": 30,
                "anomaly_score": 30,
                "cell_balance_score": 30,
                "thermal_score": 30,
            }
        )
        assert len(recs) >= 4  # one per failing component


class TestComponentScores:
    """Verify individual component scoring logic."""

    def test_soh_from_dataframe_column(self, scorer):
        df = make_telemetry(soh=88.0)
        result = scorer.compute_score(df)
        assert result["soh_score"] == pytest.approx(87.0, abs=2.0)

    def test_perfect_cell_balance(self, scorer):
        df = make_telemetry(soh=95.0)
        cells = [3.3] * 12
        result = scorer.compute_score(df, cell_voltages=cells)
        assert result["cell_balance_score"] == 100.0

    def test_critical_cell_imbalance(self, scorer):
        df = make_telemetry(soh=95.0)
        cells = [3.3] * 11 + [2.0]
        result = scorer.compute_score(df, cell_voltages=cells)
        assert result["cell_balance_score"] == 30.0

    def test_no_cell_voltages_returns_100(self, scorer):
        df = make_telemetry(soh=95.0)
        result = scorer.compute_score(df)
        assert result["cell_balance_score"] == 100.0

    def test_low_temp_thermal_score(self, scorer):
        """Constant low temperature should yield LOW risk → score 100."""
        n = 100
        df = pd.DataFrame(
            {
                "voltage": np.full(n, 400.0),
                "current": np.full(n, 100.0),
                "temp": np.full(n, 25.0),
                "soc": np.full(n, 85.0),
                "soh": np.full(n, 95.0),
            }
        )
        result = scorer.compute_score(df)
        assert result["thermal_score"] == 100.0

    def test_high_temp_thermal_score(self, scorer):
        df = make_telemetry(soh=95.0, temp=60.0, n=100)
        result = scorer.compute_score(df)
        assert result["thermal_score"] < 100.0


class TestEdgeCases:
    """Edge-case handling."""

    def test_single_row_telemetry(self, scorer):
        df = make_telemetry(n=1, soh=90.0)
        result = scorer.compute_score(df)
        assert 0.0 <= result["score"] <= 100.0

    def test_empty_cell_voltages_list(self, scorer):
        df = make_telemetry(soh=90.0)
        result = scorer.compute_score(df, cell_voltages=[])
        assert result["cell_balance_score"] == 100.0

    def test_score_clamped_to_100(self):
        """Even with perfect inputs, score should not exceed 100."""
        scorer = BatteryScorer(soh_baseline=100.0)
        df = make_telemetry(soh=100.0, temp=20.0, n=100)
        result = scorer.compute_score(df, cell_voltages=[3.3] * 12)
        assert result["score"] <= 100.0

    def test_score_clamped_to_0(self):
        """Even with terrible inputs, score should not go below 0."""
        scorer = BatteryScorer(soh_baseline=0.0)
        df = make_telemetry(soh=0.0, temp=80.0, n=100)
        result = scorer.compute_score(df, cell_voltages=[4.2] * 11 + [1.0])
        assert result["score"] >= 0.0
