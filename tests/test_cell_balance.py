"""Tests for CellBalanceAnalyzer."""

import pytest

from ev_qa_framework.cell_balance import CellBalanceAnalyzer


@pytest.fixture
def analyzer():
    return CellBalanceAnalyzer()


class TestCellBalanceAnalyzer:
    """Suite of tests for cell imbalance detection."""

    def test_normal_balance(self, analyzer):
        """All voltages close together -> NORMAL."""
        voltages = [3.30, 3.31, 3.305, 3.312]
        stats = analyzer.compute_statistics(voltages)
        assert stats["max_min_imbalance"] < analyzer.warning_threshold
        assert analyzer.classify_severity(voltages) == "NORMAL"
        assert analyzer.detect_outliers(voltages) == []

    def test_critical_imbalance(self, analyzer):
        """Large spread -> CRITICAL with outliers."""
        voltages = [3.30, 3.31, 3.50, 3.28, 3.29]
        assert analyzer.classify_severity(voltages) == "CRITICAL"
        outliers = analyzer.detect_outliers(voltages)
        assert len(outliers) > 0

    def test_warning_imbalance(self, analyzer):
        """Moderate spread -> WARNING."""
        voltages = [3.30, 3.31, 3.32, 3.28]
        assert analyzer.classify_severity(voltages) == "WARNING"

    def test_empty_list_raises(self, analyzer):
        """Empty voltage list should raise on statistics."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.compute_statistics([])

    def test_empty_outliers_return_empty(self, analyzer):
        """Empty list for outliers returns empty list."""
        assert analyzer.detect_outliers([]) == []

    def test_empty_severity_normal(self, analyzer):
        """Empty list for severity returns NORMAL."""
        assert analyzer.classify_severity([]) == "NORMAL"

    def test_custom_thresholds(self):
        """Custom thresholds change severity classification."""
        strict = CellBalanceAnalyzer(warning_threshold=0.015, critical_threshold=0.03)
        voltages = [3.30, 3.305, 3.31]
        # max-min = 0.01, below warning_threshold -> NORMAL
        assert strict.classify_severity(voltages) == "NORMAL"

        tight = CellBalanceAnalyzer(warning_threshold=0.005, critical_threshold=0.01)
        assert tight.classify_severity(voltages) == "CRITICAL"

    def test_statistics_values(self, analyzer):
        """Check computed statistics match expectations."""
        voltages = [3.2, 3.4, 3.3]
        stats = analyzer.compute_statistics(voltages)
        assert stats["mean"] == pytest.approx(3.3)
        assert stats["median"] == pytest.approx(3.3)
        assert stats["max"] == 3.4
        assert stats["min"] == 3.2
        assert stats["max_min_imbalance"] == pytest.approx(0.2)

    def test_predict_trend_rising(self, analyzer):
        """Upward trend in imbalance should have positive slope."""
        timeline = [
            [3.30, 3.31, 3.305],
            [3.30, 3.32, 3.305],
            [3.29, 3.33, 3.305],
            [3.28, 3.34, 3.305],
        ]
        slope, intercept = analyzer.predict_trend(timeline)
        assert slope > 0

    def test_predict_trend_insufficient_data(self, analyzer):
        """Fewer than 2 snapshots returns zero slope."""
        assert analyzer.predict_trend([[3.3, 3.31]]) == (0.0, 0.0)
        assert analyzer.predict_trend([]) == (0.0, 0.0)

    def test_plot_imbalance_saves_file(self, analyzer, tmp_path):
        """plot_imbalance should save a PNG file."""
        timeline = [[3.30, 3.31], [3.29, 3.33], [3.28, 3.35]]
        save_path = str(tmp_path / "test_plot.png")
        analyzer.plot_imbalance(timeline, save_path=save_path)
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_empty_raises(self, analyzer):
        """Empty timeline raises ValueError."""
        with pytest.raises(ValueError, match="No data"):
            analyzer.plot_imbalance([])
