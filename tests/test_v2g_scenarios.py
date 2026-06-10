"""Tests for V2G scenarios."""

import pandas as pd
import pytest

from ev_qa_framework.v2g_scenarios import V2GScenarioGenerator, V2GHealthAnalyzer
from ev_qa_framework.config import FrameworkConfig


class TestV2GScenarioGenerator:
    def test_generate_v2g_cycle_typical(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_v2g_cycle(24, "typical")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "current" in df.columns
        assert "duration_h" in df.columns

    def test_generate_v2g_cycle_peak_shaving(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_v2g_cycle(24, "peak_shaving")
        assert len(df) == 24

    def test_generate_v2g_cycle_frequency_regulation(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_v2g_cycle(24, "frequency_regulation")
        assert len(df) == 24

    def test_generate_v2g_cycle_unknown_profile(self):
        gen = V2GScenarioGenerator()
        with pytest.raises(ValueError, match="Unknown profile"):
            gen.generate_v2g_cycle(24, "unknown")

    def test_peak_shaving_scenario(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_peak_shaving_scenario()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        # Peak hours should have negative current (discharge)
        assert df.iloc[17]["current"] < 0

    def test_frequency_regulation_aggressive(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_frequency_regulation_scenario("aggressive", 4)
        assert len(df) == 4 * 60  # 1-minute resolution

    def test_frequency_regulation_moderate(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_frequency_regulation_scenario("moderate", 2)
        assert len(df) == 2 * 60

    def test_frequency_regulation_conservative(self):
        gen = V2GScenarioGenerator()
        df = gen.generate_frequency_regulation_scenario("conservative", 1)
        assert len(df) == 60

    def test_frequency_regulation_unknown(self):
        gen = V2GScenarioGenerator()
        with pytest.raises(ValueError, match="Unknown signal profile"):
            gen.generate_frequency_regulation_scenario("unknown")


class TestV2GHealthAnalyzer:
    def test_compute_v2g_impact(self):
        analyzer = V2GHealthAnalyzer()
        baseline = pd.DataFrame({
            "voltage": [400.0] * 10,
            "current": [50.0] * 10,
            "temperature": [30.0] * 10,
            "soh": [95.0] * 10,
        })
        v2g = pd.DataFrame({
            "voltage": [380.0] * 10,
            "current": [70.0] * 10,
            "temperature": [35.0] * 10,
            "soh": [90.0] * 10,
        })
        result = analyzer.compute_v2g_impact(baseline, v2g)
        assert "baseline_score" in result
        assert "v2g_score" in result
        assert "score_delta" in result

    def test_estimate_cycle_life_impact(self):
        analyzer = V2GHealthAnalyzer()
        result = analyzer.estimate_cycle_life_impact(1.0, 0.5)
        assert "equivalent_full_cycles_per_day" in result
        assert "estimated_total_cycles" in result
        assert "estimated_years_to_80_soh" in result
        assert result["equivalent_full_cycles_per_day"] == 0.5

    def test_estimate_high_v2g_usage(self):
        analyzer = V2GHealthAnalyzer()
        result = analyzer.estimate_cycle_life_impact(3.0, 0.8)
        assert abs(result["equivalent_full_cycles_per_day"] - 2.4) < 0.01
        assert result["estimated_years_to_80_soh"] < 5.0

    def test_recommendations_healthy(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(95.0)
        assert len(recs) > 0
        assert any("safe" in r.lower() for r in recs)

    def test_recommendations_good(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(85.0)
        assert len(recs) > 0
        assert any("moderate" in r.lower() for r in recs)

    def test_recommendations_degraded(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(75.0)
        assert len(recs) > 0
        assert any("limit" in r.lower() for r in recs)

    def test_recommendations_critical(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(65.0)
        assert len(recs) > 0
        assert any("critical" in r.lower() for r in recs)

    def test_recommendations_boundary_90(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(90.0)
        assert any("healthy" in r.lower() for r in recs)

    def test_recommendations_boundary_80(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(80.0)
        assert any("good" in r.lower() for r in recs)

    def test_recommendations_boundary_70(self):
        analyzer = V2GHealthAnalyzer()
        recs = analyzer.get_v2g_recommendations(70.0)
        assert any("degradation" in r.lower() for r in recs)
