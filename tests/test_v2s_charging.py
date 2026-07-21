"""Tests for V2SScenarioGenerator and ChargingStationSimulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.v2g_scenarios import (
    ChargingStationSimulator,
    V2SScenarioGenerator,
)


class TestV2SScenarioGenerator:
    """Test V2SScenarioGenerator class."""

    def setup_method(self):
        self.generator = V2SScenarioGenerator(battery_capacity_ah=100.0, nominal_voltage=400.0)

    def test_ac_slow_station(self):
        df = self.generator.generate_charging_station_profile("ac_slow", duration_hours=4)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 240  # 4 hours * 60 samples
        assert list(df.columns) == ["current", "voltage", "duration_h", "soc"]
        assert df["current"].mean() == pytest.approx(17.5, rel=0.1)
        assert df["soc"].iloc[0] == pytest.approx(20.0, abs=1.0)
        assert df["soc"].iloc[-1] == pytest.approx(95.0, abs=1.0)

    def test_dc_fast_station(self):
        df = self.generator.generate_charging_station_profile("dc_fast", duration_hours=2)
        assert len(df) == 120
        assert df["current"].mean() == pytest.approx(125.0, rel=0.1)

    def test_dc_ultra_station(self):
        df = self.generator.generate_charging_station_profile("dc_ultra", duration_hours=1)
        assert len(df) == 60
        assert df["current"].mean() == pytest.approx(375.0, rel=0.1)

    def test_unknown_station_raises(self):
        with pytest.raises(ValueError, match="Unknown station type"):
            self.generator.generate_charging_station_profile("unknown_type")

    def test_v2s_peak_shaving(self):
        df = self.generator.generate_v2s_dispatch("peak_shaving", duration_hours=24)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1440
        assert list(df.columns) == ["current", "voltage", "duration_h", "soc"]
        assert df["soc"].iloc[0] > df["soc"].iloc[-1]  # SOC decreases

    def test_v2s_frequency_regulation(self):
        df = self.generator.generate_v2s_dispatch("frequency_regulation", duration_hours=4)
        assert len(df) == 240
        assert np.all(df["soc"] == 50.0)  # SOC maintained at 50%

    def test_v2s_solar_buffering(self):
        df = self.generator.generate_v2s_dispatch("solar_buffering", duration_hours=24)
        assert len(df) == 1440
        assert df["soc"].iloc[-1] > df["soc"].iloc[0]  # SOC increases

    def test_unknown_grid_signal_raises(self):
        with pytest.raises(ValueError, match="Unknown grid signal"):
            self.generator.generate_v2s_dispatch("unknown_signal")


class TestChargingStationSimulator:
    """Test ChargingStationSimulator class."""

    def setup_method(self):
        self.simulator = ChargingStationSimulator(battery_capacity_ah=100.0, nominal_voltage=400.0)

    def test_basic_charging_session(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=20.0, target_soc=80.0
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ["timestamp", "current", "voltage", "soc", "energy_delivered_kwh"]
        assert df["soc"].iloc[-1] >= 80.0
        assert df["energy_delivered_kwh"].sum() > 0

    def test_soc_increases(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=20.0, target_soc=95.0
        )
        assert df["soc"].is_monotonic_increasing

    def test_cc_cv_voltage_curve(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=20.0, target_soc=95.0
        )
        # Voltage should increase then plateau
        voltage = df["voltage"].values
        assert np.all(np.diff(voltage) >= -1e-6)  # voltage non-decreasing

    def test_current_tapers(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=20.0, target_soc=95.0
        )
        current = df["current"].values
        # Current should taper in CV phase
        assert current[-1] < current[len(current) // 2]

    def test_target_soc_less_than_initial_raises(self):
        with pytest.raises(ValueError, match="target_soc must be >= initial_soc"):
            self.simulator.simulate_charging_session(
                station_power_kw=50.0, initial_soc=80.0, target_soc=50.0
            )

    def test_zero_duration_returns_empty(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=80.0, target_soc=80.0
        )
        assert len(df) == 0

    def test_energy_delivered_positive(self):
        df = self.simulator.simulate_charging_session(
            station_power_kw=50.0, initial_soc=30.0, target_soc=90.0
        )
        assert (df["energy_delivered_kwh"] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
