"""Tests for Battery Digital Twin."""

import pandas as pd

from ev_qa_framework.config import FrameworkConfig
from ev_qa_framework.digital_twin import BatteryDigitalTwin, BatteryState


class TestBatteryState:
    def test_default_values(self):
        s = BatteryState()
        assert s.voltage == 400.0
        assert s.soc == 80.0
        assert s.soh == 100.0
        assert s.cycle_count == 0.0

    def test_to_dict(self):
        s = BatteryState()
        d = s.to_dict()
        assert isinstance(d, dict)
        assert "voltage" in d
        assert "soh" in d
        assert len(d) == 8


class TestBatteryDigitalTwinInit:
    def test_default_init(self):
        twin = BatteryDigitalTwin()
        assert twin.config is not None
        assert twin.state.soh == 100.0
        assert twin.state.cycle_count == 0.0

    def test_custom_config(self):
        config = FrameworkConfig()
        twin = BatteryDigitalTwin(config=config)
        assert twin.config is config

    def test_reset(self):
        twin = BatteryDigitalTwin()
        twin.step(1.0, 50.0)
        twin.reset()
        assert twin.state.soh == 100.0
        assert twin.state.cycle_count == 0.0
        assert len(twin._history) == 0

    def test_get_state(self):
        twin = BatteryDigitalTwin()
        state = twin.get_state()
        assert isinstance(state, dict)
        assert "voltage" in state


class TestStep:
    def test_charge_increases_soc(self):
        twin = BatteryDigitalTwin()
        initial_soc = twin.state.soc
        twin.step(1.0, 50.0)  # charge
        assert twin.state.soc > initial_soc

    def test_discharge_decreases_soc(self):
        twin = BatteryDigitalTwin()
        initial_soc = twin.state.soc
        twin.step(1.0, -50.0)  # discharge
        assert twin.state.soc < initial_soc

    def test_soc_clamped_to_100(self):
        twin = BatteryDigitalTwin()
        twin.state.soc = 99.0
        twin.step(10.0, 1000.0)  # massive charge
        assert twin.state.soc <= 100.0

    def test_soc_clamped_to_0(self):
        twin = BatteryDigitalTwin()
        twin.state.soc = 1.0
        twin.step(10.0, -1000.0)  # massive discharge
        assert twin.state.soc >= 0.0

    def test_temperature_increases_with_current(self):
        twin = BatteryDigitalTwin()
        initial_temp = twin.state.temperature
        twin.step(1.0, 200.0)  # high current
        assert twin.state.temperature > initial_temp

    def test_cycle_count_increments(self):
        twin = BatteryDigitalTwin()
        initial_cycles = twin.state.cycle_count
        twin.step(1.0, 50.0)
        assert twin.state.cycle_count > initial_cycles

    def test_history_recorded(self):
        twin = BatteryDigitalTwin()
        twin.step(1.0, 10.0)
        twin.step(1.0, -10.0)
        assert len(twin._history) == 2

    def test_soh_decreases_over_time(self):
        twin = BatteryDigitalTwin()
        initial_soh = twin.state.soh
        for _ in range(100):
            twin.step(1.0, 50.0)
        assert twin.state.soh < initial_soh

    def test_resistance_increases_with_cycles(self):
        twin = BatteryDigitalTwin()
        initial_r = twin.state.internal_resistance
        for _ in range(50):
            twin.step(1.0, 50.0)
        assert twin.state.internal_resistance > initial_r


class TestSimulateDriveCycle:
    def test_returns_dataframe(self):
        twin = BatteryDigitalTwin()
        profile = pd.DataFrame({"current": [10.0, -10.0, 5.0, -5.0]})
        result = twin.simulate_drive_cycle(profile, dt=0.1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_voltage_in_reasonable_range(self):
        twin = BatteryDigitalTwin()
        profile = pd.DataFrame({"current": [50.0, -50.0, 30.0]})
        result = twin.simulate_drive_cycle(profile, dt=0.1)
        assert all(result["voltage"] >= 0)
        assert all(result["voltage"] <= 500)

    def test_empty_profile(self):
        twin = BatteryDigitalTwin()
        profile = pd.DataFrame({"current": []})
        result = twin.simulate_drive_cycle(profile)
        assert len(result) == 0


class TestPredictSOH:
    def test_prediction_returns_float(self):
        twin = BatteryDigitalTwin()
        result = twin.predict_soh(10)
        assert isinstance(result, float)

    def test_soh_decreases_with_more_cycles(self):
        twin = BatteryDigitalTwin()
        soh_10 = twin.predict_soh(10)
        soh_100 = twin.predict_soh(100)
        assert soh_100 < soh_10

    def test_prediction_does_not_modify_state(self):
        twin = BatteryDigitalTwin()
        initial_soh = twin.state.soh
        twin.predict_soh(50)
        assert twin.state.soh == initial_soh


class TestDegradationSummary:
    def test_returns_dict(self):
        twin = BatteryDigitalTwin()
        summary = twin.get_degradation_summary()
        assert isinstance(summary, dict)

    def test_has_required_keys(self):
        twin = BatteryDigitalTwin()
        summary = twin.get_degradation_summary()
        assert "current_soh" in summary
        assert "cycle_count" in summary
        assert "capacity_remaining_ah" in summary
        assert "internal_resistance" in summary
        assert "estimated_cycles_to_80" in summary
        assert "estimated_cycles_to_70" in summary

    def test_cycles_to_target_none_if_healthy(self):
        twin = BatteryDigitalTwin()
        summary = twin.get_degradation_summary()
        # Fresh battery won't reach 70% SOH in 10000 cycles
        # (depends on model params, but should be None or very large)
        assert summary["estimated_cycles_to_70"] is None or summary["estimated_cycles_to_70"] > 100

    def test_cycles_to_target_zero_if_already_below(self):
        twin = BatteryDigitalTwin()
        twin.state.soh = 65.0
        summary = twin.get_degradation_summary()
        assert summary["estimated_cycles_to_70"] == 0.0
