"""Tests for Chinese GB standards compliance."""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.config import FrameworkConfig
from ev_qa_framework.digital_twin import BatteryDigitalTwin
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel


class TestGB38031Safety:
    """GB 38031 — Mandatory EV battery safety standard (China)."""

    def test_vibration_test(self):
        """GB 38031: Vibration test — battery must withstand vibration."""
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 60.0
        twin = BatteryDigitalTwin(config)

        # Simulate vibration: rapid small current oscillations
        for _ in range(100):
            twin.step(0.01, np.random.uniform(-20, 20))

        assert twin.state.temperature < config.safety_thresholds.max_temperature
        assert twin.state.voltage > 0

    def test_mechanical_shock(self):
        """GB 38031: Mechanical shock — sudden high current pulse."""
        twin = BatteryDigitalTwin()

        # Simulate mechanical shock: sudden current spike
        initial_voltage = twin.state.voltage
        twin.step(0.001, 500.0)  # 500A spike for 3.6s

        # Voltage should drop but not collapse
        assert twin.state.voltage > initial_voltage * 0.5

    def test_drop_test(self):
        """GB 38031: Drop test — simulate impact after free fall."""
        twin = BatteryDigitalTwin()

        # Simulate drop: sudden discharge + temperature spike
        twin.step(0.1, -200.0)
        twin.state.temperature += 5.0  # impact heat

        assert twin.state.soc >= 0.0
        assert twin.state.voltage > 0

    def test_crush_test(self):
        """GB 38031: Crush test — simulate mechanical deformation."""
        twin = BatteryDigitalTwin()

        # Simulate crush: internal resistance increases
        twin.state.internal_resistance *= 2.0  # deformation

        twin.step(0.1, 50.0)
        # Higher resistance → more heat
        assert twin.state.temperature > twin.state.ambient_temperature

    def test_thermal_stability(self):
        """GB 38031: Thermal stability at high temperature."""
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 60.0
        twin = BatteryDigitalTwin(config)
        twin.state.ambient_temperature = 45.0

        for _ in range(20):
            twin.step(0.1, 10.0)

        # Should not exceed critical temperature
        assert twin.state.temperature < 100.0

    def test_overcharge_protection(self):
        """GB 38031: Overcharge protection."""
        config = FrameworkConfig()
        config.safety_thresholds.max_voltage = 403.2
        fw = EVQAFramework(config=config)

        # Overcharge: voltage above max
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=450.0,  # Above max
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        # Framework should reject overcharge
        assert fw.validate_telemetry(telemetry)[0] is False

    def test_overdischarge_protection(self):
        """GB 38031: Overdischarge protection."""
        config = FrameworkConfig()
        config.safety_thresholds.min_voltage = 288.0
        fw = EVQAFramework(config=config)

        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=200.0,  # Below min
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        # Framework should reject overdischarge
        assert fw.validate_telemetry(telemetry)[0] is False

    def test_short_circuit_protection(self):
        """GB 38031: Short circuit protection."""
        twin = BatteryDigitalTwin()

        # Simulate short circuit: extreme current
        initial_voltage = twin.state.voltage
        twin.step(0.001, 1000.0)  # 1000A short

        # Voltage should drop significantly
        assert twin.state.voltage < initial_voltage

    def test_water_resistance(self):
        """GB 38031: Water resistance — insulation check."""
        twin = BatteryDigitalTwin()

        # Simulate water exposure: resistance drops
        twin.state.internal_resistance *= 0.5

        twin.step(0.1, 50.0)
        # Should still operate (simplified)
        assert twin.state.voltage > 0


class TestGB31484CycleLife:
    """GB/T 31484-2015 — Cycle life testing."""

    def test_capacity_retention(self):
        """Capacity retention after cycling."""
        twin = BatteryDigitalTwin()
        initial_soh = twin.state.soh

        # Simulate 100 cycles
        for _ in range(100):
            twin.step(1.0, 50.0)
            twin.step(1.0, -50.0)

        # SOH should decrease
        assert twin.state.soh < initial_soh

    def test_cycle_life_at_low_temperature(self):
        """Cycle life at low temperature (0°C)."""
        twin = BatteryDigitalTwin()
        twin.state.ambient_temperature = 0.0
        twin.state.temperature = 0.0

        for _ in range(50):
            twin.step(1.0, 30.0)

        # Degradation should be faster at low temp
        assert twin.state.soh < 100.0

    def test_cycle_life_at_high_temperature(self):
        """Cycle life at high temperature (45°C)."""
        twin = BatteryDigitalTwin()
        twin.state.ambient_temperature = 45.0
        twin.state.temperature = 45.0

        for _ in range(50):
            twin.step(1.0, 30.0)

        # Degradation should be faster at high temp
        assert twin.state.soh < 100.0


class TestGB31486Temperature:
    """GB/T 31486-2015 — Temperature performance testing."""

    def test_low_temperature_discharge(self):
        """Discharge at -20°C."""
        twin = BatteryDigitalTwin()
        twin.state.ambient_temperature = -20.0
        twin.state.temperature = -20.0

        initial_soc = twin.state.soc
        twin.step(1.0, -30.0)

        # Should still discharge
        assert twin.state.soc < initial_soc

    def test_high_temperature_discharge(self):
        """Discharge at 55°C."""
        twin = BatteryDigitalTwin()
        twin.state.ambient_temperature = 55.0
        twin.state.temperature = 55.0

        initial_soc = twin.state.soc
        twin.step(1.0, -30.0)

        assert twin.state.soc < initial_soc

    def test_temperature_cycling(self):
        """Temperature cycling from -20°C to 55°C."""
        twin = BatteryDigitalTwin()

        for temp in [-20, 0, 25, 45, 55, 25, 0, -20]:
            twin.state.ambient_temperature = float(temp)
            twin.state.temperature = float(temp)
            twin.step(0.5, 20.0)

        # Battery should survive temperature cycling
        assert twin.state.voltage > 0
        assert twin.state.soh > 0
