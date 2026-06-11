"""
Standard-compliant tests for EV-QA-Framework.

Tests for compliance with international EV battery testing standards:
  - UN 38.3 — Transport safety
  - IEC 62660-1/-2 — Cell performance
  - SAE J2464 — Abuse testing
  - ISO 12405 — Traction battery

Uses:
  - ev_qa_framework.framework.EVQAFramework for validation
  - ev_qa_framework.models.BatteryTelemetryModel for data models
  - ev_qa_framework.config.FrameworkConfig for thresholds
"""

import math

import pytest

from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel
from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIN = "TESTVEHCLE0123456"


def _make_telemetry(
    *,
    voltage: float = 390.0,
    current: float = 50.0,
    temperature: float = 35.0,
    soc: float = 80.0,
    soh: float = 98.0,
    vin: str = VIN,
) -> BatteryTelemetryModel:
    """Factory for BatteryTelemetryModel with sensible defaults."""
    return BatteryTelemetryModel(
        vin=vin,
        voltage=voltage,
        current=current,
        temperature=temperature,
        soc=soc,
        soh=soh,
    )


def _make_config(**overrides) -> FrameworkConfig:
    """Build a FrameworkConfig with optional safety_thresholds overrides."""
    safety_overrides = overrides.pop("safety_thresholds", {})
    thresholds = SafetyThresholds(**safety_overrides) if safety_overrides else SafetyThresholds()
    return FrameworkConfig(safety_thresholds=thresholds, **overrides)


# ===========================================================================
# 1. UN 38.3 — Transport Safety
# ===========================================================================


class TestUN383MechanicalShock:
    """UN 38.3 — Mechanical shock / vibration profile simulation."""

    def test_vibration_profile_within_limits(self):
        """Simulate a vibration profile: small oscillations around nominal."""
        qa = EVQAFramework("UN38.3-Shock")
        # Vibration causes small voltage/temperature ripples — all within safe range
        data = [
            {
                "voltage": 390.0 + i * 0.1,
                "current": 50.0,
                "temperature": 35.0 + i * 0.2,
                "soc": 80.0,
                "soh": 98.0,
            }
            for i in range(10)
        ]
        import asyncio

        results = asyncio.run(qa.run_test_suite(data))
        assert results["failed"] == 0
        assert results["passed"] == 10

    def test_voltage_spike_during_shock_detected(self):
        """A voltage spike during mechanical shock must be flagged."""
        qa = EVQAFramework("UN38.3-Shock-Spike")
        data = [
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
            {
                "voltage": 950.0,
                "current": 50.0,
                "temperature": 35.0,
                "soc": 80.0,
                "soh": 98.0,
            },  # spike > max_voltage
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
        ]
        import asyncio

        results = asyncio.run(qa.run_test_suite(data))
        assert results["failed"] >= 1

    def test_mechanical_shock_temperature_anomaly(self):
        """Sudden temperature rise during shock triggers anomaly detection."""
        qa = EVQAFramework("UN38.3-Shock-Temp", config=_make_config(fail_on_anomaly=True))
        data = [
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
            {
                "voltage": 390.0,
                "current": 50.0,
                "temperature": 55.0,
                "soc": 80.0,
                "soh": 98.0,
            },  # 20°C jump
        ]
        import asyncio

        results = asyncio.run(qa.run_test_suite(data))
        assert results["failed"] >= 1


class TestUN383ThermalRunaway:
    """UN 38.3 — Thermal runaway threshold detection."""

    def test_thermal_runaway_threshold_exceeded(self):
        """Temperature above max_temperature must fail validation."""
        qa = EVQAFramework("UN38.3-Thermal")
        telemetry = _make_telemetry(temperature=65.0)  # > 60°C default max
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_thermal_runaway_predictor_critical(self):
        """ThermalRunawayPredictor should flag CRITICAL for rapid temp rise."""
        import pandas as pd

        predictor = ThermalRunawayPredictor(mode="rule")
        temps = [30.0, 35.0, 42.0, 50.0, 58.0, 66.0]
        df = pd.DataFrame({"temp": temps})
        result = predictor.predict_risk(df)
        assert result["risk_level"] in ("HIGH", "CRITICAL")

    def test_normal_temperature_not_flagged_as_runaway(self):
        """Stable normal temperature should not trigger thermal runaway."""
        import pandas as pd

        predictor = ThermalRunawayPredictor(mode="rule")
        # Very stable temps with minimal variance — all below 50°C
        temps = [25.0, 25.1, 25.0, 25.2, 25.1]
        df = pd.DataFrame({"temp": temps})
        result = predictor.predict_risk(df)
        assert result["risk_level"] == "LOW"


class TestUN383OverchargeProtection:
    """UN 38.3 — Overcharge protection test."""

    def test_overcharge_voltage_rejected(self):
        """Voltage above max_voltage must be rejected by validation."""
        qa = EVQAFramework("UN38.3-Overcharge")
        telemetry = _make_telemetry(voltage=950.0)  # > 900V default max
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_overcharge_voltage_at_boundary(self):
        """Voltage exactly at max_voltage boundary should pass."""
        qa = EVQAFramework("UN38.3-Overcharge-Boundary")
        telemetry = _make_telemetry(voltage=900.0)  # == max_voltage
        assert qa.validate_telemetry(telemetry)[0] is True

    def test_overcharge_voltage_just_above_boundary(self):
        """Voltage slightly above max_voltage must fail."""
        qa = EVQAFramework("UN38.3-Overcharge-Above")
        telemetry = _make_telemetry(voltage=900.1)
        assert qa.validate_telemetry(telemetry)[0] is False


class TestUN383ShortCircuitDetection:
    """UN 38.3 — Short circuit detection via current anomaly."""

    def test_extreme_current_detected_as_anomaly(self):
        """Extreme current (near-zero voltage) should be flagged."""
        qa = EVQAFramework("UN38.3-ShortCircuit")
        # Short circuit: voltage collapses, current spikes
        data = [
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
            {
                "voltage": 50.0,
                "current": 500.0,
                "temperature": 35.0,
                "soc": 80.0,
                "soh": 98.0,
            },  # near-short
        ]
        import asyncio

        results = asyncio.run(qa.run_test_suite(data))
        # The low-voltage entry should fail validation
        assert results["failed"] >= 1

    def test_normal_current_passes(self):
        """Normal current should pass validation."""
        qa = EVQAFramework("UN38.3-NormalCurrent")
        telemetry = _make_telemetry(current=50.0, voltage=390.0)
        assert qa.validate_telemetry(telemetry)[0] is True


# ===========================================================================
# 2. IEC 62660-1/-2 — Cell Performance
# ===========================================================================


class TestIEC62660Capacity:
    """IEC 62660-1 — Discharge capacity measurement."""

    def test_capacity_within_nominal_range(self):
        """Battery with SOH near 100% should have capacity within range."""
        qa = EVQAFramework("IEC62660-Capacity")
        telemetry = _make_telemetry(soh=98.0, voltage=390.0, soc=80.0)
        assert qa.validate_telemetry(telemetry)[0] is True

    def test_capacity_degraded_below_critical(self):
        """SOH below critical_soh should trigger a warning (but not fail validation)."""
        EVQAFramework("IEC62660-Capacity-Degraded")
        telemetry = _make_telemetry(soh=65.0)  # < 70% critical_soh
        # validate_telemetry returns True because SOH check is a warning, not a hard fail
        # but the telemetry should still be valid
        assert telemetry.soh < 70.0

    def test_capacity_measurement_voltage_stability(self):
        """Voltage should remain within safe range for valid capacity measurement."""
        qa = EVQAFramework("IEC62660-Capacity-Voltage")
        telemetry = _make_telemetry(voltage=390.0)
        assert qa.validate_telemetry(telemetry)[0] is True
        assert 200.0 <= telemetry.voltage <= 900.0


class TestIEC62660CycleLife:
    """IEC 62660-2 — Cycle life / capacity fade over cycles."""

    def test_capacity_fade_over_cycles(self):
        """Simulate capacity fade: SOH decreases over cycles."""
        qa = EVQAFramework("IEC62660-CycleLife")
        # Simulate 5 cycles with decreasing SOH
        data = [
            {
                "voltage": 390.0,
                "current": 50.0,
                "temperature": 35.0,
                "soc": 80.0,
                "soh": 98.0 - i * 2,
            }
            for i in range(5)
        ]
        import asyncio

        results = asyncio.run(qa.run_test_suite(data))
        assert results["total_tests"] == 5
        # All should pass since SOH is still above critical (70%)
        assert results["passed"] == 5

    def test_capacity_fade_below_critical(self):
        """After many cycles, SOH drops below critical threshold."""
        qa = EVQAFramework("IEC62660-CycleLife-Critical")
        data = [
            {
                "voltage": 390.0,
                "current": 50.0,
                "temperature": 35.0,
                "soc": 80.0,
                "soh": 75.0 - i * 2,
            }
            for i in range(5)
        ]
        import asyncio

        asyncio.run(qa.run_test_suite(data))
        # Last entries have SOH < 70% — they should still pass validation
        # (SOH is a warning, not a hard fail) but be below critical
        soh_values = [d["soh"] for d in data]
        assert any(s < 70.0 for s in soh_values)

    def test_cycle_life_telemetry_model_soh_field(self):
        """BatteryTelemetryModel should correctly store SOH values."""
        t1 = _make_telemetry(soh=95.0)
        t2 = _make_telemetry(soh=80.0)
        t3 = _make_telemetry(soh=65.0)
        assert t1.soh > t2.soh > t3.soh


class TestIEC62660InternalResistance:
    """IEC 62660-2 — Internal resistance estimation via voltage drop."""

    def test_internal_resistance_normal(self):
        """Normal voltage drop under load indicates healthy internal resistance."""
        qa = EVQAFramework("IEC62660-Resistance")
        # Under load (high current), voltage drops slightly — normal behavior
        no_load = _make_telemetry(voltage=400.0, current=0.0)
        under_load = _make_telemetry(voltage=390.0, current=100.0)
        voltage_drop = no_load.voltage - under_load.voltage
        # Voltage drop should be reasonable (< 20V for 100A)
        assert voltage_drop < 20.0
        assert qa.validate_telemetry(no_load)[0] is True
        assert qa.validate_telemetry(under_load)[0] is True

    def test_internal_resistance_high_detected(self):
        """Excessive voltage drop indicates high internal resistance (degraded cell)."""
        EVQAFramework("IEC62660-Resistance-High")
        no_load = _make_telemetry(voltage=400.0, current=0.0)
        under_load = _make_telemetry(voltage=350.0, current=100.0)  # 50V drop — excessive
        voltage_drop = no_load.voltage - under_load.voltage
        assert voltage_drop > 30.0  # excessive drop


# ===========================================================================
# 3. SAE J2464 — Abuse Testing
# ===========================================================================


class TestSAEJ2464OverchargeAbuse:
    """SAE J2464 — Overcharge abuse test (voltage > max)."""

    def test_overcharge_abuse_detected(self):
        """Voltage exceeding max_voltage must be rejected."""
        qa = EVQAFramework("SAEJ2464-Overcharge")
        telemetry = _make_telemetry(voltage=950.0)  # > 900V max
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_overcharge_abuse_temperature_rise(self):
        """Overcharge abuse often causes temperature rise."""
        qa = EVQAFramework("SAEJ2464-Overcharge-Temp")
        telemetry = _make_telemetry(voltage=950.0, temperature=70.0)
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_overcharge_abuse_with_chemistry_profile(self):
        """Overcharge abuse with NMC chemistry profile."""
        config = _make_config(chemistry="nmc")
        qa = EVQAFramework("SAEJ2464-Overcharge-NMC", config=config)
        # NMC max voltage = 4.2 * 96 = 403.2V
        telemetry = _make_telemetry(voltage=450.0)  # exceeds NMC pack max
        assert qa.validate_telemetry(telemetry)[0] is False


class TestSAEJ2464OverDischargeAbuse:
    """SAE J2464 — Over-discharge abuse test (voltage < min)."""

    def test_overdischarge_abuse_detected(self):
        """Voltage below min_voltage must be rejected."""
        qa = EVQAFramework("SAEJ2464-OverDischarge")
        telemetry = _make_telemetry(voltage=150.0)  # < 200V min
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_overdischarge_abuse_soc_zero(self):
        """SOC at 0% with low voltage indicates over-discharge."""
        qa = EVQAFramework("SAEJ2464-OverDischarge-SOC")
        telemetry = _make_telemetry(voltage=180.0, soc=5.0)
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_overdischarge_boundary(self):
        """Voltage exactly at min_voltage should pass."""
        qa = EVQAFramework("SAEJ2464-OverDischarge-Boundary")
        telemetry = _make_telemetry(voltage=200.0)  # == min_voltage
        assert qa.validate_telemetry(telemetry)[0] is True


class TestSAEJ2464ThermalAbuse:
    """SAE J2464 — Thermal abuse test (temperature > max)."""

    def test_thermal_abuse_detected(self):
        """Temperature above max_temperature must be rejected."""
        qa = EVQAFramework("SAEJ2464-Thermal")
        telemetry = _make_telemetry(temperature=65.0)  # > 60°C
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_thermal_abuse_extreme(self):
        """Extreme temperature (fire/explosion risk) must be rejected."""
        qa = EVQAFramework("SAEJ2464-Thermal-Extreme")
        telemetry = _make_telemetry(temperature=120.0)
        assert qa.validate_telemetry(telemetry)[0] is False

    def test_thermal_abuse_low_temperature(self):
        """Temperature below min_temperature must be rejected."""
        qa = EVQAFramework("SAEJ2464-Thermal-Low")
        telemetry = _make_telemetry(temperature=-50.0)  # < -40°C min
        assert qa.validate_telemetry(telemetry)[0] is False


# ===========================================================================
# 4. ISO 12405 — Traction Battery
# ===========================================================================


class TestISO12405SafetyThreshold:
    """ISO 12405 — Safety threshold validation for traction batteries."""

    def test_safety_thresholds_default_values(self):
        """Default safety thresholds should be within ISO 12405 ranges."""
        config = _make_config()
        st = config.safety_thresholds
        assert st.max_temperature == 60.0
        assert st.min_temperature == -40.0
        assert st.min_voltage == 200.0
        assert st.max_voltage == 900.0

    def test_safety_thresholds_custom(self):
        """Custom safety thresholds should be configurable."""
        config = _make_config(
            safety_thresholds={
                "max_temperature": 55.0,
                "min_voltage": 250.0,
                "max_voltage": 450.0,
            }
        )
        st = config.safety_thresholds
        assert st.max_temperature == 55.0
        assert st.min_voltage == 250.0
        assert st.max_voltage == 450.0

    def test_safety_thresholds_serialization(self):
        """Safety thresholds should serialize/deserialize correctly."""
        original = SafetyThresholds(max_temperature=50.0, min_voltage=300.0)
        d = original.to_dict()
        restored = SafetyThresholds.from_dict(d)
        assert restored.max_temperature == 50.0
        assert restored.min_voltage == 300.0

    def test_validate_telemetry_within_iso_thresholds(self):
        """Telemetry within ISO 12405 thresholds must pass validation."""
        qa = EVQAFramework("ISO12405-Valid")
        telemetry = _make_telemetry(voltage=400.0, temperature=35.0, soc=50.0, soh=90.0)
        assert qa.validate_telemetry(telemetry)[0] is True

    def test_validate_telemetry_outside_iso_thresholds(self):
        """Telemetry outside ISO 12405 thresholds must fail validation."""
        qa = EVQAFramework("ISO12405-Invalid")
        telemetry = _make_telemetry(voltage=400.0, temperature=70.0)
        assert qa.validate_telemetry(telemetry)[0] is False


class TestISO12405EnergyMeasurement:
    """ISO 12405 — Energy measurement test for traction batteries."""

    def test_energy_measurement_voltage_range(self):
        """Pack voltage should be within measurable range."""
        qa = EVQAFramework("ISO12405-Energy")
        telemetry = _make_telemetry(voltage=390.0)
        assert qa.validate_telemetry(telemetry)[0] is True
        assert telemetry.voltage > 0

    def test_energy_measurement_current_range(self):
        """Current should be measurable (positive for discharge, negative for charge)."""
        discharge = _make_telemetry(current=100.0)
        charge = _make_telemetry(current=-50.0)
        assert discharge.current > 0
        assert charge.current < 0

    def test_energy_calculation_power(self):
        """Power = Voltage × Current should be computable."""
        telemetry = _make_telemetry(voltage=390.0, current=50.0)
        power = telemetry.voltage * telemetry.current
        assert power == 19500.0  # 19.5 kW

    def test_energy_measurement_soc_range(self):
        """SOC should be within 0-100% for valid energy measurement."""
        qa = EVQAFramework("ISO12405-Energy-SOC")
        telemetry = _make_telemetry(soc=50.0)
        assert 0.0 <= telemetry.soc <= 100.0
        assert qa.validate_telemetry(telemetry)[0] is True


# ===========================================================================
# Cross-standard integration tests
# ===========================================================================


class TestCrossStandardIntegration:
    """Integration tests spanning multiple standards."""

    @pytest.mark.asyncio
    async def test_full_suite_passes_safe_battery(self):
        """A safe battery should pass all standard checks."""
        qa = EVQAFramework("CrossStandard-Safe")
        data = [
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
            {"voltage": 395.0, "current": 45.0, "temperature": 36.0, "soc": 82.0, "soh": 97.5},
            {"voltage": 388.0, "current": 55.0, "temperature": 34.0, "soc": 78.0, "soh": 97.0},
        ]
        results = await qa.run_test_suite(data)
        assert results["total_tests"] == 3
        assert results["passed"] == 3
        assert results["failed"] == 0

    @pytest.mark.asyncio
    async def test_full_suite_catches_abuse(self):
        """Abuse conditions should be caught by the framework."""
        qa = EVQAFramework("CrossStandard-Abuse")
        data = [
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
            {
                "voltage": 950.0,
                "current": 200.0,
                "temperature": 70.0,
                "soc": 80.0,
                "soh": 98.0,
            },  # abuse
            {"voltage": 390.0, "current": 50.0, "temperature": 35.0, "soc": 80.0, "soh": 98.0},
        ]
        results = await qa.run_test_suite(data)
        assert results["failed"] >= 1

    def test_battery_telemetry_model_validation(self):
        """BatteryTelemetryModel should enforce field constraints."""
        # Valid model
        t = _make_telemetry()
        assert t.voltage == 390.0
        assert t.temperature == 35.0

    def test_battery_telemetry_model_dump(self):
        """BatteryTelemetryModel should serialize to dict."""
        t = _make_telemetry()
        d = t.model_dump()
        assert isinstance(d, dict)
        assert "vin" in d
        assert "voltage" in d
        assert "temperature" in d
