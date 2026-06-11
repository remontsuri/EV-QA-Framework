"""
Edge case tests for Pydantic models.

Covers:
- BatteryTelemetryModel boundary validation
- BatteryCellDataModel edge cases
- VIN validation (forbidden chars, case handling)
- Voltage/current/temperature boundary values
- validate_telemetry helper function
"""

import pytest
from pydantic import ValidationError

from ev_qa_framework.models import (
    BatteryCellDataModel,
    BatteryTelemetryModel,
    validate_telemetry,
)


class TestBatteryTelemetryModel:
    def test_valid_telemetry(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=396.5,
            current=125.3,
            temperature=35.2,
            soc=78.5,
            soh=96.2,
        )
        assert m.vin == "1HGBH41JXMN109186"
        assert m.voltage == 396.5

    def test_vin_must_be_17_chars(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="SHORT",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_vin_forbidden_i(self):
        with pytest.raises(ValidationError, match="I, O, Q"):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN10918I",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_vin_forbidden_o(self):
        with pytest.raises(ValidationError, match="I, O, Q"):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN10918O",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_vin_forbidden_q(self):
        with pytest.raises(ValidationError, match="I, O, Q"):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN10918Q",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_vin_lowercase_converted_to_upper(self):
        m = BatteryTelemetryModel(
            vin="1hgbh41jxmn109186",
            voltage=400.0,
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.vin == "1HGBH41JXMN109186"

    def test_vin_non_alphanumeric(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN10918!",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_voltage_boundary_zero(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=0.0,
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.voltage == 0.0

    def test_voltage_boundary_max(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=1000.0,
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.voltage == 1000.0

    def test_voltage_above_max(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=1000.1,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_voltage_negative(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=-1.0,
                current=50.0,
                temperature=30.0,
                soc=80.0,
                soh=95.0,
            )

    def test_temperature_boundary_min(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=50.0,
            temperature=-50.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.temperature == -50.0

    def test_temperature_boundary_max(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=50.0,
            temperature=150.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.temperature == 150.0

    def test_temperature_below_min(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=400.0,
                current=50.0,
                temperature=-50.1,
                soc=80.0,
                soh=95.0,
            )

    def test_temperature_above_max(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=400.0,
                current=50.0,
                temperature=150.1,
                soc=80.0,
                soh=95.0,
            )

    def test_soc_boundary_zero(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=50.0,
            temperature=30.0,
            soc=0.0,
            soh=95.0,
        )
        assert m.soc == 0.0

    def test_soc_boundary_hundred(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=50.0,
            temperature=30.0,
            soc=100.0,
            soh=95.0,
        )
        assert m.soc == 100.0

    def test_soc_negative(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=-0.1,
                soh=95.0,
            )

    def test_soc_above_hundred(self):
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin="1HGBH41JXMN109186",
                voltage=400.0,
                current=50.0,
                temperature=30.0,
                soc=100.1,
                soh=95.0,
            )

    def test_soh_boundary(self):
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=50.0,
            temperature=30.0,
            soc=80.0,
            soh=0.0,
        )
        assert m.soh == 0.0

    def test_negative_current(self):
        """Negative current should be allowed (discharge)."""
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=400.0,
            current=-200.0,
            temperature=30.0,
            soc=80.0,
            soh=95.0,
        )
        assert m.current == -200.0

    def test_validate_telemetry_helper(self):
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 396.5,
            "current": 125.3,
            "temperature": 35.2,
            "soc": 78.5,
            "soh": 96.2,
        }
        m = validate_telemetry(data)
        assert isinstance(m, BatteryTelemetryModel)

    def test_validate_telemetry_invalid(self):
        with pytest.raises(ValidationError):
            validate_telemetry({"vin": "SHORT"})


class TestBatteryCellDataModel:
    def test_valid_cell_data(self):
        m = BatteryCellDataModel(
            vin="1HGBH41JXMN109186",
            cell_voltages=[3.7] * 96,
        )
        assert len(m.cell_voltages) == 96

    def test_empty_cell_voltages_raises(self):
        with pytest.raises(ValidationError):
            BatteryCellDataModel(
                vin="1HGBH41JXMN109186",
                cell_voltages=[],
            )

    def test_negative_cell_voltage_raises(self):
        with pytest.raises(ValidationError):
            BatteryCellDataModel(
                vin="1HGBH41JXMN109186",
                cell_voltages=[-0.1] + [3.7] * 95,
            )

    def test_cell_voltage_above_5v_raises(self):
        with pytest.raises(ValidationError):
            BatteryCellDataModel(
                vin="1HGBH41JXMN109186",
                cell_voltages=[5.1] + [3.7] * 95,
            )

    def test_cell_voltage_boundary_zero(self):
        m = BatteryCellDataModel(
            vin="1HGBH41JXMN109186",
            cell_voltages=[0.0] * 96,
        )
        assert all(v == 0.0 for v in m.cell_voltages)

    def test_cell_voltage_boundary_5v(self):
        m = BatteryCellDataModel(
            vin="1HGBH41JXMN109186",
            cell_voltages=[5.0] * 96,
        )
        assert all(v == 5.0 for v in m.cell_voltages)

    def test_single_cell(self):
        m = BatteryCellDataModel(
            vin="1HGBH41JXMN109186",
            cell_voltages=[3.7],
        )
        assert m.cell_voltages == [3.7]
