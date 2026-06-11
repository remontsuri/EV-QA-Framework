"""
Parameterized tests for SOC (State of Charge) and Pydantic model validation
"""

import pytest
from pydantic import ValidationError

from ev_qa_framework.models import validate_telemetry


class TestSOCValidation:
    """Tests for SOC (State of Charge) validation checks"""

    @pytest.mark.parametrize(
        "soc,expected_valid",
        [
            # Boundary values
            (-1, False),  # Below minimum
            (0, True),  # Minimum valid value
            (0.1, True),  # Slightly above minimum
            (1, True),  # Small value
            (50, True),  # Middle of range
            (99, True),  # Almost maximum
            (99.9, True),  # Slightly below maximum
            (100, True),  # Maximum valid value
            (100.1, False),  # Slightly above maximum
            (101, False),  # Above maximum
        ],
    )
    def test_soc_boundary_values(self, soc, expected_valid):
        """SOC boundary values: -1, 0, 1, 99, 100, 101"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": 100,
            "temperature": 35,
            "soc": soc,
            "soh": 95,
        }

        if expected_valid:
            telemetry = validate_telemetry(data)
            assert telemetry.soc == soc
        else:
            with pytest.raises(ValidationError) as exc_info:
                validate_telemetry(data)
            assert "soc" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "soc_value",
        [
            "string",  # String instead of number
            "50",  # Number as string (Pydantic can convert)
            None,  # Empty value
            [],  # List
            {},  # Dictionary
            True,  # Boolean value (can be coerced to 1)
        ],
    )
    def test_soc_invalid_types(self, soc_value):
        """Type checks: string instead of number, None, other types"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": 100,
            "temperature": 35,
            "soc": soc_value,
            "soh": 95,
        }

        # Pydantic can convert some types (e.g., "50" -> 50.0)
        # But strings like "string", None, [], {} will raise an error
        if soc_value in [None, [], {}]:
            with pytest.raises(ValidationError):
                validate_telemetry(data)
        elif soc_value == "50":
            # Pydantic automatically converts string "50" to float
            telemetry = validate_telemetry(data)
            assert telemetry.soc == 50.0
        elif soc_value is True:
            # True is converted to 1.0
            telemetry = validate_telemetry(data)
            assert telemetry.soc == 1.0
        else:
            # "string" and other non-convertible types
            with pytest.raises(ValidationError):
                validate_telemetry(data)


class TestPydanticTelemetryModel:
    """Tests for the Pydantic BatteryTelemetryModel model"""

    def test_valid_telemetry_creation(self):
        """Creating valid telemetry"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 396.5,
            "current": 125.3,
            "temperature": 35.2,
            "soc": 78.5,
            "soh": 96.2,
        }
        telemetry = validate_telemetry(data)
        assert telemetry.vin == "1HGBH41JXMN109186"
        assert telemetry.voltage == 396.5
        assert telemetry.soc == 78.5

    def test_vin_validation_length(self):
        """VIN must be exactly 17 characters"""
        # Short VIN
        with pytest.raises(ValidationError) as exc_info:
            validate_telemetry(
                {
                    "vin": "SHORT",
                    "voltage": 400,
                    "current": 100,
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95,
                }
            )
        assert "vin" in str(exc_info.value).lower()

        # Long VIN
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "TOOLONGVINCODE123456",
                    "voltage": 400,
                    "current": 100,
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95,
                }
            )

    def test_vin_forbidden_characters(self):
        """VIN must not contain I, O, Q"""
        for forbidden_char in ["I", "O", "Q"]:
            vin = f"1HGBH41JXMN10918{forbidden_char}"
            with pytest.raises(ValidationError) as exc_info:
                validate_telemetry(
                    {
                        "vin": vin,
                        "voltage": 400,
                        "current": 100,
                        "temperature": 35,
                        "soc": 80,
                        "soh": 95,
                    }
                )
            assert "VIN" in str(exc_info.value) or "vin" in str(exc_info.value)

    def test_voltage_range_validation(self):
        """Voltage must be 0-1000V"""
        # Negative voltage
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "1HGBH41JXMN109186",
                    "voltage": -10,
                    "current": 100,
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95,
                }
            )

        # Too high voltage
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "1HGBH41JXMN109186",
                    "voltage": 1500,
                    "current": 100,
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95,
                }
            )

    def test_temperature_range_validation(self):
        """Temperature must be in range -50 to +150°C"""
        # Too low temperature
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "1HGBH41JXMN109186",
                    "voltage": 400,
                    "current": 100,
                    "temperature": -100,
                    "soc": 80,
                    "soh": 95,
                }
            )

        # Too high temperature
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "1HGBH41JXMN109186",
                    "voltage": 400,
                    "current": 100,
                    "temperature": 200,
                    "soc": 80,
                    "soh": 95,
                }
            )

    def test_negative_current_allowed(self):
        """Negative current is allowed (battery discharge)"""
        telemetry = validate_telemetry(
            {
                "vin": "1HGBH41JXMN109186",
                "voltage": 400,
                "current": -50,  # Discharge
                "temperature": 35,
                "soc": 80,
                "soh": 95,
            }
        )
        assert telemetry.current == -50

    def test_timestamp_auto_generation(self):
        """Timestamp is automatically generated if not provided"""
        telemetry = validate_telemetry(
            {
                "vin": "1HGBH41JXMN109186",
                "voltage": 400,
                "current": 100,
                "temperature": 35,
                "soc": 80,
                "soh": 95,
            }
        )
        assert telemetry.timestamp is not None

    def test_missing_required_fields(self):
        """Missing required fields raises an error"""
        with pytest.raises(ValidationError):
            validate_telemetry(
                {
                    "vin": "1HGBH41JXMN109186",
                    "voltage": 400,
                    # current is missing
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95,
                }
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
