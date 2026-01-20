import pytest
from ev_qa_framework.framework import BatteryTelemetry, EVQAFramework

class TestEVQAFrameworkLimts:
    
    def setup_method(self):
        self.qa = EVQAFramework("Limit-Tester")

    @pytest.mark.parametrize("temp, expected", [
        (59.9, True),
        (60.0, True),  # Boundary logic check: code says > 60 is warning, so 60 is OK?
        # Code: if telemetry.temperature > 60: return False
        # So 60 is True (Valid), 60.1 is False (Invalid)
        (60.1, False),
        (100.0, False),
        (-20.0, True), # Assuming cold is ok for now, code only checks > 60
        (25.0, True)
    ])
    def test_temperature_limits(self, temp, expected):
        """Boundary tests for Temperature"""
        t = BatteryTelemetry(3.9, 10, temp, 50, 100)
        assert self.qa.validate_telemetry(t) == expected

    @pytest.mark.parametrize("voltage, expected", [
        (3.0, True),
        (2.99, False),
        (2.9, False),
        (3.01, True),
        (4.3, True),
        (4.31, False),
        (4.4, False),
        (3.7, True)
    ])
    def test_voltage_limits(self, voltage, expected):
        """Boundary tests for Voltage"""
        t = BatteryTelemetry(voltage, 10, 25, 50, 100)
        assert self.qa.validate_telemetry(t) == expected

    @pytest.mark.parametrize("soc, expected", [
        (0, True),
        (-0.1, False),
        (-1, False),
        (0.1, True),
        (100, True),
        (100.1, False),
        (101, False),
        (50, True)
    ])
    def test_soc_limits(self, soc, expected):
        """Boundary tests for SOC"""
        t = BatteryTelemetry(3.9, 10, 25, soc, 100)
        assert self.qa.validate_telemetry(t) == expected

    def test_invalid_telemetry_types(self):
        """Negative test for invalid types"""
        # Python doesn't enforce types at runtime, but operations might fail
        # The Validate method uses comparison operators
        # Comparing string ">" int in Python 3 raises TypeError
        with pytest.raises(TypeError):
            t = BatteryTelemetry("high", 10, 25, 50, 100)
            self.qa.validate_telemetry(t)

