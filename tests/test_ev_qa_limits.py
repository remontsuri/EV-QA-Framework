from typing import Literal
import pytest
from pydantic import ValidationError
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel

class TestEVQAFrameworkLimts:
    
    def setup_method(self):
        self.qa = EVQAFramework("Limit-Tester")
        self.vin = "TESTVEHCLE0123456"

    @pytest.mark.parametrize("temp, expected", [
        (59.9, True),
        (60.0, True),
        (60.1, False),
        (100.0, False),
        (-20.0, True),
        (25.0, True)
    ])
    def test_temperature_limits(self, temp: float, expected: bool):
        """Boundary tests for Temperature"""
        # Voltage must be safe (e.g. 390)
        t = BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=10, temperature=temp, soc=50, soh=100)
        assert self.qa.validate_telemetry(t) == expected

    @pytest.mark.parametrize("voltage, expected", [
        (200.0, True),
        (199.9, False),
        (100.0, False),
        (390.0, True),
        (900.0, True),
        (900.1, False),
        (950.0, False),
        (800.0, True)
    ])
    def test_voltage_limits(self, voltage: float, expected: bool):
        """Boundary tests for Voltage (200V - 900V is safe)"""
        t = BatteryTelemetryModel(vin=self.vin, voltage=voltage, current=10, temperature=25, soc=50, soh=100)
        assert self.qa.validate_telemetry(t) == expected

    @pytest.mark.parametrize("soc", [0, 0.1, 50, 99.9, 100])
    def test_soc_valid(self, soc):
        """Valid SOC values"""
        t = BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=10, temperature=25, soc=soc, soh=100)
        assert self.qa.validate_telemetry(t) is True

    @pytest.mark.parametrize("soc", [-0.1, -1, 100.1, 101])
    def test_soc_invalid(self, soc):
        """Invalid SOC values should raise ValidationError"""
        with pytest.raises(ValidationError):
             BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=10, temperature=25, soc=soc, soh=100)

    def test_invalid_telemetry_types(self):
        """Negative test for invalid types"""
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(vin=self.vin, voltage="high", current=10, temperature=25, soc=50, soh=100)
