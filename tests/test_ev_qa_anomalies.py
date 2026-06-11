"""Extended tests for anomaly detection in EV-QA-Framework"""

import pytest
from pydantic import ValidationError

from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel


class TestAnomalyDetection:
    """Tests for rule-based anomaly detection"""

    def setup_method(self):
        self.qa = EVQAFramework("Anomaly-Tester")
        self.vin = "TESTVEHCLE0123456"

    def test_no_temperature_jump_stable(self):
        """Stable temperature without jumps"""
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35.5, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=36, soc=80, soh=98
            ),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0

    def test_exact_5_degree_jump(self):
        """Exactly 5°C jump - detection boundary"""
        # Code checks > 5, so 5.0 should not be detected
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98
            ),  # Exactly 5°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0  # > 5, not >= 5

    def test_5_1_degree_jump(self):
        """5.1°C jump - should be detected"""
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35.1, soc=80, soh=98
            ),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1
        assert "Sharp temperature jump" in anomalies[0]

    def test_multiple_temperature_jumps(self):
        """Multiple temperature jumps"""
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=37, soc=80, soh=98
            ),  # +7°C
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=32, soc=80, soh=98
            ),  # -5°C (not detected, abs change = 5 not > 5)
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=40.1, soc=80, soh=98
            ),  # +8.1°C (32->40.1)
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        # Should be 2 anomalies: 30->37 (+7) and 32->40.1 (+8.1)
        assert len(anomalies) >= 2

    def test_temperature_drop(self):
        """Sharp temperature drop"""
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=50, soc=80, soh=98
            ),
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=43, soc=80, soh=98
            ),  # -7°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1

    def test_single_telemetry_no_anomaly(self):
        """Single data point - no anomalies"""
        telemetries = [
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98
            )
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0

    def test_empty_telemetry_list(self):
        """Empty list - no anomalies"""
        anomalies = self.qa.detect_anomalies([])
        assert len(anomalies) == 0


class TestNegativeScenarios:
    """Negative tests for handling invalid data"""

    def setup_method(self):
        self.qa = EVQAFramework("Negative-Tester")
        self.vin = "TESTVEHCLE0123456"

    def test_extreme_negative_temperature(self):
        """Extremely low temperature should cause model validation error"""
        # Pydantic limit is -50
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=-273, soc=80, soh=98
            )

    def test_extreme_high_voltage(self):
        """Extremely high voltage (within Pydantic limits, but above Warning)"""
        # Pydantic 0-1000. Test 1000.
        t = BatteryTelemetryModel(
            vin=self.vin, voltage=1000, current=50, temperature=35, soc=80, soh=98
        )
        assert self.qa.validate_telemetry(t) is False

    def test_negative_soc(self):
        """Negative SOC should cause validation error"""
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35, soc=-10, soh=98
            )

    def test_soc_over_100(self):
        """SOC over 100% should cause validation error"""
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(
                vin=self.vin, voltage=390.0, current=50, temperature=35, soc=150, soh=98
            )

    def test_zero_voltage(self):
        """Zero voltage"""
        t = BatteryTelemetryModel(
            vin=self.vin, voltage=0, current=50, temperature=35, soc=80, soh=98
        )
        assert self.qa.validate_telemetry(t) is False

    def test_negative_current(self):
        """Negative current (discharge)"""
        # Current can be negative during discharge - this is normal
        t = BatteryTelemetryModel(
            vin=self.vin, voltage=390.0, current=-50, temperature=35, soc=80, soh=98
        )
        assert self.qa.validate_telemetry(t) is True


@pytest.mark.asyncio
class TestAsyncTestSuite:
    """Tests for async test suite execution"""

    def setup_method(self):
        self.qa = EVQAFramework("Async-Tester")

    async def test_all_valid_telemetry(self):
        """All data is valid"""
        test_data = [
            {"voltage": 370.0, "current": 50, "temperature": 30, "soc": 75, "soh": 98},
            {"voltage": 380.0, "current": 45, "temperature": 31, "soc": 78, "soh": 98},
            {"voltage": 390.0, "current": 40, "temperature": 32, "soc": 80, "soh": 98},
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results["total_tests"] == 3
        # If data is valid and VIN was added automatically
        assert results["passed"] == 3
        assert results["failed"] == 0

    async def test_mixed_valid_invalid(self):
        """Mixed valid and invalid data"""
        test_data = [
            {"voltage": 390.0, "current": 50, "temperature": 35, "soc": 80, "soh": 98},  # OK
            {
                "voltage": 1000.0,
                "current": 50,
                "temperature": 35,
                "soc": 80,
                "soh": 98,
            },  # voltage too high (warning > 900)
            {
                "voltage": 390.0,
                "current": 50,
                "temperature": 70,
                "soc": 80,
                "soh": 98,
            },  # temp too high (warning)
            {
                "voltage": 390.0,
                "current": 50,
                "temperature": 35,
                "soc": 105,
                "soh": 98,
            },  # SOC invalid (pydantic error)
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results["total_tests"] == 4
        assert results["passed"] == 1
        assert results["failed"] == 3

    async def test_with_anomalies(self):
        """Data with detectable anomalies"""
        test_data = [
            {"voltage": 390.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
            {
                "voltage": 390.0,
                "current": 50,
                "temperature": 40,
                "soc": 80,
                "soh": 98,
            },  # +10°C jump
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results["passed"] == 2  # Both are valid individually
        assert len(results["anomalies"]) > 0  # But there is an anomaly

    async def test_anomaly_flag_causes_failure(self):
        """With fail_on_anomaly enabled, telemetry with anomaly is considered failed"""
        self.qa.config.fail_on_anomaly = True
        test_data = [
            {"voltage": 390.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
            {
                "voltage": 390.0,
                "current": 50,
                "temperature": 40,
                "soc": 80,
                "soh": 98,
            },  # +10°C jump
        ]
        results = await self.qa.run_test_suite(test_data)
        # first element succeeded, second failed due to jump
        assert results["passed"] == 1
        assert results["failed"] == 1
        assert len(results["anomalies"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
