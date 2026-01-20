"""pytest test suite for EV-QA-Framework

Tests for battery telemetry validation, anomaly detection,
and QA framework functionality.
"""

import pytest
import asyncio
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel


class TestBatteryTelemetry:
    """Tests for BatteryTelemetryModel class"""
    
    def test_battery_telemetry_creation(self):
        """Test creation of battery telemetry object"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=390.0, # Используем реальные значения (300V-400V) так как модель валидирует 0-1000
            current=50,
            temperature=35,
            soc=80,
            soh=98
        )
        assert telemetry.voltage == 390.0
        assert telemetry.current == 50
        assert telemetry.temperature == 35
        assert telemetry.soc == 80
        assert telemetry.soh == 98
        assert telemetry.timestamp is not None
    
    def test_battery_telemetry_to_dict(self):
        """Test conversion to dictionary"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=390.0, 
            current=50, 
            temperature=35, 
            soc=80, 
            soh=98
        )
        data = telemetry.model_dump()
        assert isinstance(data, dict)
        assert data['voltage'] == 390.0
        assert 'timestamp' in data


class TestEVQAFramework:
    """Tests for EVQAFramework class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.qa = EVQAFramework("Test-QA")
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        assert self.qa.name == "Test-QA"
        assert isinstance(self.qa.telemetry_data, list)
        assert isinstance(self.qa.test_results, dict)
    
    def test_validate_valid_telemetry(self):
        """Test validation of valid telemetry"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456", 
            voltage=400.0, # Внимание: Pydantic позволяет 0-1000, но Framework warning < 3.0
                         # Здесь ставим 4.0 чтобы пройти warning check
            current=50, 
            temperature=35, 
            soc=80, 
            soh=98
        )
        assert self.qa.validate_telemetry(telemetry) is True
    
    def test_validate_high_temperature(self):
        """Test rejection of high temperature"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=400.0, 
            current=50, 
            temperature=65, # > 60 warning
            soc=80, 
            soh=98
        )
        assert self.qa.validate_telemetry(telemetry) is False
    
    def test_validate_invalid_voltage_low(self):
        """Test rejection of low voltage"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=2.5, # < 3.0 warning
            current=50, 
            temperature=35, 
            soc=80, 
            soh=98
        )
        assert self.qa.validate_telemetry(telemetry) is False
    
    
    def test_detect_anomalies_empty(self):
        """Test anomaly detection with empty list"""
        anomalies = self.qa.detect_anomalies([])
        assert anomalies == []
    
    def test_detect_anomalies_single_item(self):
        """Test anomaly detection with single item"""
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=390.0, 
            current=50, 
            temperature=35, 
            soc=80, 
            soh=98
        )
        anomalies = self.qa.detect_anomalies([telemetry])
        assert anomalies == []
    
    def test_detect_anomalies_temperature_jump(self):
        """Test detection of sudden temperature change"""
        telemetries = [
            BatteryTelemetryModel(vin="TESTVEHCLE0123456", voltage=390.0, current=50, temperature=35, soc=80, soh=98),
            BatteryTelemetryModel(vin="TESTVEHCLE0123456", voltage=390.0, current=45, temperature=42, soc=85, soh=98),  # 7°C jump
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) > 0
        assert "Резкий скачок температуры" in anomalies[0]
    
    @pytest.mark.asyncio
    async def test_run_test_suite_async(self):
        """Test async test suite execution"""
        test_data = [
            {'voltage': 390.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},
            {'voltage': 395.0, 'current': 45, 'temperature': 36, 'soc': 85, 'soh': 98},
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['total_tests'] == 2
        assert results['passed'] == 2
        assert results['failed'] == 0
    
    @pytest.mark.asyncio
    async def test_run_test_suite_with_failures(self):
        """Test test suite with invalid data"""
        test_data = [
            {'voltage': 390.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},
            {'voltage': 100.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},  # Invalid voltage (< 200V)
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['passed'] == 1
        assert results['failed'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
