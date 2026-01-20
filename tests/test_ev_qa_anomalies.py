"""Расширенные тесты для детекции аномалий в EV-QA-Framework"""

import pytest
from pydantic import ValidationError
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel


class TestAnomalyDetection:
    """Тесты для rule-based детекции аномалий"""
    
    def setup_method(self):
        self.qa = EVQAFramework("Anomaly-Tester")
        self.vin = "TESTVEHCLE0123456"
    
    def test_no_temperature_jump_stable(self):
        """Стабильная температура без скачков"""
        telemetries = [
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35.5, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=36, soc=80, soh=98),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0
    
    def test_exact_5_degree_jump(self):
        """Ровно 5°C скачок - граница детекции"""
        # Код проверяет > 5, значит 5.0 не должно детектироваться
        telemetries = [
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98),  # Ровно 5°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0  # > 5, а не >= 5
    
    def test_5_1_degree_jump(self):
        """5.1°C скачок - должен детектироваться"""
        telemetries = [
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35.1, soc=80, soh=98),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1
        assert "Резкий скачок температуры" in anomalies[0]
    
    def test_multiple_temperature_jumps(self):
        """Множественные скачки температуры"""
        telemetries = [
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=30, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=37, soc=80, soh=98),  # +7°C
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=32, soc=80, soh=98),  # -5°C (не детектируется, т.к. abs change = 5 not > 5)
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=40.1, soc=80, soh=98),  # +8.1°C (32->40.1)
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        # Должно быть 2 аномалии: 30->37 (+7) и 32->40.1 (+8.1)
        assert len(anomalies) >= 2
    
    def test_temperature_drop(self):
        """Резкое падение температуры"""
        telemetries = [
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=50, soc=80, soh=98),
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=43, soc=80, soh=98),  # -7°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1
    
    def test_single_telemetry_no_anomaly(self):
        """Одна точка данных - нет аномалий"""
        telemetries = [BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35, soc=80, soh=98)]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0
    
    def test_empty_telemetry_list(self):
        """Пустой список - нет аномалий"""
        anomalies = self.qa.detect_anomalies([])
        assert len(anomalies) == 0


class TestNegativeScenarios:
    """Негативные тесты для обработки некорректных данных"""
    
    def setup_method(self):
        self.qa = EVQAFramework("Negative-Tester")
        self.vin = "TESTVEHCLE0123456"
    
    def test_extreme_negative_temperature(self):
        """Экстремально низкая температура должна вызывать ошибку валидации модели"""
        # Pydantic limit is -50
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=-273, soc=80, soh=98)
    
    def test_extreme_high_voltage(self):
        """Экстремально высокое напряжение (в пределах Pydantic, но выше Warning)"""
        # Pydantic 0-1000. Test 1000.
        t = BatteryTelemetryModel(vin=self.vin, voltage=1000, current=50, temperature=35, soc=80, soh=98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_negative_soc(self):
        """Отрицательный SOC должен вызывать ошибку валидации"""
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35, soc=-10, soh=98)
    
    def test_soc_over_100(self):
        """SOC больше 100% должен вызывать ошибку валидации"""
        with pytest.raises(ValidationError):
            BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=50, temperature=35, soc=150, soh=98)
    
    def test_zero_voltage(self):
        """Нулевое напряжение"""
        t = BatteryTelemetryModel(vin=self.vin, voltage=0, current=50, temperature=35, soc=80, soh=98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_negative_current(self):
        """Отрицательный ток (разряд)"""
        # Ток может быть отрицательным при разряде - это нормально
        t = BatteryTelemetryModel(vin=self.vin, voltage=390.0, current=-50, temperature=35, soc=80, soh=98)
        assert self.qa.validate_telemetry(t) is True


@pytest.mark.asyncio
class TestAsyncTestSuite:
    """Тесты для асинхронного запуска тестового набора"""
    
    def setup_method(self):
        self.qa = EVQAFramework("Async-Tester")
    
    async def test_all_valid_telemetry(self):
        """Все данные валидны"""
        test_data = [
            {'voltage': 370.0, 'current': 50, 'temperature': 30, 'soc': 75, 'soh': 98},
            {'voltage': 380.0, 'current': 45, 'temperature': 31, 'soc': 78, 'soh': 98},
            {'voltage': 390.0, 'current': 40, 'temperature': 32, 'soc': 80, 'soh': 98},
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['total_tests'] == 3
        # Если данные валидны и VIN добавился автоматически
        assert results['passed'] == 3
        assert results['failed'] == 0
    
    async def test_mixed_valid_invalid(self):
        """Смешанные валидные и невалидные данные"""
        test_data = [
            {'voltage': 390.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},  # OK
            {'voltage': 1000.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},  # voltage too high (warning > 900)
            {'voltage': 390.0, 'current': 50, 'temperature': 70, 'soc': 80, 'soh': 98},  # temp too high (warning)
            {'voltage': 390.0, 'current': 50, 'temperature': 35, 'soc': 105, 'soh': 98}, # SOC invalid (pydantic error)
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['total_tests'] == 4
        assert results['passed'] == 1
        assert results['failed'] == 3
    
    async def test_with_anomalies(self):
        """Данные с детектируемыми аномалиями"""
        test_data = [
            {'voltage': 390.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98},
            {'voltage': 390.0, 'current': 50, 'temperature': 40, 'soc': 80, 'soh': 98},  # +10°C jump
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['passed'] == 2  # Оба валидны по отдельности
        assert len(results['anomalies']) > 0  # Но есть аномалия


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
