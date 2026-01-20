"""Расширенные тесты для детекции аномалий в EV-QA-Framework"""

import pytest
from ev_qa_framework.framework import BatteryTelemetry, EVQAFramework


class TestAnomalyDetection:
    """Тесты для rule-based детекции аномалий"""
    
    def setup_method(self):
        self.qa = EVQAFramework("Anomaly-Tester")
    
    def test_no_temperature_jump_stable(self):
        """Стабильная температура без скачков"""
        telemetries = [
            BatteryTelemetry(3.9, 50, 35, 80, 98),
            BatteryTelemetry(3.9, 50, 35.5, 80, 98),
            BatteryTelemetry(3.9, 50, 36, 80, 98),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0
    
    def test_exact_5_degree_jump(self):
        """Ровно 5°C скачок - граница детекции"""
        # Код проверяет > 5, значит 5.0 не должно детектироваться
        telemetries = [
            BatteryTelemetry(3.9, 50, 30, 80, 98),
            BatteryTelemetry(3.9, 50, 35, 80, 98),  # Ровно 5°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 0  # > 5, а не >= 5
    
    def test_5_1_degree_jump(self):
        """5.1°C скачок - должен детектироваться"""
        telemetries = [
            BatteryTelemetry(3.9, 50, 30, 80, 98),
            BatteryTelemetry(3.9, 50, 35.1, 80, 98),
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1
        assert "Sudden temp change" in anomalies[0]
    
    def test_multiple_temperature_jumps(self):
        """Множественные скачки температуры"""
        telemetries = [
            BatteryTelemetry(3.9, 50, 30, 80, 98),
            BatteryTelemetry(3.9, 50, 37, 80, 98),  # +7°C
            BatteryTelemetry(3.9, 50, 32, 80, 98),  # -5°C (не детектируется, т.к. abs)
            BatteryTelemetry(3.9, 50, 40, 80, 98),  # +8°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        # Должно быть 2 аномалии: +7 и +8
        assert len(anomalies) >= 2
    
    def test_temperature_drop(self):
        """Резкое падение температуры"""
        telemetries = [
            BatteryTelemetry(3.9, 50, 50, 80, 98),
            BatteryTelemetry(3.9, 50, 43, 80, 98),  # -7°C
        ]
        anomalies = self.qa.detect_anomalies(telemetries)
        assert len(anomalies) == 1
    
    def test_single_telemetry_no_anomaly(self):
        """Одна точка данных - нет аномалий"""
        telemetries = [BatteryTelemetry(3.9, 50, 35, 80, 98)]
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
    
    def test_extreme_negative_temperature(self):
        """Экстремально низкая температура"""
        t = BatteryTelemetry(3.9, 50, -273, 80, 98)  # Абсолютный ноль
        # Код не проверяет нижний лимит температуры, только >60
        result = self.qa.validate_telemetry(t)
        assert result is True  # Технически пройдет по текущей логике
    
    def test_extreme_high_voltage(self):
        """Экстремально высокое напряжение"""
        t = BatteryTelemetry(1000, 50, 35, 80, 98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_negative_soc(self):
        """Отрицательный SOC"""
        t = BatteryTelemetry(3.9, 50, 35, -10, 98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_soc_over_100(self):
        """SOC больше 100%"""
        t = BatteryTelemetry(3.9, 50, 35, 150, 98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_zero_voltage(self):
        """Нулевое напряжение"""
        t = BatteryTelemetry(0, 50, 35, 80, 98)
        assert self.qa.validate_telemetry(t) is False
    
    def test_negative_current(self):
        """Отрицательный ток (разряд)"""
        # Ток может быть отрицательным при разряде - это нормально
        t = BatteryTelemetry(3.9, -50, 35, 80, 98)
        # Код не проверяет ток, только voltage, temp, soc
        assert self.qa.validate_telemetry(t) is True


@pytest.mark.asyncio
class TestAsyncTestSuite:
    """Тесты для асинхронного запуска тестового набора"""
    
    def setup_method(self):
        self.qa = EVQAFramework("Async-Tester")
    
    async def test_all_valid_telemetry(self):
        """Все данные валидны"""
        test_data = [
            {'voltage': 3.7, 'current': 50, 'temperature': 30, 'soc': 75, 'soh': 98},
            {'voltage': 3.8, 'current': 45, 'temperature': 31, 'soc': 78, 'soh': 98},
            {'voltage': 3.9, 'current': 40, 'temperature': 32, 'soc': 80, 'soh': 98},
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['total_tests'] == 3
        assert results['passed'] == 3
        assert results['failed'] == 0
    
    async def test_mixed_valid_invalid(self):
        """Смешанные валидные и невалидные данные"""
        test_data = [
            {'voltage': 3.9, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},  # OK
            {'voltage': 5.0, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},  # voltage too high
            {'voltage': 3.9, 'current': 50, 'temperature': 70, 'soc': 80, 'soh': 98},  # temp too high
            {'voltage': 3.9, 'current': 50, 'temperature': 35, 'soc': 105, 'soh': 98}, # SOC invalid
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['total_tests'] == 4
        assert results['passed'] == 1
        assert results['failed'] == 3
    
    async def test_with_anomalies(self):
        """Данные с детектируемыми аномалиями"""
        test_data = [
            {'voltage': 3.9, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98},
            {'voltage': 3.9, 'current': 50, 'temperature': 40, 'soc': 80, 'soh': 98},  # +10°C jump
        ]
        results = await self.qa.run_test_suite(test_data)
        assert results['passed'] == 2  # Оба валидны по отдельности
        assert len(results['anomalies']) > 0  # Но есть аномалия


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
