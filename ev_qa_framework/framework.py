from __future__ import annotations

"""
EV-QA-Framework: Mini QA Framework for EV & IoT Testing

Author: Remontsuri
License: MIT

AI-powered battery management system testing framework with pytest,
CAN protocol support, telemetry monitoring, and ML-based anomaly detection.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Set
import logging
import pandas as pd
from .analysis import EVBatteryAnalyzer
from .config import FrameworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from .models import BatteryTelemetryModel

class EVQAFramework:
    """Main QA Framework for EV & IoT testing"""
    
    # Default VIN for testing legacy data without VINs (deprecated, use config.default_vin)
    DEFAULT_TEST_VIN = "TESTVEHCLE0123456"
    
    def __init__(self, name: str = "EV-QA-Tester", config: Optional[FrameworkConfig] = None):
        """
        Инициализация QA Framework.
        
        Args:
            name: Название экземпляра фреймворка
            config: Кастомная конфигурация (если None, используется дефолтная)
        """
        self.name = name
        self.telemetry_data: List[BatteryTelemetryModel] = []
        # generic results dictionary with mixed values
        self.test_results: Dict[str, Any] = {}
        
        # Загрузка конфигурации
        self.config = config if config is not None else FrameworkConfig()
        # Проверяем, что default_vin пройдет валидацию Pydantic.
        # Используем простой dummy-телеметрию, чтобы воспользоваться проверками модели.
        try:
            BatteryTelemetryModel(
                vin=self.config.default_vin,
                voltage=0.0,
                current=0.0,
                temperature=0.0,
                soc=0.0,
                soh=0.0,
            )
        except Exception as e:
            logger.warning(
                f"default_vin '{self.config.default_vin}' невалиден ({e}), "
                f"заменяем на DEFAULT_TEST_VIN ({self.DEFAULT_TEST_VIN})"
            )
            self.config.default_vin = self.DEFAULT_TEST_VIN
        
        # Инициализация ML-анализатора с параметрами из конфига
        self.ml_analyzer = EVBatteryAnalyzer(
            contamination=self.config.ml_config.contamination,
            n_estimators=self.config.ml_config.n_estimators,
            random_state=self.config.ml_config.random_state
        )
        logger.info(f"Initialized {self.name} with ML analyzer (contamination={self.config.ml_config.contamination})")
    
    def validate_telemetry(self, telemetry: BatteryTelemetryModel) -> bool:
        """
        Валидация телеметрии батареи относительно порогов безопасности.
        
        Использует пороги из self.config.safety_thresholds.
        """
        thresholds = self.config.safety_thresholds
        
        # Проверка температуры
        if telemetry.temperature > thresholds.max_temperature:
            logger.warning(
                f"ПРЕДУПРЕЖДЕНИЕ Температуры: {telemetry.temperature}°C "
                f"(порог: {thresholds.max_temperature}°C)"
            )
            return False
        
        if telemetry.temperature < thresholds.min_temperature:
            logger.warning(
                f"ПРЕДУПРЕЖДЕНИЕ Температуры: {telemetry.temperature}°C "
                f"(минимум: {thresholds.min_temperature}°C)"
            )
            return False
        
        # Проверка напряжения
        if telemetry.voltage < thresholds.min_voltage or telemetry.voltage > thresholds.max_voltage:
            logger.warning(
                f"ПРЕДУПРЕЖДЕНИЕ Напряжения: {telemetry.voltage}V "
                f"(диапазон: {thresholds.min_voltage}-{thresholds.max_voltage}V)"
            )
            return False
        
        # Дополнительные проверки
        if telemetry.soc < thresholds.min_soc:
            logger.warning(f"Низкий уровень заряда: {telemetry.soc}%")
        
        if telemetry.soh < thresholds.critical_soh:
            logger.warning(f"Критическое состояние батареи: {telemetry.soh}%")
        
        return True
    
    def detect_anomalies(self, telemetry_list: List[BatteryTelemetryModel]) -> List[str]:
        """
        Rule-based детектирование аномалий в телеметрии.

        Выявляем два типа событий:
        1. Перегрев свыше max_temperature (порог из safety_thresholds)
        2. Резкий скачок температуры между соседними точками больше max_temperature_jump

        Возвращает список сообщений об аномалиях (строк).
        """
        anomalies: List[str] = []
        if not telemetry_list:
            return anomalies

        thresholds = self.config.safety_thresholds
        # 1. Проверка значений температуры на каждом шаге
        for t in telemetry_list:
            if t.temperature > thresholds.max_temperature:
                anomalies.append(
                    f"Температуры: {t.temperature}°C (порог: {thresholds.max_temperature}°C)"
                )

        # 2. Скачки температуры
        if len(telemetry_list) >= 2:
            temp_jump_threshold = thresholds.max_temperature_jump
            for i in range(1, len(telemetry_list)):
                temp_change = abs(telemetry_list[i].temperature - telemetry_list[i-1].temperature)
                if temp_change > temp_jump_threshold:
                    anomalies.append(
                        f"Резкий скачок температуры: {temp_change}°C "
                        f"(порог: {temp_jump_threshold}°C)"
                    )
        return anomalies
    
    async def run_test_suite(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Запуск полного набора QA-тестов с ML-анализом.

        Args:
            telemetry_data: Список словарей с данными телеметрии.

        Returns:
            Словарь с результатами тестов и анализа.
        """
        results: Dict[str, Any] = {
            'total_tests': len(telemetry_data),
            'passed': 0,
            'failed': 0,
            'anomalies': [],
            'ml_analysis': None,
            'critical_issues': []
        }
        
        if not telemetry_data:
            return results

        telemetries: List[BatteryTelemetryModel] = []
        status: List[bool] = []

        for data in telemetry_data:
            # Compatibility layer: Inject VIN if missing
            current_data = data.copy()
            if 'vin' not in current_data:
                current_data['vin'] = self.config.default_vin
                
            try:
                telemetry = BatteryTelemetryModel(**current_data)
                telemetries.append(telemetry)
                
                # Rule-based threshold validation
                is_valid = self.validate_telemetry(telemetry)
                status.append(is_valid)
                if not is_valid:
                    results['critical_issues'].append(
                        f"Safety threshold violation: {telemetry.model_dump(exclude={'timestamp', 'vin'})}"
                    )
            except Exception as e:
                msg = f"Validation failed - {e}"
                logger.error(msg)
                results['critical_issues'].append(msg)
                status.append(False)
                # Attempt to create model with partial data or defaults to keep indices synced
                # if possible, otherwise we might have issues with jump detection
                continue
        
        # Rule-based anomaly detection (e.g. temperature jumps)
        anomalies = self.detect_anomalies(telemetries)
        results['anomalies'] = anomalies

        # Adjust pass/fail based on anomalies if configured
        if self.config.fail_on_anomaly:
            jump_threshold = self.config.safety_thresholds.max_temperature_jump
            for i in range(1, len(telemetries)):
                # If the previous point was already failed, we ignore the jump from it
                # to avoid cascading failures from a single bad reading.
                if i > 0 and i-1 < len(status) and not status[i-1]:
                    continue

                if abs(telemetries[i].temperature - telemetries[i-1].temperature) > jump_threshold:
                    if i < len(status):
                        status[i] = False

        # compute counts
        results['passed'] = sum(1 for s in status if s)
        results['failed'] = results['total_tests'] - results['passed']
        
        # ML-based analysis
        if telemetries:
            df = pd.DataFrame([t.model_dump() for t in telemetries])
            # Ensure column names match what analyzer expects
            if 'temperature' in df.columns:
                df.rename(columns={'temperature': 'temp'}, inplace=True)

            try:
                ml_results = self.ml_analyzer.analyze_telemetry(df)
                results['ml_analysis'] = ml_results
            except Exception as e:
                logger.error(f"ML Analysis failed: {e}")
                results['ml_analysis'] = {"error": str(e)}
        
        self.test_results = results
        logger.info(f"Test suite finished: {results['passed']} passed, {results['failed']} failed")
        return results


# Example usage
if __name__ == "__main__":
    # Create QA framework instance
    qa = EVQAFramework("ChargePoint-QA")
    
    # Sample telemetry data
    test_data: list[dict[str, Any]] = [
        {'voltage': 3.9, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},
        {'voltage': 3.95, 'current': 45, 'temperature': 36, 'soc': 85, 'soh': 98},
        {'voltage': 3.85, 'current': 60, 'temperature': 45, 'soc': 75, 'soh': 97},
    ]
    
    # Run tests
    result = asyncio.run(qa.run_test_suite(test_data))
    print(json.dumps(result, indent=2))
