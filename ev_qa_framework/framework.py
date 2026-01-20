"""
EV-QA-Framework: Mini QA Framework for EV & IoT Testing
Author: Remontsuri
License: MIT

AI-powered battery management system testing framework with pytest,
CAN protocol support, telemetry monitoring, and ML-based anomaly detection.
"""

import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging
import pandas as pd
from .analysis import EVBatteryAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from .models import BatteryTelemetryModel

class EVQAFramework:
    """Main QA Framework for EV & IoT testing"""
    
    # Default VIN for testing legacy data without VINs
    DEFAULT_TEST_VIN = "TESTVEHCLE0123456"
    
    def __init__(self, name: str = "EV-QA-Tester"):
        self.name = name
        self.telemetry_data: List[BatteryTelemetryModel] = []
        self.test_results: Dict = {}
        self.ml_analyzer = EVBatteryAnalyzer()
        logger.info(f"Initialized {self.name} with ML analyzer")
    
    def validate_telemetry(self, telemetry: BatteryTelemetryModel) -> bool:
        """
        Валидация телеметрии батареи относительно порогов безопасности.
        """
        # Пороги безопасности для батарей EV
        if telemetry.temperature > 60:
            logger.warning(f"ПРЕДУПРЕЖДЕНИЕ Температуры: {telemetry.temperature}°C")
            return False
        
        if telemetry.voltage < 200.0 or telemetry.voltage > 900.0:
            logger.warning(f"ПРЕДУПРЕЖДЕНИЕ Напряжения: {telemetry.voltage}V")
            return False
        
        return True
    
    def detect_anomalies(self, telemetry_list: List[BatteryTelemetryModel]) -> List[str]:
        """ML-детектирование аномалий с использованием простого статистического анализа"""
        anomalies = []
        
        if len(telemetry_list) < 2:
            return anomalies
        
        # Расчет средней температуры
        temps = [t.temperature for t in telemetry_list]
        if not temps:
             return anomalies
             
        avg_temp = sum(temps) / len(temps)
        
        # Обнаружение резких скачков температуры
        for i in range(1, len(telemetry_list)):
            temp_change = abs(telemetry_list[i].temperature - telemetry_list[i-1].temperature)
            if temp_change > 5:  # Резкий скачок более 5°C
                anomalies.append(f"Резкий скачок температуры: {temp_change}°C")
        
        return anomalies
    
    async def run_test_suite(self, telemetry_data: List[Dict]) -> Dict:
        """Run full QA test suite with ML analysis"""
        results = {
            'total_tests': len(telemetry_data),
            'passed': 0,
            'failed': 0,
            'anomalies': [],
            'ml_analysis': None
        }
        
        telemetries = []
        for data in telemetry_data:
            # Compatibility layer: Inject VIN if missing
            if 'vin' not in data:
                data['vin'] = self.DEFAULT_TEST_VIN
                
            try:
                # Use Pydantic for validation
                telemetry = BatteryTelemetryModel(**data)
                telemetries.append(telemetry)
                
                if self.validate_telemetry(telemetry):
                    results['passed'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Validation Error: {e}")
                results['failed'] += 1
                
        # Rule-based anomaly detection
        results['anomalies'] = self.detect_anomalies(telemetries)
        
        # ML-based analysis
        if telemetries:
            # Convert Pydantic models to dicts for DataFrame
            df = pd.DataFrame([t.model_dump() for t in telemetries])
            df.rename(columns={'temperature': 'temp'}, inplace=True)
            ml_results = self.ml_analyzer.analyze_telemetry(df)
            results['ml_analysis'] = ml_results
        
        self.test_results = results
        logger.info(f"Test Results: {results}")
        return results


# Example usage
if __name__ == "__main__":
    # Create QA framework instance
    qa = EVQAFramework("ChargePoint-QA")
    
    # Sample telemetry data
    test_data = [
        {'voltage': 3.9, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},
        {'voltage': 3.95, 'current': 45, 'temperature': 36, 'soc': 85, 'soh': 98},
        {'voltage': 3.85, 'current': 60, 'temperature': 45, 'soc': 75, 'soh': 97},
    ]
    
    # Run tests
    result = asyncio.run(qa.run_test_suite(test_data))
    print(json.dumps(result, indent=2))
