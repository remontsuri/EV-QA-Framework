"""
Тесты для модуля конфигурации EV-QA-Framework
"""

import pytest
import json
import os
import tempfile
from ev_qa_framework.config import (
    SafetyThresholds, 
    MLConfig, 
    FrameworkConfig
)
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel


class TestSafetyThresholds:
    """Тесты для класса SafetyThresholds"""
    
    def test_default_initialization(self):
        """Тест инициализации с дефолтными значениями"""
        thresholds = SafetyThresholds()
        assert thresholds.max_temperature == 60.0
        assert thresholds.min_voltage == 200.0
        assert thresholds.max_voltage == 900.0
        assert thresholds.max_temperature_jump == 5.0
    
    def test_custom_initialization(self):
        """Тест инициализации с кастомными значениями"""
        thresholds = SafetyThresholds(
            max_temperature=55.0,
            min_voltage=250.0,
            max_voltage=450.0
        )
        assert thresholds.max_temperature == 55.0
        assert thresholds.min_voltage == 250.0
        assert thresholds.max_voltage == 450.0
    
    def test_to_dict(self):
        """Тест конвертации в словарь"""
        thresholds = SafetyThresholds()
        data = thresholds.to_dict()
        assert isinstance(data, dict)
        assert 'max_temperature' in data
        assert 'min_voltage' in data
        assert data['max_temperature'] == 60.0
    
    def test_from_dict(self):
        """Тест создания из словаря"""
        data = {
            'max_temperature': 70.0,
            'min_voltage': 100.0,
            'max_voltage': 800.0,
            'min_temperature': -40.0,
            'max_temperature_jump': 5.0,
            'min_soc': 10.0,
            'critical_soh': 70.0,
            'max_current': 500.0
        }
        thresholds = SafetyThresholds.from_dict(data)
        assert thresholds.max_temperature == 70.0
        assert thresholds.min_voltage == 100.0
    
    def test_save_and_load_from_file(self):
        """Тест сохранения и загрузки из файла"""
        thresholds = SafetyThresholds(max_temperature=55.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            thresholds.save_to_file(filepath)
            loaded = SafetyThresholds.load_from_file(filepath)
            assert loaded.max_temperature == 55.0
        finally:
            os.unlink(filepath)


class TestMLConfig:
    """Тесты для класса MLConfig"""
    
    def test_default_initialization(self):
        """Тест дефолтной инициализации"""
        config = MLConfig()
        assert config.contamination == 0.1
        assert config.n_estimators == 200
        assert config.random_state == 42
        assert config.critical_score_threshold == -0.8
        assert config.warning_score_threshold == -0.5
    
    def test_custom_initialization(self):
        """Тест кастомной инициализации"""
        config = MLConfig(
            contamination=0.05,
            n_estimators=300,
            critical_score_threshold=-0.9
        )
        assert config.contamination == 0.05
        assert config.n_estimators == 300
        assert config.critical_score_threshold == -0.9
    
    def test_to_dict(self):
        """Тест конвертации в словарь"""
        config = MLConfig()
        data = config.to_dict()
        assert 'contamination' in data
        assert 'n_estimators' in data
        assert data['contamination'] == 0.1


class TestFrameworkConfig:
    """Тесты для класса FrameworkConfig"""
    
    def test_default_initialization(self):
        """Тест дефолтной инициализации"""
        config = FrameworkConfig()
        assert isinstance(config.safety_thresholds, SafetyThresholds)
        assert isinstance(config.ml_config, MLConfig)
        assert config.default_vin == "TESTVEHCLE0123456"
    
    def test_custom_initialization(self):
        """Тест кастомной инициализации"""
        thresholds = SafetyThresholds(max_temperature=55.0)
        ml_config = MLConfig(contamination=0.05)
        
        config = FrameworkConfig(
            safety_thresholds=thresholds,
            ml_config=ml_config,
            default_vin="CUSTOM123VIN45678"
        )
        
        assert config.safety_thresholds.max_temperature == 55.0
        assert config.ml_config.contamination == 0.05
        assert config.default_vin == "CUSTOM123VIN45678"
    
    def test_to_dict(self):
        """Тест конвертации в словарь"""
        config = FrameworkConfig()
        data = config.to_dict()
        assert 'safety_thresholds' in data
        assert 'ml_config' in data
        assert 'default_vin' in data
    
    def test_save_and_load_from_file(self):
        """Тест сохранения и загрузки из файла"""
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 55.0
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            config.save_to_file(filepath)
            loaded = FrameworkConfig.load_from_file(filepath)
            assert loaded.safety_thresholds.max_temperature == 55.0
        finally:
            os.unlink(filepath)
    
    def test_load_from_nonexistent_file(self):
        """Тест загрузки из несуществующего файла (должен вернуть дефолт)"""
        config = FrameworkConfig.load_from_file('nonexistent_file.json')
        assert isinstance(config, FrameworkConfig)
        assert config.safety_thresholds.max_temperature == 60.0


class TestFrameworkIntegration:
    """Интеграционные тесты для EVQAFramework с конфигурацией"""
    
    def test_framework_with_default_config(self):
        """Тест фреймворка с дефолтной конфигурацией"""
        qa = EVQAFramework("Test-QA")
        assert qa.config is not None
        assert qa.config.safety_thresholds.max_temperature == 60.0
    
    def test_framework_with_custom_config(self):
        """Тест фреймворка с кастомной конфигурацией"""
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 55.0
        
        qa = EVQAFramework("Test-QA", config=config)
        assert qa.config.safety_thresholds.max_temperature == 55.0
    
    def test_validation_with_custom_thresholds(self):
        """Тест валидации с кастомными порогами"""
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 50.0  # Строгий порог
        
        qa = EVQAFramework("Test-QA", config=config)
        
        # Температура 55°C должна быть отклонена (> 50°C)
        telemetry = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=400.0,
            current=50,
            temperature=55.0,
            soc=80,
            soh=98
        )
        assert qa.validate_telemetry(telemetry) is False
        
        # Температура 45°C должна пройти
        telemetry2 = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=400.0,
            current=50,
            temperature=45.0,
            soc=80,
            soh=98
        )
        assert qa.validate_telemetry(telemetry2) is True
    
    def test_voltage_validation_with_custom_thresholds(self):
        """Тест валидации напряжения с кастомными порогами"""
        config = FrameworkConfig()
        config.safety_thresholds.min_voltage = 300.0
        config.safety_thresholds.max_voltage = 500.0
        
        qa = EVQAFramework("Test-QA", config=config)
        
        # 250V должно быть отклонено (< 300V)
        telemetry1 = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=250.0,
            current=50,
            temperature=35,
            soc=80,
            soh=98
        )
        assert qa.validate_telemetry(telemetry1) is False
        
        # 400V должно пройти
        telemetry2 = BatteryTelemetryModel(
            vin="TESTVEHCLE0123456",
            voltage=400.0,
            current=50,
            temperature=35,
            soc=80,
            soh=98
        )
        assert qa.validate_telemetry(telemetry2) is True
    
    def test_ml_analyzer_with_custom_config(self):
        """Тест ML-анализатора с кастомной конфигурацией"""
        config = FrameworkConfig()
        config.ml_config.contamination = 0.05
        config.ml_config.n_estimators = 100
        
        qa = EVQAFramework("Test-QA", config=config)
        assert qa.ml_analyzer.contamination == 0.05
        assert qa.ml_analyzer.model.n_estimators == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
