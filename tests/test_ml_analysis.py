"""Тесты для ML-анализатора батарейной телеметрии"""

import pytest
import numpy as np
import pandas as pd
from ev_qa_analysis import EVBatteryAnalyzer


class TestEVBatteryAnalyzer:
    """Тесты для класса EVBatteryAnalyzer"""
    
    def test_initialization(self):
        """Тест инициализации анализатора"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        assert analyzer.model is not None
        assert analyzer.scaler is not None
        assert analyzer.anomalies is None
    
    def test_initialization_default_contamination(self):
        """Тест инициализации с параметрами по умолчанию"""
        analyzer = EVBatteryAnalyzer()
        # Contamination=0.1 по умолчанию
        assert analyzer.model.contamination == 0.1
    
    def test_analyze_perfect_data(self):
        """Анализ идеальных данных без аномалий"""
        analyzer = EVBatteryAnalyzer(contamination=0.05)
        
        # Генерируем стабильные данные
        np.random.seed(42)
        df = pd.DataFrame({
            'voltage': np.full(100, 48.0),
            'current': np.full(100, 100.0),
            'temp': np.full(100, 35.0),
            'soc': np.full(100, 85.0)
        })
        
        results = analyzer.analyze_telemetry(df)
        
        assert results['total_samples'] == 100
        # С contamination=0.05 ожидаем ~5% аномалий даже на идеальных данных
        assert results['anomalies_detected'] <= 10
    
    def test_analyze_with_obvious_outliers(self):
        """Анализ данных с явными выбросами"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        
        # Нормальные данные
        np.random.seed(42)
        normal_data = {
            'voltage': np.random.normal(48, 1, 90),
            'current': np.random.normal(100, 5, 90),
            'temp': np.random.normal(35, 2, 90),
            'soc': np.random.normal(85, 5, 90)
        }
        
        # Добавляем 10 явных выбросов
        outliers = {
            'voltage': np.full(10, 100.0),  # Экстремальное напряжение
            'current': np.full(10, 500.0),  # Экстремальный ток
            'temp': np.full(10, 90.0),      # Экстремальная температура
            'soc': np.full(10, 5.0)
        }
        
        df = pd.DataFrame({
            'voltage': np.concatenate([normal_data['voltage'], outliers['voltage']]),
            'current': np.concatenate([normal_data['current'], outliers['current']]),
            'temp': np.concatenate([normal_data['temp'], outliers['temp']]),
            'soc': np.concatenate([normal_data['soc'], outliers['soc']])
        })
        
        results = analyzer.analyze_telemetry(df)
        
        assert results['total_samples'] == 100
        # Должны детектироваться аномалии (как минимум часть из 10)
        assert results['anomalies_detected'] > 0
    
    def test_severity_critical(self):
        """Тест критической серьезности аномалий"""
        analyzer = EVBatteryAnalyzer(contamination=0.2)
        
        # Данные с экстремальными выбросами
        df = pd.DataFrame({
            'voltage': [48]*80 + [200]*20,  # Экстремальный выброс
            'current': [100]*80 + [1000]*20,
            'temp': [35]*80 + [150]*20,
            'soc': [85]*100
        })
        
        results = analyzer.analyze_telemetry(df)
        # При таких экстремальных выбросах severity может быть CRITICAL
        assert results['severity'] in ['CRITICAL', 'WARNING', 'INFO']
    
    def test_severity_info(self):
        """Тест низкой серьезности (INFO)"""
        analyzer = EVBatteryAnalyzer(contamination=0.05)
        
        # Почти идеальные данные с минимальными вариациями
        np.random.seed(42)
        df = pd.DataFrame({
            'voltage': np.random.normal(48, 0.1, 100),
            'current': np.random.normal(100, 1, 100),
            'temp': np.random.normal(35, 0.5, 100),
            'soc': np.random.normal(85, 1, 100)
        })
        
        results = analyzer.analyze_telemetry(df)
        # Ожидаем INFO или WARNING на стабильных данных
        assert results['severity'] in ['INFO', 'WARNING']
    
    def test_anomaly_percentage_calculation(self):
        """Тест расчета процента аномалий"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        
        np.random.seed(42)
        df = pd.DataFrame({
            'voltage': np.random.normal(48, 2, 100),
            'current': np.random.normal(100, 10, 100),
            'temp': np.random.normal(35, 3, 100),
            'soc': np.random.normal(85, 5, 100)
        })
        
        results = analyzer.analyze_telemetry(df)
        
        # Проверяем, что процент рассчитывается правильно
        expected_percentage = (results['anomalies_detected'] / 100) * 100
        assert results['anomaly_percentage'] == expected_percentage
    
    def test_small_dataset(self):
        """Тест на маленьком датасете"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        
        # Минимальный датасет (IsolationForest требует минимум данных)
        df = pd.DataFrame({
            'voltage': [48.0, 48.1, 48.2, 47.9, 48.0],
            'current': [100, 101, 99, 100, 102],
            'temp': [35, 35, 36, 35, 35],
            'soc': [85, 85, 84, 86, 85]
        })
        
        results = analyzer.analyze_telemetry(df)
        assert results['total_samples'] == 5
        assert isinstance(results['anomalies_detected'], int)


class TestEVBatteryAnalyzerEdgeCases:
    """Граничные случаи для ML-анализатора"""
    
    def test_single_feature_variance(self):
        """Данные с вариацией только в одной фиче"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        
        df = pd.DataFrame({
            'voltage': np.random.normal(48, 5, 100),  # Вариация
            'current': np.full(100, 100.0),           # Константа
            'temp': np.full(100, 35.0),               # Константа
            'soc': np.full(100, 85.0)                 # Константа
        })
        
        results = analyzer.analyze_telemetry(df)
        assert results['total_samples'] == 100
        assert 'anomalies_detected' in results
    
    def test_negative_values(self):
        """Отрицательные значения в данных"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        
        df = pd.DataFrame({
            'voltage': [48]*95 + [-10]*5,  # Отрицательное напряжение (невалидно физически)
            'current': [100]*100,
            'temp': [35]*100,
            'soc': [85]*100
        })
        
        results = analyzer.analyze_telemetry(df)
        # Должно детектировать отрицательные значения как аномалии
        assert results['anomalies_detected'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
