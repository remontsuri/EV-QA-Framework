"""
Тесты для персистентности ML-моделей (save/load)
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from ev_qa_framework.analysis import EVBatteryAnalyzer, AnomalyDetector


class TestModelPersistence:
    """Тесты для сохранения и загрузки моделей"""
    
    def setup_method(self):
        """Подготовка тестовых данных"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'voltage': np.random.normal(48, 2, 100),
            'current': np.random.normal(100, 10, 100),
            'temp': np.random.normal(35, 3, 100),
            'soc': np.random.normal(85, 5, 100)
        })
    
    def test_save_model_basic(self):
        """Тест базового сохранения модели"""
        analyzer = EVBatteryAnalyzer(contamination=0.1)
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_save_model_with_metadata(self):
        """Тест сохранения модели с метаданными"""
        analyzer = EVBatteryAnalyzer()
        analyzer.analyze_telemetry(self.test_data)
        
        metadata = {
            'version': '1.0',
            'dataset': 'test_battery_data',
            'contamination': 0.1
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath, metadata=metadata)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_save_model_without_training(self):
        """Тест ошибки при сохранении необученной модели"""
        analyzer = EVBatteryAnalyzer()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Модель не обучена"):
                analyzer.save_model(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_model_basic(self):
        """Тест базовой загрузки модели"""
        # Обучаем и сохраняем
        analyzer = EVBatteryAnalyzer(contamination=0.1, 
                                     critical_threshold=-0.9)
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath)
            
            # Загружаем
            loaded_analyzer = EVBatteryAnalyzer.load_model(filepath)
            
            # Проверяем параметры
            assert loaded_analyzer.contamination == 0.1
            assert loaded_analyzer.critical_threshold == -0.9
            assert hasattr(loaded_analyzer.scaler, 'mean_')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_model_inference(self):
        """Тест использования загруженной модели для inference"""
        # Обучаем на одной части данных
        train_data = self.test_data.iloc[:80]
        test_data = self.test_data.iloc[80:]
        
        analyzer = EVBatteryAnalyzer()
        results1 = analyzer.analyze_telemetry(train_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath)
            
            # Загружаем и тестируем
            loaded_analyzer = EVBatteryAnalyzer.load_model(filepath)
            results2 = loaded_analyzer.analyze_telemetry(test_data)
            
            # Проверяем, что модель работает
            assert 'total_samples' in results2
            assert 'anomalies_detected' in results2
            assert results2['total_samples'] == len(test_data)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_nonexistent_file(self):
        """Тест ошибки при загрузке несуществующего файла"""
        with pytest.raises(FileNotFoundError):
            EVBatteryAnalyzer.load_model('nonexistent_model.joblib')
    
    def test_model_info(self):
        """Тест получения информации о модели"""
        analyzer = EVBatteryAnalyzer(contamination=0.15, n_estimators=150)
        
        # До обучения
        info_before = analyzer.get_model_info()
        assert info_before['is_fitted'] is False
        assert info_before['contamination'] == 0.15
        assert info_before['n_estimators'] == 150
        
        # После обучения
        analyzer.analyze_telemetry(self.test_data)
        info_after = analyzer.get_model_info()
        assert info_after['is_fitted'] is True
    
    def test_save_without_extension(self):
        """Тест автоматического добавления расширения .joblib"""
        analyzer = EVBatteryAnalyzer()
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath_base = f.name
        
        # Удаляем временный файл, используем только путь
        os.unlink(filepath_base)
        
        try:
            # Сохраняем без расширения
            analyzer.save_model(filepath_base)
            
            # Проверяем, что добавлено .joblib
            expected_filepath = filepath_base + '.joblib'
            assert os.path.exists(expected_filepath)
        finally:
            if os.path.exists(filepath_base + '.joblib'):
                os.unlink(filepath_base + '.joblib')
    
    def test_save_create_directory(self):
        """Тест автоматического создания директории"""
        analyzer = EVBatteryAnalyzer()
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Указываем путь с несуществующей поддиректорией
            filepath = os.path.join(tmpdir, 'models', 'subdir', 'model.joblib')
            
            analyzer.save_model(filepath)
            assert os.path.exists(filepath)


class TestAnomalyDetectorPersistence:
    """Тесты для сохранения/загрузки AnomalyDetector"""
    
    def setup_method(self):
        """Подготовка тестовых данных"""
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'voltage': np.random.normal(48, 1, 100),
            'current': np.random.normal(100, 5, 100),
            'temp': np.random.normal(35, 2, 100),
            'soc': np.random.normal(85, 3, 100)
        })
        
        self.test_data = pd.DataFrame({
            'voltage': [48, 48, 100],  # 100 - аномалия
            'current': [100, 100, 100],
            'temp': [35, 35, 35],
            'soc': [85, 85, 85]
        })
    
    def test_anomaly_detector_save_load(self):
        """Тест save/load для AnomalyDetector"""
        # Обучаем детектор
        detector = AnomalyDetector(contamination=0.01)
        detector.train(self.train_data)
        
        # Детекция до сохранения
        predictions_before, scores_before = detector.detect(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            # Сохраняем
            detector.save_model(filepath, metadata={'type': 'AnomalyDetector'})
            
            # Загружаем (как EVBatteryAnalyzer, т.к. это базовый класс)
            # В реальности нужно будет создать load_model и для AnomalyDetector
            loaded = EVBatteryAnalyzer.load_model(filepath)
            
            # Проверяем, что модель загрузилась
            assert loaded.contamination == 0.01
            assert hasattr(loaded.scaler, 'mean_')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestModelVersioning:
    """Тесты для версионирования моделей"""
    
    def setup_method(self):
        """Подготовка тестовых данных"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'voltage': np.random.normal(48, 2, 50),
            'current': np.random.normal(100, 10, 50),
            'temp': np.random.normal(35, 3, 50),
            'soc': np.random.normal(85, 5, 50)
        })
    
    def test_multiple_versions(self):
        """Тест сохранения нескольких версий модели"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Версия 1.0
            analyzer_v1 = EVBatteryAnalyzer(contamination=0.1)
            analyzer_v1.analyze_telemetry(self.data)
            
            filepath_v1 = os.path.join(tmpdir, 'model_v1.0.joblib')
            analyzer_v1.save_model(filepath_v1, metadata={'version': '1.0'})
            
            # Версия 2.0 с другими параметрами
            analyzer_v2 = EVBatteryAnalyzer(contamination=0.05)
            analyzer_v2.analyze_telemetry(self.data)
            
            filepath_v2 = os.path.join(tmpdir, 'model_v2.0.joblib')
            analyzer_v2.save_model(filepath_v2, metadata={'version': '2.0'})
            
            # Проверяем, что обе версии сохранены
            assert os.path.exists(filepath_v1)
            assert os.path.exists(filepath_v2)
            
            # Загружаем обе версии
            loaded_v1 = EVBatteryAnalyzer.load_model(filepath_v1)
            loaded_v2 = EVBatteryAnalyzer.load_model(filepath_v2)
            
            # Проверяем различия
            assert loaded_v1.contamination == 0.1
            assert loaded_v2.contamination == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
