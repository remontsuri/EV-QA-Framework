"""
Tests for ML model persistence (save/load)
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from ev_qa_framework.analysis import EVBatteryAnalyzer, AnomalyDetector


class TestModelPersistence:
    """Tests for saving and loading models"""
    
    def setup_method(self):
        """Prepare test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'voltage': np.random.normal(48, 2, 100),
            'current': np.random.normal(100, 10, 100),
            'temp': np.random.normal(35, 3, 100),
            'soc': np.random.normal(85, 5, 100)
        })
    
    def test_save_model_basic(self):
        """Test basic model saving"""
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
        """Test saving model with metadata"""
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
        """Test error when saving untrained model"""
        analyzer = EVBatteryAnalyzer()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Model not trained"):
                analyzer.save_model(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_model_basic(self):
        """Test basic model loading"""
        # Train and save
        analyzer = EVBatteryAnalyzer(contamination=0.1, 
                                     critical_threshold=-0.9)
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath)
            
            # Load
            loaded_analyzer = EVBatteryAnalyzer.load_model(filepath)
            
            # Verify parameters
            assert loaded_analyzer.contamination == 0.1
            assert loaded_analyzer.critical_threshold == -0.9
            assert hasattr(loaded_analyzer.scaler, 'mean_')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_model_inference(self):
        """Test using loaded model for inference"""
        # Train on one part of the data
        train_data = self.test_data.iloc[:80]
        test_data = self.test_data.iloc[80:]
        
        analyzer = EVBatteryAnalyzer()
        results1 = analyzer.analyze_telemetry(train_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            analyzer.save_model(filepath)
            
            # Load and test
            loaded_analyzer = EVBatteryAnalyzer.load_model(filepath)
            results2 = loaded_analyzer.analyze_telemetry(test_data)
            
            # Verify model works
            assert 'total_samples' in results2
            assert 'anomalies_detected' in results2
            assert results2['total_samples'] == len(test_data)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_nonexistent_file(self):
        """Test error when loading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            EVBatteryAnalyzer.load_model('nonexistent_model.joblib')
    
    def test_model_info(self):
        """Test getting model info"""
        analyzer = EVBatteryAnalyzer(contamination=0.15, n_estimators=150)
        
        # Before training
        info_before = analyzer.get_model_info()
        assert info_before['is_fitted'] is False
        assert info_before['contamination'] == 0.15
        assert info_before['n_estimators'] == 150
        
        # After training
        analyzer.analyze_telemetry(self.test_data)
        info_after = analyzer.get_model_info()
        assert info_after['is_fitted'] is True
    
    def test_save_without_extension(self):
        """Test automatic .joblib extension addition"""
        analyzer = EVBatteryAnalyzer()
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath_base = f.name
        
        # Delete temporary file, use only the path
        os.unlink(filepath_base)
        
        try:
            # Save without extension
            analyzer.save_model(filepath_base)
            
            # Verify .joblib was added
            expected_filepath = filepath_base + '.joblib'
            assert os.path.exists(expected_filepath)
        finally:
            if os.path.exists(filepath_base + '.joblib'):
                os.unlink(filepath_base + '.joblib')
    
    def test_save_create_directory(self):
        """Test automatic directory creation"""
        analyzer = EVBatteryAnalyzer()
        analyzer.analyze_telemetry(self.test_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Specify path with nonexistent subdirectory
            filepath = os.path.join(tmpdir, 'models', 'subdir', 'model.joblib')
            
            analyzer.save_model(filepath)
            assert os.path.exists(filepath)


class TestAnomalyDetectorPersistence:
    """Tests for saving/loading AnomalyDetector"""
    
    def setup_method(self):
        """Prepare test data"""
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'voltage': np.random.normal(48, 1, 100),
            'current': np.random.normal(100, 5, 100),
            'temp': np.random.normal(35, 2, 100),
            'soc': np.random.normal(85, 3, 100)
        })
        
        self.test_data = pd.DataFrame({
            'voltage': [48, 48, 100],  # 100 - anomaly
            'current': [100, 100, 100],
            'temp': [35, 35, 35],
            'soc': [85, 85, 85]
        })
    
    def test_anomaly_detector_save_load(self):
        """Test save/load for AnomalyDetector"""
        # Train detector
        detector = AnomalyDetector(contamination=0.01)
        detector.train(self.train_data)
        
        # Detection before saving
        predictions_before, scores_before = detector.detect(self.test_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            filepath = f.name
        
        try:
            # Save
            detector.save_model(filepath, metadata={'type': 'AnomalyDetector'})
            
            # Load (as EVBatteryAnalyzer, since it is the base class)
            # In reality you'd need to create load_model for AnomalyDetector too
            loaded = EVBatteryAnalyzer.load_model(filepath)
            
            # Verify model loaded
            assert loaded.contamination == 0.01
            assert hasattr(loaded.scaler, 'mean_')
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestModelVersioning:
    """Tests for model versioning"""
    
    def setup_method(self):
        """Prepare test data"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'voltage': np.random.normal(48, 2, 50),
            'current': np.random.normal(100, 10, 50),
            'temp': np.random.normal(35, 3, 50),
            'soc': np.random.normal(85, 5, 50)
        })
    
    def test_multiple_versions(self):
        """Test saving multiple model versions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Version 1.0
            analyzer_v1 = EVBatteryAnalyzer(contamination=0.1)
            analyzer_v1.analyze_telemetry(self.data)
            
            filepath_v1 = os.path.join(tmpdir, 'model_v1.0.joblib')
            analyzer_v1.save_model(filepath_v1, metadata={'version': '1.0'})
            
            # Version 2.0 with different parameters
            analyzer_v2 = EVBatteryAnalyzer(contamination=0.05)
            analyzer_v2.analyze_telemetry(self.data)
            
            filepath_v2 = os.path.join(tmpdir, 'model_v2.0.joblib')
            analyzer_v2.save_model(filepath_v2, metadata={'version': '2.0'})
            
            # Verify both versions are saved
            assert os.path.exists(filepath_v1)
            assert os.path.exists(filepath_v2)
            
            # Load both versions
            loaded_v1 = EVBatteryAnalyzer.load_model(filepath_v1)
            loaded_v2 = EVBatteryAnalyzer.load_model(filepath_v2)
            
            # Verify differences
            assert loaded_v1.contamination == 0.1
            assert loaded_v2.contamination == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
