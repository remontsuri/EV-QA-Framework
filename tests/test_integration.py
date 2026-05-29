"""
Integration tests for EV-QA-Framework.
Verifying interaction of all components: Config -> Framework -> Model -> Results.
"""

import pytest
import asyncio
import os
import tempfile
from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.analysis import EVBatteryAnalyzer


class TestIntegrationFlow:
    """Integration tests for the full cycle"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_custom_config(self):
        """
        Testing the full cycle:
        1. Creating a custom config
        2. Framework initialization
        3. Running test on mixed data (normal + anomalies)
        4. Checking results
        """
        # 1. Setup strict config
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 45.0  # Very strict
        config.default_vin = "INTEGRATION_TEST_VIN"
        # in the integration scenario we want rule-based anomalies
        # to also count as failures
        config.fail_on_anomaly = True
        
        # 2. Initialization
        qa = EVQAFramework(name="Integration-QA", config=config)
        
        # 3. Data preparation
        test_data = [
            # Normal data (for training and validation)
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98},
            {'voltage': 401.0, 'current': 51, 'temperature': 31, 'soc': 79, 'soh': 98},
            {'voltage': 402.0, 'current': 52, 'temperature': 32, 'soc': 78, 'soh': 98},
            # Rule-based anomaly (temperature > 45)
            {'voltage': 400.0, 'current': 50, 'temperature': 50, 'soc': 77, 'soh': 98},
            # Rule-based anomaly (temperature jump > 5)
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 76, 'soh': 98},
            {'voltage': 400.0, 'current': 50, 'temperature': 40, 'soc': 75, 'soh': 98}, # Jump +10
        ]
        
        # 4. Execution
        results = await qa.run_test_suite(test_data)
        
        # 5. Verification
        assert results['total_tests'] == 6
        # First 3 passed, 4th failed due to temperature, 5th passed (baseline for jump), 6th failed due to jump
        # But wait: 4th failed (50 > 45). 6th failed (jump 40-30=10 > 5).
        # Total 2 failed, 4 passed.
        assert results['passed'] == 4
        assert results['failed'] == 2
        
        # Check anomaly messages
        anomaly_list = results['anomalies']
        assert any("Temperature: 50.0" in msg for msg in anomaly_list)
        assert any("Sudden temperature jump: 10.0" in msg for msg in anomaly_list)

    @pytest.mark.asyncio
    async def test_ml_persistence_integration(self):
        """
        ML persistence integration:
        1. Training model on normal data
        2. Saving model
        3. Loading into framework
        4. Verifying detection on new data without retraining
        """
        # Training data
        train_data = [
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 95}
        ] * 20 # 20 identical points for stability
        
        # 1. Train
        qa_train = EVQAFramework("Trainer")
        await qa_train.run_test_suite(train_data)
        
        # 2. Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            model_path = f.name
            
        try:
            qa_train.ml_analyzer.save_model(model_path, metadata={'task': 'integration_test'})
            
            # 3. Create new framework and load model
            loaded_analyzer = EVBatteryAnalyzer.load_model(model_path)
            qa_prod = EVQAFramework("Production")
            qa_prod.ml_analyzer = loaded_analyzer
            
            # 4. Check for anomalies (without additional training)
            anomaly_data = [
                {'voltage': 800.0, 'current': 300, 'temperature': 55, 'soc': 20, 'soh': 90} # Strong deviation
            ]
            
            # We use ml_analyzer directly, since run_test_suite calls analyze_telemetry
            # which in the current AnomalyDetector implementation does fit if there's a lot of data or if analyze_telemetry is called.
            # But if we loaded the model, it is already fitted.
            
            import pandas as pd
            df_anomaly = pd.DataFrame(anomaly_data)
            ml_results = qa_prod.ml_analyzer.analyze_telemetry(df_anomaly)
            
            # Since we trained on 400V/50A, 800V/300A will definitely be an anomaly
            assert ml_results['anomalies_detected'] > 0
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    @pytest.mark.asyncio
    async def test_config_hot_reload_simulation(self):
        """
        Simulation of "on-the-fly" configuration change
        """
        qa = EVQAFramework("Hot-Reload")
        
        # First, default thresholds (60 degrees)
        telemetry = {'voltage': 400.0, 'current': 50, 'temperature': 55, 'soc': 80, 'soh': 98}
        results_1 = await qa.run_test_suite([telemetry])
        assert results_1['passed'] == 1
        
        # Change config to stricter (50 degrees)
        new_config = FrameworkConfig()
        new_config.safety_thresholds.max_temperature = 50.0
        qa.config = new_config
        
        results_2 = await qa.run_test_suite([telemetry])
        assert results_2['failed'] == 1
        assert "Temperature: 55.0" in results_2['anomalies'][0]

    def test_error_handling_invalid_telemetry_format(self):
        """
        Test error handling for invalid data format
        """
        qa = EVQAFramework("Error-Handler")
        
        # Data with missing required field 'voltage'
        invalid_data = [
            {'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98}
        ]
        
        # run_test_suite returns results even if there were validation errors,
        # logging them as failed tests.
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(qa.run_test_suite(invalid_data))
        
        assert results['total_tests'] == 1
        assert results['failed'] == 1
        assert any("Validation failed" in msg for msg in results['critical_issues'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
