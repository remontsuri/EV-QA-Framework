"""
Example of using ML model persistence
Demonstrates the train-once, deploy-many pattern for production
"""

import numpy as np
import pandas as pd
from ev_qa_framework.analysis import EVBatteryAnalyzer, AnomalyDetector


def generate_normal_telemetry(n_samples=1000):
    """Generate normal telemetry data"""
    np.random.seed(42)
    return pd.DataFrame({
        'voltage': np.random.normal(48, 1, n_samples),
        'current': np.random.normal(100, 5, n_samples),
        'temp': np.random.normal(35, 2, n_samples),
        'soc': np.random.normal(85, 5, n_samples)
    })


def generate_test_telemetry_with_anomalies():
    """Generate test data with anomalies"""
    normal_data = pd.DataFrame({
        'voltage': [48, 48, 48, 48, 48],
        'current': [100, 100, 100, 100, 100],
        'temp': [35, 35, 35, 35, 35],
        'soc': [85, 85, 85, 85, 85]
    })
    
    # Add anomalies
    anomalies = pd.DataFrame({
        'voltage': [200, 10],  # Extreme values
        'current': [500, 600],
        'temp': [90, -20],
        'soc': [5, 100]
    })
    
    return pd.concat([normal_data, anomalies], ignore_index=True)


def example_1_train_and_save():
    """
    Example 1: Training the model and saving
    
    Typical scenario: train a model on historical data
    and save for use in production.
    """
    print("=" * 70)
    print("Example 1: Training and saving the model")
    print("=" * 70)
    
    # Generate training data (historical "normal" data)
    print("📊 Generating training data (1000 normal points)...")
    train_data = generate_normal_telemetry(1000)
    
    # Create and train the analyzer
    print("🧠 Training ML model (Isolation Forest)...")
    analyzer = EVBatteryAnalyzer(
        contamination=0.05,  # Expect 5% anomalies
        n_estimators=200,
        critical_threshold=-0.9,
        warning_threshold=-0.6
    )
    
    results = analyzer.analyze_telemetry(train_data)
    print(f"   Trained on {results['total_samples']} points")
    print(f"   Anomalies detected: {results['anomalies_detected']}")
    print(f"   Severity: {results['severity']}")
    
    # Save model with metadata
    print("\n💾 Saving model...")
    metadata = {
        'version': '1.0',
        'dataset': 'historical_battery_data_2024',
        'contamination': 0.05,
        'date': '2024-01-28',
        'description': 'Baseline model trained on 1000 normal samples'
    }
    
    analyzer.save_model('models/battery_analyzer_baseline', metadata=metadata)
    print()


def example_2_load_and_infer():
    """
    Example 2: Loading a saved model and inference
    
    Production scenario: load a pre-trained model
    and use it for anomaly detection on new data.
    """
    print("=" * 70)
    print("Example 2: Loading model and inference")
    print("=" * 70)
    
    # Load the saved model
    print("📥 Loading saved model...")
    analyzer = EVBatteryAnalyzer.load_model('models/battery_analyzer_baseline')
    
    # Get model information
    print("\n📋 Loaded model info:")
    info = analyzer.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test on new data
    print("\n🔍 Testing on new data with anomalies...")
    test_data = generate_test_telemetry_with_anomalies()
    
    results = analyzer.analyze_telemetry(test_data)
    print(f"   Total points: {results['total_samples']}")
    print(f"   Anomalies: {results['anomalies_detected']}")
    print(f"   Anomaly percentage: {results['anomaly_percentage']:.2f}%")
    print(f"   Severity: {results['severity']}")
    print()


def example_3_model_versioning():
    """
    Example 3: Model versioning
    
    Scenario: create multiple model versions with different parameters
    for A/B testing.
    """
    print("=" * 70)
    print("Example 3: Model versioning (A/B testing)")
    print("=" * 70)
    
    train_data = generate_normal_telemetry(500)
    test_data = generate_test_telemetry_with_anomalies()
    
    # Model A: conservative (low contamination)
    print("\n🅰️  Creating model A (conservative, contamination=0.01)...")
    model_a = EVBatteryAnalyzer(contamination=0.01, n_estimators=200)
    model_a.analyze_telemetry(train_data)
    model_a.save_model('models/model_a_conservative', 
                       metadata={'version': 'A', 'type': 'conservative'})
    
    # Model B: tolerant (high contamination)
    print("🅱️  Creating model B (tolerant, contamination=0.15)...")
    model_b = EVBatteryAnalyzer(contamination=0.15, n_estimators=200)
    model_b.analyze_telemetry(train_data)
    model_b.save_model('models/model_b_tolerant', 
                       metadata={'version': 'B', 'type': 'tolerant'})
    
    # Compare models on test data
    print("\n📊 Model comparison on test data:")
    
    loaded_a = EVBatteryAnalyzer.load_model('models/model_a_conservative')
    results_a = loaded_a.analyze_telemetry(test_data)
    print(f"   Model A: {results_a['anomalies_detected']} anomalies, {results_a['severity']}")
    
    loaded_b = EVBatteryAnalyzer.load_model('models/model_b_tolerant')
    results_b = loaded_b.analyze_telemetry(test_data)
    print(f"   Model B: {results_b['anomalies_detected']} anomalies, {results_b['severity']}")
    print()


def example_4_anomaly_detector_pattern():
    """
    Example 4: Train/Detect pattern with AnomalyDetector
    
    More advanced pattern for production with separate
    training and detection phases.
    """
    print("=" * 70)
    print("Example 4: AnomalyDetector - Train/Detect pattern")
    print("=" * 70)
    
    # Train on "clean" data
    print("🧠 Training AnomalyDetector on clean data...")
    clean_data = generate_normal_telemetry(1000)
    
    detector = AnomalyDetector(contamination=0.01, n_estimators=250)
    detector.train(clean_data)
    
    # Save
    print("\n💾 Saving trained detector...")
    detector.save_model('models/anomaly_detector_prod', 
                       metadata={'purpose': 'production_detector', 'trained_samples': 1000})
    
    # In another session/service: load and detect
    print("\n📥 Loading detector (simulating production environment)...")
    # Note: in the current version, load_model returns EVBatteryAnalyzer
    # In production, consider adding AnomalyDetector.load_model
    loaded_detector = EVBatteryAnalyzer.load_model('models/anomaly_detector_prod')
    
    # Real-time detection
    print("\n🔍 Real-time detection on streaming data...")
    
    # Simulate data stream (batches of 10 points)
    for batch_num in range(3):
        batch_data = generate_test_telemetry_with_anomalies() if batch_num == 1 else generate_normal_telemetry(10)
        results = loaded_detector.analyze_telemetry(batch_data)
        
        print(f"   Batch {batch_num + 1}: "
              f"{results['anomalies_detected']}/{results['total_samples']} anomalies "
              f"({results['severity']})")
    print()


def example_5_production_workflow():
    """
    Example 5: Full production workflow
    
    1. Train offline on historical data
    2. Save model
    3. Deploy to production
    4. Load and inference in real-time
    5. Monitoring and versioning
    """
    print("=" * 70)
    print("Example 5: Production Workflow")
    print("=" * 70)
    
    # OFFLINE TRAINING
    print("\n📚 PHASE 1: Offline Training")
    print("-" * 70)
    
    historical_data = generate_normal_telemetry(2000)
    
    # Train baseline model
    baseline = EVBatteryAnalyzer(
        contamination=0.05,
        n_estimators=300,  # More trees for production
        critical_threshold=-0.85,
        warning_threshold=-0.55
    )
    
    print("   Training baseline model on 2000 historical points...")
    baseline.analyze_telemetry(historical_data)
    
    baseline.save_model(
        'models/production/baseline_v1.0',
        metadata={
            'version': '1.0',
            'stage': 'production',
            'performance': 'baseline',
            'trained_samples': 2000,
            'last_updated': '2024-01-28'
        }
    )
    print("   ✅ Model saved to production")
    
    # PRODUCTION DEPLOYMENT
    print("\n🚀 PHASE 2: Production Deployment")
    print("-" * 70)
    
    production_model = EVBatteryAnalyzer.load_model('models/production/baseline_v1.0')
    print("   ✅ Model loaded into production environment")
    
    # REAL-TIME INFERENCE
    print("\n⚡ PHASE 3: Real-time Inference")
    print("-" * 70)
    
    # Simulate real-time monitoring
    print("   Monitoring battery telemetry...")
    for minute in range(1, 4):
        incoming_data = generate_normal_telemetry(60)  # 60 points per minute
        results = production_model.analyze_telemetry(incoming_data)
        
        status = "🟢 NORMAL" if results['severity'] == 'INFO' else "🔴 ALERT"
        print(f"   Minute {minute}: {status} - "
              f"{results['anomalies_detected']} anomalies ({results['severity']})")
    
    print("\n✅ Production workflow completed!")
    print()


def main():
    """Main function - run all examples"""
    print("\n" + "=" * 70)
    print(" 🔋 EV-QA-Framework: ML Model Persistence Examples")
    print("=" * 70 + "\n")
    
    # Create model directory
    import os
    os.makedirs('models/production', exist_ok=True)
    
    # Run examples
    example_1_train_and_save()
    example_2_load_and_infer()
    example_3_model_versioning()
    example_4_anomaly_detector_pattern()
    example_5_production_workflow()
    
    print("=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print("\n💡 Saved models:")
    print("   - models/battery_analyzer_baseline.joblib")
    print("   - models/model_a_conservative.joblib")
    print("   - models/model_b_tolerant.joblib")
    print("   - models/anomaly_detector_prod.joblib")
    print("   - models/production/baseline_v1.0.joblib")
    print("\n📖 Use EVBatteryAnalyzer.load_model() to load")


if __name__ == "__main__":
    main()
