"""
Простой пример использования EV-QA-Framework
Демонстрирует базовую валидацию телеметрии и ML-детекцию аномалий
"""

from ev_qa_framework.models import validate_telemetry, BatteryTelemetryModel
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd
import numpy as np

def main():
    print("=== EV-QA-Framework Demo ===\n")
    
    # 1. ВАЛИДАЦИЯ ОДНОЙ ТОЧКИ ТЕЛЕМЕТРИИ
    print("1️⃣ Pydantic Validation Example:")
    print("-" * 50)
    
    # Valid telemetry
    valid_data = {
        "vin": "1HGBH41JXMN109186",
        "voltage": 396.5,
        "current": 125.3,
        "temperature": 35.2,
        "soc": 78.5,
        "soh": 96.2
    }
    
    try:
        telemetry = validate_telemetry(valid_data)
        print(f"✅ Valid telemetry accepted:")
        print(f"   VIN: {telemetry.vin}")
        print(f"   Voltage: {telemetry.voltage}V")
        print(f"   Temperature: {telemetry.temperature}°C")
        print(f"   SOC: {telemetry.soc}%\n")
    except Exception as e:
        print(f"❌ Validation failed: {e}\n")
    
    # Invalid telemetry (voltage out of range)
    invalid_data = {
        "vin": "1HGBH41JXMN109186",
        "voltage": 1500,  # Too high!
        "current": 125.3,
        "temperature": 35.2,
        "soc": 78.5,
        "soh": 96.2
    }
    
    try:
        telemetry = validate_telemetry(invalid_data)
        print("✅ This shouldn't print")
    except Exception as e:
        print(f"✅ Invalid data correctly rejected:")
        print(f"   Error: {str(e)[:100]}...\n")
    
    # 2. ML ANOMALY DETECTION
    print("2️⃣ ML Anomaly Detection Example:")
    print("-" * 50)
    
    # Generate normal battery telemetry
    np.random.seed(42)
    normal_telemetry = pd.DataFrame({
        'voltage': np.random.normal(400, 5, 500),   # 400V ± 5V
        'current': np.random.normal(120, 10, 500),  # 120A ± 10A
        'temp': np.random.normal(35, 3, 500),       # 35°C ± 3°C
        'soc': np.random.normal(80, 10, 500)        # 80% ± 10%
    })
    
    print(f"📊 Training data: {len(normal_telemetry)} samples of normal behavior")
    
    # Train detector
    detector = AnomalyDetector(contamination=0.01, n_estimators=200)
    detector.train(normal_telemetry)
    
    # Test data with anomalies
    test_data = pd.DataFrame({
        'voltage': [400, 405, 398, 600, 402],  # 600V is anomaly
        'current': [120, 118, 122, 120, 119],
        'temp': [35, 36, 34, 35, 35],
        'soc': [80, 79, 81, 80, 82]
    })
    
    print(f"\n🔍 Testing on {len(test_data)} samples (1 anomaly expected)...")
    predictions, scores = detector.detect(test_data)
    
    # Display results
    print("\n📋 Detection Results:")
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        status = "🚨 ANOMALY" if pred == -1 else "✅ Normal"
        print(f"   Sample {i+1}: {status} (score: {score:.3f})")
        if pred == -1:
            print(f"      → Voltage: {test_data.iloc[i]['voltage']}V (out of normal range)")
    
    # 3. INTEGRATION EXAMPLE
    print("\n3️⃣ Real-World Integration Example:")
    print("-" * 50)
    print("In production, you would:")
    print("1. Read CAN bus data → python-can library")
    print("2. Validate with Pydantic → validate_telemetry()")
    print("3. Store in DataFrame → pandas")
    print("4. Run ML detector → detector.detect()")
    print("5. Alert if anomaly → Send to Grafana/PagerDuty")
    
    print("\n✨ Demo Complete! Check README.md for full documentation.")

if __name__ == "__main__":
    main()
