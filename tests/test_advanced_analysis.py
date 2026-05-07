import pytest
import pandas as pd
import numpy as np
from ev_qa_framework.analysis import EVBatteryAnalyzer

def test_cell_imbalance_normal():
    analyzer = EVBatteryAnalyzer()
    voltages = [3.8, 3.81, 3.79, 3.8]
    results = analyzer.detect_cell_imbalance(voltages)
    assert results['severity'] == "NORMAL"
    assert results['max_imbalance'] <= 0.05

def test_cell_imbalance_critical():
    analyzer = EVBatteryAnalyzer()
    voltages = [3.8, 3.8, 3.65, 3.8] # 0.15 imbalance
    results = analyzer.detect_cell_imbalance(voltages)
    assert results['severity'] == "CRITICAL"
    assert results['max_imbalance'] > 0.1

def test_thermal_runaway_risk_low():
    analyzer = EVBatteryAnalyzer()
    df = pd.DataFrame({
        'temp': [35.0, 35.1, 35.0, 35.2],
        'voltage': [400] * 4,
        'current': [100] * 4
    })
    results = analyzer.predict_thermal_runaway_risk(df)
    assert results['risk_level'] == "LOW"

def test_thermal_runaway_risk_critical():
    analyzer = EVBatteryAnalyzer()
    df = pd.DataFrame({
        'temp': [35.0, 45.0, 55.0, 68.0],
        'voltage': [400] * 4,
        'current': [100] * 4
    })
    results = analyzer.predict_thermal_runaway_risk(df)
    assert results['risk_level'] == "CRITICAL"
