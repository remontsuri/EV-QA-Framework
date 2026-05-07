"""
Tesla-Specific Advanced Analysis Demo
Demonstrates cell imbalance detection and thermal runaway risk prediction.
"""

import pandas as pd
import numpy as np
from ev_qa_framework.analysis import EVBatteryAnalyzer
from ev_qa_framework.models import BatteryCellDataModel

def simulate_tesla_pack_data(vin="5YJSA1E26HF000337", num_cells=96):
    """Simulates 96 cell group voltages for a Tesla Model S/X pack."""
    base_v = 3.85
    voltages = np.random.normal(base_v, 0.01, num_cells).tolist()
    # Introduce one lagging cell group (imbalance)
    voltages[42] = 3.72
    return BatteryCellDataModel(vin=vin, cell_voltages=voltages)

def main():
    print("=== Tesla Advanced Battery Analysis Demo ===\n")
    analyzer = EVBatteryAnalyzer()

    # 1. Cell Imbalance Detection
    print("1️⃣ Analyzing Cell Imbalance (96 Cell Groups):")
    pack_data = simulate_tesla_pack_data()
    imbalance_results = analyzer.detect_cell_imbalance(pack_data.cell_voltages)

    print(f"   VIN: {pack_data.vin}")
    print(f"   Avg Voltage: {imbalance_results['average_voltage']:.3f}V")
    print(f"   Max Imbalance: {imbalance_results['max_imbalance']:.3f}V")
    print(f"   Severity: {imbalance_results['severity']}")
    if imbalance_results['severity'] != "NORMAL":
        print(f"   🚨 Warning: Cell imbalance detected above Tesla safety limits!")
    print()

    # 2. Thermal Runaway Risk Prediction
    print("2️⃣ Predicting Thermal Runaway Risk (Simulated Overheating):")
    # Simulate rapidly rising temperature
    time_series = pd.DataFrame({
        'voltage': [400] * 10,
        'current': [150] * 10,
        'temp': [35.0, 36.5, 38.2, 40.5, 43.1, 46.8, 51.2, 56.5, 62.1, 68.5]
    })

    risk_results = analyzer.predict_thermal_runaway_risk(time_series)
    print(f"   Current Temp: {risk_results['current_temp']}°C")
    print(f"   Temp Rise Rate: {risk_results['avg_temp_rise_rate']:.2f}°C/sample")
    print(f"   Risk Level: {risk_results['risk_level']}")
    print(f"   Risk Score: {risk_results['risk_score']:.2f}")

    if risk_results['risk_level'] == "CRITICAL":
        print("   🔥 EMERGENCY: High risk of Thermal Runaway! Disconnecting contactors...")

if __name__ == "__main__":
    main()
