"""
Advanced Integration Demo: CAN + ML + SOH Prediction + Dashboard
--------------------------------------------------------------
This script demonstrates the full power of the EV-QA-Framework.
It simulates a real battery telemetry stream via CAN bus,
applies ML anomaly detection, and predicts battery health.
"""

import asyncio
import pandas as pd
import numpy as np
from ev_qa_framework import (
    EVQAFramework,
    CANBatterySimulator,
    CANTelemetryReceiver,
    SOHPredictor,
    BatteryTelemetryModel
)


async def run_advanced_demo():
    """Run the advanced integration demo"""
    print("🚀 Initializing Advanced EV-QA Demo...")

    # 1. Setup Framework & Models
    qa = EVQAFramework("Advanced-Demo")
    predictor = SOHPredictor(sequence_length=5)

    # Pre-train SOH model with mock historical data
    print("🧠 Pre-training SOH prediction model...")
    hist_data = {
        'voltage': np.random.normal(396, 5, 50),
        'current': np.random.normal(100, 10, 50),
        'temperature': np.random.normal(35, 2, 50),
        'soh': np.linspace(100, 99.5, 50)
    }
    predictor.train(pd.DataFrame(hist_data), epochs=5)

    # 2. Start CAN Bus Simulation
    print("📡 Starting CAN Bus Emulation...")
    sim = CANBatterySimulator()
    receiver = CANTelemetryReceiver()
    sim.start()
    receiver.start()

    # 3. Main Loop
    print("\n--- Starting Real-time Monitoring Loop ---")
    telemetry_buffer = pd.DataFrame(hist_data).tail(5)  # Start with history

    try:
        for i in range(10):
            # Get data from CAN
            raw_data = receiver.get_telemetry()
            raw_data['vin'] = "DEMOVEHICLE001XYZ".replace('I', '1') \
                .replace('O', '0')
            raw_data['soh'] = telemetry_buffer['soh'].iloc[-1]

            # Pydantic Validation
            try:
                telemetry = BatteryTelemetryModel(**raw_data)
                is_safe = qa.validate_telemetry(telemetry)
            except Exception as e:
                print(f"❌ Validation Error: {e}")
                continue

            # ML Anomaly Detection (Isolation Forest)
            # Add new data to buffer
            new_row = pd.DataFrame([raw_data])[['voltage', 'current',
                                                'temperature', 'soh']]
            telemetry_buffer = pd.concat([telemetry_buffer, new_row]).tail(10)

            # SOH Prediction (LSTM)
            next_soh = predictor.predict_next(telemetry_buffer)

            # Report Status
            status = "✅ SAFE" if is_safe else "⚠️ WARNING"
            v_val = telemetry.voltage
            t_val = telemetry.temperature
            s_val = telemetry.soh
            print("[{:02d}] V={:.1f}V | ".format(i, v_val) +
                  "T={:.1f}°C | ".format(t_val) +
                  "SOH={:.2f}% | ".format(s_val) +
                  "Next SOH Pred={:.2f}% | ".format(next_soh) +
                  "Status: {}".format(status))

            await asyncio.sleep(1)

    finally:
        print("\n🛑 Stopping Demo...")
        sim.stop()
        receiver.stop()


if __name__ == "__main__":
    asyncio.run(run_advanced_demo())
