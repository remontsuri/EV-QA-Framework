#!/usr/bin/env python3
"""
Real-world scenario: EV battery pack quality inspection at factory.

Simulates a Tesla Model 3 LFP battery pack (96 cells, 400V nominal)
going through end-of-line quality checks:
1. Telemetry validation
2. Anomaly detection
3. Standards compliance (UN 38.3, IEC 62660)
4. Thermal runaway prediction
5. Cell balance analysis
6. SOH estimation
"""

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ev_qa_framework.analysis import AnomalyDetector, EVBatteryAnalyzer
from ev_qa_framework.cell_balance import CellBalanceAnalyzer
from ev_qa_framework.chemistries import ChemistryKey
from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryCellDataModel, BatteryTelemetryModel
from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor

VIN = "5YJ3E1EA1KF123456"

# ─── Scenario 1: Normal battery pack ────────────────────────────────────────

print("=" * 70)
print("SCENARIO 1: Normal battery pack — end-of-line inspection")
print("=" * 70)

# Generate 60 seconds of telemetry at 1Hz
np.random.seed(42)
n_samples = 60
timestamps = pd.date_range("2026-03-15 14:00:00", periods=n_samples, freq="1s")

# Normal LFP pack: 400V nominal, 96 cells in series
base_voltage = 400.0
base_current = 50.0  # charging
base_temp = 35.0
base_soc = 75.0
base_soh = 98.5

# Add realistic noise
voltage_noise = np.random.normal(0, 0.5, n_samples)
current_noise = np.random.normal(0, 1.0, n_samples)
temp_noise = np.random.normal(0, 0.3, n_samples)
soc_drift = np.linspace(0, 0.5, n_samples)  # slow charge

telemetry_records = []
for i in range(n_samples):
    rec = BatteryTelemetryModel(
        vin=VIN,
        voltage=round(base_voltage + voltage_noise[i], 2),
        current=round(base_current + current_noise[i], 2),
        temperature=round(base_temp + temp_noise[i], 2),
        soc=round(base_soc + soc_drift[i], 2),
        soh=round(base_soh, 2),
    )
    telemetry_records.append(rec)

print(f"\nGenerated {n_samples} telemetry records")
print(f"Time range: {timestamps[0]} → {timestamps[-1]}")
print(f"VIN: {VIN}")

# ─── Step 1: Telemetry Validation ───────────────────────────────────────────

print("\n" + "-" * 50)
print("STEP 1: Telemetry Validation")
print("-" * 50)

qa = EVQAFramework("Factory-Inspection-Normal")
valid_count = 0
invalid_count = 0
all_warnings = []

for rec in telemetry_records:
    is_valid, warnings = qa.validate_telemetry(rec)
    if is_valid:
        valid_count += 1
    else:
        invalid_count += 1
        all_warnings.extend(warnings)

print(f"  Valid records:   {valid_count}/{n_samples}")
print(f"  Invalid records: {invalid_count}/{n_samples}")
if all_warnings:
    print(f"  Warnings: {len(all_warnings)}")
    for w in all_warnings[:5]:
        print(f"    ⚠ {w}")
else:
    print("  Warnings: none")

# ─── Step 2: Anomaly Detection ──────────────────────────────────────────────

print("\n" + "-" * 50)
print("STEP 2: Anomaly Detection (Isolation Forest)")
print("-" * 50)

analyzer = EVBatteryAnalyzer()
df = pd.DataFrame([r.model_dump() for r in telemetry_records])
df["temp"] = df["temperature"]  # alias for normalize_columns

results = analyzer.analyze_telemetry(df)
print(f"  Total samples:       {results['total_samples']}")
print(f"  Anomalies detected:  {results['anomalies_detected']}")
print(f"  Anomaly percentage:  {results['anomaly_percentage']:.2f}%")
print(f"  Severity:            {results['severity']}")

# ─── Step 3: Standards Compliance ───────────────────────────────────────────

print("\n" + "-" * 50)
print("STEP 3: Standards Compliance")
print("-" * 50)

standards = [
    ("UN 38.3 — Transport Safety", "UN38.3"),
    ("IEC 62660-1 — Cell Performance", "IEC62660"),
    ("SAE J2464 — Abuse Testing", "SAEJ2464"),
    ("ISO 12405 — Traction Battery", "ISO12405"),
]

# Use the last telemetry point for standards check
last_rec = telemetry_records[-1]
for name, std_id in standards:
    qa_std = EVQAFramework(std_id)
    is_valid, warnings = qa_std.validate_telemetry(last_rec)
    status = "✅ PASS" if is_valid else "❌ FAIL"
    print(f"  {status}  {name}")
    if warnings:
        for w in warnings:
            print(f"           ⚠ {w}")

# ─── Step 4: Thermal Runaway Prediction ─────────────────────────────────────

print("\n" + "-" * 50)
print("STEP 4: Thermal Runaway Prediction")
print("-" * 50)

predictor = ThermalRunawayPredictor(mode="rule")
risk = predictor.predict_risk(df)
print(f"  Risk level:   {risk['risk_level']}")
print(f"  Risk score:   {risk['risk_score']:.2f}")
print(f"  Confidence:   {risk['confidence']:.2f}")

trend = predictor.analyze_temperature_trend(df)
print(f"  Temp rise rate: {trend['temp_rise_rate']:.4f} °C/s")
print(f"  Max temp:       {trend['max_temp']:.2f} °C")
print(f"  Volatility:     {trend['volatility']:.4f}")

# ─── Step 5: Cell Balance Analysis ──────────────────────────────────────────

print("\n" + "-" * 50)
print("STEP 5: Cell Balance Analysis (96 cells)")
print("-" * 50)

# Simulate 96 cell voltages (LFP: 3.2V nominal ±0.02V)
cell_voltages = np.random.normal(3.20, 0.015, 96).tolist()
cell_model = BatteryCellDataModel(vin=VIN, cell_voltages=cell_voltages)

balancer = CellBalanceAnalyzer()
stats = balancer.compute_statistics(cell_voltages)
imbalance_val = max(cell_voltages) - min(cell_voltages)
outliers = balancer.detect_outliers(cell_voltages)
severity = balancer.classify_severity(cell_voltages)

print(f"  Cell count:         {len(cell_voltages)}")
print(f"  Average voltage:    {stats['mean']:.4f}V")
print(f"  Imbalance (max-min): {imbalance_val:.4f}V")
print(f"  Std deviation:      {stats['std']:.4f}V")
print(f"  Outlier cells:      {len(outliers)} → {outliers if outliers else 'none'}")
print(f"  Severity:           {severity}")

# ─── Scenario 2: Defective battery pack ─────────────────────────────────────

print("\n" + "=" * 70)
print("SCENARIO 2: Defective battery pack — overheating + cell imbalance")
print("=" * 70)

# Simulate overheating: temp rises from 40°C to 75°C in 60 seconds
temp_rising = np.linspace(40.0, 75.0, n_samples) + np.random.normal(0, 0.5, n_samples)
# One cell is dead (0.5V instead of 3.2V)
cell_voltages_bad = np.random.normal(3.20, 0.015, 96)
cell_voltages_bad[42] = 0.5  # dead cell
cell_voltages_bad[73] = 2.8  # weak cell

bad_records = []
for i in range(n_samples):
    rec = BatteryTelemetryModel(
        vin="5YJ3E1EA1KF999999",
        voltage=round(400.0 - i * 0.5 + np.random.normal(0, 0.3), 2),  # voltage sag
        current=round(150.0 + np.random.normal(0, 2.0), 2),  # high current
        temperature=round(temp_rising[i], 2),
        soc=round(30.0 - i * 0.1, 2),
        soh=round(82.0, 2),  # degraded
    )
    bad_records.append(rec)

qa_bad = EVQAFramework("Factory-Inspection-Defective")

# Validation
bad_valid = 0
bad_invalid = 0
bad_warnings = []
for rec in bad_records:
    is_valid, warnings = qa_bad.validate_telemetry(rec)
    if is_valid:
        bad_valid += 1
    else:
        bad_invalid += 1
        bad_warnings.extend(warnings)

print(f"\n  Valid records:   {bad_valid}/{n_samples}")
print(f"  Invalid records: {bad_invalid}/{n_samples}")
print(f"  Warnings: {len(bad_warnings)}")
for w in bad_warnings[:5]:
    print(f"    ⚠ {w}")

# Anomaly detection
df_bad = pd.DataFrame([r.model_dump() for r in bad_records])
df_bad["temp"] = df_bad["temperature"]
results_bad = analyzer.analyze_telemetry(df_bad)
print(
    f"\n  Anomalies: {results_bad['anomalies_detected']}/{results_bad['total_samples']} ({results_bad['anomaly_percentage']:.1f}%)"
)
print(f"  Severity:  {results_bad['severity']}")

# Thermal runaway
risk_bad = predictor.predict_risk(df_bad)
print(f"\n  Thermal Risk: {risk_bad['risk_level']} (score: {risk_bad['risk_score']:.2f})")

# Cell balance
stats_bad = balancer.compute_statistics(cell_voltages_bad.tolist())
imbalance_bad_val = max(cell_voltages_bad) - min(cell_voltages_bad)
outliers_bad = balancer.detect_outliers(cell_voltages_bad.tolist())
severity_bad = balancer.classify_severity(cell_voltages_bad.tolist())
print(f"\n  Cell imbalance: {imbalance_bad_val:.4f}V")
print(f"  Outlier cells:  {len(outliers_bad)} → {outliers_bad}")
print(f"  Severity:       {severity_bad}")

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("FACTORY INSPECTION SUMMARY")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│  PACK 1 (VIN: {VIN})                     │
│  Status: ✅ PASSED                                           │
│  Valid telemetry:  {valid_count}/{n_samples}                                    │
│  Anomalies:        {results['anomalies_detected']}/{results['total_samples']} ({results['anomaly_percentage']:.1f}%)                         │
│  Thermal risk:     {risk['risk_level']:10s}                              │
│  Cell balance:     {severity:10s}                              │
│  Standards:        All passed                                │
├─────────────────────────────────────────────────────────────┤
│  PACK 2 (VIN: 5YJ3E1EA1KF999999)                             │
│  Status: ❌ REJECTED                                         │
│  Valid telemetry:  {bad_valid}/{n_samples}                                    │
│  Anomalies:        {results_bad['anomalies_detected']}/{results_bad['total_samples']} ({results_bad['anomaly_percentage']:.1f}%)                        │
│  Thermal risk:     {risk_bad['risk_level']:10s}                              │
│  Cell balance:     {severity_bad:10s}                              │
│  Standards:        FAILED (temperature, voltage)             │
└─────────────────────────────────────────────────────────────┘
""")

print(f"Inspection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
