"""
Tesla Factory Simulation — 1500 cars, 3 battery chemistries, full QA pipeline.
Demonstrates EV-QA-Framework capabilities at scale.
"""
import random
import time
import json
import numpy as np
import pandas as pd
from ev_qa_framework import (
    EVQAFramework, BatteryTelemetryModel, FrameworkConfig,
    EVBatteryAnalyzer, ThermalRunawayPredictor, BatteryScorer,
    FleetAnalytics, BatteryDigitalTwin,
)
from ev_qa_framework.config import SafetyThresholds
from ev_qa_framework.chemistries import get_profile, ChemistryKey
from ev_qa_framework.cell_balance import CellBalanceAnalyzer

random.seed(42)
np.random.seed(42)

# === CONFIGURATION ===
TOTAL_CARS = 1500
CELLS_PER_PACK = 96
READINGS_PER_CAR = 50  # telemetry readings per car

BATTERY_PROFILES = {
    "LFP_Cooper": {"chemistry": "lfp", "share": 0.40, "model": "Model 2",
                   "voltage_nom": 3.2, "voltage_min": 2.5, "voltage_max": 3.65,
                   "temp_nom": 30, "temp_range": 5, "soh_start": 99.5,
                   "defect_rate": 0.02},
    "NMC_Panasonic": {"chemistry": "nmc", "share": 0.35, "model": "Model 3",
                      "voltage_nom": 3.6, "voltage_min": 3.0, "voltage_max": 4.2,
                      "temp_nom": 32, "temp_range": 6, "soh_start": 99.8,
                      "defect_rate": 0.03},
    "NCA_Tesla": {"chemistry": "nca", "share": 0.25, "model": "Model S/X",
                  "voltage_nom": 3.7, "voltage_min": 3.0, "voltage_max": 4.2,
                  "temp_nom": 33, "temp_range": 7, "soh_start": 99.9,
                  "defect_rate": 0.015},
}

# === GENERATE TELEMETRY ===
def generate_car_telemetry(car_id: int, profile: dict, defective: bool = False) -> list[dict]:
    """Generate realistic telemetry for one car."""
    readings = []
    soh = profile["soh_start"]
    base_temp = profile["temp_nom"]

    for i in range(READINGS_PER_CAR):
        # Normal degradation over readings
        soh -= random.uniform(0.001, 0.005)

        temp = base_temp + random.gauss(0, profile["temp_range"] * 0.3)
        voltage = profile["voltage_nom"] + random.gauss(0, 0.05)
        current = 50 + random.gauss(0, 5)
        soc = max(10, min(100, 80 - i * 0.5 + random.gauss(0, 2)))

        # Inject defects
        if defective:
            defect_type = random.choice(["thermal_spike", "voltage_drop", "cell_imbalance"])
            if defect_type == "thermal_spike" and i > READINGS_PER_CAR // 2:
                temp += random.uniform(20, 40)  # thermal anomaly
            elif defect_type == "voltage_drop":
                voltage -= random.uniform(0.3, 0.8)  # voltage sag
            elif defect_type == "cell_imbalance":
                pass  # handled in cell voltages

        # Clamp values
        temp = max(-10, min(80, temp))
        voltage = max(profile["voltage_min"] - 0.1, min(profile["voltage_max"] + 0.1, voltage))

        readings.append({
            "car_id": car_id,
            "battery_type": profile["model"],
            "chemistry": profile["chemistry"],
            "reading_num": i,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "temperature": round(temp, 2),
            "soc": round(soc, 1),
            "soh": round(max(50, soh), 2),
        })

    return readings

def generate_cell_voltages(profile: dict, defective: bool = False) -> list[float]:
    """Generate cell voltages for one pack."""
    base = profile["voltage_nom"]
    voltages = [base + random.gauss(0, 0.01) for _ in range(CELLS_PER_PACK)]

    if defective:
        # Inject 3-5 outlier cells
        n_outliers = random.randint(3, 5)
        for _ in range(n_outliers):
            idx = random.randint(0, CELLS_PER_PACK - 1)
            voltages[idx] += random.choice([-1, 1]) * random.uniform(0.05, 0.15)

    return [round(v, 4) for v in voltages]

# === MAIN SIMULATION ===
print("=" * 70)
print("  TESLA FACTORY SIMULATION — 1500 CARS")
print("  EV-QA-Framework v2.5.0 Full Pipeline Test")
print("=" * 70)

t_start = time.time()

# Create configs for each chemistry
configs = {}
for name, prof in BATTERY_PROFILES.items():
    chem = get_profile(prof["chemistry"])
    configs[name] = FrameworkConfig(
        safety_thresholds=SafetyThresholds(
            max_temperature=45.0,
            min_voltage=prof["voltage_min"] * CELLS_PER_PACK * 0.9,
            max_voltage=prof["voltage_max"] * CELLS_PER_PACK * 1.1,
            max_temperature_jump=8.0,
        )
    )

# Generate all car data
print("\n[1/5] Generating telemetry for 1500 cars...")
all_readings = []
car_metadata = []

for car_id in range(TOTAL_CARS):
    # Select battery type based on share
    r = random.random()
    cumulative = 0
    selected = None
    for name, prof in BATTERY_PROFILES.items():
        cumulative += prof["share"]
        if r <= cumulative:
            selected = name
            break

    prof = BATTERY_PROFILES[selected]
    defective = random.random() < prof["defect_rate"]

    readings = generate_car_telemetry(car_id, prof, defective)
    all_readings.extend(readings)

    car_metadata.append({
        "car_id": car_id,
        "battery_type": prof["model"],
        "chemistry": prof["chemistry"],
        "defective": defective,
        "n_readings": len(readings),
    })

df_all = pd.DataFrame(all_readings)
df_meta = pd.DataFrame(car_metadata)

print(f"  Generated {len(all_readings)} readings for {TOTAL_CARS} cars")
print(f"  Battery mix: Model 2 (LFP)={sum(1 for m in car_metadata if m['battery_type']=='Model 2')}, "
      f"Model 3 (NMC)={sum(1 for m in car_metadata if m['battery_type']=='Model 3')}, "
      f"Model S/X (NCA)={sum(1 for m in car_metadata if m['battery_type']=='Model S/X')}")
print(f"  Defective cars: {sum(1 for m in car_metadata if m['defective'])} "
      f"({sum(1 for m in car_metadata if m['defective'])/TOTAL_CARS*100:.1f}%)")

# === PHASE 2: ANOMALY DETECTION ===
print("\n[2/5] Running ML anomaly detection...")
analyzer = EVBatteryAnalyzer(contamination=0.05, n_estimators=200)

# Analyze each car
anomaly_results = []
for car_id in range(TOTAL_CARS):
    car_df = df_all[df_all["car_id"] == car_id][["voltage", "current", "temperature"]].copy()
    if len(car_df) >= 10:
        result = analyzer.analyze_telemetry(car_df)
        if result:
            anomaly_results.append({
                "car_id": car_id,
                "anomalies_detected": result.get("anomalies_detected", 0),
                "anomaly_pct": result.get("anomaly_percentage", 0),
                "severity": result.get("severity", "UNKNOWN"),
            })

df_anomalies = pd.DataFrame(anomaly_results)
n_flagged = len(df_anomalies[df_anomalies["anomalies_detected"] > 0])
print(f"  Analyzed {len(anomaly_results)} cars")
print(f"  Cars with anomalies: {n_flagged} ({n_flagged/len(anomaly_results)*100:.1f}%)")
print(f"  Severity distribution: {df_anomalies['severity'].value_counts().to_dict()}")

# === PHASE 3: THERMAL RUNAWAY DETECTION ===
print("\n[3/5] Running thermal runaway prediction...")
thermal_pred = ThermalRunawayPredictor(mode="rule")

thermal_results = []
for car_id in range(TOTAL_CARS):
    car_temps = df_all[df_all["car_id"] == car_id]["temperature"].tolist()
    if len(car_temps) >= 3:
        temp_df = pd.DataFrame({"temp": car_temps})
        result = thermal_pred.predict_risk(temp_df)
        thermal_results.append({
            "car_id": car_id,
            "risk_level": result.get("risk_level", "UNKNOWN"),
            "risk_score": result.get("risk_score", 0),
        })

df_thermal = pd.DataFrame(thermal_results)
print(f"  Analyzed {len(thermal_results)} cars")
risk_dist = df_thermal["risk_level"].value_counts().to_dict()
print(f"  Risk distribution: {risk_dist}")
critical_cars = df_thermal[df_thermal["risk_level"] == "CRITICAL"]["car_id"].tolist()
if critical_cars:
    print(f"  CRITICAL risk cars: {critical_cars[:10]}{'...' if len(critical_cars) > 10 else ''}")

# === PHASE 4: CELL BALANCE + SCORING ===
print("\n[4/5] Running cell balance analysis + battery scoring...")
cell_analyzer = CellBalanceAnalyzer()
scorer = BatteryScorer()

score_results = []
# Map model name to profile key
model_to_profile = {prof["model"]: name for name, prof in BATTERY_PROFILES.items()}

for car_id in range(500):  # Score first 500 for speed
    car_df = df_all[df_all["car_id"] == car_id].copy()
    car_df["temp"] = car_df["temperature"]

    # Cell balance
    car_meta = df_meta[df_meta["car_id"] == car_id]
    profile_key = model_to_profile[car_meta["battery_type"].iloc[0]]
    cell_voltages = generate_cell_voltages(
        BATTERY_PROFILES[profile_key],
        defective=car_meta["defective"].iloc[0]
    )
    balance = cell_analyzer.classify_severity(cell_voltages)

    # Battery score
    score_result = scorer.compute_score(car_df, cell_voltages)
    score_results.append({
        "car_id": car_id,
        "score": score_result["score"],
        "grade": score_result["grade"],
        "cell_balance": balance,
    })

df_scores = pd.DataFrame(score_results)
print(f"  Scored {len(score_results)} cars")
print(f"  Average score: {df_scores['score'].mean():.1f}/100")
print(f"  Grade distribution: {df_scores['grade'].value_counts().to_dict()}")
print(f"  Cell balance: {df_scores['cell_balance'].value_counts().to_dict()}")

# === PHASE 5: SUMMARY REPORT ===
print("\n[5/5] Generating summary report...")
t_end = time.time()
elapsed = t_end - t_start

# Merge all results
summary = df_meta.merge(df_anomalies, on="car_id", how="left") \
    .merge(df_thermal, on="car_id", how="left")

# Per-chemistry breakdown
print("\n" + "=" * 70)
print("  SIMULATION RESULTS")
print("=" * 70)

print(f"\n  Total cars:          {TOTAL_CARS}")
print(f"  Total readings:      {len(all_readings):,}")
print(f"  Processing time:     {elapsed:.1f}s")
print(f"  Readings/sec:        {len(all_readings)/elapsed:,.0f}")

print(f"\n--- Battery Mix ---")
for name, prof in BATTERY_PROFILES.items():
    count = sum(1 for m in car_metadata if m["battery_type"] == prof["model"])
    defect = sum(1 for m in car_metadata if m["battery_type"] == prof["model"] and m["defective"])
    print(f"  {prof['model']:12s} ({prof['chemistry'].upper():3s}): {count:4d} cars, {defect:3d} defective ({defect/count*100:.1f}%)")

print(f"\n--- Anomaly Detection ---")
print(f"  Cars analyzed:      {len(anomaly_results)}")
print(f"  Flagged:            {n_flagged} ({n_flagged/len(anomaly_results)*100:.1f}%)")
for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]:
    count = len(df_anomalies[df_anomalies["severity"] == sev])
    if count > 0:
        print(f"    {sev:10s}: {count:4d}")

print(f"\n--- Thermal Runaway ---")
for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    count = risk_dist.get(level, 0)
    if count > 0:
        print(f"    {level:10s}: {count:4d}")

print(f"\n--- Battery Scoring (first 500 cars) ---")
print(f"  Average score:      {df_scores['score'].mean():.1f}/100")
for grade in ["A+", "A", "B", "C", "D", "F"]:
    count = len(df_scores[df_scores["grade"] == grade])
    if count > 0:
        print(f"    Grade {grade:2s}: {count:4d} cars")

print(f"\n--- Cell Balance ---")
for bal in ["NORMAL", "WARNING", "CRITICAL"]:
    count = len(df_scores[df_scores["cell_balance"] == bal])
    if count > 0:
        print(f"    {bal:10s}: {count:4d}")

# Top defective cars
defective_cars = summary[summary["defective"] == True].sort_values("anomaly_pct", ascending=False).head(10)
if len(defective_cars) > 0:
    print(f"\n--- Top 10 Defective Cars ---")
    print(f"  {'Car ID':>6s} {'Battery':>12s} {'Anomaly%':>9s} {'Severity':>10s} {'Risk':>10s}")
    for _, row in defective_cars.iterrows():
        print(f"  {row['car_id']:6d} {row['battery_type']:>12s} {row.get('anomaly_pct', 0):8.1f}% {row.get('severity', '?'):>10s} {row.get('risk_level', '?'):>10s}")

# Export results
report = {
    "simulation": "Tesla Factory QA Pipeline",
    "version": "2.5.0",
    "total_cars": TOTAL_CARS,
    "total_readings": len(all_readings),
    "processing_time_s": round(elapsed, 1),
    "battery_mix": {name: {"count": sum(1 for m in car_metadata if m["battery_type"] == prof["model"]),
                           "defective": sum(1 for m in car_metadata if m["battery_type"] == prof["model"] and m["defective"])}
                    for name, prof in BATTERY_PROFILES.items()},
    "anomaly_detection": {"analyzed": len(anomaly_results), "flagged": n_flagged,
                          "severity_distribution": df_anomalies["severity"].value_counts().to_dict()},
    "thermal_runaway": {"analyzed": len(thermal_results), "risk_distribution": risk_dist},
    "battery_scoring": {"scored": len(score_results), "avg_score": round(df_scores["score"].mean(), 1),
                        "grade_distribution": df_scores["grade"].value_counts().to_dict()},
}

with open("tesla_factory_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Report saved: tesla_factory_report.json")
print("=" * 70)
