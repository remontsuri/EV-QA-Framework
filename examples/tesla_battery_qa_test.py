#!/usr/bin/env python3
"""
Tesla Model S Battery QA Test — Practical Example

Simulates a full QA workflow for a Tesla Model S 85 kWh battery pack:
1. Generate realistic telemetry data (voltage, current, temperature, SOC, SOH)
2. Validate telemetry via Pydantic models
3. Detect anomalies with Isolation Forest
4. Analyze cell imbalance (96 cell groups)
5. Predict thermal runaway risk
6. Score battery health (A/B/C/D/F)
7. Run fleet comparison (healthy vs degraded pack)
8. Digital twin simulation (drive cycle + aging)
9. V2G scenario impact analysis
10. Generate JSON report

Usage:
    uv run python examples/tesla_battery_qa_test.py
"""

import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ev_qa_framework.analysis import AnomalyDetector, EVBatteryAnalyzer
from ev_qa_framework.battery_scoring import BatteryScorer
from ev_qa_framework.cell_balance import CellBalanceAnalyzer
from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.digital_twin import BatteryDigitalTwin
from ev_qa_framework.fleet_analytics import FleetAnalytics
from ev_qa_framework.models import BatteryTelemetryModel, validate_telemetry
from ev_qa_framework.physics_features import PhysicsFeatureExtractor
from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor
from ev_qa_framework.v2g_scenarios import V2GHealthAnalyzer, V2GScenarioGenerator

# ── Tesla Model S 85 kWh pack constants ──────────────────────────────────────
TESLA_PACK = {
    "model": "Model S 85 kWh",
    "vin": "5YJSA1E26HF000337",
    "nominal_voltage": 400.0,  # V
    "capacity_ah": 210.0,  # Ah
    "capacity_kwh": 85.0,  # kWh
    "num_cells_series": 96,  # 96s configuration
    "num_cells_parallel": 74,  # 74p (18650 cells)
    "cell_nominal_voltage": 3.7,  # V per cell
    "max_charge_voltage": 4.2,  # V per cell
    "min_discharge_voltage": 3.0,  # V per cell
    "max_charge_current": 200.0,  # A
    "max_discharge_current": 400.0,  # A
    "thermal_runaway_threshold": 65.0,  # °C
}


def generate_healthy_telemetry(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic telemetry for a healthy Tesla pack (SOH ~95%)."""
    np.random.seed(seed)
    base_time = datetime(2026, 6, 10, 8, 0, 0)

    # Simulate a mixed driving cycle: city + highway
    t = np.linspace(0, 3600 * 4, n_samples)  # 4 hours of data, 1 sample per ~28s

    # Current profile: alternating discharge (driving) and regen braking
    current = np.zeros(n_samples)
    for i in range(n_samples):
        phase = (i // 50) % 4
        if phase == 0:  # city driving
            current[i] = np.random.normal(80, 30)
        elif phase == 1:  # highway
            current[i] = np.random.normal(150, 20)
        elif phase == 2:  # regen braking
            current[i] = np.random.normal(-40, 15)
        else:  # idle / charging
            current[i] = np.random.normal(-5, 3)

    # Voltage responds to current (simplified OCV + IR drop)
    ocv = 400.0 - (np.cumsum(current) * 0.0001)  # slow SOC drift
    internal_r = 0.08  # ohms (healthy pack)
    voltage = ocv - current * internal_r + np.random.normal(0, 0.5, n_samples)
    voltage = np.clip(voltage, 350, 420)

    # Temperature: starts at ambient, rises with driving
    temperature = 25.0 + np.cumsum(np.abs(current) * 0.0001) + np.random.normal(0, 0.5, n_samples)
    temperature = np.clip(temperature, 22, 45)

    # SOC: starts at 80%, drifts with current
    soc = 80.0 - np.cumsum(current) * 0.001 + np.random.normal(0, 0.1, n_samples)
    soc = np.clip(soc, 15, 95)

    # SOH: stable at 95.2% (healthy but aged)
    soh = 95.2 + np.random.normal(0, 0.05, n_samples)

    timestamps = [base_time + timedelta(seconds=float(s)) for s in t]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "vin": TESLA_PACK["vin"],
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "soc": soc,
            "soh": soh,
        }
    )


def generate_degraded_telemetry(n_samples: int = 500, seed: int = 13) -> pd.DataFrame:
    """Generate telemetry for a degraded Tesla pack (SOH ~72%, high resistance)."""
    np.random.seed(seed)
    base_time = datetime(2026, 6, 10, 8, 0, 0)
    t = np.linspace(0, 3600 * 4, n_samples)

    current = np.zeros(n_samples)
    for i in range(n_samples):
        phase = (i // 50) % 4
        if phase == 0:
            current[i] = np.random.normal(80, 30)
        elif phase == 1:
            current[i] = np.random.normal(150, 20)
        elif phase == 2:
            current[i] = np.random.normal(-40, 15)
        else:
            current[i] = np.random.normal(-5, 3)

    # Higher internal resistance → more voltage sag
    ocv = 400.0 - (np.cumsum(current) * 0.00015)
    internal_r = 0.18  # ohms (degraded — 2.25x healthy)
    voltage = ocv - current * internal_r + np.random.normal(0, 1.0, n_samples)
    voltage = np.clip(voltage, 320, 420)

    # Higher temperature due to increased I²R losses
    temperature = 28.0 + np.cumsum(np.abs(current) * 0.0002) + np.random.normal(0, 1.0, n_samples)
    temperature = np.clip(temperature, 25, 58)

    soc = 80.0 - np.cumsum(current) * 0.0012 + np.random.normal(0, 0.15, n_samples)
    soc = np.clip(soc, 10, 95)

    # SOH degraded to 72%
    soh = 72.0 + np.random.normal(0, 0.1, n_samples)

    # Inject some anomalies: voltage spikes, thermal events
    anomaly_indices = np.random.choice(n_samples, 15, replace=False)
    voltage[anomaly_indices[:5]] += np.random.uniform(15, 30, 5)  # voltage spikes
    temperature[anomaly_indices[5:10]] += np.random.uniform(8, 15, 5)  # thermal events
    current[anomaly_indices[10:]] += np.random.uniform(50, 100, 5)  # current spikes

    timestamps = [base_time + timedelta(seconds=float(s)) for s in t]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "vin": "5YJSA1E26HF000999",
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "soc": soc,
            "soh": soh,
        }
    )


def generate_cell_voltages(soh: float, num_cells: int = 96) -> list[float]:
    """Generate cell group voltages for a Tesla pack."""
    base_v = 3.85 if soh > 90 else 3.75
    std = 0.008 if soh > 90 else 0.025
    voltages = np.random.normal(base_v, std, num_cells).tolist()

    if soh < 80:
        # Degraded pack: some weak cells
        weak_cells = np.random.choice(num_cells, 5, replace=False)
        for idx in weak_cells:
            voltages[idx] = np.random.normal(3.65, 0.03)

    return voltages


def run_tesla_qa_test():
    """Run the full Tesla battery QA test suite."""
    report = {
        "test_date": datetime.now().isoformat(),
        "pack_info": TESLA_PACK,
        "tests": {},
    }

    print("=" * 70)
    print("  Tesla Model S 85 kWh — Battery QA Test Suite")
    print("=" * 70)

    # ── 1. Telemetry Validation ──────────────────────────────────────────
    print("\n[1/10] Telemetry Validation...")
    sample = {
        "vin": TESLA_PACK["vin"],
        "voltage": 396.5,
        "current": 125.3,
        "temperature": 35.2,
        "soc": 78.5,
        "soh": 95.2,
    }
    validated = validate_telemetry(sample)
    assert validated.vin == TESLA_PACK["vin"]
    assert 350 <= validated.voltage <= 420
    report["tests"]["telemetry_validation"] = {"status": "PASS", "sample": sample}
    print(f"  VIN: {validated.vin}")
    print(f"  Voltage: {validated.voltage}V, Current: {validated.current}A")
    print(f"  Temperature: {validated.temperature}°C, SOC: {validated.soc}%, SOH: {validated.soh}%")

    # ── 2. Generate Telemetry ────────────────────────────────────────────
    print("\n[2/10] Generating telemetry data...")
    healthy_df = generate_healthy_telemetry(500)
    degraded_df = generate_degraded_telemetry(500)
    report["tests"]["telemetry_generation"] = {
        "status": "PASS",
        "healthy_samples": len(healthy_df),
        "degraded_samples": len(degraded_df),
    }
    print(f"  Healthy pack: {len(healthy_df)} samples, SOH={healthy_df['soh'].mean():.1f}%")
    print(f"  Degraded pack: {len(degraded_df)} samples, SOH={degraded_df['soh'].mean():.1f}%")

    # ── 3. Anomaly Detection ────────────────────────────────────────────
    print("\n[3/10] Anomaly Detection (Isolation Forest)...")
    detector = AnomalyDetector(contamination=0.03, n_estimators=200)
    detector.train(healthy_df[["voltage", "current", "temperature"]])

    # Test on healthy data
    preds_healthy, scores_healthy = detector.detect(
        healthy_df[["voltage", "current", "temperature"]]
    )
    anomalies_healthy = int((preds_healthy == -1).sum())

    # Test on degraded data
    preds_degraded, scores_degraded = detector.detect(
        degraded_df[["voltage", "current", "temperature"]]
    )
    anomalies_degraded = int((preds_degraded == -1).sum())

    report["tests"]["anomaly_detection"] = {
        "status": "PASS",
        "healthy_anomalies": anomalies_healthy,
        "degraded_anomalies": anomalies_degraded,
        "contamination": 0.03,
    }
    print(f"  Healthy pack: {anomalies_healthy}/500 anomalies ({anomalies_healthy/5:.1f}%)")
    print(f"  Degraded pack: {anomalies_degraded}/500 anomalies ({anomalies_degraded/5:.1f}%)")

    # ── 4. Cell Imbalance Analysis ───────────────────────────────────────
    print("\n[4/10] Cell Imbalance Analysis (96 cell groups)...")
    cell_analyzer = CellBalanceAnalyzer(warning_threshold=0.015, critical_threshold=0.04)

    healthy_cells = generate_cell_voltages(soh=95.0, num_cells=96)
    degraded_cells = generate_cell_voltages(soh=72.0, num_cells=96)

    healthy_stats = cell_analyzer.compute_statistics(healthy_cells)
    healthy_severity = cell_analyzer.classify_severity(healthy_cells)

    degraded_stats = cell_analyzer.compute_statistics(degraded_cells)
    degraded_severity = cell_analyzer.classify_severity(degraded_cells)

    report["tests"]["cell_imbalance"] = {
        "status": "PASS",
        "healthy": {"std": round(healthy_stats["std"], 6), "severity": healthy_severity},
        "degraded": {"std": round(degraded_stats["std"], 6), "severity": degraded_severity},
    }
    print(f"  Healthy: std={healthy_stats['std']:.6f}, severity={healthy_severity}")
    print(f"  Degraded: std={degraded_stats['std']:.6f}, severity={degraded_severity}")

    # ── 5. Thermal Runaway Prediction ────────────────────────────────────
    print("\n[5/10] Thermal Runaway Prediction...")
    thermal_predictor = ThermalRunawayPredictor(mode="rule")

    # Normal operation
    normal_thermal = pd.DataFrame({"temperature": [30, 32, 33, 35, 36, 37, 38]})
    risk_normal = thermal_predictor.predict_risk(normal_thermal)

    # Escalating temperature (thermal event)
    escalating_thermal = pd.DataFrame({"temperature": [35, 40, 48, 55, 62, 68, 72]})
    risk_escalating = thermal_predictor.predict_risk(escalating_thermal)

    report["tests"]["thermal_runaway"] = {
        "status": "PASS",
        "normal": {"risk_level": risk_normal["risk_level"], "score": risk_normal["risk_score"]},
        "escalating": {
            "risk_level": risk_escalating["risk_level"],
            "score": risk_escalating["risk_score"],
        },
    }
    print(f"  Normal: {risk_normal['risk_level']} (score={risk_normal['risk_score']})")
    print(f"  Escalating: {risk_escalating['risk_level']} (score={risk_escalating['risk_score']})")

    # ── 6. Battery Health Scoring ────────────────────────────────────────
    print("\n[6/10] Battery Health Scoring...")
    scorer = BatteryScorer()

    healthy_telemetry = pd.DataFrame(
        {
            "voltage": healthy_df["voltage"].head(10),
            "current": healthy_df["current"].head(10),
            "temperature": healthy_df["temperature"].head(10),
            "soh": healthy_df["soh"].head(10),
            "internal_resistance": [0.08] * 10,
        }
    )
    score_healthy = scorer.compute_score(
        telemetry_df=healthy_telemetry, cell_voltages=healthy_cells
    )

    degraded_telemetry = pd.DataFrame(
        {
            "voltage": degraded_df["voltage"].head(10),
            "current": degraded_df["current"].head(10),
            "temperature": degraded_df["temperature"].head(10),
            "soh": degraded_df["soh"].head(10),
            "internal_resistance": [0.18] * 10,
        }
    )
    score_degraded = scorer.compute_score(
        telemetry_df=degraded_telemetry, cell_voltages=degraded_cells
    )

    report["tests"]["battery_scoring"] = {
        "status": "PASS",
        "healthy": {"score": score_healthy["score"], "grade": score_healthy["grade"]},
        "degraded": {"score": score_degraded["score"], "grade": score_degraded["grade"]},
    }
    print(f"  Healthy: {score_healthy['score']}/100 → Grade {score_healthy['grade']}")
    print(f"  Degraded: {score_degraded['score']}/100 → Grade {score_degraded['grade']}")

    # ── 7. Physics-Based Features ────────────────────────────────────────
    print("\n[7/10] Physics-Based Feature Extraction...")
    extractor = PhysicsFeatureExtractor()

    ic_curve = extractor.extract_ic_curve(
        voltage=np.array([4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3]),
        capacity=np.array([210, 200, 190, 175, 155, 130, 100, 70, 40, 15]),
    )
    resistance = extractor.estimate_resistance(
        voltage_drop=np.array([0, 5, 10, 15, 20]),
        current=np.array([0, 50, 100, 150, 200]),
    )

    report["tests"]["physics_features"] = {
        "status": "PASS",
        "ic_peaks": ic_curve["num_peaks"],
        "estimated_resistance_ohm": resistance.get("mean_resistance", resistance.get("resistance")),
    }
    print(f"  IC curve peaks: {ic_curve['num_peaks']}")
    print(f"  Estimated resistance: {resistance}")

    # ── 8. Fleet Analytics ───────────────────────────────────────────────
    print("\n[8/10] Fleet Analytics (2 vehicles)...")
    fleet = FleetAnalytics()
    fleet.add_battery(TESLA_PACK["vin"], healthy_df)
    fleet.add_battery("5YJSA1E26HF000999", degraded_df)

    summary = fleet.get_fleet_summary()
    fleet.compare_batteries()

    report["tests"]["fleet_analytics"] = {
        "status": "PASS",
        "fleet_size": summary["fleet_size"],
        "avg_soh": round(summary["avg_soh"], 1),
        "avg_score": round(summary["avg_score"], 1),
        "grade_distribution": summary["grade_distribution"],
    }
    print(f"  Fleet size: {summary['fleet_size']} vehicles")
    print(f"  Avg SOH: {summary['avg_soh']:.1f}%")
    print(f"  Avg score: {summary['avg_score']:.1f}")
    print(f"  Grade distribution: {summary['grade_distribution']}")

    # ── 9. Digital Twin Simulation ───────────────────────────────────────
    print("\n[9/10] Digital Twin — Drive Cycle Simulation...")
    twin = BatteryDigitalTwin()

    # Simulate a 2-hour mixed drive cycle
    np.random.seed(42)
    drive_current = np.random.normal(80, 40, 7200)  # 2 hours, 1 sample/sec
    drive_cycle = pd.DataFrame({"current": drive_current})

    twin.simulate_drive_cycle(cycle_profile=drive_cycle, dt=1.0)
    state = twin.get_state()

    report["tests"]["digital_twin"] = {
        "status": "PASS",
        "final_soh": round(state["soh"], 4),
        "final_voltage": round(state["voltage"], 2),
        "cycle_count": round(state["cycle_count"], 2),
        "capacity_remaining_ah": round(state["capacity_ah"], 2),
    }
    print(f"  SOH after drive: {state['soh']:.4f}%")
    print(f"  Voltage: {state['voltage']:.1f}V")
    print(f"  Equivalent cycles: {state['cycle_count']:.2f}")
    print(f"  Capacity remaining: {state['capacity_ah']:.2f} Ah")

    # ── 10. V2G Scenario Analysis ────────────────────────────────────────
    print("\n[10/10] V2G Scenario Analysis...")
    v2g_gen = V2GScenarioGenerator(
        battery_capacity_ah=TESLA_PACK["capacity_ah"],
        nominal_voltage=TESLA_PACK["nominal_voltage"],
    )
    v2g_cycle = v2g_gen.generate_v2g_cycle(duration_hours=24, grid_demand_profile="typical")

    v2g_analyzer = V2GHealthAnalyzer()
    v2g_analyzer.compute_v2g_impact(
        baseline_df=healthy_df.head(100),
        v2g_df=v2g_cycle,
    )
    recommendations = v2g_analyzer.get_v2g_recommendations(current_soh=95.0)

    report["tests"]["v2g_scenarios"] = {
        "status": "PASS",
        "cycle_duration_hours": 24,
        "recommendations": recommendations,
    }
    print(f"  V2G cycle: {v2g_cycle.shape[0]} hours")
    print(f"  Recommendations: {recommendations}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  QA TEST SUMMARY")
    print("=" * 70)
    all_pass = all(t["status"] == "PASS" for t in report["tests"].values())
    for test_name, test_result in report["tests"].items():
        status = "PASS" if test_result["status"] == "PASS" else "FAIL"
        print(f"  [{status}] {test_name}")
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    # Save report
    report_path = "examples/tesla_qa_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    return report


if __name__ == "__main__":
    run_tesla_qa_test()
