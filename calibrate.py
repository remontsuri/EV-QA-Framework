"""
Calibration script — tune thresholds from battery telemetry data.

Usage:
  python calibrate.py simulate --model "Model 3" --output data.csv
  python calibrate.py analyze --input data.csv --output recommendations.json

For real CAN data collection, use the BMS adapters:
  from ev_qa_framework.bms_adapters import TeslaBMSAdapter
  adapter = TeslaBMSAdapter(channel="vcan0")
  adapter.connect()
  telemetry = adapter.read_telemetry()
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def simulate_data(model: str, n_readings: int = 500, output: str = "simulated.csv"):
    """Generate realistic synthetic data for calibration testing."""
    profiles = {
        "Model 2": {"vnom": 3.2, "vstd": 0.03, "tnom": 32, "tstd": 2.0, "inom": 50, "istd": 5},
        "Model 3": {"vnom": 3.6, "vstd": 0.04, "tnom": 33, "tstd": 2.5, "inom": 55, "istd": 6},
        "Model S/X": {"vnom": 3.7, "vstd": 0.05, "tnom": 34, "tstd": 3.0, "inom": 60, "istd": 7},
    }
    p = profiles.get(model, profiles["Model 3"])
    rng = np.random.default_rng(42)

    data = []
    soc = 80.0
    for i in range(n_readings):
        temp = p["tnom"] + rng.normal(0, p["tstd"])
        volt = p["vnom"] + rng.normal(0, p["vstd"])
        curr = p["inom"] + rng.normal(0, p["istd"])
        soc = max(10, min(100, soc - 0.1 + rng.normal(0, 0.5)))
        data.append({"voltage": round(volt, 3), "current": round(curr, 2),
                     "temperature": round(temp, 2), "soc": round(soc, 1)})

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Generated {n_readings} readings -> {output}")
    return df


def analyze_thresholds(csv_path: str, output_path: str = "calibration_recommendations.json"):
    """Analyze data and recommend calibration thresholds."""
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} readings from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Basic statistics
    print(f"\n--- Voltage ---")
    print(f"  Mean: {df['voltage'].mean():.3f} V")
    print(f"  Std:  {df['voltage'].std():.3f} V")
    print(f"  Min:  {df['voltage'].min():.3f} V")
    print(f"  Max:  {df['voltage'].max():.3f} V")
    print(f"  Range: {df['voltage'].max() - df['voltage'].min():.3f} V")

    print(f"\n--- Temperature ---")
    print(f"  Mean: {df['temperature'].mean():.1f} °C")
    print(f"  Std:  {df['temperature'].std():.2f} °C")
    print(f"  Min:  {df['temperature'].min():.1f} °C")
    print(f"  Max:  {df['temperature'].max():.1f} °C")
    print(f"  P95:  {df['temperature'].quantile(0.95):.1f} °C")
    print(f"  P99:  {df['temperature'].quantile(0.99):.1f} °C")

    print(f"\n--- Current ---")
    print(f"  Mean: {df['current'].mean():.1f} A")
    print(f"  Std:  {df['current'].std():.1f} A")

    # Temperature jump analysis
    temp_diffs = df["temperature"].diff().dropna().abs()
    print(f"\n--- Temperature Jumps ---")
    print(f"  Mean jump: {temp_diffs.mean():.2f} °C")
    print(f"  Max jump:  {temp_diffs.max():.2f} °C")
    print(f"  P95 jump:  {temp_diffs.quantile(0.95):.2f} °C")
    print(f"  P99 jump:  {temp_diffs.quantile(0.99):.2f} °C")

    # Recommended thresholds (3-sigma rule)
    t_mean = df["temperature"].mean()
    t_std = df["temperature"].std()
    v_mean = df["voltage"].mean()
    v_std = df["voltage"].std()

    recommended = {
        "max_temperature": round(t_mean + 3 * t_std, 1),
        "min_temperature": round(t_mean - 3 * t_std, 1),
        "max_temperature_jump": round(max(temp_diffs.quantile(0.99), 5.0), 1),
        "max_voltage": round(v_mean + 3 * v_std, 2),
        "min_voltage": round(v_mean - 3 * v_std, 2),
        "anomaly_contamination": 0.05,
        "thermal_critical_temp": round(t_mean + 5 * t_std, 1),
    }

    print(f"\n{'='*50}")
    print(f"  RECOMMENDED THRESHOLDS (3-sigma)")
    print(f"{'='*50}")
    for k, v in recommended.items():
        print(f"  {k:30s} = {v}")

    print(f"\n  Apply to FrameworkConfig:")
    print(f"  from ev_qa_framework import FrameworkConfig, SafetyThresholds")
    print(f"  config = FrameworkConfig(safety_thresholds=SafetyThresholds(")
    for k, v in recommended.items():
        if "voltage" in k or "temperature" in k:
            print(f"      {k}={v},")
    print(f"  ))")

    # Save recommendations
    with open(output_path, "w") as f:
        json.dump({
            "source": csv_path,
            "n_readings": len(df),
            "statistics": {
                "voltage": {"mean": round(df["voltage"].mean(), 3), "std": round(df["voltage"].std(), 3)},
                "temperature": {"mean": round(df["temperature"].mean(), 1), "std": round(df["temperature"].std(), 2)},
                "current": {"mean": round(df["current"].mean(), 1), "std": round(df["current"].std(), 1)},
                "temp_jumps": {"mean": round(temp_diffs.mean(), 2), "max": round(temp_diffs.max(), 2), "p99": round(temp_diffs.quantile(0.99), 2)},
            },
            "recommended": recommended,
        }, f, indent=2)
    print(f"\n  Saved: calibration_recommendations.json")


def main():
    parser = argparse.ArgumentParser(description="EV-QA-Framework Calibration Tool")
    sub = parser.add_subparsers(dest="command")

    # simulate
    sim = sub.add_parser("simulate", help="Generate synthetic data")
    sim.add_argument("--model", default="Model 3", choices=["Model 2", "Model 3", "Model S/X"])
    sim.add_argument("--readings", type=int, default=500)
    sim.add_argument("--output", "-o", default="simulated.csv")

    # analyze
    ana = sub.add_parser("analyze", help="Analyze data and recommend thresholds")
    ana.add_argument("--input", "-i", required=True)
    ana.add_argument("--output", "-o", default="calibration_recommendations.json")

    args = parser.parse_args()
    if args.command == "simulate":
        simulate_data(args.model, args.readings, args.output)
    elif args.command == "analyze":
        analyze_thresholds(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
