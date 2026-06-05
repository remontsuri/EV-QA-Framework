#!/usr/bin/env python3
"""CLI interface for EV-QA-Framework"""

import argparse
import json
import time

import pandas as pd

from .analysis import EVBatteryAnalyzer
from .can_bus import CANBatterySimulator, CANTelemetryReceiver, DBCFileSimulator
from .soh_predictor import SOHPredictor


def analyze_csv(file_path: str, output: str = None):
    """Analyze telemetry from CSV file"""
    df = pd.read_csv(file_path)

    # Rename 'temperature' to 'temp' if needed
    if 'temperature' in df.columns and 'temp' not in df.columns:
        df = df.rename(columns={'temperature': 'temp'})

    # Initialize framework with ML
    analyzer = EVBatteryAnalyzer()

    # Run ML analysis
    results = analyzer.analyze_telemetry(df)

    print("✅ Analysis complete:")
    print(f"   Total samples: {results['total_samples']}")
    print(f"   Anomalies: {results['anomalies_detected']}")
    print(f"   Severity: {results['severity']}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to {output}")


def run_can_demo(duration: int = 10):
    """Run a live CAN bus emulation demo"""
    print(f"Starting CAN Bus Emulation Demo ({duration}s)...")
    sim = CANBatterySimulator()
    receiver = CANTelemetryReceiver()

    sim.start()
    receiver.start()

    try:
        for _ in range(duration):
            data = receiver.get_telemetry()
            v_val = data["voltage"]
            i_val = data["current"]
            t_val = data["temperature"]
            s_val = data["soc"]
            print(f"CAN Telemetry: V={v_val:.1f}V | "
                  f"I={i_val:.1f}A | "
                  f"T={t_val:.0f}C | "
                  f"SOC={s_val:.0f}%")
            time.sleep(1)
    finally:
        sim.stop()
        receiver.stop()
    print("CAN Demo finished.")


def run_dbc_emulate(dbc_path: str = None, duration: int = 10):
    """Run DBC-driven CAN emulation."""
    if dbc_path:
        print(f"Loading DBC: {dbc_path}")
    else:
        print("Using built-in battery DBC")
    sim = DBCFileSimulator(dbc_path=dbc_path)
    sim.start()

    try:
        for _ in range(duration):
            print(f"Sent {len(sim.dbc.messages)} messages")
            time.sleep(1)
    finally:
        sim.stop()
    print("DBC Emulation finished.")


def train_soh_model(csv_path: str, model_path: str):
    """Train SOH LSTM model from historical CSV data"""
    print(f"🧠 Training SOH Predictor from {csv_path}...")
    df = pd.read_csv(csv_path)
    predictor = SOHPredictor()
    predictor.train(df, epochs=5)
    predictor.save(model_path)
    print(f"✅ Model saved to {model_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="EV Battery QA Analysis tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    ana_p = subparsers.add_parser("analyze", help="Analyze CSV telemetry")
    ana_p.add_argument("--input", "-i", required=True, help="Input CSV file")
    ana_p.add_argument("--output", "-o", help="Output JSON file")

    # CAN command
    can_p = subparsers.add_parser("can-demo", help="Run CAN emulation demo")
    can_p.add_argument("--duration", "-d", type=int, default=10,
                       help="Demo duration in seconds")

    # DBC emulate command
    dbc_p = subparsers.add_parser("emulate", help="Run DBC-driven CAN emulation")
    dbc_p.add_argument("--dbc", help="Path to .dbc file (default: built-in)")
    dbc_p.add_argument("--duration", "-d", type=int, default=10,
                       help="Duration in seconds")

    # Train command
    train_p = subparsers.add_parser("train-soh", help="Train SOH model")
    train_p.add_argument("--input", "-i", required=True, help="Historical CSV")
    train_p.add_argument("--model-dir", "-m", required=True,
                         help="Directory to save model")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Start web dashboard")

    args = parser.parse_args()

    if args.command == "dashboard":
        print("🌐 Starting dashboard...")
        import uvicorn  # pylint: disable=C0415

        from dashboard.app import app as dash_app  # pylint: disable=C0415
        uvicorn.run(dash_app, host="0.0.0.0", port=8000)
    elif args.command == "analyze":
        analyze_csv(args.input, args.output)
    elif args.command == "can-demo":
        run_can_demo(args.duration)
    elif args.command == "emulate":
        run_dbc_emulate(dbc_path=args.dbc, duration=args.duration)
    elif args.command == "train-soh":
        train_soh_model(args.input, args.model_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
