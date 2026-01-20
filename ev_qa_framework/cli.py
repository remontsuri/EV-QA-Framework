#!/usr/bin/env python3
"""CLI interface for EV-QA-Framework"""

import argparse
import json
import pandas as pd
from pathlib import Path
from .framework import EVQAFramework
from .analysis import EVBatteryAnalyzer

def analyze_csv(file_path: str, output: str = None):
    """Analyze telemetry from CSV file"""
    df = pd.read_csv(file_path)
    
    # Initialize framework with ML
    qa = EVQAFramework("CLI-Analyzer")
    analyzer = EVBatteryAnalyzer()
    
    # Run ML analysis
    results = analyzer.analyze_telemetry(df)
    
    print(f"✅ Analysis complete:")
    print(f"   Total samples: {results['total_samples']}")
    print(f"   Anomalies: {results['anomalies_detected']}")
    print(f"   Severity: {results['severity']}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to {output}")

def main():
    parser = argparse.ArgumentParser(description='EV Battery QA Analysis Tool')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with telemetry')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    
    args = parser.parse_args()
    
    if args.dashboard:
        print("🌐 Starting dashboard...")
        import subprocess
        subprocess.run(['python', 'dashboard/app.py'])
    else:
        analyze_csv(args.input, args.output)

if __name__ == '__main__':
    main()