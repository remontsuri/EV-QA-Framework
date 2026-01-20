#!/usr/bin/env python3
"""Quick start script for EV-QA-Framework"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_tests():
    """Run all tests"""
    print("🧪 Running tests...")
    subprocess.run([sys.executable, "-m", "pytest", "-v"])

def start_dashboard():
    """Start web dashboard"""
    print("🌐 Starting dashboard...")
    subprocess.run([sys.executable, "dashboard/app.py"])

def analyze_sample():
    """Analyze sample data"""
    print("📊 Analyzing sample data...")
    subprocess.run([
        sys.executable, "-m", "ev_qa_framework.cli",
        "--input", "examples/test_telemetry.csv",
        "--output", "results.json"
    ])

def main():
    parser = argparse.ArgumentParser(description='EV-QA-Framework Quick Start')
    parser.add_argument('action', choices=['test', 'dashboard', 'analyze'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'test':
        run_tests()
    elif args.action == 'dashboard':
        start_dashboard()
    elif args.action == 'analyze':
        analyze_sample()

if __name__ == '__main__':
    main()