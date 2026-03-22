"""Unit test for Tesla analysis script"""

import os
import json
from ev_qa_framework import framework

from scripts.test_tesla_battery import analyze_tesla_battery
import pathlib


def test_analyze_tesla_battery_returns_expected_report(tmp_path, monkeypatch):
    """Run the script function and inspect returned results and generated file."""
    # ensure working directory and data file are available
    report_path = tmp_path / "tesla_battery_report.json"
    # determine absolute path to csv
    repo_root = pathlib.Path(__file__).parents[1]
    csv_path = str(repo_root / 'examples' / 'tesla_model_s_defective.csv')

    # monkeypatch file write path to temp directory
    monkeypatch.chdir(tmp_path)

    results = analyze_tesla_battery(csv_path=csv_path)
    assert isinstance(results, dict)
    assert results['total_tests'] == 50
    assert results['passed'] + results['failed'] == results['total_tests']
    # check some failure conditions
    assert results['failed'] >= 0

    # verify report file exists and contains the same data
    assert report_path.exists()
    with open(report_path, 'r') as f:
        report = json.load(f)
    assert report['test_results']['total_tests'] == 50
    assert report['test_results']['failed'] == results['failed']
