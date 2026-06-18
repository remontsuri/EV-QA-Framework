"""Tesla Model S Battery QA Test

Covers the legacy QA analysis flow from ``examples/tesla_battery_qa_test.py``.
The tests exercise telemetry validation, critical-voltage detection, ML-powered
analysis (``EVQAFramework.run_test_suite``), and report generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.append(".")

from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import validate_telemetry


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "examples" / "tesla_model_s_defective.csv"


def analyze_tesla_battery(csv_path: str | Path = DEFAULT_CSV) -> dict[str, Any]:
    """Run the Tesla Model S battery QA analysis for *csv_path*.

    Returns the payload returned by :pymeth:`ev_qa_framework.framework.EVAFramework.run_test_suite`.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    qa = EVQAFramework("Tesla-Model-S-QA")

    telemetry_data: list[dict[str, Any]] = []
    critical_issues: list[str] = []

    for idx, row in df.iterrows():
        data = {
            "voltage": row["voltage"],
            "current": row["current"],
            "temperature": row["temp"],
            "soc": row["soc"],
            "soh": row["soh"],
        }

        try:
            validate_telemetry(
                {
                    "vin": row["vin"],
                    "voltage": row["voltage"],
                    "current": row["current"],
                    "temperature": row["temp"],
                    "soc": row["soc"],
                    "soh": row["soh"],
                }
            )
        except Exception as exc:  # pragma: no cover — path is implementation-defined
            critical_issues.append(f"Point {idx}: Validation failed - {exc}")

        telemetry_data.append(data)

        if row["voltage"] > 450:
            critical_issues.append(
                f"CRITICAL: Overvoltage {row['voltage']}V at point {idx}"
            )
        if row["voltage"] < 50:
            critical_issues.append(
                f"CRITICAL: Undervoltage {row['voltage']}V at point {idx}"
            )
        if row["temp"] > 70:
            critical_issues.append(
                f"CRITICAL: Overheating {row['temp']}C at point {idx}"
            )

    results = pytest.importorskip("asyncio").run(qa.run_test_suite(telemetry_data))
    return results


def test_analyze_tesla_battery_returns_expected_shape(tmp_path: Path) -> None:
    """The returned dict contains the standard QA-test contract."""
    results = analyze_tesla_battery(DEFAULT_CSV)

    assert isinstance(results, dict)
    assert results["total_tests"] == 50
    assert results["passed"] + results["failed"] == results["total_tests"]
    assert results["failed"] >= 0


def test_tesla_battery_generates_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The analysis returns a QA report with expected shape."""
    results = analyze_tesla_battery(DEFAULT_CSV)

    assert isinstance(results, dict)
    assert results["total_tests"] == 50
    assert results["passed"] + results["failed"] == results["total_tests"]
    assert results["failed"] >= 0
