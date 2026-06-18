"""Tests for "Tesla Potesti" test cases."""

import pytest

from ev_qa_framework.config import get_tesla_config
from ev_qa_framework.framework import EVQAFramework


def test_tesla_configuration_valid():
    """Tesla config is created with correct NCA parameters."""
    # NCA 108s: 4.2 * 108 = 453.6 V (chemically correct vs old hardcoded 450.0)
    assert get_tesla_config().safety_thresholds.max_voltage == pytest.approx(453.6)
    assert get_tesla_config().fail_on_anomaly is True
    assert len(get_tesla_config().default_vin) == 17


def test_tesla_normal_and_spike():
    """Normal cycle with temperature spike."""
    qa = EVQAFramework(name="Tesla-Potesti-QA", config=get_tesla_config())
    normal = {"voltage": 360.0, "current": 120.0, "temperature": 35.0, "soc": 80.0, "soh": 95.0}
    spike = {"voltage": 400.0, "current": 200.0, "temperature": 65.0, "soc": 60.0, "soh": 90.0}
    results = qa.run_test_suite([normal, spike])
    # first  should ,
    assert results["passed"] == 1
    assert results["failed"] == 1
    assert any("Temperature" in msg for msg in results["anomalies"])


def test_tesla_vin_format():
    """VIN format should pass validation."""
    qa = EVQAFramework(config=get_tesla_config())
    data = {"voltage": 350.0, "current": 50.0, "temperature": 25.0, "soc": 90.0, "soh": 98.0}
    result = qa.run_test_suite([data])
    assert result["passed"] == 1
