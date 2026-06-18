"""Cross-field SOC/SOH validation tests."""

import warnings

import pytest

from ev_qa_framework.models import BatteryTelemetryModel


@pytest.mark.parametrize(
    "soc,soh",
    [
        (99.0, 12.0),
        (85.0, 25.0),
        (90.0, 29.0),
    ],
)
def test_soh_low_soc_high_raises(soc: float, soh: float):
    with pytest.raises(ValueError, match="SOH too low to hold high SOC"):
        BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=390.0,
            current=10.0,
            temperature=25.0,
            soc=soc,
            soh=soh,
        )


@pytest.mark.parametrize(
    "soc,soh",
    [
        (5.0, 85.0),
        (9.0, 90.0),
        (0.0, 80.1),
    ],
)
def test_soh_high_soc_low_warns(soc: float, soh: float):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=390.0,
            current=10.0,
            temperature=25.0,
            soc=soc,
            soh=soh,
        )
        assert m.soc == soc
        assert m.soh == soh
        assert len(w) >= 1
        assert "critically low charge" in str(w[0].message).lower()


def test_violation_example_from_task():
    with pytest.raises(ValueError):
        BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=390.0,
            current=10.0,
            temperature=25.0,
            soc=99.0,
            soh=12.0,
        )


def test_valid_mid_range_passes():
    telemetry = BatteryTelemetryModel(
        vin="1HGBH41JXMN109186",
        voltage=390.0,
        current=10.0,
        temperature=25.0,
        soc=50.0,
        soh=90.0,
    )
    assert telemetry.soc == 50.0
    assert telemetry.soh == 90.0


def test_valid_low_soh_low_soc_passes():
    BatteryTelemetryModel(
        vin="1HGBH41JXMN109186",
        voltage=390.0,
        current=10.0,
        temperature=25.0,
        soc=15.0,
        soh=20.0,
    )


def test_valid_high_soh_high_soc_passes():
    BatteryTelemetryModel(
        vin="1HGBH41JXMN109186",
        voltage=390.0,
        current=10.0,
        temperature=25.0,
        soc=75.0,
        soh=95.0,
    )
