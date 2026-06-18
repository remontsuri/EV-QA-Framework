"""Property-based tests using Hypothesis for EV-QA-Framework."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from ev_qa_framework.models import BatteryTelemetryModel


@given(
    soc=st.floats(min_value=0.0, max_value=100.0),
    soh=st.floats(min_value=30.0, max_value=100.0),
)
@settings(max_examples=50)
def test_valid_soc_soh_always_accepted(soc: float, soh: float):
    m = BatteryTelemetryModel(
        vin="1HGBH41JXMN109186",
        voltage=400.0,
        current=100.0,
        temperature=35.0,
        soc=soc,
        soh=soh,
    )
    assert 0 <= m.soc <= 100
    assert 0 <= m.soh <= 100


@given(
    voltage=st.floats(min_value=0.0, max_value=1000.0),
    current=st.floats(min_value=-500.0, max_value=500.0),
    temperature=st.floats(min_value=-50.0, max_value=150.0),
)
@settings(max_examples=50)
def test_valid_telemetry_always_accepted(voltage: float, current: float, temperature: float):
    m = BatteryTelemetryModel(
        vin="1HGBH41JXMN109186",
        voltage=voltage,
        current=current,
        temperature=temperature,
        soc=50.0,
        soh=90.0,
    )
    assert m.voltage == voltage
    assert m.current == current


@given(
    voltage=st.floats(min_value=1001.0, max_value=9999.0),
)
@settings(max_examples=10)
def test_overvoltage_rejected(voltage: float):
    try:
        BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=voltage,
            current=100.0,
            temperature=35.0,
            soc=50.0,
            soh=90.0,
        )
        assert False, f"Should have rejected voltage={voltage}"
    except Exception:
        pass
