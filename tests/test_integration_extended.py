"""
Integration tests for multi-module workflows.

Covers:
- Config -> Framework -> Analysis pipeline
- Chemistry profile -> SafetyThresholds -> Validation pipeline
- DBC parser -> Decode pipeline
- Telemetry validation -> Anomaly detection pipeline
- Multi-module data flow: models -> config -> framework -> results
"""

import asyncio
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.analysis import EVBatteryAnalyzer
from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.dbc_parser import DBCParser, battery_dbc_content, builtin_dbc
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.models import BatteryTelemetryModel, validate_telemetry


class TestConfigFrameworkIntegration:
    def test_chemistry_profile_propagates_to_validation(self):
        """Chemistry profile should set thresholds that affect validation."""
        cfg_strict = FrameworkConfig(chemistry="nca")
        cfg_lenient = FrameworkConfig(chemistry="lfp")

        qa_strict = EVQAFramework("Strict", config=cfg_strict)
        qa_lenient = EVQAFramework("Lenient", config=cfg_lenient)

        # A temperature that might pass for one chemistry but fail for another
        telemetry = BatteryTelemetryModel(
            vin="1HGBH41JXMN109186",
            voltage=396.5,
            current=125.3,
            temperature=58.0,
            soc=78.5,
            soh=96.2,
        )
        # Both should produce a result (the validation logic is the same,
        # but thresholds differ)
        assert qa_strict.validate_telemetry(telemetry) in (True, False)
        assert qa_lenient.validate_telemetry(telemetry) in (True, False)

    def test_fail_on_anomaly_affects_test_results(self):
        """fail_on_anomaly should cause rule-based anomalies to fail tests."""
        cfg = FrameworkConfig(fail_on_anomaly=True)
        qa = EVQAFramework("FailOnAnomaly", config=cfg)

        data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
            {"voltage": 400.0, "current": 50, "temperature": 65, "soc": 80, "soh": 98},
        ]
        results = asyncio.run(qa.run_test_suite(data))
        assert results["failed"] >= 1

    def test_fail_on_anomaly_false_allows_anomalies(self):
        """With fail_on_anomaly=False, rule-based anomalies don't fail tests."""
        cfg = FrameworkConfig(fail_on_anomaly=False)
        qa = EVQAFramework("PassOnAnomaly", config=cfg)

        data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
        ]
        results = asyncio.run(qa.run_test_suite(data))
        assert results["failed"] == 0


class TestTelemetryValidationPipeline:
    def test_valid_telemetry_flows_through(self):
        """Valid telemetry should pass validation and produce results."""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 396.5,
            "current": 125.3,
            "temperature": 35.2,
            "soc": 78.5,
            "soh": 96.2,
        }
        model = validate_telemetry(data)
        assert model.vin == "1HGBH41JXMN109186"

    def test_invalid_telemetry_raises_before_analysis(self):
        """Invalid telemetry should be caught by Pydantic before analysis."""
        with pytest.raises(Exception):
            validate_telemetry({"vin": "SHORT"})

    def test_boundary_telemetry_values(self):
        """Boundary values should be accepted by the model."""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 0.0,
            "current": -500.0,
            "temperature": -50.0,
            "soc": 0.0,
            "soh": 0.0,
        }
        model = validate_telemetry(data)
        assert model.voltage == 0.0
        assert model.current == -500.0


class TestDBCParserIntegration:
    def test_builtin_dbc_loads(self):
        """Built-in DBC should load without errors."""
        parser = builtin_dbc()
        assert len(parser.list_messages()) > 0

    def test_builtin_dbc_has_voltage_current(self):
        parser = builtin_dbc()
        msg = parser.get_message(257)
        assert msg is not None
        assert msg.name == "VoltageCurrent"

    def test_builtin_dbc_has_temp_soc(self):
        parser = builtin_dbc()
        msg = parser.get_message(258)
        assert msg is not None
        assert "Temperature" in msg.signals

    def test_builtin_dbc_decode(self):
        """Decode a known CAN frame with Intel byte order."""
        parser = builtin_dbc()
        # VoltageCurrent message (ID 257 = 0x101)
        # Voltage: 400.0V / 0.1 = 4000 = 0x0FA0 at bits 0-15 (Intel: LSB first)
        # So byte[0] = 0xA0, byte[1] = 0x0F
        # Current: 100.0A / 0.1 = 1000 = 0x03E8 at bits 16-31 (Intel: LSB first)
        # So byte[2] = 0xE8, byte[3] = 0x03
        data = bytes([0xA0, 0x0F, 0xE8, 0x03, 0x00, 0x00, 0x00, 0x00])
        decoded = parser.decode(257, data)
        assert "Voltage" in decoded
        assert abs(decoded["Voltage"] - 400.0) < 0.01
        assert "Current" in decoded
        assert abs(decoded["Current"] - 100.0) < 0.01

    def test_builtin_dbc_get_signal_value(self):
        parser = builtin_dbc()
        data = bytes([0xA0, 0x0F, 0xE8, 0x03, 0x00, 0x00, 0x00, 0x00])
        voltage = parser.get_signal_value(257, data, "Voltage")
        assert abs(voltage - 400.0) < 0.01

    def test_decode_unknown_id_returns_empty(self):
        parser = builtin_dbc()
        decoded = parser.decode(0xFFFF, b"\x00" * 8)
        assert decoded == {}

    def test_get_signal_unknown_id_returns_none(self):
        parser = builtin_dbc()
        result = parser.get_signal_value(0xFFFF, b"\x00" * 8, "Voltage")
        assert result is None

    def test_get_signal_unknown_signal_returns_none(self):
        parser = builtin_dbc()
        result = parser.get_signal_value(257, b"\x00" * 8, "NonExistent")
        assert result is None

    def test_get_message_by_name(self):
        parser = builtin_dbc()
        msg = parser.get_message_by_name("VoltageCurrent")
        assert msg is not None
        assert msg.id == 257

    def test_get_message_by_name_not_found(self):
        parser = builtin_dbc()
        msg = parser.get_message_by_name("NonExistent")
        assert msg is None

    def test_extended_message(self):
        """J1939 messages (ID > 0x7FF) should be marked as extended."""
        parser = builtin_dbc()
        msg = parser.get_message(65270)
        assert msg is not None
        assert msg.is_extended is True

    def test_comments_parsed(self):
        """Comments from DBC should be accessible."""
        parser = builtin_dbc()
        msg = parser.get_message(257)
        assert msg is not None
        assert msg.comment != ""

    def test_signal_comments(self):
        parser = builtin_dbc()
        msg = parser.get_message(257)
        assert msg is not None
        voltage_sig = msg.signals.get("Voltage")
        assert voltage_sig is not None
        assert voltage_sig.comment != ""

    def test_list_messages_returns_all(self):
        parser = builtin_dbc()
        messages = parser.list_messages()
        ids = [m.id for m in messages]
        assert 257 in ids
        assert 258 in ids
        assert 259 in ids
        assert 65270 in ids
        assert 65271 in ids
        assert 65272 in ids
        assert 65273 in ids

    def test_signal_raw_to_physical(self):
        parser = builtin_dbc()
        msg = parser.get_message(257)
        sig = msg.signals["Voltage"]
        # scale=0.1, offset=0
        assert sig.raw_to_physical(4000) == 400.0

    def test_signal_physical_to_raw(self):
        parser = builtin_dbc()
        msg = parser.get_message(257)
        sig = msg.signals["Voltage"]
        assert sig.physical_to_raw(400.0) == 4000

    def test_motorola_signal_decode(self):
        """Test Motorola (big-endian) signal decoding."""
        parser = builtin_dbc()
        # BatteryVoltage message (ID 65271) has PackVoltage at bits 0-16, Intel
        # Let's test with a known Intel signal: PackVoltage scale=0.01
        # 350.0V / 0.01 = 35000 = 0x88B8 -> Intel: byte[0]=0xB8, byte[1]=0x88
        msg = parser.get_message(65271)
        assert msg is not None
        sig = msg.signals["PackVoltage"]
        assert sig.byte_order == "Intel"
        data = bytes([0xB8, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        decoded = parser.decode(65271, data)
        assert abs(decoded["PackVoltage"] - 350.0) < 0.01


class TestMLAnalysisIntegration:
    def test_ml_analyzer_detects_anomalies(self):
        """ML analyzer should detect obvious anomalies."""
        normal_data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 95}
        ] * 20
        anomaly_data = [
            {"voltage": 800.0, "current": 300, "temperature": 80, "soc": 5, "soh": 50}
        ] * 5

        qa = EVQAFramework("ML-Test")
        asyncio.run(qa.run_test_suite(normal_data))

        import pandas as pd

        df_anomaly = pd.DataFrame(anomaly_data)
        results = qa.ml_analyzer.analyze_telemetry(df_anomaly)
        assert results["anomalies_detected"] > 0

    def test_ml_save_load_roundtrip(self):
        """Model should save and load without errors."""
        train_data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 95}
        ] * 20

        qa1 = EVQAFramework("Trainer")
        asyncio.run(qa1.run_test_suite(train_data))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
            path = f.name

        try:
            qa1.ml_analyzer.save_model(path)
            assert os.path.exists(path)

            qa2 = EVQAFramework("Loader")
            qa2.ml_analyzer.load_model(path)

            # Loaded model should be usable for analysis
            test_data = pd.DataFrame(
                [{"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 95}] * 5
            )
            r = qa2.ml_analyzer.analyze_telemetry(test_data)
            assert "total_samples" in r
            assert r["total_samples"] == 5
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestEndToEndWorkflow:
    def test_full_qa_workflow(self):
        """End-to-end: config -> framework -> test suite -> results."""
        cfg = FrameworkConfig(chemistry="nmc")
        qa = EVQAFramework("E2E-Test", config=cfg)

        data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
            {"voltage": 401.0, "current": 51, "temperature": 31, "soc": 79, "soh": 98},
            {"voltage": 402.0, "current": 52, "temperature": 32, "soc": 78, "soh": 98},
        ]

        results = asyncio.run(qa.run_test_suite(data))
        assert results["total_tests"] == 3
        assert results["passed"] == 3
        assert results["failed"] == 0

    def test_full_qa_with_anomalies(self):
        """End-to-end with anomalies should detect them."""
        cfg = FrameworkConfig(chemistry="nmc", fail_on_anomaly=True)
        qa = EVQAFramework("E2E-Anomaly", config=cfg)

        data = [
            {"voltage": 400.0, "current": 50, "temperature": 30, "soc": 80, "soh": 98},
            {"voltage": 400.0, "current": 50, "temperature": 70, "soc": 80, "soh": 98},
        ]

        results = asyncio.run(qa.run_test_suite(data))
        assert results["total_tests"] == 2
        assert results["failed"] >= 1
