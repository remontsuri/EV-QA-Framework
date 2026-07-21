"""
Edge case tests for the HIL (Hardware-in-the-Loop) module.

Covers:
- CANMessage creation and conversion
- HILTestResult to_dict
- HILInterface in simulation mode
- BMSHardwareEmulator message generation
- HILTestRunner test execution
- HILTestRunner compare_expected_vs_actual
- HILTestRunner generate_hil_report
"""

import pandas as pd
import pytest

from ev_qa_framework.config import FrameworkConfig
from ev_qa_framework.hil import (
    BMSHardwareEmulator,
    CANMessage,
    HILInterface,
    HILTestResult,
    HILTestRunner,
)


class TestCANMessage:
    def test_default_creation(self):
        msg = CANMessage(arbitration_id=0x100, data=b"\x01\x02\x03\x04")
        assert msg.arbitration_id == 0x100
        assert msg.data == b"\x01\x02\x03\x04"
        assert msg.timestamp == 0.0
        assert msg.is_extended is False
        assert msg.dlc == 8

    def test_custom_creation(self):
        msg = CANMessage(
            arbitration_id=0x1FFFFFFF,
            data=b"\x01\x02",
            timestamp=1234.5,
            is_extended=True,
            dlc=2,
        )
        assert msg.is_extended is True
        assert msg.dlc == 2
        assert msg.timestamp == 1234.5

    def test_from_can_msg_without_timestamp(self):
        """from_can_msg should handle messages without timestamp attribute."""
        mock_msg = type(
            "MockMsg",
            (),
            {
                "arbitration_id": 0x200,
                "data": [1, 2, 3, 4],
                "is_extended_id": True,
                "dlc": 4,
            },
        )()
        msg = CANMessage.from_can_msg(mock_msg)
        assert msg.arbitration_id == 0x200
        assert msg.data == bytes([1, 2, 3, 4])
        assert msg.is_extended is True

    def test_from_can_msg_with_timestamp(self):
        mock_msg = type(
            "MockMsg",
            (),
            {
                "arbitration_id": 0x200,
                "data": [1, 2, 3, 4],
                "timestamp": 999.9,
                "is_extended_id": False,
                "dlc": 4,
            },
        )()
        msg = CANMessage.from_can_msg(mock_msg)
        assert msg.timestamp == 999.9

    def test_to_can_msg_with_python_can(self):
        """to_can_msg should work when python-can is installed."""
        msg = CANMessage(arbitration_id=0x100, data=b"\x01\x02\x03\x04")
        # In test environment, python-can IS installed (HAS_CAN=True)
        result = msg.to_can_msg()
        assert result is not None
        assert result.arbitration_id == 0x100


class TestHILTestResult:
    def test_default_creation(self):
        r = HILTestResult(test_name="test1", passed=True, duration_s=1.5)
        assert r.test_name == "test1"
        assert r.passed is True
        assert r.duration_s == 1.5
        assert r.messages_sent == 0
        assert r.messages_received == 0
        assert r.errors == []
        assert r.warnings == []
        assert r.data == {}

    def test_to_dict(self):
        r = HILTestResult(
            test_name="test1",
            passed=False,
            duration_s=2.0,
            messages_sent=10,
            messages_received=8,
            errors=["err1"],
            warnings=["warn1"],
            data={"key": "val"},
        )
        d = r.to_dict()
        assert d["test_name"] == "test1"
        assert d["passed"] is False
        assert d["duration_s"] == 2.0
        assert d["messages_sent"] == 10
        assert d["messages_received"] == 8
        assert d["errors"] == ["err1"]
        assert d["warnings"] == ["warn1"]
        assert d["data"] == {"key": "val"}


class TestHILInterfaceSimulation:
    def test_simulation_mode_default(self):
        hil = HILInterface(simulation=True)
        assert hil.simulation is True

    def test_send_and_receive(self):
        hil = HILInterface(simulation=True)
        msg = CANMessage(arbitration_id=0x100, data=b"\x01\x02\x03\x04")
        hil.send(msg)
        received = hil.receive()
        assert received is not None
        assert received.arbitration_id == 0x100
        assert received.data == b"\x01\x02\x03\x04"

    def test_receive_empty_returns_none(self):
        hil = HILInterface(simulation=True)
        assert hil.receive() is None

    def test_send_telemetry(self):
        hil = HILInterface(simulation=True)
        hil.send_telemetry(voltage=400.0, current=50.0, temperature=30.0, soc=80.0)
        msg = hil.receive()
        assert msg is not None
        assert msg.arbitration_id == 0x100
        assert len(msg.data) == 8

    def test_send_telemetry_custom_id(self):
        hil = HILInterface(simulation=True)
        hil.send_telemetry(voltage=400.0, current=50.0, temperature=30.0, soc=80.0, msg_id=0x200)
        msg = hil.receive()
        assert msg.arbitration_id == 0x200

    def test_fifo_order(self):
        """Messages should be received in FIFO order."""
        hil = HILInterface(simulation=True)
        for i in range(5):
            hil.send(CANMessage(arbitration_id=0x100 + i, data=bytes([i])))
        for i in range(5):
            msg = hil.receive()
            assert msg.arbitration_id == 0x100 + i
            assert msg.data == bytes([i])
        assert hil.receive() is None

    def test_context_manager(self):
        hil = HILInterface(simulation=True)
        with hil as h:
            assert h is hil
        # After context exit, bus should be None (closed)

    def test_close_idempotent(self):
        """Closing multiple times should not raise."""
        hil = HILInterface(simulation=True)
        hil.close()
        hil.close()  # Should not raise


class TestBMSHardwareEmulator:
    def test_generate_telemetry_message(self):
        emu = BMSHardwareEmulator()
        msg = emu.generate_telemetry_message()
        assert msg.arbitration_id == 0x100
        assert len(msg.data) == 8

    def test_generate_telemetry_message_custom_id(self):
        emu = BMSHardwareEmulator()
        msg = emu.generate_telemetry_message(msg_id=0x300)
        assert msg.arbitration_id == 0x300

    def test_generate_cycle(self):
        emu = BMSHardwareEmulator()
        messages = emu.generate_cycle(n_messages=50)
        assert len(messages) == 50
        for msg in messages:
            assert len(msg.data) == 8

    def test_generate_cycle_default(self):
        emu = BMSHardwareEmulator()
        messages = emu.generate_cycle()
        assert len(messages) == 100

    def test_generate_cycle_single(self):
        emu = BMSHardwareEmulator()
        messages = emu.generate_cycle(n_messages=1)
        assert len(messages) == 1

    def test_with_custom_config(self):
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 50.0
        emu = BMSHardwareEmulator(config=config)
        msg = emu.generate_telemetry_message()
        assert msg is not None


class TestHILTestRunner:
    def test_run_hil_test(self):
        runner = HILTestRunner(simulation=True)
        profile = {"name": "test_battery_qa"}
        result = runner.run_hil_test(profile, duration=0.5)
        assert result.test_name == "test_battery_qa"
        assert result.passed is True
        assert result.duration_s >= 0
        assert result.messages_sent > 0

    def test_run_hil_test_zero_duration(self):
        runner = HILTestRunner(simulation=True)
        profile = {"name": "zero_duration_test"}
        result = runner.run_hil_test(profile, duration=0.0)
        assert result.test_name == "zero_duration_test"

    def test_run_hil_test_default_name(self):
        runner = HILTestRunner(simulation=True)
        result = runner.run_hil_test({})
        assert result.test_name == "unnamed_test"

    def test_compare_expected_vs_actual(self):
        runner = HILTestRunner(simulation=True)
        expected = pd.DataFrame({"voltage": [400.0, 401.0, 402.0]})
        actual = pd.DataFrame({"voltage": [400.5, 401.5, 402.5]})
        diff = runner.compare_expected_vs_actual(expected, actual)
        assert "voltage" in diff
        assert diff["voltage"]["mae"] == pytest.approx(0.5)
        assert diff["voltage"]["max_error"] == pytest.approx(0.5)

    def test_compare_different_columns(self):
        runner = HILTestRunner(simulation=True)
        expected = pd.DataFrame({"voltage": [400.0], "current": [50.0]})
        actual = pd.DataFrame({"voltage": [400.5]})
        diff = runner.compare_expected_vs_actual(expected, actual)
        assert "voltage" in diff
        assert "current" not in diff

    def test_compare_different_lengths(self):
        runner = HILTestRunner(simulation=True)
        expected = pd.DataFrame({"voltage": [400.0, 401.0, 402.0]})
        actual = pd.DataFrame({"voltage": [400.5, 401.5]})
        diff = runner.compare_expected_vs_actual(expected, actual)
        assert "voltage" in diff
        # Should compare only first 2
        assert diff["voltage"]["mae"] == pytest.approx(0.5)

    def test_compare_empty_dataframe(self):
        runner = HILTestRunner(simulation=True)
        expected = pd.DataFrame({"voltage": []})
        actual = pd.DataFrame({"voltage": []})
        diff = runner.compare_expected_vs_actual(expected, actual)
        assert diff == {}

    def test_generate_hil_report(self):
        runner = HILTestRunner(simulation=True)
        results = [
            HILTestResult("test1", True, 1.0, messages_sent=10, messages_received=10),
            HILTestResult("test2", True, 2.0, messages_sent=20, messages_received=18),
            HILTestResult("test3", False, 0.5, messages_sent=5, messages_received=0),
        ]
        report = runner.generate_hil_report(results)
        assert report["total_tests"] == 3
        assert report["passed"] == 2
        assert report["failed"] == 1
        assert report["pass_rate"] == pytest.approx(2 / 3)
        assert report["total_duration_s"] == pytest.approx(3.5)
        assert report["total_messages_sent"] == 35
        assert report["total_messages_received"] == 28
        assert len(report["results"]) == 3

    def test_generate_hil_report_empty(self):
        runner = HILTestRunner(simulation=True)
        report = runner.generate_hil_report([])
        assert report["total_tests"] == 0
        assert report["pass_rate"] == 0
