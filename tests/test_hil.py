"""Tests for HIL interface."""

import pandas as pd
import pytest

from ev_qa_framework.hil import (
    BMSHardwareEmulator,
    CANMessage,
    HILInterface,
    HILTestResult,
    HILTestRunner,
)


class TestCANMessage:
    def test_creation(self):
        msg = CANMessage(arbitration_id=0x100, data=bytes([1, 2, 3, 4]))
        assert msg.arbitration_id == 0x100
        assert msg.data == bytes([1, 2, 3, 4])
        assert msg.dlc == 8

    def test_default_values(self):
        msg = CANMessage(arbitration_id=0x200, data=bytes([0] * 8))
        assert msg.is_extended is False
        assert msg.timestamp == 0.0


class TestHILInterface:
    def test_simulation_mode(self):
        hil = HILInterface(simulation=True)
        assert hil.simulation is True

    def test_send_simulation(self):
        hil = HILInterface(simulation=True)
        msg = CANMessage(arbitration_id=0x100, data=bytes([1, 2, 3, 4]))
        hil.send(msg)
        assert len(hil._sim_messages) == 1

    def test_receive_simulation(self):
        hil = HILInterface(simulation=True)
        msg = CANMessage(arbitration_id=0x100, data=bytes([1, 2, 3, 4]))
        hil.send(msg)
        received = hil.receive()
        assert received is not None
        assert received.arbitration_id == 0x100

    def test_receive_empty(self):
        hil = HILInterface(simulation=True)
        assert hil.receive() is None

    def test_send_telemetry(self):
        hil = HILInterface(simulation=True)
        hil.send_telemetry(voltage=400.0, current=50.0, temperature=30.0, soc=80.0)
        assert len(hil._sim_messages) == 1
        msg = hil._sim_messages[0]
        assert msg.arbitration_id == 0x100
        assert len(msg.data) == 8

    def test_context_manager(self):
        with HILInterface(simulation=True) as hil:
            assert hil.simulation is True


class TestBMSHardwareEmulator:
    def test_creation(self):
        emu = BMSHardwareEmulator()
        assert emu.config is not None

    def test_generate_telemetry_message(self):
        emu = BMSHardwareEmulator()
        msg = emu.generate_telemetry_message()
        assert isinstance(msg, CANMessage)
        assert len(msg.data) == 8

    def test_generate_cycle(self):
        emu = BMSHardwareEmulator()
        messages = emu.generate_cycle(50)
        assert len(messages) == 50
        assert all(isinstance(m, CANMessage) for m in messages)

    def test_custom_msg_id(self):
        emu = BMSHardwareEmulator()
        msg = emu.generate_telemetry_message(msg_id=0x200)
        assert msg.arbitration_id == 0x200


class TestHILTestRunner:
    def test_creation(self):
        runner = HILTestRunner(simulation=True)
        assert runner.hil.simulation is True

    def test_run_hil_test(self):
        runner = HILTestRunner(simulation=True)
        profile = {"name": "test_telemetry"}
        result = runner.run_hil_test(profile, duration=0.5)
        assert isinstance(result, HILTestResult)
        assert result.test_name == "test_telemetry"
        assert result.passed is True
        assert result.messages_sent > 0

    def test_compare_expected_vs_actual(self):
        runner = HILTestRunner(simulation=True)
        expected = pd.DataFrame(
            {
                "voltage": [400.0, 401.0, 402.0],
                "current": [50.0, 51.0, 52.0],
            }
        )
        actual = pd.DataFrame(
            {
                "voltage": [400.5, 401.5, 402.5],
                "current": [50.5, 51.5, 52.5],
            }
        )
        diff = runner.compare_expected_vs_actual(expected, actual)
        assert "voltage" in diff
        assert "current" in diff
        assert diff["voltage"]["mae"] == pytest.approx(0.5, abs=0.01)

    def test_generate_hil_report(self):
        runner = HILTestRunner(simulation=True)
        results = [
            HILTestResult(test_name="t1", passed=True, duration_s=1.0, messages_sent=10),
            HILTestResult(
                test_name="t2", passed=False, duration_s=2.0, messages_sent=20, errors=["err"]
            ),
        ]
        report = runner.generate_hil_report(results)
        assert report["total_tests"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1
        assert report["pass_rate"] == 0.5


class TestHILTestResult:
    def test_to_dict(self):
        result = HILTestResult(
            test_name="test",
            passed=True,
            duration_s=1.5,
            messages_sent=100,
            messages_received=95,
        )
        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["passed"] is True
        assert d["messages_sent"] == 100
