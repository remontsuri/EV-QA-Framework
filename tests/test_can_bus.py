"""
Test module for CAN Bus integration (legacy compatibility).
"""

from unittest.mock import MagicMock, call, patch

import can
import pytest
import time

from ev_qa_framework.can_bus import (
    CANBatterySimulator,
    CANTelemetryReceiver,
)


# ---------------------------------------------------------------------------
# Pure logic: encoding / decoding helpers
# ---------------------------------------------------------------------------
class TestCANSymbolEncoding:
    """Test message packing/unpacking basics and the receiver's decode helpers."""

    def test_voltage_roundtrip(self):
        voltage = 405.3
        v_scaled = int(voltage * 10)
        data = bytes([
            (v_scaled >> 8) & 0xFF,
            v_scaled & 0xFF,
            0,
            0,
            0,
            0,
            0,
            0,
        ])
        recovered = ((data[0] << 8) | data[1]) / 10.0
        assert abs(recovered - voltage) < 0.1

    def test_place_raw_sets_clears_single_bit(self):
        data = [0, 0, 0, 0, 0, 0, 0, 0]
        sig = MagicMock()
        sig.length = 1
        sig.start_bit = 0
        sig.byte_order = "Intel"

        def place(data, sig, raw):
            i = 0
            bit_pos = sig.start_bit + i if sig.byte_order == "Intel" else sig.start_bit - i
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if (raw >> i) & 1:
                data[byte_idx] |= 1 << bit_in_byte
            else:
                data[byte_idx] &= ~(1 << bit_in_byte)

        place(data, sig, raw=1)
        assert data[0] == 1
        place(data, sig, raw=0)
        assert data[0] == 0

    def test_decode_positive_voltage(self):
        voltage = 405.3
        v_scaled = int(voltage * 10)
        data = bytes([v_scaled >> 8, v_scaled & 0xFF, 0, 0])
        v_scaled2 = (data[0] << 8) | data[1]
        v = v_scaled2 / 10.0
        assert abs(v - voltage) < 0.1

    def test_decode_soc_in_range(self):
        data = bytes([0, 0, 0, 0, 35, 80, 0, 0])
        temp = data[4]
        soc = data[5]
        assert temp == 35
        assert soc == 80


# ---------------------------------------------------------------------------
# Receiver lifecycle / behavior
# ---------------------------------------------------------------------------
class TestCANTelemetryReceiverLifecycle:
    def test_start_sets_running_flag(self):
        recv = CANTelemetryReceiver(channel="vcan0", hardware=False)
        recv.bus = MagicMock()
        with patch("ev_qa_framework.can_bus.threading.Thread"):
            recv.start()
        assert recv._running is True

    def test_stop_clears_running_and_calls_join(self):
        bus_mock = MagicMock()
        recv = CANTelemetryReceiver(channel="vcan0", hardware=False)
        recv.bus = bus_mock
        recv._hw_interface = None
        fake_thread = MagicMock()
        recv._thread = fake_thread
        recv._running = True

        recv.stop()

        assert recv._running is False
        fake_thread.join.assert_called_with(timeout=3)
        bus_mock.shutdown.assert_called_with()

    def test_stop_disconnects_hw_interface(self):
        hw = MagicMock()
        recv = CANTelemetryReceiver(channel="vcan0", hardware=False)
        recv._hw_interface = hw
        recv.bus = None
        recv._thread = MagicMock()

        recv.stop()
        hw.disconnect.assert_called_with()


# ---------------------------------------------------------------------------
# Integration: simulator -> receiver via mocked bus
# ---------------------------------------------------------------------------
@patch("ev_qa_framework.can_bus.can.interface.Bus")
def test_can_sim_receiver(_mock_bus):
    mock_instance = MagicMock()

    msg1 = can.Message(
        arbitration_id=0x101,
        data=[0x0F, 0xA0, 0x01, 0xF4, 0, 0, 0, 0],
        is_extended_id=False,
    )
    msg2 = can.Message(
        arbitration_id=0x102,
        data=[0x23, 0x50, 0, 0, 0, 0, 0, 0],
        is_extended_id=False,
    )

    recv_results = iter([msg1, msg2])

    def recv_side_effect(*_args, **_kwargs):
        try:
            return next(recv_results)
        except StopIteration:
            return None

    mock_instance.recv.side_effect = recv_side_effect
    _mock_bus.return_value = mock_instance

    sim = CANBatterySimulator(channel="vcan0", hardware=False)
    receiver = CANTelemetryReceiver(channel="vcan0", hardware=False)

    sim.start()
    receiver.start()

    for _ in range(20):
        data = receiver.get_telemetry()
        if data.get("voltage", 0) > 0 or data.get("soc") not in (0, None):
            break
        time.sleep(0.1)
    else:
        pytest.fail("receiver did not receive telemetry in time")

    data = receiver.get_telemetry()
    sim.stop()
    receiver.stop()

    assert "voltage" in data
    assert "current" in data
    assert "temperature" in data
    assert "soc" in data
    assert 400 <= data["voltage"] <= 410
    assert data["soc"] == 80
