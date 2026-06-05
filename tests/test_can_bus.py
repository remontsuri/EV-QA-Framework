"""
Test module for CAN Bus integration (legacy compatibility).
"""
from unittest.mock import MagicMock, patch

import can
import pytest

from ev_qa_framework.can_bus import CANBatterySimulator, CANTelemetryReceiver


@patch("ev_qa_framework.can_bus.can.interface.Bus")
def test_can_sim_receiver(mock_bus):
    """Test CAN simulation and reception with mocked bus"""
    mock_instance = MagicMock()

    # Simulate messages arriving on the bus
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

    # Return msg1, msg2 once, then None forever
    recv_results = iter([msg1, msg2])

    def recv_side_effect(timeout=1.0):
        try:
            return next(recv_results)
        except StopIteration:
            return None

    mock_instance.recv.side_effect = recv_side_effect
    mock_bus.return_value = mock_instance

    sim = CANBatterySimulator(channel="vcan0", hardware=False)
    receiver = CANTelemetryReceiver(channel="vcan0", hardware=False)

    sim.start()
    receiver.start()

    import time
    time.sleep(0.5)

    data = receiver.get_telemetry()
    sim.stop()
    receiver.stop()

    assert "voltage" in data
    assert "current" in data
    assert "temperature" in data
    assert "soc" in data

    assert 400 <= data["voltage"] <= 410
    assert data["soc"] == 80
