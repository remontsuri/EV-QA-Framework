import time
from ev_qa_framework.can_bus import CANBatterySimulator, CANTelemetryReceiver


def test_can_sim_receiver():
    """Test CAN simulation and reception"""
    # Test on a virtual interface
    sim = CANBatterySimulator(interface='virtual', channel='test_channel')
    receiver = CANTelemetryReceiver(interface='virtual',
                                    channel='test_channel')

    sim.start()
    receiver.start()

    # Wait for a few messages
    time.sleep(3.0)

    data = receiver.get_telemetry()
    sim.stop()
    receiver.stop()

    assert 'voltage' in data
    assert 'current' in data
    assert 'temperature' in data
    assert 'soc' in data

    # Values should be within reasonable bounds
    assert 390 <= data['voltage'] <= 410
    assert 40 <= data['current'] <= 60
    assert 30 <= data['temperature'] <= 40
    assert data['soc'] == 80
