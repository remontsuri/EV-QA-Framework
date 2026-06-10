"""
Tests for hardware CAN support (socketCAN, OBD-II, error handling).

Run with: .venv/bin/python -m pytest tests/test_hardware_can.py -v
"""

import builtins
import time
from unittest.mock import MagicMock, patch

import can as python_can
import pytest

from ev_qa_framework.can_bus import (
    CANBatterySimulator,
    CANBusOffError,
    CANConnectionError,
    CANHardwareInterface,
    CANHardwareNotFoundError,
    CANTelemetryReceiver,
    CANTimeoutError,
    DBCFileSimulator,
    HardwareCANError,
    OBD2Adapter,
    OBD2ConnectionError,
    OBD2ProtocolError,
    detect_can_interfaces,
    find_available_can_channel,
    find_hardware_can_interfaces,
)

# ═══════════════════════════════════════════════════════════════════
# Exception hierarchy tests
# ═══════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    def test_hardware_can_error_base(self):
        assert issubclass(CANConnectionError, HardwareCANError)
        assert issubclass(CANBusOffError, HardwareCANError)
        assert issubclass(CANTimeoutError, HardwareCANError)
        assert issubclass(CANHardwareNotFoundError, HardwareCANError)
        assert issubclass(OBD2ConnectionError, HardwareCANError)
        assert issubclass(OBD2ProtocolError, HardwareCANError)

    def test_exceptions_can_be_raised_and_caught(self):
        with pytest.raises(HardwareCANError):
            raise CANConnectionError("test")
        with pytest.raises(HardwareCANError):
            raise CANBusOffError("bus off")
        with pytest.raises(HardwareCANError):
            raise CANHardwareNotFoundError("no interface")

    def test_exception_message_preserved(self):
        msg = "Custom CAN error message"
        try:
            raise CANConnectionError(msg)
        except HardwareCANError as e:
            assert str(e) == msg


# ═══════════════════════════════════════════════════════════════════
# CAN interface detection tests
# ═══════════════════════════════════════════════════════════════════


class TestDetectCANInterfaces:
    @patch("ev_qa_framework.can_bus.platform.system")
    @patch("ev_qa_framework.can_bus.os.listdir")
    @patch("ev_qa_framework.can_bus.os.path.isdir")
    @patch("ev_qa_framework.can_bus.os.path.isfile")
    def test_detect_vcan_only(
        self,
        mock_isfile,
        mock_isdir,
        mock_listdir,
        mock_system,
    ):
        """Detect only virtual CAN interfaces"""
        mock_system.return_value = "Linux"
        mock_listdir.return_value = ["vcan0", "vcan1"]
        mock_isdir.return_value = True
        mock_isfile.side_effect = lambda p: "operstate" in p

        with patch.object(builtins, "open") as mock_open:
            file_handle = MagicMock()
            file_handle.__enter__.return_value = file_handle
            file_handle.read.return_value = "up\n"
            mock_open.return_value = file_handle

            result = detect_can_interfaces()
            assert len(result) == 2
            assert result[0]["name"] == "vcan0"
            assert result[0]["type"] == "virtual"
            assert result[0]["up"] is True
            assert result[1]["name"] == "vcan1"

    @patch("ev_qa_framework.can_bus.platform.system")
    def test_non_linux_returns_empty(self, mock_system):
        mock_system.return_value = "Windows"
        assert detect_can_interfaces() == []

    @patch("ev_qa_framework.can_bus.platform.system")
    @patch("ev_qa_framework.can_bus.os.listdir")
    @patch("ev_qa_framework.can_bus.os.path.isdir")
    @patch("ev_qa_framework.can_bus.os.path.isfile")
    def test_detect_hardware_can(
        self,
        mock_isfile,
        mock_isdir,
        mock_listdir,
        mock_system,
    ):
        mock_system.return_value = "Linux"
        mock_listdir.return_value = ["can0", "vcan0"]
        mock_isdir.return_value = True
        mock_isfile.side_effect = lambda p: "operstate" in p

        with patch.object(builtins, "open") as mock_open:
            file_handle = MagicMock()
            file_handle.__enter__.return_value = file_handle
            file_handle.read.return_value = "up\n"
            mock_open.return_value = file_handle

            result = detect_can_interfaces()
            names = [i["name"] for i in result]
            assert "can0" in names
            assert "vcan0" in names

            can0 = [i for i in result if i["name"] == "can0"][0]
            assert can0["type"] == "hardware"

    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_find_hardware_can_filters_virtual(self, mock_detect):
        mock_detect.return_value = [
            {"name": "can0", "type": "hardware", "up": True, "driver": "mcp251x"},
            {"name": "vcan0", "type": "virtual", "up": True, "driver": None},
        ]
        result = find_hardware_can_interfaces()
        assert len(result) == 1
        assert result[0]["name"] == "can0"

    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_find_available_prefers_hardware(self, mock_detect):
        mock_detect.return_value = [
            {"name": "can0", "type": "hardware", "up": True, "driver": "mcp251x"},
            {"name": "vcan0", "type": "virtual", "up": True, "driver": None},
        ]
        assert find_available_can_channel(prefer_hardware=True) == "can0"

    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_find_available_fallsback_to_virtual(self, mock_detect):
        mock_detect.return_value = [
            {"name": "vcan0", "type": "virtual", "up": True, "driver": None},
        ]
        assert find_available_can_channel(prefer_hardware=True) == "vcan0"

    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_find_available_none_found(self, mock_detect):
        mock_detect.return_value = []
        with pytest.raises(CANHardwareNotFoundError):
            find_available_can_channel()


# ═══════════════════════════════════════════════════════════════════
# CANHardwareInterface tests
# ═══════════════════════════════════════════════════════════════════


class TestCANHardwareInterface:
    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_context_manager(self, mock_bus):
        mock_bus.return_value = MagicMock()
        with CANHardwareInterface(channel="vcan0", auto_reconnect=False) as hw:
            assert hw.is_connected is True

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_connect_virtual_success(self, mock_bus):
        mock_bus.return_value = MagicMock()
        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        assert hw.connect() is True
        assert hw.is_connected is True
        assert hw.is_hardware is False
        hw.disconnect()

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_connect_hardware_success(self, mock_bus):
        mock_bus.return_value = MagicMock()
        hw = CANHardwareInterface(channel="can0", auto_reconnect=False)
        assert hw.connect() is True
        assert hw.is_connected is True
        assert hw.is_hardware is True
        hw.disconnect()

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_connect_failure_raises_without_autoreconnect(self, mock_bus):
        mock_bus.side_effect = python_can.CanError("test error")
        hw = CANHardwareInterface(channel="can0", auto_reconnect=False)
        with pytest.raises(CANConnectionError):
            hw.connect()

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_connect_failure_returns_false_with_autoreconnect(self, mock_bus):
        mock_bus.side_effect = python_can.CanError("test error")
        hw = CANHardwareInterface(channel="can0", auto_reconnect=True)
        assert hw.connect() is False
        assert hw.is_connected is False

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_send_success(self, mock_bus):
        mock_instance = MagicMock()
        mock_bus.return_value = mock_instance
        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        hw.connect()

        msg = MagicMock()
        result = hw.send(msg)
        assert result is True
        mock_instance.send.assert_called_once()

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_send_bus_off_raises(self, mock_bus):
        mock_instance = MagicMock()
        mock_instance.send.side_effect = python_can.CanError("bus-off detected")
        mock_bus.return_value = mock_instance

        hw = CANHardwareInterface(channel="can0", auto_reconnect=False)
        hw.connect()

        with pytest.raises(CANBusOffError):
            hw.send(MagicMock())

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_recv_returns_message(self, mock_bus):
        mock_instance = MagicMock()
        expected_msg = MagicMock()
        mock_instance.recv.return_value = expected_msg
        mock_bus.return_value = mock_instance

        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        hw.connect()

        msg = hw.recv(timeout=0.5)
        assert msg is expected_msg

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_recv_timeout_returns_none(self, mock_bus):
        mock_instance = MagicMock()
        mock_instance.recv.return_value = None
        mock_bus.return_value = mock_instance

        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        hw.connect()

        msg = hw.recv(timeout=0.1)
        assert msg is None

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_disconnect_shuts_down_bus(self, mock_bus):
        mock_instance = MagicMock()
        mock_bus.return_value = mock_instance

        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        hw.connect()
        hw.disconnect()

        mock_instance.shutdown.assert_called_once()
        assert hw.is_connected is False

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_health_check_disconnected(self, mock_bus):
        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        result = hw.health_check()
        assert result["status"] == "disconnected"
        assert result["connected"] is False

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_health_check_connected_virtual(self, mock_bus):
        mock_instance = MagicMock()
        mock_instance.send.return_value = None
        mock_bus.return_value = mock_instance

        hw = CANHardwareInterface(channel="vcan0", auto_reconnect=False)
        hw.connect()
        result = hw.health_check()
        assert result["connected"] is True

    @patch("ev_qa_framework.can_bus.can.interface.Bus")
    def test_auto_reconnect_on_send_failure(self, mock_bus):
        # Always raise on Bus creation so reconnect also fails
        mock_bus.side_effect = python_can.CanError("transient error")

        hw = CANHardwareInterface(
            channel="vcan0",
            auto_reconnect=True,
            max_reconnect_attempts=1,
            reconnect_delay=0.01,
        )
        hw.connect()

        result = hw.send(MagicMock())
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# OBD2Adapter tests
# ═══════════════════════════════════════════════════════════════════


class TestOBD2Adapter:
    def test_init_defaults(self):
        adapter = OBD2Adapter()
        assert adapter.port is None
        assert adapter.baudrate == 38400
        assert adapter.timeout == 2.0
        assert adapter.auto_reconnect is True
        assert adapter.is_connected is False

    @patch("serial.Serial")
    @patch("ev_qa_framework.can_bus.os.path.exists")
    def test_connect_auto_port(self, mock_exists, mock_serial):
        mock_exists.side_effect = lambda p: p == "/dev/ttyUSB0"

        mock_ser = MagicMock()
        mock_ser.is_open = True

        # ELM327 handshake: return version string then prompts
        read_data = list(b"ELM327 v1.5\r>")
        read_iter = iter(read_data)

        def read_fn(size=1):
            try:
                return bytes([next(read_iter)])
            except StopIteration:
                return b">"

        mock_ser.read = read_fn
        mock_ser.in_waiting = 1
        mock_serial.return_value = mock_ser

        adapter = OBD2Adapter(auto_reconnect=False)
        result = adapter.connect()
        assert result is True, "OBD2Adapter should auto-detect and connect"

    @patch("serial.Serial")
    def test_connect_explicit_port_success(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.side_effect = lambda s=1: b">"
        mock_serial.return_value = mock_ser

        adapter = OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False)
        result = adapter.connect()
        assert result is True

    @patch("serial.Serial")
    def test_disconnect_closes_serial(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.side_effect = lambda s=1: b">"
        mock_serial.return_value = mock_ser

        adapter = OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False)
        adapter.connect()
        adapter.disconnect()
        mock_ser.close.assert_called_once()
        assert adapter.is_connected is False

    @patch("serial.Serial")
    def test_context_manager(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.side_effect = lambda s=1: b">"
        mock_serial.return_value = mock_ser

        with OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False) as adapter:
            assert adapter is not None

    @patch("serial.Serial")
    def test_connect_failure(self, mock_serial):
        mock_serial.side_effect = OSError("Port not found")
        adapter = OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False)
        assert adapter.connect() is False

    @patch("serial.Serial")
    def test_send_command_returns_none_when_disconnected(self, mock_serial):
        adapter = OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False)
        result = adapter.send_command("AT RV")
        assert result is None

    @patch("serial.Serial")
    def test_get_telemetry_returns_dict(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.side_effect = lambda s=1: b">"
        mock_serial.return_value = mock_ser

        adapter = OBD2Adapter(port="/dev/ttyUSB0", auto_reconnect=False)
        adapter.connect()

        telemetry = adapter.get_telemetry()
        assert isinstance(telemetry, dict)
        expected_keys = [
            "battery_voltage",
            "battery_current",
            "battery_temperature",
            "soc",
            "odometer",
        ]
        for key in expected_keys:
            assert key in telemetry

    def test_auto_detect_port_no_hardware(self):
        adapter = OBD2Adapter(auto_reconnect=False)
        port = adapter._auto_detect_port()
        assert port is None or isinstance(port, str)


# ═══════════════════════════════════════════════════════════════════
# CANBatterySimulator tests (enhanced)
# ═══════════════════════════════════════════════════════════════════


class TestCANBatterySimulatorEnhanced:
    def test_init_defaults(self):
        sim = CANBatterySimulator()
        assert sim.channel == "vcan0"
        assert sim.hardware is False
        assert sim.bus is None

    def test_hardware_mode_flag(self):
        sim = CANBatterySimulator(channel="can0", hardware=True)
        assert sim.channel == "can0"
        assert sim.hardware is True

    def test_start_stop_virtual(self):
        sim = CANBatterySimulator(channel="test_vcan")
        receiver = CANTelemetryReceiver(channel="test_vcan")

        sim.start()
        receiver.start()

        time.sleep(0.5)

        data = receiver.get_telemetry()
        sim.stop()
        receiver.stop()

        assert "voltage" in data
        assert "current" in data

    def test_is_hardware_property(self):
        sim_virtual = CANBatterySimulator(channel="vcan0", hardware=False)
        assert sim_virtual.is_hardware is False

        # hardware=True -> is_hardware returns True before start (from self.hardware)
        sim_hw = CANBatterySimulator(channel="can0", hardware=True)
        assert sim_hw.is_hardware is True


# ═══════════════════════════════════════════════════════════════════
# CANTelemetryReceiver tests (enhanced)
# ═══════════════════════════════════════════════════════════════════


class TestCANTelemetryReceiverEnhanced:
    def test_init_defaults(self):
        receiver = CANTelemetryReceiver()
        assert receiver.channel == "vcan0"
        assert receiver.hardware is False

    def test_hardware_mode_flag(self):
        receiver = CANTelemetryReceiver(channel="can0", hardware=True)
        assert receiver.channel == "can0"
        assert receiver.hardware is True

    def test_get_telemetry_empty_before_start(self):
        receiver = CANTelemetryReceiver()
        data = receiver.get_telemetry()
        assert data == {"voltage": 0.0, "current": 0.0, "temperature": 0.0, "soc": 0.0}


# ═══════════════════════════════════════════════════════════════════
# DBCFileSimulator tests (enhanced)
# ═══════════════════════════════════════════════════════════════════


class TestDBCFileSimulatorEnhanced:
    def test_start_with_hardware_flag(self):
        sim = DBCFileSimulator()
        sim.start(channel="test_dbc", hardware=False)
        sim.stop()
        assert True

    def test_start_stop_default(self):
        sim = DBCFileSimulator()
        sim.start()
        time.sleep(0.2)
        sim.stop()
        assert sim.running is False


# ═══════════════════════════════════════════════════════════════════
# Module-level function tests
# ═══════════════════════════════════════════════════════════════════


class TestModuleFunctions:
    def test_detect_can_interfaces_is_callable(self):
        assert callable(detect_can_interfaces)

    def test_find_hardware_can_interfaces_is_callable(self):
        assert callable(find_hardware_can_interfaces)

    def test_find_available_can_channel_is_callable(self):
        assert callable(find_available_can_channel)
