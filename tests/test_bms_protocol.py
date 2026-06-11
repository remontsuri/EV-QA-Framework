"""
Comprehensive tests for the BMS Protocol abstraction layer.

Tests cover:
- BMSTelemetry data model
- BMSCANInterface (with mock CAN receiver)
- BMSModbusTCPInterface (with mock Modbus TCP client)
- BMSModbusRTUInterface (with mock Modbus RTU client)
- Protocol auto-detection (scan functions)
- BMSProtocolManager (unified interface, auto-detect, fallback)
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from ev_qa_framework.bms_protocol import (
    BMSCANInterface,
    # Interfaces
    BMSInterface,
    BMSModbusRTUInterface,
    BMSModbusTCPInterface,
    # Manager
    BMSProtocolManager,
    # Data model
    BMSTelemetry,
    # Auto-detection
    DetectedBMS,
    # Enums
    ProtocolType,
    _auto_detect_serial_ports,
    scan_can_interfaces,
    scan_modbus_rtu,
    scan_modbus_tcp,
)

# ═══════════════════════════════════════════════════════════════════
# BMSTelemetry Data Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSTelemetry:
    def test_default_values(self):
        t = BMSTelemetry()
        assert t.pack_voltage is None
        assert t.pack_current is None
        assert t.soc is None
        assert t.soh is None
        assert t.fault_flags == []
        assert t.cell_voltages == []
        assert t.is_balancing is False

    def test_to_dict(self):
        t = BMSTelemetry(
            pack_voltage=400.0,
            pack_current=50.0,
            soc=80.0,
            protocol="can",
            source="vcan0",
        )
        d = t.to_dict()
        assert d["pack_voltage"] == 400.0
        assert d["pack_current"] == 50.0
        assert d["soc"] == 80.0
        assert d["protocol"] == "can"
        assert d["source"] == "vcan0"

    def test_has_faults(self):
        t_ok = BMSTelemetry()
        assert t_ok.has_faults is False

        t_fault = BMSTelemetry(fault_flags=["Overvoltage", "Overtemperature"])
        assert t_fault.has_faults is True

    def test_is_healthy(self):
        t_healthy = BMSTelemetry(soc=80.0, fault_flags=[])
        assert t_healthy.is_healthy is True

        t_fault = BMSTelemetry(soc=80.0, fault_flags=["Overvoltage"])
        assert t_fault.is_healthy is False

        t_no_soc = BMSTelemetry(soc=None)
        assert t_no_soc.is_healthy is False

    def test_custom_values(self):
        t = BMSTelemetry(
            pack_voltage=360.5,
            pack_current=-25.3,
            soc=95.0,
            soh=88.5,
            temperature_max=45.0,
            temperature_min=20.0,
            temperature_avg=32.5,
            cell_voltage_min=3.2,
            cell_voltage_max=3.8,
            cell_voltage_delta=0.6,
            cell_voltages=[3.5, 3.6, 3.7, 3.8],
            charge_cycle_count=1500,
            fault_flags=["Cell imbalance"],
            status_flags=0x0003,
            is_balancing=True,
            protocol="modbus_tcp",
            timestamp=time.time(),
            source="192.168.1.100:502",
        )
        assert t.pack_voltage == 360.5
        assert t.pack_current == -25.3
        assert len(t.cell_voltages) == 4
        assert t.charge_cycle_count == 1500
        assert t.is_balancing is True


# ═══════════════════════════════════════════════════════════════════
# BMSCANInterface Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSCANInterface:
    def test_init_defaults(self):
        iface = BMSCANInterface()
        assert iface.channel == "vcan0"
        assert iface.hardware is False
        assert iface.protocol == ProtocolType.CAN
        assert iface.is_connected is False

    def test_init_custom(self):
        iface = BMSCANInterface(channel="can0", hardware=True, bitrate=250000)
        assert iface.channel == "can0"
        assert iface.hardware is True
        assert iface.bitrate == 250000

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_connect_success(self, mock_receiver_cls):
        mock_recv = MagicMock()
        mock_receiver_cls.return_value = mock_recv

        iface = BMSCANInterface(channel="vcan0")
        result = iface.connect()

        assert result is True
        assert iface.is_connected is True
        mock_recv.start.assert_called_once()

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_connect_failure(self, mock_receiver_cls):
        mock_receiver_cls.side_effect = RuntimeError("CAN not available")

        iface = BMSCANInterface(channel="vcan0")
        result = iface.connect()

        assert result is False
        assert iface.is_connected is False

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_disconnect(self, mock_receiver_cls):
        mock_recv = MagicMock()
        mock_receiver_cls.return_value = mock_recv

        iface = BMSCANInterface()
        iface.connect()
        iface.disconnect()

        mock_recv.stop.assert_called_once()
        assert iface.is_connected is False

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_read_telemetry(self, mock_receiver_cls):
        mock_recv = MagicMock()
        mock_recv.get_telemetry.return_value = {
            "voltage": 400.0,
            "current": 50.0,
            "temperature": 30.0,
            "soc": 80.0,
            "soh": 95.0,
        }
        mock_receiver_cls.return_value = mock_recv

        iface = BMSCANInterface(channel="vcan0")
        iface.connect()

        telemetry = iface.read_telemetry()

        assert telemetry.pack_voltage == 400.0
        assert telemetry.pack_current == 50.0
        assert telemetry.soc == 80.0
        assert telemetry.soh == 95.0
        assert telemetry.temperature_avg == 30.0
        assert telemetry.protocol == "can"
        assert telemetry.source == "vcan0"

    def test_read_telemetry_disconnected(self):
        iface = BMSCANInterface()
        telemetry = iface.read_telemetry()
        assert telemetry.pack_voltage is None
        assert telemetry.protocol == "can"

    def test_health_check(self):
        iface = BMSCANInterface(channel="vcan0")
        result = iface.health_check()
        assert result["protocol"] == "can"
        assert result["channel"] == "vcan0"
        assert result["connected"] is False
        assert result["status"] == "disconnected"

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_context_manager(self, mock_receiver_cls):
        mock_recv = MagicMock()
        mock_receiver_cls.return_value = mock_recv

        with BMSCANInterface() as iface:
            assert iface.is_connected is True

        mock_recv.stop.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# BMSModbusTCPInterface Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSModbusTCPInterface:
    def test_init_defaults(self):
        iface = BMSModbusTCPInterface("192.168.1.100")
        assert iface.host == "192.168.1.100"
        assert iface.port == 502
        assert iface.unit_id == 1
        assert iface.protocol == ProtocolType.MODBUS_TCP

    def test_init_custom(self):
        iface = BMSModbusTCPInterface("10.0.0.1", port=5020, unit_id=5)
        assert iface.port == 5020
        assert iface.unit_id == 5

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_connect_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        result = iface.connect()

        assert result is True
        assert iface.is_connected is True
        mock_client.connect.assert_called_once()

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_connect_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        result = iface.connect()

        assert result is False
        assert iface.is_connected is False

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_disconnect(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        iface.connect()
        iface.disconnect()

        mock_client.disconnect.assert_called_once()
        assert iface.is_connected is False

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_read_telemetry(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.read_battery_telemetry.return_value = {
            "pack_voltage": 400.0,
            "pack_current": 50.0,
            "soc": 80.0,
            "soh": 95.0,
            "temperature_max": 45.0,
            "temperature_min": 20.0,
            "temperature_avg": 32.5,
            "cell_voltage_min": 3.2,
            "cell_voltage_max": 3.8,
            "cell_voltage_delta": 0.6,
            "charge_cycle_count": 1500,
            "fault_flags": ["Overvoltage"],
            "status_flags": 0x0003,
        }
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        iface.connect()

        telemetry = iface.read_telemetry()

        assert telemetry.pack_voltage == 400.0
        assert telemetry.pack_current == 50.0
        assert telemetry.soc == 80.0
        assert telemetry.soh == 95.0
        assert telemetry.temperature_max == 45.0
        assert telemetry.temperature_min == 20.0
        assert telemetry.temperature_avg == 32.5
        assert telemetry.cell_voltage_min == 3.2
        assert telemetry.cell_voltage_max == 3.8
        assert telemetry.cell_voltage_delta == 0.6
        assert telemetry.charge_cycle_count == 1500
        assert "Overvoltage" in telemetry.fault_flags
        assert telemetry.status_flags == 0x0003
        assert telemetry.protocol == "modbus_tcp"
        assert telemetry.source == "192.168.1.100:502"

    def test_read_telemetry_disconnected(self):
        iface = BMSModbusTCPInterface("192.168.1.100")
        telemetry = iface.read_telemetry()
        assert telemetry.pack_voltage is None
        assert telemetry.protocol == "modbus_tcp"

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_read_telemetry_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.read_battery_telemetry.side_effect = Exception("Modbus error")
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        iface.connect()

        telemetry = iface.read_telemetry()
        assert telemetry.pack_voltage is None
        assert telemetry.protocol == "modbus_tcp"

    @patch("ev_qa_framework.modbus.ModbusTCPClient")
    def test_health_check(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.health_check.return_value = {
            "status": "healthy",
            "connected": True,
        }
        mock_client_cls.return_value = mock_client

        iface = BMSModbusTCPInterface("192.168.1.100")
        iface.connect()

        result = iface.health_check()
        assert result["protocol"] == "modbus_tcp"
        assert result["status"] == "healthy"

    def test_health_check_disconnected(self):
        iface = BMSModbusTCPInterface("192.168.1.100")
        result = iface.health_check()
        assert result["status"] == "disconnected"


# ═══════════════════════════════════════════════════════════════════
# BMSModbusRTUInterface Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSModbusRTUInterface:
    def test_init_defaults(self):
        iface = BMSModbusRTUInterface()
        assert iface.port == "/dev/ttyUSB0"
        assert iface.baudrate == 9600
        assert iface.unit_id == 1
        assert iface.protocol == ProtocolType.MODBUS_RTU

    def test_init_custom(self):
        iface = BMSModbusRTUInterface(port="COM3", baudrate=19200, unit_id=5)
        assert iface.port == "COM3"
        assert iface.baudrate == 19200

    @patch("ev_qa_framework.modbus.ModbusRTUClient")
    def test_connect_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client_cls.return_value = mock_client

        iface = BMSModbusRTUInterface("/dev/ttyUSB0")
        result = iface.connect()

        assert result is True
        assert iface.is_connected is True

    @patch("ev_qa_framework.modbus.ModbusRTUClient")
    def test_connect_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception("Port not found")
        mock_client_cls.return_value = mock_client

        iface = BMSModbusRTUInterface("/dev/ttyUSB0")
        result = iface.connect()

        assert result is False
        assert iface.is_connected is False

    @patch("ev_qa_framework.modbus.ModbusRTUClient")
    def test_disconnect(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        iface = BMSModbusRTUInterface()
        iface.connect()
        iface.disconnect()

        mock_client.disconnect.assert_called_once()
        assert iface.is_connected is False

    @patch("ev_qa_framework.modbus.ModbusRTUClient")
    def test_read_telemetry(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.read_battery_telemetry.return_value = {
            "pack_voltage": 360.0,
            "pack_current": -25.0,
            "soc": 60.0,
            "soh": 90.0,
            "temperature_max": 35.0,
            "temperature_min": 15.0,
            "temperature_avg": 25.0,
            "cell_voltage_min": 3.3,
            "cell_voltage_max": 3.6,
            "cell_voltage_delta": 0.3,
            "charge_cycle_count": 800,
            "fault_flags": [],
            "status_flags": 0x0001,
        }
        mock_client_cls.return_value = mock_client

        iface = BMSModbusRTUInterface("/dev/ttyUSB0")
        iface.connect()

        telemetry = iface.read_telemetry()

        assert telemetry.pack_voltage == 360.0
        assert telemetry.pack_current == -25.0
        assert telemetry.soc == 60.0
        assert telemetry.protocol == "modbus_rtu"
        assert telemetry.source == "/dev/ttyUSB0"

    def test_read_telemetry_disconnected(self):
        iface = BMSModbusRTUInterface()
        telemetry = iface.read_telemetry()
        assert telemetry.pack_voltage is None
        assert telemetry.protocol == "modbus_rtu"

    def test_health_check_disconnected(self):
        iface = BMSModbusRTUInterface()
        result = iface.health_check()
        assert result["status"] == "disconnected"


# ═══════════════════════════════════════════════════════════════════
# Protocol Auto-Detection Tests
# ═══════════════════════════════════════════════════════════════════


class TestAutoDetection:
    def test_detected_bms_defaults(self):
        d = DetectedBMS(
            protocol=ProtocolType.CAN,
            description="test",
            config={"channel": "vcan0"},
        )
        assert d.priority == 0

    def test_detected_bms_custom_priority(self):
        d = DetectedBMS(
            protocol=ProtocolType.MODBUS_TCP,
            description="test",
            config={},
            priority=10,
        )
        assert d.priority == 10

    @patch("ev_qa_framework.bms_protocol.platform.system")
    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_scan_can_linux(self, mock_detect, mock_system):
        mock_system.return_value = "Linux"
        mock_detect.return_value = [
            {"name": "can0", "type": "hardware", "up": True, "driver": "mcp251x"},
            {"name": "vcan0", "type": "virtual", "up": True, "driver": None},
        ]

        result = scan_can_interfaces()
        assert len(result) == 2
        assert result[0].protocol == ProtocolType.CAN
        assert result[0].config["channel"] == "can0"
        assert result[0].priority == 10  # hardware
        assert result[1].priority == 5  # virtual

    @patch("ev_qa_framework.bms_protocol.platform.system")
    @patch("ev_qa_framework.can_bus.detect_can_interfaces")
    def test_scan_can_exception(self, mock_detect, mock_system):
        mock_system.return_value = "Linux"
        mock_detect.side_effect = RuntimeError("CAN error")

        result = scan_can_interfaces()
        assert result == []

    @patch("ev_qa_framework.bms_protocol.platform.system")
    def test_scan_can_non_linux(self, mock_system):
        mock_system.return_value = "Windows"
        result = scan_can_interfaces()
        assert result == []

    @patch("socket.socket")
    def test_scan_modbus_tcp(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0  # Success
        mock_socket_cls.return_value = mock_sock

        result = scan_modbus_tcp(hosts=["192.168.1.100"], timeout=0.1)
        assert len(result) == 1
        assert result[0].protocol == ProtocolType.MODBUS_TCP
        assert result[0].config["host"] == "192.168.1.100"

    @patch("socket.socket")
    def test_scan_modbus_tcp_no_devices(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 111  # Connection refused
        mock_socket_cls.return_value = mock_sock

        result = scan_modbus_tcp(hosts=["192.168.1.100"], timeout=0.1)
        assert result == []

    @patch("socket.socket")
    def test_scan_modbus_tcp_os_error(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.side_effect = OSError("Network unreachable")
        mock_socket_cls.return_value = mock_sock

        result = scan_modbus_tcp(hosts=["192.168.1.100"], timeout=0.1)
        assert result == []

    @patch("ev_qa_framework.bms_protocol.platform.system")
    def test_auto_detect_serial_ports_linux(self, mock_system):
        mock_system.return_value = "Linux"
        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda p: ["/dev/ttyUSB0"] if "ttyUSB" in p else []
            ports = _auto_detect_serial_ports()
            assert "/dev/ttyUSB0" in ports

    @patch("ev_qa_framework.bms_protocol.platform.system")
    def test_auto_detect_serial_ports_windows(self, mock_system):
        mock_system.return_value = "Windows"
        ports = _auto_detect_serial_ports()
        assert "COM1" in ports
        assert "COM16" in ports

    @patch("ev_qa_framework.bms_protocol.platform.system")
    def test_auto_detect_serial_ports_unknown(self, mock_system):
        mock_system.return_value = "UnknownOS"
        ports = _auto_detect_serial_ports()
        assert ports == []


# ═══════════════════════════════════════════════════════════════════
# BMSProtocolManager Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSProtocolManager:
    def test_init_defaults(self):
        mgr = BMSProtocolManager()
        assert mgr.protocol == ProtocolType.AUTO
        assert mgr.auto_fallback is True
        assert mgr.is_connected is False
        assert mgr.active_protocol == "none"

    def test_init_explicit(self):
        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        assert mgr.protocol == ProtocolType.CAN

    @patch.object(BMSProtocolManager, "auto_detect")
    def test_auto_detect_delegates(self, mock_scan):
        mock_scan.return_value = []
        mgr = BMSProtocolManager()
        result = mgr.auto_detect()
        assert result == []
        mock_scan.assert_called_once()

    def test_get_detected_empty(self):
        mgr = BMSProtocolManager()
        assert mgr.get_detected() == []

    def _make_mock_interface(self, protocol, connected=True):
        """Create a properly configured mock interface."""
        mock_iface = MagicMock()
        mock_iface.connect.return_value = connected
        mock_iface.is_connected = connected
        # Set protocol as a proper enum value
        if protocol == "can":
            mock_iface.protocol = ProtocolType.CAN
        elif protocol == "modbus_tcp":
            mock_iface.protocol = ProtocolType.MODBUS_TCP
        elif protocol == "modbus_rtu":
            mock_iface.protocol = ProtocolType.MODBUS_RTU
        return mock_iface

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_connect_can(self, mock_can_cls):
        mock_can_cls.return_value = self._make_mock_interface("can")

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0", "hardware": False},
        )
        result = mgr.connect()

        assert result is True
        assert mgr.is_connected is True
        assert mgr.active_protocol == "can"
        mock_can_cls.assert_called_once_with(channel="vcan0", hardware=False)

    @patch("ev_qa_framework.bms_protocol.BMSModbusTCPInterface")
    def test_connect_modbus_tcp(self, mock_tcp_cls):
        mock_tcp_cls.return_value = self._make_mock_interface("modbus_tcp")

        mgr = BMSProtocolManager(
            protocol=ProtocolType.MODBUS_TCP,
            config={"host": "192.168.1.100", "port": 502, "unit_id": 1},
        )
        result = mgr.connect()

        assert result is True
        assert mgr.active_protocol == "modbus_tcp"

    @patch("ev_qa_framework.bms_protocol.BMSModbusRTUInterface")
    def test_connect_modbus_rtu(self, mock_rtu_cls):
        mock_rtu_cls.return_value = self._make_mock_interface("modbus_rtu")

        mgr = BMSProtocolManager(
            protocol=ProtocolType.MODBUS_RTU,
            config={"port": "/dev/ttyUSB0", "baudrate": 9600, "unit_id": 1},
        )
        result = mgr.connect()

        assert result is True
        assert mgr.active_protocol == "modbus_rtu"

    @patch("ev_qa_framework.can_bus.CANTelemetryReceiver")
    def test_connect_failure(self, mock_receiver_cls):
        mock_receiver_cls.side_effect = RuntimeError("CAN interface not available")
        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        result = mgr.connect()
        assert result is False
        assert mgr.is_connected is False

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_disconnect(self, mock_can_cls):
        mock_iface = self._make_mock_interface("can")
        mock_can_cls.return_value = mock_iface

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        mgr.connect()
        mgr.disconnect()

        mock_iface.disconnect.assert_called_once()
        assert mgr.is_connected is False
        assert mgr.active_protocol == "none"

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_context_manager(self, mock_can_cls):
        mock_iface = self._make_mock_interface("can")
        mock_can_cls.return_value = mock_iface

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        mgr.connect()
        assert mgr.is_connected is True
        assert len(mgr._interfaces) == 1
        mgr.disconnect()
        mock_iface.disconnect.assert_called_once()
        assert len(mgr._interfaces) == 0

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_read_telemetry(self, mock_can_cls):
        mock_iface = self._make_mock_interface("can")
        mock_iface.read_telemetry.return_value = BMSTelemetry(
            pack_voltage=400.0,
            soc=80.0,
            protocol="can",
        )
        mock_can_cls.return_value = mock_iface

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        mgr.connect()

        telemetry = mgr.read_telemetry()
        assert telemetry.pack_voltage == 400.0
        assert telemetry.soc == 80.0

    def test_read_telemetry_not_connected(self):
        mgr = BMSProtocolManager()
        telemetry = mgr.read_telemetry()
        assert telemetry.protocol == "none"
        assert telemetry.source == "disconnected"

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_health_check(self, mock_can_cls):
        mock_iface = self._make_mock_interface("can")
        mock_iface.health_check.return_value = {
            "protocol": "can",
            "status": "healthy",
        }
        mock_can_cls.return_value = mock_iface

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        mgr.connect()

        result = mgr.health_check()
        assert result["connected"] is True
        assert result["active_protocol"] == "can"
        assert len(result["interfaces"]) == 1

    def test_health_check_not_connected(self):
        mgr = BMSProtocolManager()
        result = mgr.health_check()
        assert result["connected"] is False
        assert result["active_protocol"] == "none"

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_context_manager_with_data(self, mock_can_cls):
        mock_iface = self._make_mock_interface("can")
        mock_can_cls.return_value = mock_iface

        with BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        ) as mgr:
            assert mgr.is_connected is True

        mock_iface.disconnect.assert_called_once()

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_auto_fallback(self, mock_can_cls):
        """Test that manager falls back to next interface on failure."""
        mock_iface1 = self._make_mock_interface("can")
        mock_iface1.read_telemetry.side_effect = Exception("Connection lost")

        mock_iface2 = self._make_mock_interface("can")
        mock_iface2.read_telemetry.return_value = BMSTelemetry(
            pack_voltage=350.0, soc=70.0, protocol="can"
        )

        mock_can_cls.side_effect = [mock_iface1, mock_iface2]

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
            auto_fallback=True,
        )
        mgr.connect()

        # Manually add second interface to simulate multiple detected
        mgr._interfaces.append(mock_iface2)

        telemetry = mgr.read_telemetry()
        assert telemetry.pack_voltage == 350.0
        assert mgr._active_interface is mock_iface2

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    def test_auto_fallback_disabled(self, mock_can_cls):
        """Test that fallback is skipped when auto_fallback=False."""
        mock_iface = self._make_mock_interface("can")
        mock_iface.read_telemetry.side_effect = Exception("Connection lost")
        mock_can_cls.return_value = mock_iface

        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
            auto_fallback=False,
        )
        mgr.connect()

        telemetry = mgr.read_telemetry()
        assert telemetry.pack_voltage is None

    @patch("ev_qa_framework.bms_protocol.scan_can_interfaces")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_tcp")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_rtu")
    def test_auto_detect_all_protocols(self, mock_rtu, mock_tcp, mock_can):
        mock_can.return_value = [
            DetectedBMS(
                protocol=ProtocolType.CAN,
                description="can0",
                config={"channel": "can0", "hardware": True},
                priority=10,
            ),
        ]
        mock_tcp.return_value = [
            DetectedBMS(
                protocol=ProtocolType.MODBUS_TCP,
                description="192.168.1.100:502",
                config={"host": "192.168.1.100", "port": 502},
                priority=8,
            ),
        ]
        mock_rtu.return_value = []

        mgr = BMSProtocolManager()
        detected = mgr.auto_detect()

        assert len(detected) == 2
        assert detected[0].protocol == ProtocolType.CAN
        assert detected[1].protocol == ProtocolType.MODBUS_TCP

    @patch("ev_qa_framework.bms_protocol.BMSCANInterface")
    @patch("ev_qa_framework.bms_protocol.scan_can_interfaces")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_tcp")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_rtu")
    def test_connect_auto(self, mock_rtu, mock_tcp, mock_can, mock_can_cls):
        mock_can.return_value = [
            DetectedBMS(
                protocol=ProtocolType.CAN,
                description="can0",
                config={"channel": "can0", "hardware": True},
                priority=10,
            ),
        ]
        mock_tcp.return_value = []
        mock_rtu.return_value = []

        mock_can_cls.return_value = self._make_mock_interface("can")

        mgr = BMSProtocolManager(protocol=ProtocolType.AUTO)
        result = mgr.connect()

        assert result is True
        assert mgr.is_connected is True

    @patch("ev_qa_framework.bms_protocol.scan_can_interfaces")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_tcp")
    @patch("ev_qa_framework.bms_protocol.scan_modbus_rtu")
    def test_connect_auto_no_devices(self, mock_rtu, mock_tcp, mock_can):
        mock_can.return_value = []
        mock_tcp.return_value = []
        mock_rtu.return_value = []

        mgr = BMSProtocolManager(protocol=ProtocolType.AUTO)
        result = mgr.connect()

        assert result is False
        assert mgr.is_connected is False
