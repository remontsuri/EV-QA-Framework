"""
Tests for the BMS Protocol abstraction layer.

Covers:
- ProtocolType enum
- BMSTelemetry data model (to_dict, has_faults, is_healthy)
- DetectedBMS dataclass
- BMSProtocolManager initialization and properties
- scan functions (with mocking)
"""

from unittest.mock import MagicMock, patch

from ev_qa_framework.bms_protocol import (
    BMSProtocolManager,
    BMSTelemetry,
    DetectedBMS,
    ProtocolType,
    _auto_detect_serial_ports,
    scan_can_interfaces,
    scan_modbus_tcp,
)


class TestProtocolType:
    def test_values(self):
        assert ProtocolType.CAN.value == "can"
        assert ProtocolType.MODBUS_TCP.value == "modbus_tcp"
        assert ProtocolType.MODBUS_RTU.value == "modbus_rtu"
        assert ProtocolType.AUTO.value == "auto"


class TestBMSTelemetry:
    def test_default_creation(self):
        t = BMSTelemetry()
        assert t.pack_voltage is None
        assert t.pack_current is None
        assert t.soc is None
        assert t.soh is None
        assert t.cell_voltages == []
        assert t.fault_flags == []
        assert t.is_balancing is False

    def test_to_dict(self):
        t = BMSTelemetry(
            pack_voltage=400.0,
            pack_current=50.0,
            soc=80.0,
            soh=95.0,
            protocol="can",
            source="vcan0",
        )
        d = t.to_dict()
        assert d["pack_voltage"] == 400.0
        assert d["pack_current"] == 50.0
        assert d["soc"] == 80.0
        assert d["protocol"] == "can"
        assert d["source"] == "vcan0"

    def test_has_faults_empty(self):
        t = BMSTelemetry()
        assert t.has_faults is False

    def test_has_faults_with_faults(self):
        t = BMSTelemetry(fault_flags=["Overvoltage"])
        assert t.has_faults is True

    def test_is_healthy_no_faults_with_soc(self):
        t = BMSTelemetry(soc=80.0, fault_flags=[])
        assert t.is_healthy is True

    def test_is_healthy_with_faults(self):
        t = BMSTelemetry(soc=80.0, fault_flags=["Overvoltage"])
        assert t.is_healthy is False

    def test_is_healthy_no_soc(self):
        t = BMSTelemetry(soc=None)
        assert t.is_healthy is False

    def test_all_fields_set(self):
        t = BMSTelemetry(
            pack_voltage=400.0,
            pack_current=50.0,
            soc=80.0,
            soh=95.0,
            temperature_max=35.0,
            temperature_min=25.0,
            temperature_avg=30.0,
            cell_voltage_min=3.5,
            cell_voltage_max=4.2,
            cell_voltage_delta=0.7,
            cell_voltages=[3.7] * 96,
            charge_cycle_count=100,
            fault_flags=[],
            status_flags=0,
            is_balancing=False,
            protocol="can",
            timestamp=1234567890.0,
            source="vcan0",
        )
        d = t.to_dict()
        assert len(d) == 18
        assert d["charge_cycle_count"] == 100
        assert len(d["cell_voltages"]) == 96


class TestDetectedBMS:
    def test_creation(self):
        d = DetectedBMS(
            protocol=ProtocolType.CAN,
            description="CAN interface vcan0",
            config={"channel": "vcan0"},
        )
        assert d.protocol == ProtocolType.CAN
        assert d.priority == 0

    def test_creation_with_priority(self):
        d = DetectedBMS(
            protocol=ProtocolType.MODBUS_TCP,
            description="Modbus TCP at 192.168.1.100",
            config={"host": "192.168.1.100", "port": 502},
            priority=8,
        )
        assert d.priority == 8


class TestBMSProtocolManager:
    def test_default_init(self):
        mgr = BMSProtocolManager()
        assert mgr.protocol == ProtocolType.AUTO
        assert mgr.config == {}
        assert mgr.auto_fallback is True
        assert mgr.is_connected is False

    def test_explicit_protocol(self):
        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0"},
        )
        assert mgr.protocol == ProtocolType.CAN
        assert mgr.config["channel"] == "vcan0"

    def test_is_connected_no_active_interface(self):
        mgr = BMSProtocolManager()
        assert mgr.is_connected is False


class TestScanFunctions:
    def test_scan_modbus_tcp_no_hosts(self):
        """Scanning with no responsive hosts should return empty list."""
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 1  # Connection refused
            mock_socket_cls.return_value = mock_sock
            results = scan_modbus_tcp(hosts=["192.168.1.200"], port=502, timeout=0.1)
            assert results == []

    def test_scan_modbus_tcp_with_responsive_host(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0  # Success
            mock_socket_cls.return_value = mock_sock
            results = scan_modbus_tcp(hosts=["192.168.1.100"], port=502, timeout=0.1)
            assert len(results) == 1
            assert results[0].protocol == ProtocolType.MODBUS_TCP
            assert results[0].config["host"] == "192.168.1.100"

    def test_scan_modbus_tcp_oserror_handled(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect_ex.side_effect = OSError("Network unreachable")
            mock_socket_cls.return_value = mock_sock
            results = scan_modbus_tcp(hosts=["192.168.1.100"], port=502, timeout=0.1)
            assert results == []

    def test_scan_can_interfaces_non_linux(self):
        with patch("platform.system", return_value="Windows"):
            results = scan_can_interfaces()
            assert results == []

    def test_auto_detect_serial_ports_linux(self):
        with patch("platform.system", return_value="Linux"):
            with patch("glob.glob", return_value=["/dev/ttyUSB0", "/dev/ttyUSB1"]):
                ports = _auto_detect_serial_ports()
                assert "/dev/ttyUSB0" in ports
                assert "/dev/ttyUSB1" in ports

    def test_auto_detect_serial_ports_windows(self):
        with patch("platform.system", return_value="Windows"):
            ports = _auto_detect_serial_ports()
            assert "COM1" in ports
            assert "COM16" in ports

    def test_auto_detect_serial_ports_unknown(self):
        with patch("platform.system", return_value="UnknownOS"):
            ports = _auto_detect_serial_ports()
            assert ports == []
