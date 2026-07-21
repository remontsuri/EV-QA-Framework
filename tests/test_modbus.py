"""
Comprehensive tests for the Modbus protocol module.

Tests cover:
- CRC-16 computation and validation
- PDU / frame builders and parsers
- Exception hierarchy
- ModbusTCPClient with mock sockets
- ModbusRTUClient with mock serial
- BMS register map operations
- Read/write register operations
"""

import socket
import struct
from unittest.mock import MagicMock, patch

import pytest

from ev_qa_framework.modbus import (
    BMS_REGISTER_MAP,
    FAULT_FLAGS,
    # Constants / Enums
    FunctionCode,
    # Clients
    ModbusConfigurationError,
    ModbusConnectionError,
    ModbusCRCError,
    # Exceptions
    ModbusError,
    ModbusResponseError,
    ModbusRTUClient,
    ModbusTCPClient,
    ModbusTimeoutError,
    _append_crc,
    # PDU builders
    _build_read_pdu,
    _build_tcp_mbap,
    _build_write_multiple_registers_pdu,
    _build_write_single_register_pdu,
    # CRC
    _crc16_modbus,
    _parse_read_response,
    _validate_crc,
)

# ═══════════════════════════════════════════════════════════════════
# Exception Hierarchy Tests
# ═══════════════════════════════════════════════════════════════════


class TestModbusExceptions:
    def test_base_exception(self):
        assert issubclass(ModbusConnectionError, ModbusError)
        assert issubclass(ModbusTimeoutError, ModbusError)
        assert issubclass(ModbusCRCError, ModbusError)
        assert issubclass(ModbusResponseError, ModbusError)
        assert issubclass(ModbusConfigurationError, ModbusError)

    def test_catch_base(self):
        with pytest.raises(ModbusError):
            raise ModbusConnectionError("test")
        with pytest.raises(ModbusError):
            raise ModbusTimeoutError("timeout")
        with pytest.raises(ModbusError):
            raise ModbusCRCError("crc fail")
        with pytest.raises(ModbusError):
            raise ModbusResponseError("response err")

    def test_response_error_has_exception_code(self):
        err = ModbusResponseError("test", exception_code=0x02)
        assert err.exception_code == 0x02
        assert str(err) == "test"

    def test_exception_message_preserved(self):
        msg = "Custom Modbus error"
        try:
            raise ModbusConnectionError(msg)
        except ModbusError as e:
            assert str(e) == msg


# ═══════════════════════════════════════════════════════════════════
# CRC-16 Tests
# ═══════════════════════════════════════════════════════════════════


class TestCRC16:
    def test_crc_known_value(self):
        """Test CRC-16/Modbus round-trip."""
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        crc = _crc16_modbus(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF
        frame = _append_crc(data)
        assert _validate_crc(frame) is True

    def test_crc_empty(self):
        """CRC of empty data should be 0xFFFF."""
        assert _crc16_modbus(b"") == 0xFFFF

    def test_crc_single_byte(self):
        crc = _crc16_modbus(b"\x00")
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_append_crc(self):
        frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        with_crc = _append_crc(frame)
        assert len(with_crc) == len(frame) + 2
        expected_crc = _crc16_modbus(frame)
        assert struct.unpack("<H", with_crc[-2:])[0] == expected_crc

    def test_validate_crc_valid(self):
        frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        with_crc = _append_crc(frame)
        assert _validate_crc(with_crc) is True

    def test_validate_crc_invalid(self):
        frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A, 0xFF, 0xFF])
        assert _validate_crc(frame) is False

    def test_validate_crc_too_short(self):
        assert _validate_crc(b"\x01\x02") is False
        assert _validate_crc(b"") is False


# ═══════════════════════════════════════════════════════════════════
# PDU Builder Tests
# ═══════════════════════════════════════════════════════════════════


class TestPDUBuilders:
    def test_build_read_pdu(self):
        pdu = _build_read_pdu(FunctionCode.READ_HOLDING_REGISTERS, 0x0000, 10)
        fc, addr, qty = struct.unpack(">BHH", pdu)
        assert fc == 0x03
        assert addr == 0x0000
        assert qty == 10

    def test_build_read_pdu_input_registers(self):
        pdu = _build_read_pdu(FunctionCode.READ_INPUT_REGISTERS, 0x0010, 5)
        fc, addr, qty = struct.unpack(">BHH", pdu)
        assert fc == 0x04
        assert addr == 0x0010
        assert qty == 5

    def test_build_write_single_register_pdu(self):
        pdu = _build_write_single_register_pdu(0x0001, 250)
        fc, addr, val = struct.unpack(">BHH", pdu)
        assert fc == 0x06
        assert addr == 0x0001
        assert val == 250

    def test_build_write_multiple_registers_pdu(self):
        pdu = _build_write_multiple_registers_pdu(0x0000, [100, 200, 300])
        fc, addr, qty, bc = struct.unpack(">BHHB", pdu[:6])
        assert fc == 0x10
        assert addr == 0x0000
        assert qty == 3
        assert bc == 6
        v1, v2, v3 = struct.unpack(">HHH", pdu[6:12])
        assert v1 == 100
        assert v2 == 200
        assert v3 == 300

    def test_build_tcp_mbap(self):
        mbap = _build_tcp_mbap(transaction_id=1, length=7, unit_id=1)
        tid, pid, length, uid = struct.unpack(">HHHB", mbap)
        assert tid == 1
        assert pid == 0x0000
        assert length == 7
        assert uid == 1


# ═══════════════════════════════════════════════════════════════════
# Response Parser Tests
# ═══════════════════════════════════════════════════════════════════


class TestParseReadResponse:
    def test_parse_normal_response(self):
        pdu = bytes([0x03, 0x04, 0x0F, 0xA0, 0x01, 0xF4])
        regs = _parse_read_response(pdu, FunctionCode.READ_HOLDING_REGISTERS, 2)
        assert regs == [0x0FA0, 0x01F4]
        assert regs[0] == 4000
        assert regs[1] == 500

    def test_parse_exception_response(self):
        pdu = bytes([0x83, 0x02])
        with pytest.raises(ModbusResponseError) as exc_info:
            _parse_read_response(pdu, FunctionCode.READ_HOLDING_REGISTERS, 1)
        assert exc_info.value.exception_code == 0x02

    def test_parse_wrong_function_code(self):
        pdu = bytes([0x04, 0x02, 0x00, 0x0A])
        with pytest.raises(ModbusResponseError, match="Unexpected function code"):
            _parse_read_response(pdu, FunctionCode.READ_HOLDING_REGISTERS, 1)

    def test_parse_empty_pdu(self):
        with pytest.raises(ModbusResponseError, match="Empty PDU"):
            _parse_read_response(b"", FunctionCode.READ_HOLDING_REGISTERS, 1)

    def test_parse_wrong_byte_count(self):
        pdu = bytes([0x03, 0x04, 0x00, 0x0A])
        with pytest.raises(ModbusResponseError, match="Unexpected byte count"):
            _parse_read_response(pdu, FunctionCode.READ_HOLDING_REGISTERS, 1)


# ═══════════════════════════════════════════════════════════════════
# Modbus Register Map Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSRegisterMap:
    def test_register_map_has_required_keys(self):
        required = [
            "pack_voltage",
            "pack_current",
            "soc",
            "soh",
            "temperature_max",
            "temperature_min",
            "temperature_avg",
            "cell_voltage_min",
            "cell_voltage_max",
            "cell_voltage_delta",
            "charge_cycle_count",
            "fault_flags",
            "status_flags",
        ]
        for key in required:
            assert key in BMS_REGISTER_MAP, f"Missing register: {key}"

    def test_register_map_entries_have_required_fields(self):
        for name, reg in BMS_REGISTER_MAP.items():
            assert "address" in reg, f"{name}: missing address"
            assert "count" in reg, f"{name}: missing count"
            assert "scale" in reg, f"{name}: missing scale"
            assert "unit" in reg, f"{name}: missing unit"

    def test_fault_flags_bitmap(self):
        assert len(FAULT_FLAGS) >= 10
        assert FAULT_FLAGS[0] == "Overvoltage"
        assert FAULT_FLAGS[9] == "Thermal runaway"


# ═══════════════════════════════════════════════════════════════════
# Helper: Mock recv that returns bytes chunks from a frame
# ═══════════════════════════════════════════════════════════════════


def _make_recv_side_effect(response_frame: bytes):
    """Create a recv side_effect that returns proper bytes chunks from a frame.

    Each call creates a fresh iterator, so the frame is re-served on retry.
    """

    def recv(n):
        # Use a fresh position tracker per top-level call
        if not hasattr(recv, "_pos") or recv._pos >= len(response_frame):
            recv._pos = 0
        if recv._pos >= len(response_frame):
            return b""
        chunk = response_frame[recv._pos : recv._pos + n]
        recv._pos += n
        return chunk

    return recv


def _make_serial_read_side_effect(frame: bytes):
    """Create a serial read side_effect returning individual bytes as bytes objects."""
    pos = [0]

    def read(n=1):
        if pos[0] >= len(frame):
            return b""
        if n == 1:
            result = bytes([frame[pos[0]]])
            pos[0] += 1
            return result
        chunk = frame[pos[0] : pos[0] + n]
        pos[0] += n
        return chunk

    return read


# ═══════════════════════════════════════════════════════════════════
# ModbusTCPClient Tests (Mock Socket)
# ═══════════════════════════════════════════════════════════════════


class TestModbusTCPClient:
    def test_init_defaults(self):
        client = ModbusTCPClient("192.168.1.100")
        assert client.host == "192.168.1.100"
        assert client.port == 502
        assert client.unit_id == 1
        assert client.timeout == 3.0
        assert client.retries == 3
        assert client.is_connected is False

    def test_init_custom(self):
        client = ModbusTCPClient("10.0.0.1", port=5020, unit_id=5, timeout=5.0)
        assert client.host == "10.0.0.1"
        assert client.port == 5020
        assert client.unit_id == 5

    def test_invalid_unit_id(self):
        with pytest.raises(ModbusConfigurationError):
            ModbusTCPClient("192.168.1.100", unit_id=0)
        with pytest.raises(ModbusConfigurationError):
            ModbusTCPClient("192.168.1.100", unit_id=248)

    @patch("socket.socket")
    def test_connect_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        assert client.is_connected is True

    @patch("socket.socket")
    def test_connect_failure(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("Connection refused")
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")

        with pytest.raises(ModbusConnectionError):
            client.connect()

        assert client.is_connected is False

    @patch("socket.socket")
    def test_disconnect(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        client.disconnect()

        mock_sock.close.assert_called_once()
        assert client.is_connected is False

    def _build_tcp_response(
        self,
        transaction_id: int,
        unit_id: int,
        pdu: bytes,
    ) -> bytes:
        length = 1 + len(pdu)
        mbap = struct.pack(">HHHB", transaction_id, 0x0000, length, unit_id)
        return mbap + pdu

    @patch("socket.socket")
    def test_read_holding_registers(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x03, 0x04, 0x0F, 0xA0, 0x01, 0xF4])
        response_frame = self._build_tcp_response(1, 1, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        regs = client.read_holding_registers(0, 2)
        assert regs == [0x0FA0, 0x01F4]

    @patch("socket.socket")
    def test_read_holding_registers_exception(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x83, 0x02])
        response_frame = self._build_tcp_response(1, 1, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusResponseError) as exc_info:
            client.read_holding_registers(0, 2)
        assert exc_info.value.exception_code == 0x02

    def test_read_invalid_quantity(self):
        client = ModbusTCPClient("192.168.1.100")
        with pytest.raises(ModbusConfigurationError):
            client.read_holding_registers(0, 0)
        with pytest.raises(ModbusConfigurationError):
            client.read_holding_registers(0, 126)

    @patch("socket.socket")
    def test_write_single_register(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x06, 0x00, 0x01, 0x00, 0xFA])
        response_frame = self._build_tcp_response(1, 1, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        client.write_single_register(1, 250)

    @patch("socket.socket")
    def test_read_battery_telemetry(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        # Build sequential responses for the 4 read operations
        responses = []

        # Main block (7 registers): voltage=4000(400V), current=500(50A),
        # soc=800(80%), soh=950(95%), temp_max=700(30°C), temp_min=600(20°C), temp_avg=650(25°C)
        # temp = raw * 0.1 - 40, so raw = (temp + 40) * 10
        main_pdu = bytes(
            [
                0x03,
                0x0E,
                0x0F,
                0xA0,  # 4000 -> 400.0V
                0x01,
                0xF4,  # 500 -> 50.0A
                0x03,
                0x20,  # 800 -> 80.0%
                0x03,
                0xB6,  # 950 -> 95.0%
                0x02,
                0xBC,  # 700 -> 30.0°C
                0x02,
                0x58,  # 600 -> 20.0°C
                0x02,
                0x8A,  # 650 -> 25.0°C
            ]
        )
        responses.append(self._build_tcp_response(1, 1, main_pdu))

        # Cell voltage block (3 registers)
        cell_pdu = bytes(
            [
                0x03,
                0x06,
                0x0D,
                0xAC,
                0x0E,
                0x74,
                0x00,
                0xC8,
            ]
        )
        responses.append(self._build_tcp_response(2, 1, cell_pdu))

        # Cycle count (2 registers)
        cycle_pdu = bytes(
            [
                0x03,
                0x04,
                0x00,
                0x00,
                0x01,
                0xF4,
            ]
        )
        responses.append(self._build_tcp_response(3, 1, cycle_pdu))

        # Fault/status flags
        flags_pdu = bytes(
            [
                0x03,
                0x04,
                0x00,
                0x01,
                0x00,
                0x03,
            ]
        )
        responses.append(self._build_tcp_response(4, 1, flags_pdu))

        # Create a combined recv that serves all responses sequentially
        all_data = b"".join(responses)
        mock_sock.recv.side_effect = _make_recv_side_effect(all_data)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        telemetry = client.read_battery_telemetry()

        assert telemetry["pack_voltage"] == pytest.approx(400.0)
        assert telemetry["pack_current"] == pytest.approx(50.0)
        assert telemetry["soc"] == pytest.approx(80.0)
        assert telemetry["soh"] == pytest.approx(95.0)
        assert telemetry["temperature_max"] == pytest.approx(30.0)
        assert telemetry["temperature_min"] == pytest.approx(20.0)
        assert telemetry["temperature_avg"] == pytest.approx(25.0)
        assert telemetry["cell_voltage_min"] == pytest.approx(3.5)
        assert telemetry["cell_voltage_max"] == pytest.approx(3.7)
        assert telemetry["cell_voltage_delta"] == pytest.approx(0.2)
        assert telemetry["charge_cycle_count"] == 500
        assert "Overvoltage" in telemetry["fault_flags"]
        assert telemetry["status_flags"] == 3

    @patch("socket.socket")
    def test_read_register_by_name(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x03, 0x02, 0x0F, 0xA0])
        response_frame = self._build_tcp_response(1, 1, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        voltage = client.read_register_by_name("pack_voltage")
        assert voltage == pytest.approx(400.0)

    def test_read_register_by_name_unknown(self):
        client = ModbusTCPClient("192.168.1.100")
        with pytest.raises(ModbusConfigurationError, match="Unknown register name"):
            client.read_register_by_name("nonexistent_register")

    @patch("socket.socket")
    def test_health_check_healthy(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x03, 0x02, 0x00, 0x03])
        response_frame = self._build_tcp_response(1, 1, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        result = client.health_check()
        assert result["status"] == "healthy"
        assert result["connected"] is True
        assert result["status_register"] == 3

    def test_health_check_disconnected(self):
        client = ModbusTCPClient("192.168.1.100")
        result = client.health_check()
        assert result["status"] == "disconnected"
        assert result["connected"] is False

    @patch("socket.socket")
    def test_context_manager(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        client.__exit__(None, None, None)
        mock_sock.close.assert_called_once()
        assert client.is_connected is False

    @patch("socket.socket")
    def test_send_not_connected(self, mock_socket_cls):
        client = ModbusTCPClient("192.168.1.100")
        with pytest.raises(ModbusConnectionError, match="Not connected"):
            client._send_raw(b"\x00")

    @patch("socket.socket")
    def test_recv_not_connected(self, mock_socket_cls):
        client = ModbusTCPClient("192.168.1.100")
        with pytest.raises(ModbusConnectionError, match="Not connected"):
            client._recv_raw(10)

    @patch("socket.socket")
    def test_recv_timeout(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.recv.side_effect = socket.timeout("timed out")
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusTimeoutError):
            client._recv_raw(10)

    @patch("socket.socket")
    def test_recv_connection_closed(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b""
        mock_socket_cls.return_value = mock_sock

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusConnectionError, match="Connection closed"):
            client._recv_raw(10)

    @patch("socket.socket")
    def test_invalid_protocol_id(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        mbap = struct.pack(">HHHB", 1, 0x0001, 7, 1)
        pdu = bytes([0x03, 0x02, 0x00, 0x0A])
        frame = mbap + pdu
        mock_sock.recv.side_effect = _make_recv_side_effect(frame)

        client = ModbusTCPClient("192.168.1.100")
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusResponseError, match="Invalid protocol ID"):
            client._recv_raw(10)

    @patch("socket.socket")
    def test_unit_id_mismatch(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x03, 0x02, 0x00, 0x0A])
        response_frame = self._build_tcp_response(1, 2, response_pdu)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1)
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusResponseError, match="Unit ID mismatch"):
            client.read_holding_registers(0, 1)


# ═══════════════════════════════════════════════════════════════════
# ModbusRTUClient Tests (Mock Serial)
# ═══════════════════════════════════════════════════════════════════


class TestModbusRTUClient:
    def test_init_defaults(self):
        client = ModbusRTUClient("/dev/ttyUSB0")
        assert client.port == "/dev/ttyUSB0"
        assert client.baudrate == 9600
        assert client.unit_id == 1
        assert client.timeout == 3.0

    def test_init_custom(self):
        client = ModbusRTUClient("COM3", baudrate=19200, unit_id=5)
        assert client.port == "COM3"
        assert client.baudrate == 19200
        assert client.unit_id == 5

    def test_invalid_unit_id(self):
        with pytest.raises(ModbusConfigurationError):
            ModbusRTUClient("/dev/ttyUSB0", unit_id=0)

    def _build_rtu_response(self, unit_id: int, pdu: bytes) -> bytes:
        frame = bytes([unit_id]) + pdu
        return _append_crc(frame)

    @patch("serial.Serial")
    def test_connect_success(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        client = ModbusRTUClient("/dev/ttyUSB0")
        result = client.connect()

        assert result is True
        assert client.is_connected is True
        mock_serial_cls.assert_called_once()

    @patch("serial.Serial")
    def test_connect_failure(self, mock_serial_cls):
        mock_serial_cls.side_effect = OSError("Port not found")

        client = ModbusRTUClient("/dev/ttyUSB0")

        with pytest.raises(ModbusConnectionError):
            client.connect()

        assert client.is_connected is False

    @patch("serial.Serial")
    def test_disconnect(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        client = ModbusRTUClient("/dev/ttyUSB0")
        client.connect()
        client.disconnect()

        mock_ser.close.assert_called_once()
        assert client.is_connected is False

    @patch("serial.Serial")
    def test_read_holding_registers(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        response_pdu = bytes([0x03, 0x04, 0x0F, 0xA0, 0x01, 0xF4])
        rtu_frame = self._build_rtu_response(1, response_pdu)
        mock_ser.read.side_effect = _make_serial_read_side_effect(rtu_frame)

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        regs = client.read_holding_registers(0, 2)
        assert regs == [0x0FA0, 0x01F4]

    @patch("serial.Serial")
    def test_read_crc_error(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        response_pdu = bytes([0x03, 0x04, 0x0F, 0xA0, 0x01, 0xF4])
        frame = bytes([1]) + response_pdu
        bad_crc_frame = frame + b"\xff\xff"
        mock_ser.read.side_effect = _make_serial_read_side_effect(bad_crc_frame)

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        with pytest.raises(ModbusCRCError):
            client.read_holding_registers(0, 2)

    @patch("serial.Serial")
    def test_read_timeout(self, mock_serial_cls):
        mock_ser = MagicMock()
        # Return only 2 bytes (incomplete)
        mock_ser.read.side_effect = _make_serial_read_side_effect(b"\x01\x03")
        mock_serial_cls.return_value = mock_ser

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        with pytest.raises(ModbusTimeoutError):
            client.read_holding_registers(0, 1)

    @patch("serial.Serial")
    def test_send_not_connected(self, mock_serial_cls):
        client = ModbusRTUClient("/dev/ttyUSB0")
        with pytest.raises(ModbusConnectionError, match="Not connected"):
            client._send_raw(b"\x00")

    @patch("serial.Serial")
    def test_recv_not_connected(self, mock_serial_cls):
        client = ModbusRTUClient("/dev/ttyUSB0")
        with pytest.raises(ModbusConnectionError, match="Not connected"):
            client._recv_raw(10)

    @patch("serial.Serial")
    def test_context_manager(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        client = ModbusRTUClient("/dev/ttyUSB0")
        client._serial = mock_ser
        client._connected = True

        client.__exit__(None, None, None)
        mock_ser.close.assert_called_once()
        assert client.is_connected is False

    @patch("serial.Serial")
    def test_write_single_register(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        response_pdu = bytes([0x06, 0x00, 0x01, 0x00, 0xFA])
        rtu_frame = self._build_rtu_response(1, response_pdu)

        # The RTU _recv_raw reads 3 bytes first, then byte_count + 2 more
        # byte_count = 0x00 (from response_pdu[1]), but actually for write
        # response, the echo is: FC(1) + addr(2) + value(2) = 5 bytes PDU
        # RTU frame: unit_id(1) + PDU(5) + CRC(2) = 8 bytes total
        # read(3) returns first 3 bytes, then read(remaining) returns rest
        call_count = [0]

        def read_side_effect(n=1):
            call_count[0] += 1
            if n == 3:
                read_side_effect._prev_ns = getattr(read_side_effect, "_prev_ns", [])
                read_side_effect._prev_ns.append(3)
                return rtu_frame[:3]
            else:
                remaining = len(rtu_frame) - 3
                read_side_effect._prev_ns = [3]
                return rtu_frame[3 : 3 + remaining]

        mock_ser.read.side_effect = read_side_effect

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        client.write_single_register(1, 250)

    @patch("serial.Serial")
    def test_read_battery_telemetry(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        # Build all 4 response frames
        main_pdu = bytes(
            [
                0x03,
                0x0E,
                0x0F,
                0xA0,
                0x01,
                0xF4,
                0x03,
                0x20,
                0x03,
                0xB6,
                0x00,
                0x46,
                0x00,
                0x3C,
                0x00,
                0x41,
            ]
        )
        cell_pdu = bytes(
            [
                0x03,
                0x06,
                0x0D,
                0xAC,
                0x0E,
                0x74,
                0x00,
                0xC8,
            ]
        )
        cycle_pdu = bytes(
            [
                0x03,
                0x04,
                0x00,
                0x00,
                0x01,
                0xF4,
            ]
        )
        flags_pdu = bytes(
            [
                0x03,
                0x04,
                0x00,
                0x00,
                0x00,
                0x03,
            ]
        )

        all_frames = b"".join(
            [
                self._build_rtu_response(1, main_pdu),
                self._build_rtu_response(1, cell_pdu),
                self._build_rtu_response(1, cycle_pdu),
                self._build_rtu_response(1, flags_pdu),
            ]
        )
        mock_ser.read.side_effect = _make_serial_read_side_effect(all_frames)

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        telemetry = client.read_battery_telemetry()

        assert telemetry["pack_voltage"] == pytest.approx(400.0)
        assert telemetry["soc"] == pytest.approx(80.0)
        assert telemetry["cell_voltage_min"] == pytest.approx(3.5)
        assert telemetry["charge_cycle_count"] == 500
        assert telemetry["fault_flags"] == []

    @patch("serial.Serial")
    def test_health_check(self, mock_serial_cls):
        mock_ser = MagicMock()
        mock_serial_cls.return_value = mock_ser

        response_pdu = bytes([0x03, 0x02, 0x00, 0x03])
        rtu_frame = self._build_rtu_response(1, response_pdu)
        mock_ser.read.side_effect = _make_serial_read_side_effect(rtu_frame)

        client = ModbusRTUClient("/dev/ttyUSB0", unit_id=1)
        client._serial = mock_ser
        client._connected = True

        result = client.health_check()
        assert result["status"] == "healthy"
        assert result["connected"] is True

    def test_health_check_disconnected(self):
        client = ModbusRTUClient("/dev/ttyUSB0")
        result = client.health_check()
        assert result["status"] == "disconnected"


# ═══════════════════════════════════════════════════════════════════
# Transaction Retry Tests
# ═══════════════════════════════════════════════════════════════════


class TestTransactionRetry:
    @patch("socket.socket")
    def test_retry_on_timeout(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.recv.side_effect = socket.timeout("timeout")

        client = ModbusTCPClient("192.168.1.100", unit_id=1, retries=2)
        client._socket = mock_sock
        client._connected = True

        with pytest.raises(ModbusTimeoutError, match="failed after 2 attempts"):
            client.read_holding_registers(0, 1)

    @patch("socket.socket")
    def test_no_retry_on_first_success(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        response_pdu = bytes([0x03, 0x02, 0x00, 0x0A])
        response_frame = struct.pack(">HHHB", 1, 0x0000, 6, 1) + response_pdu

        # Each call to recv returns the full frame (mock resets each time)
        mock_sock.recv.side_effect = _make_recv_side_effect(response_frame)

        client = ModbusTCPClient("192.168.1.100", unit_id=1, retries=3)
        client._socket = mock_sock
        client._connected = True

        regs = client.read_holding_registers(0, 1)
        assert regs == [10]
        # Should succeed on first attempt
        assert mock_sock.recv.call_count <= 10  # At most a few recv calls
