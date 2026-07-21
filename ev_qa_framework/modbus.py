"""
Modbus Protocol Module for BMS Communication.

Supports:

- Modbus TCP (port 502) for Ethernet-connected BMS
- Modbus RTU (RS-485/RS-232) for serial BMS connections
- Function codes: 01 (Read Coils), 02 (Read Discrete Inputs),
  03 (Read Holding Registers), 04 (Read Input Registers),
  05 (Write Single Coil), 06 (Write Single Register),
  15 (Write Multiple Coils), 16 (Write Multiple Registers)
- CRC-16 validation for RTU frames
- Unit ID routing for multi-device RS-485 networks
- BMS register map for common battery telemetry

Architecture:
    ModbusClient (base)
    ├── ModbusTCPClient  — TCP socket transport
    └── ModbusRTUClient  — serial (RS-485) transport

Usage:
    # Modbus TCP
    client = ModbusTCPClient("192.168.1.100", port=502, unit_id=1)
    client.connect()
    registers = client.read_holding_registers(0, 10)
    voltage = registers[0] / 10.0  # 0.1V per LSB
    client.disconnect()

    # Modbus RTU
    client = ModbusRTUClient("/dev/ttyUSB0", baudrate=9600, unit_id=1)
    client.connect()
    telemetry = client.read_battery_telemetry()
    client.disconnect()
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# ── Custom Exception Hierarchy ──────────────────────────────────────────────


class ModbusError(Exception):
    """Base exception for all Modbus errors."""


class ModbusConnectionError(ModbusError):
    """Raised when a Modbus connection fails."""


class ModbusTimeoutError(ModbusError):
    """Raised when a Modbus operation times out."""


class ModbusCRCError(ModbusError):
    """Raised when RTU frame CRC validation fails."""


class ModbusResponseError(ModbusError):
    """Raised when the BMS returns an exception response."""

    def __init__(self, message: str, exception_code: int | None = None):
        super().__init__(message)
        self.exception_code = exception_code


class ModbusConfigurationError(ModbusError):
    """Raised for invalid configuration parameters."""


# ── Modbus Function Codes ───────────────────────────────────────────────────


class FunctionCode(IntEnum):
    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10


# Modbus Exception Codes (offset by 0x80 in response)
class ModbusExceptionCode(IntEnum):
    ILLEGAL_FUNCTION = 0x01
    ILLEGAL_DATA_ADDRESS = 0x02
    ILLEGAL_DATA_VALUE = 0x03
    SLAVE_DEVICE_FAILURE = 0x04
    ACKNOWLEDGE = 0x05
    SLAVE_DEVICE_BUSY = 0x06
    MEMORY_PARITY_ERROR = 0x08
    GATEWAY_PATH_UNAVAILABLE = 0x0A
    GATEWAY_TARGET_FAILED = 0x0B


# ── BMS Modbus Register Map ────────────────────────────────────────────────
# Standard register layout for BMS Modbus devices.
# All registers are 16-bit unsigned unless noted.

BMS_REGISTER_MAP: dict[str, dict[str, Any]] = {
    "pack_voltage": {
        "address": 0x0000,
        "count": 1,
        "scale": 0.1,
        "unit": "V",
        "description": "Total pack voltage",
    },
    "pack_current": {
        "address": 0x0001,
        "count": 1,
        "scale": 0.1,
        "unit": "A",
        "description": "Pack current (signed, offset -32768)",
        "signed": True,
    },
    "soc": {
        "address": 0x0002,
        "count": 1,
        "scale": 0.1,
        "unit": "%",
        "description": "State of Charge",
    },
    "soh": {
        "address": 0x0003,
        "count": 1,
        "scale": 0.1,
        "unit": "%",
        "description": "State of Health",
    },
    "temperature_max": {
        "address": 0x0004,
        "count": 1,
        "scale": 0.1,
        "unit": "°C",
        "description": "Maximum cell temperature (offset -40)",
        "offset": -40,
    },
    "temperature_min": {
        "address": 0x0005,
        "count": 1,
        "scale": 0.1,
        "unit": "°C",
        "description": "Minimum cell temperature (offset -40)",
        "offset": -40,
    },
    "temperature_avg": {
        "address": 0x0006,
        "count": 1,
        "scale": 0.1,
        "unit": "°C",
        "description": "Average cell temperature (offset -40)",
        "offset": -40,
    },
    "cell_voltage_min": {
        "address": 0x0010,
        "count": 1,
        "scale": 0.001,
        "unit": "V",
        "description": "Minimum cell voltage",
    },
    "cell_voltage_max": {
        "address": 0x0011,
        "count": 1,
        "scale": 0.001,
        "unit": "V",
        "description": "Maximum cell voltage",
    },
    "cell_voltage_delta": {
        "address": 0x0012,
        "count": 1,
        "scale": 0.001,
        "unit": "V",
        "description": "Cell voltage spread (max - min)",
    },
    "charge_cycle_count": {
        "address": 0x0020,
        "count": 2,
        "scale": 1,
        "unit": "cycles",
        "description": "Total charge cycles (32-bit)",
        "wide": True,
    },
    "fault_flags": {
        "address": 0x0030,
        "count": 1,
        "scale": 1,
        "unit": "bitmap",
        "description": "Fault status bitmap (see FAULT_FLAGS)",
    },
    "status_flags": {
        "address": 0x0031,
        "count": 1,
        "scale": 1,
        "unit": "bitmap",
        "description": "Operational status bitmap",
    },
}

# Fault flag bit positions
FAULT_FLAGS = {
    0: "Overvoltage",
    1: "Undervoltage",
    2: "Overcurrent charge",
    3: "Overcurrent discharge",
    4: "Overtemperature",
    5: "Undertemperature",
    6: "Cell imbalance",
    7: "Communication failure",
    8: "Short circuit",
    9: "Thermal runaway",
    10: "BMS hardware fault",
    11: "Sensor fault",
}


# ── CRC-16 (Modbus) ────────────────────────────────────────────────────────


def _crc16_modbus(data: bytes) -> int:
    """Compute Modbus CRC-16 (polynomial 0x8001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _validate_crc(frame: bytes) -> bool:
    """Validate CRC of an RTU response frame."""
    if len(frame) < 4:
        return False
    received_crc = struct.unpack("<H", frame[-2:])[0]
    computed_crc = _crc16_modbus(frame[:-2])
    return received_crc == computed_crc


def _append_crc(frame: bytes) -> bytes:
    """Append CRC-16 to an RTU frame."""
    crc = _crc16_modbus(frame)
    return frame + struct.pack("<H", crc)


# ── PDU / Frame Builders ───────────────────────────────────────────────────


def _build_read_pdu(function_code: int, start_address: int, quantity: int) -> bytes:
    """Build a read request PDU."""
    return struct.pack(">BHH", function_code, start_address, quantity)


def _build_write_single_register_pdu(register_address: int, value: int) -> bytes:
    """Build a Write Single Register (FC 06) PDU."""
    return struct.pack(">BHH", FunctionCode.WRITE_SINGLE_REGISTER, register_address, value)


def _build_write_multiple_registers_pdu(start_address: int, values: list[int]) -> bytes:
    """Build a Write Multiple Registers (FC 16) PDU."""
    quantity = len(values)
    byte_count = quantity * 2
    pdu = struct.pack(
        ">BHHB",
        FunctionCode.WRITE_MULTIPLE_REGISTERS,
        start_address,
        quantity,
        byte_count,
    )
    for v in values:
        pdu += struct.pack(">H", v & 0xFFFF)
    return pdu


def _build_tcp_mbap(transaction_id: int, length: int, unit_id: int) -> bytes:
    """Build Modbus TCP MBAP header."""
    return struct.pack(">HHHB", transaction_id, 0x0000, length, unit_id)


def _parse_read_response(pdu: bytes, function_code: int, expected_count: int) -> list[int]:
    """Parse a read response PDU, returning register values."""
    if len(pdu) < 1:
        raise ModbusResponseError("Empty PDU in response")

    received_fc = pdu[0]

    # Exception response
    if received_fc & 0x80:
        if len(pdu) >= 2:
            exc_code = pdu[1]
            raise ModbusResponseError(
                f"Modbus exception: {ModbusExceptionCode(exc_code).name} (0x{exc_code:02X})",
                exception_code=exc_code,
            )
        raise ModbusResponseError("Modbus exception with no code")

    if received_fc != function_code:
        raise ModbusResponseError(
            f"Unexpected function code: 0x{received_fc:02X} (expected 0x{function_code:02X})"
        )

    byte_count = pdu[1]
    data = pdu[2:]

    if byte_count != expected_count * 2:
        raise ModbusResponseError(
            f"Unexpected byte count: {byte_count} (expected {expected_count * 2})"
        )

    if len(data) < byte_count:
        raise ModbusResponseError(f"Response too short: {len(data)} bytes, expected {byte_count}")

    registers = []
    for i in range(0, byte_count, 2):
        registers.append(struct.unpack(">H", data[i : i + 2])[0])

    return registers


# ── Base Modbus Client ──────────────────────────────────────────────────────


class ModbusClient(ABC):
    """Abstract base class for Modbus clients.

    Provides common BMS telemetry reading logic. Subclasses implement
    the transport layer (TCP or RTU).
    """

    def __init__(
        self,
        unit_id: int = 1,
        timeout: float = 3.0,
        retries: int = 3,
    ):
        if not 1 <= unit_id <= 247:
            raise ModbusConfigurationError(f"unit_id must be 1-247, got {unit_id}")
        self.unit_id = unit_id
        self.timeout = timeout
        self.retries = retries
        self._connected = False
        self._transaction_id = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the Modbus device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    def _send_raw(self, data: bytes) -> None:
        """Send raw bytes to the transport."""

    @abstractmethod
    def _recv_raw(self, expected_min_length: int) -> bytes:
        """Receive raw bytes from the transport."""

    def _next_tid(self) -> int:
        """Get next transaction ID (wraps at 65535)."""
        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        return self._transaction_id

    def _execute_transaction(self, request_frame: bytes) -> bytes:
        """Send a request and receive the response, with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                self._send_raw(request_frame)
                response = self._recv_raw(4)
                return response
            except (ModbusTimeoutError, ModbusConnectionError, OSError) as e:
                last_error = e
                logger.warning(
                    "Modbus transaction attempt %d/%d failed: %s",
                    attempt + 1,
                    self.retries,
                    e,
                )
                if attempt < self.retries - 1:
                    time.sleep(0.1 * (attempt + 1))

        raise ModbusTimeoutError(f"Transaction failed after {self.retries} attempts: {last_error}")

    # ── Standard Modbus Operations ──────────────────────────────────────

    def read_holding_registers(self, start_address: int, quantity: int) -> list[int]:
        """Read holding registers (Function Code 03).

        Args:
            start_address: First register address (0-based).
            quantity: Number of registers to read (1-125).

        Returns:
            List of register values.
        """
        if not 1 <= quantity <= 125:
            raise ModbusConfigurationError(f"quantity must be 1-125, got {quantity}")
        pdu = _build_read_pdu(FunctionCode.READ_HOLDING_REGISTERS, start_address, quantity)
        return self._read_registers(pdu, FunctionCode.READ_HOLDING_REGISTERS, quantity)

    def read_input_registers(self, start_address: int, quantity: int) -> list[int]:
        """Read input registers (Function Code 04).

        Args:
            start_address: First register address (0-based).
            quantity: Number of registers to read (1-123).

        Returns:
            List of register values.
        """
        if not 1 <= quantity <= 123:
            raise ModbusConfigurationError(f"quantity must be 1-123, got {quantity}")
        pdu = _build_read_pdu(FunctionCode.READ_INPUT_REGISTERS, start_address, quantity)
        return self._read_registers(pdu, FunctionCode.READ_INPUT_REGISTERS, quantity)

    def write_single_register(self, register_address: int, value: int) -> None:
        """Write a single holding register (Function Code 06).

        Args:
            register_address: Register address to write.
            value: 16-bit value to write.
        """
        pdu = _build_write_single_register_pdu(register_address, value)
        self._write_register(pdu, FunctionCode.WRITE_SINGLE_REGISTER)

    def write_multiple_registers(self, start_address: int, values: list[int]) -> None:
        """Write multiple holding registers (Function Code 16).

        Args:
            start_address: First register address.
            values: List of 16-bit values to write.
        """
        if not 1 <= len(values) <= 123:
            raise ModbusConfigurationError(f"values length must be 1-123, got {len(values)}")
        pdu = _build_write_multiple_registers_pdu(start_address, values)
        self._write_registers(pdu, FunctionCode.WRITE_MULTIPLE_REGISTERS)

    # ── BMS-specific Operations ─────────────────────────────────────────

    def read_battery_telemetry(self) -> dict[str, Any]:
        """Read all battery telemetry from the BMS using the standard register map.

        Returns:
            dict with keys matching BMS_REGISTER_MAP, values are scaled
            engineering units.
        """
        telemetry: dict[str, Any] = {}

        # Read main telemetry block (registers 0x0000-0x0006, 7 registers)
        main_block = self.read_holding_registers(0x0000, 7)
        telemetry["pack_voltage"] = main_block[0] * 0.1
        # Signed current with offset
        raw_current = main_block[1]
        if raw_current >= 0x8000:
            raw_current -= 0x10000
        telemetry["pack_current"] = raw_current * 0.1
        telemetry["soc"] = main_block[2] * 0.1
        telemetry["soh"] = main_block[3] * 0.1
        telemetry["temperature_max"] = main_block[4] * 0.1 - 40.0
        telemetry["temperature_min"] = main_block[5] * 0.1 - 40.0
        telemetry["temperature_avg"] = main_block[6] * 0.1 - 40.0

        # Read cell voltage block (registers 0x0010-0x0012, 3 registers)
        cell_block = self.read_holding_registers(0x0010, 3)
        telemetry["cell_voltage_min"] = cell_block[0] * 0.001
        telemetry["cell_voltage_max"] = cell_block[1] * 0.001
        telemetry["cell_voltage_delta"] = cell_block[2] * 0.001

        # Read cycle count (registers 0x0020-0x0021, 2 registers = 32-bit)
        cycle_block = self.read_holding_registers(0x0020, 2)
        telemetry["charge_cycle_count"] = (cycle_block[0] << 16) | cycle_block[1]

        # Read fault and status flags
        flags = self.read_holding_registers(0x0030, 2)
        telemetry["fault_flags"] = self._decode_fault_flags(flags[0])
        telemetry["status_flags"] = flags[1]

        return telemetry

    def read_register_by_name(self, name: str) -> Any:
        """Read a single register by its BMS_REGISTER_MAP name.

        Args:
            name: Register name from BMS_REGISTER_MAP.

        Returns:
            Scaled value.
        """
        if name not in BMS_REGISTER_MAP:
            raise ModbusConfigurationError(
                f"Unknown register name: {name}. Available: {list(BMS_REGISTER_MAP.keys())}"
            )
        reg = BMS_REGISTER_MAP[name]
        raw = self.read_holding_registers(reg["address"], reg["count"])

        if reg.get("wide"):
            value = (raw[0] << 16) | raw[1]
        elif reg.get("signed"):
            value = raw[0]
            if value >= 0x8000:
                value -= 0x10000
        else:
            value = raw[0]

        scaled = value * reg["scale"]
        if "offset" in reg:
            scaled += reg["offset"]
        return scaled

    def health_check(self) -> dict[str, Any]:
        """Perform a health check by reading the status register.

        Returns:
            dict with connection status and device info.
        """
        result: dict[str, Any] = {
            "connected": self._connected,
            "unit_id": self.unit_id,
        }
        if not self._connected:
            result["status"] = "disconnected"
            return result

        try:
            regs = self.read_holding_registers(0x0031, 1)
            result["status"] = "healthy"
            result["status_register"] = regs[0]
        except ModbusError as e:
            result["status"] = "degraded"
            result["error"] = str(e)

        return result

    # ── Internal Methods ────────────────────────────────────────────────

    def _read_registers(self, pdu: bytes, function_code: int, quantity: int) -> list[int]:
        """Execute a read transaction."""
        frame = self._build_frame(pdu)
        response_frame = self._execute_transaction(frame)
        response_pdu = self._extract_pdu(response_frame, function_code)
        return _parse_read_response(response_pdu, function_code, quantity)

    def _write_register(self, pdu: bytes, function_code: int) -> None:
        """Execute a write single register transaction."""
        frame = self._build_frame(pdu)
        response_frame = self._execute_transaction(frame)
        response_pdu = self._extract_pdu(response_frame, function_code)
        # Echo response: check address and value match
        if len(response_pdu) < 5:
            raise ModbusResponseError("Write response too short")
        if response_pdu[0] & 0x80:
            exc = response_pdu[1] if len(response_pdu) > 1 else 0
            raise ModbusResponseError(
                f"Write exception: code 0x{exc:02X}",
                exception_code=exc,
            )

    def _write_registers(self, pdu: bytes, function_code: int) -> None:
        """Execute a write multiple registers transaction."""
        frame = self._build_frame(pdu)
        response_frame = self._execute_transaction(frame)
        response_pdu = self._extract_pdu(response_frame, function_code)
        if len(response_pdu) < 5:
            raise ModbusResponseError("Write multiple response too short")
        if response_pdu[0] & 0x80:
            exc = response_pdu[1] if len(response_pdu) > 1 else 0
            raise ModbusResponseError(
                f"Write multiple exception: code 0x{exc:02X}",
                exception_code=exc,
            )

    @abstractmethod
    def _build_frame(self, pdu: bytes) -> bytes:
        """Wrap PDU in transport-specific frame."""

    @abstractmethod
    def _extract_pdu(self, frame: bytes, expected_fc: int) -> bytes:
        """Extract PDU from transport-specific frame."""

    @staticmethod
    def _decode_fault_flags(bitmap: int) -> list[str]:
        """Decode fault flag bitmap into list of fault names."""
        faults = []
        for bit, name in FAULT_FLAGS.items():
            if bitmap & (1 << bit):
                faults.append(name)
        return faults

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ── Modbus TCP Client ───────────────────────────────────────────────────────


class ModbusTCPClient(ModbusClient):
    """Modbus TCP client for Ethernet-connected BMS devices.

    Uses standard Modbus TCP framing (MBAP header + PDU) over
    a TCP socket connection to port 502 (default).

    Args:
        host: IP address or hostname of the BMS.
        port: TCP port (default 502).
        unit_id: Modbus unit/slave ID (1-247).
        timeout: Socket timeout in seconds.
        retries: Number of retry attempts.
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        timeout: float = 3.0,
        retries: int = 3,
    ):
        super().__init__(unit_id=unit_id, timeout=timeout, retries=retries)
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None

    def connect(self) -> bool:
        """Connect to the Modbus TCP device."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self._connected = True
            logger.info("Modbus TCP connected to %s:%d", self.host, self.port)
            return True
        except OSError as e:
            self._connected = False
            self._socket = None
            raise ModbusConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}") from e

    def disconnect(self) -> None:
        """Close the TCP connection."""
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self._connected = False
        logger.info("Modbus TCP disconnected from %s:%d", self.host, self.port)

    def _send_raw(self, data: bytes) -> None:
        """Send raw TCP data."""
        if not self._socket:
            raise ModbusConnectionError("Not connected")
        try:
            self._socket.sendall(data)
        except OSError as e:
            self._connected = False
            raise ModbusConnectionError(f"Send failed: {e}") from e

    def _recv_raw(self, expected_min_length: int) -> bytes:
        """Receive raw TCP data."""
        if not self._socket:
            raise ModbusConnectionError("Not connected")
        try:
            # Read MBAP header (7 bytes) first to get length
            header = b""
            while len(header) < 7:
                chunk = self._socket.recv(7 - len(header))
                if not chunk:
                    raise ModbusConnectionError("Connection closed by remote")
                header += chunk

            tid, pid, length, uid = struct.unpack(">HHHB", header)
            if pid != 0x0000:
                raise ModbusResponseError(f"Invalid protocol ID: 0x{pid:04X}")

            # Read remaining PDU bytes
            remaining = length - 1  # subtract unit_id byte
            body = b""
            while len(body) < remaining:
                chunk = self._socket.recv(remaining - len(body))
                if not chunk:
                    raise ModbusConnectionError("Connection closed during PDU")
                body += chunk

            return header + body
        except socket.timeout as e:
            raise ModbusTimeoutError(f"Receive timeout: {e}") from e
        except OSError as e:
            self._connected = False
            raise ModbusConnectionError(f"Receive failed: {e}") from e

    def _build_frame(self, pdu: bytes) -> bytes:
        """Build Modbus TCP frame (MBAP + PDU)."""
        tid = self._next_tid()
        length = 1 + len(pdu)  # unit_id + PDU
        mbap = _build_tcp_mbap(tid, length, self.unit_id)
        return mbap + pdu

    def _extract_pdu(self, frame: bytes, expected_fc: int) -> bytes:
        """Extract PDU from Modbus TCP frame."""
        if len(frame) < 8:
            raise ModbusResponseError(f"TCP frame too short: {len(frame)} bytes")
        tid, pid, length, uid = struct.unpack(">HHHB", frame[:7])
        if uid != self.unit_id:
            raise ModbusResponseError(f"Unit ID mismatch: got {uid}, expected {self.unit_id}")
        pdu = frame[7:]
        if len(pdu) != length - 1:
            raise ModbusResponseError(f"PDU length mismatch: got {len(pdu)}, expected {length - 1}")
        return pdu


# ── Modbus RTU Client ───────────────────────────────────────────────────────


class ModbusRTUClient(ModbusClient):
    """Modbus RTU client for RS-485/RS-232 connected BMS devices.

    Uses standard Modbus RTU framing (address + PDU + CRC-16) over
    a serial port connection.

    Args:
        port: Serial port device path (e.g. '/dev/ttyUSB0', 'COM3').
        baudrate: Serial baud rate (default 9600).
        bytesize: Data bits (default 8).
        parity: Parity ('N', 'E', 'O').
        stopbits: Stop bits (default 1).
        unit_id: Modbus unit/slave ID (1-247).
        timeout: Serial read timeout in seconds.
        retries: Number of retry attempts.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        unit_id: int = 1,
        timeout: float = 3.0,
        retries: int = 3,
    ):
        super().__init__(unit_id=unit_id, timeout=timeout, retries=retries)
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self._serial = None

    def connect(self) -> bool:
        """Connect to the Modbus RTU device via serial port."""
        try:
            import serial  # type: ignore

            parity_map = {"N": "N", "E": "E", "O": "O"}
            ser_parity = parity_map.get(self.parity, "N")

            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=ser_parity,
                stopbits=self.stopbits,
                timeout=self.timeout,
            )
            self._connected = True
            logger.info(
                "Modbus RTU connected to %s @ %d baud",
                self.port,
                self.baudrate,
            )
            return True
        except Exception as e:
            self._connected = False
            self._serial = None
            raise ModbusConnectionError(f"Failed to open serial port {self.port}: {e}") from e

    def disconnect(self) -> None:
        """Close the serial connection."""
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False
        logger.info("Modbus RTU disconnected from %s", self.port)

    def _send_raw(self, data: bytes) -> None:
        """Send raw RTU frame over serial."""
        if not self._serial:
            raise ModbusConnectionError("Not connected")
        try:
            # RTU inter-frame delay (3.5 character times)
            char_time = (
                self.bytesize + (1 if self.parity != "N" else 0) + self.stopbits + 1
            ) / self.baudrate
            time.sleep(char_time * 3.5)
            self._serial.write(data)
            self._serial.flush()
        except Exception as e:
            self._connected = False
            raise ModbusConnectionError(f"Serial send failed: {e}") from e

    def _recv_raw(self, expected_min_length: int) -> bytes:
        """Receive raw RTU frame from serial."""
        if not self._serial:
            raise ModbusConnectionError("Not connected")
        try:
            # Read slave ID + function code + third byte (byte count OR exception code)
            header = self._serial.read(3)
            if len(header) < 3:
                raise ModbusTimeoutError(f"RTU response too short: {len(header)} bytes")

            # Check if exception response: high bit set in function code (header[1])
            fc = header[1]
            is_exception = (fc & 0x80) != 0

            if is_exception:
                # Exception response format: [addr][fc|0x80][exception_code][CRC]
                # third byte is exception code, then 2 bytes CRC
                rest = self._serial.read(2)  # CRC
                frame = header + rest

                # Validate CRC for exception responses too
                if not _validate_crc(frame):
                    raise ModbusCRCError("CRC mismatch in RTU exception response")

                return frame

            # Normal response: third byte is byte count, then data + 2 CRC
            byte_count = header[2]
            remaining = byte_count + 2
            body = self._serial.read(remaining)
            if len(body) < remaining:
                raise ModbusTimeoutError(
                    f"RTU response incomplete: got {len(body) + 3} bytes, expected {remaining + 3}"
                )

            frame = header + body

            # Validate CRC
            if not _validate_crc(frame):
                raise ModbusCRCError("CRC mismatch in RTU response")

            return frame
        except (ModbusTimeoutError, ModbusCRCError):
            raise
        except Exception as e:
            self._connected = False
            raise ModbusConnectionError(f"Serial receive failed: {e}") from e

    def _build_frame(self, pdu: bytes) -> bytes:
        """Build Modbus RTU frame (unit_id + PDU + CRC)."""
        frame = struct.pack("B", self.unit_id) + pdu
        return _append_crc(frame)

    def _extract_pdu(self, frame: bytes, expected_fc: int) -> bytes:
        """Extract PDU from Modbus RTU frame (strip unit_id and CRC)."""
        if len(frame) < 4:
            raise ModbusResponseError(f"RTU frame too short: {len(frame)} bytes")
        # Strip unit_id (first byte) and CRC (last 2 bytes)
        return frame[1:-2]
