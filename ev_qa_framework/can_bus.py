"""
CAN Bus Module: Hardware-level battery telemetry simulation and reception.

Supports:
- Virtual CAN (vcan0) for simulation/development
- Real socketCAN hardware interfaces (can0, can1, ...)
- OBD-II ELM327 adapters over serial or Bluetooth
- Automatic detection of available CAN interfaces
- Error handling with reconnection, bus-off monitoring, and timeouts
"""
from __future__ import annotations

import logging
import os
import platform
import random
import re
import threading
import time
from typing import Any

import can

from .dbc_parser import DBCParser, battery_dbc_content

logger = logging.getLogger(__name__)

# ── Custom Exception Hierarchy ──────────────────────────────────────────────


class HardwareCANError(Exception):
    """Base exception for all hardware CAN errors."""


class CANConnectionError(HardwareCANError):
    """Raised when a CAN bus connection fails."""


class CANBusOffError(HardwareCANError):
    """Raised when the CAN controller enters bus-off state."""


class CANTimeoutError(HardwareCANError):
    """Raised when a CAN operation times out on hardware."""


class CANHardwareNotFoundError(HardwareCANError):
    """Raised when no suitable CAN hardware interface is found."""


class OBD2ConnectionError(HardwareCANError):
    """Raised when OBD-II adapter connection fails."""


class OBD2ProtocolError(HardwareCANError):
    """Raised on unexpected OBD-II protocol responses."""


# ── CAN Interface Detection ────────────────────────────────────────────────


def detect_can_interfaces() -> list[dict[str, str]]:
    """Detect available CAN interfaces on the system.

    Returns a list of dicts with keys: 'name', 'type' ('virtual' or 'hardware'),
    'up' (bool), and 'driver' (if detectable).

    Works on Linux by inspecting /sys/class/net/can* and running ``ip link``.
    On non-Linux systems returns an empty list.
    """
    interfaces: list[dict[str, str]] = []

    if platform.system() != "Linux":
        return interfaces

    # Check /sys/class/net for CAN interfaces
    net_dir = "/sys/class/net"
    if not os.path.isdir(net_dir):
        return interfaces

    for iface in os.listdir(net_dir):
        if not iface.startswith("can") and not iface.startswith("vcan"):
            continue

        iface_path = os.path.join(net_dir, iface)
        if not os.path.isdir(iface_path):
            continue

        # Determine type
        is_vcan = iface.startswith("vcan")
        iface_type = "virtual" if is_vcan else "hardware"

        # Check if interface is up
        operstate_path = os.path.join(iface_path, "operstate")
        is_up = False
        if os.path.isfile(operstate_path):
            with open(operstate_path) as f:
                is_up = f.read().strip() == "up"

        # Try to get driver information for hardware CAN interfaces
        driver = None
        if not is_vcan:
            device_path = os.path.join(iface_path, "device")
            if os.path.isdir(device_path):
                uevent_path = os.path.join(device_path, "uevent")
                if os.path.isfile(uevent_path):
                    try:
                        with open(uevent_path) as f:
                            for line in f:
                                if line.startswith("DRIVER="):
                                    driver = line.split("=", 1)[1].strip()
                    except OSError:
                        pass

        interfaces.append({
            "name": iface,
            "type": iface_type,
            "up": is_up,
            "driver": driver,
        })

    return interfaces


def find_hardware_can_interfaces() -> list[dict[str, str]]:
    """Return only physical (non-virtual) CAN interfaces."""
    return [i for i in detect_can_interfaces() if i["type"] == "hardware"]


def find_available_can_channel(
    prefer_hardware: bool = True,
) -> str:
    """Find an available CAN channel suitable for connection.

    Args:
        prefer_hardware: If True, prefer real CAN hardware over virtual.

    Returns:
        Channel name string (e.g. 'can0', 'vcan0').

    Raises:
        CANHardwareNotFoundError: If no suitable interface is found.
    """
    all_ifaces = detect_can_interfaces()

    if not all_ifaces:
        # Try common default names
        for name in ["can0", "vcan0"]:
            try:
                bus = can.interface.Bus(channel=name, interface="socketcan")
                bus.shutdown()
                return name
            except (can.CanError, OSError, ValueError):
                pass
        raise CANHardwareNotFoundError(
            "No CAN interfaces found. Ensure socketCAN is loaded "
            "(sudo modprobe can && sudo modprobe vcan)"
        )

    hardware = [i for i in all_ifaces if i["type"] == "hardware"]
    virtual = [i for i in all_ifaces if i["type"] == "virtual"]

    if prefer_hardware and hardware:
        up_hw = [i for i in hardware if i["up"]]
        if up_hw:
            return up_hw[0]["name"]
        # Return first hardware even if down — caller can bring it up
        return hardware[0]["name"]

    if virtual:
        up_v = [i for i in virtual if i["up"]]
        if up_v:
            return up_v[0]["name"]
        return virtual[0]["name"]

    if hardware:
        return hardware[0]["name"]

    raise CANHardwareNotFoundError(
        "No CAN interfaces found after full scan."
    )


# ── CAN Hardware Interface Manager ─────────────────────────────────────────


class CANHardwareInterface:
    """Manages connection to real CAN hardware via socketCAN.

    Handles:
    - Connecting to socketCAN interfaces (can0, can1, etc.)
    - Health monitoring and bus-off detection
    - Automatic reconnection on failure
    - Graceful degradation to virtual/log-only mode

    Args:
        channel: CAN interface name (e.g. 'can0', 'vcan0'). Auto-detect if None.
        bitrate: Bus bitrate in bps (only relevant for real hardware bring-up).
        auto_reconnect: If True, attempt reconnection on bus failure.
        max_reconnect_attempts: Max reconnection attempts (0 = infinite).
        reconnect_delay: Seconds between reconnection attempts.
        timeout: Default timeout for send/recv operations (seconds).
    """

    def __init__(
        self,
        channel: str | None = None,
        bitrate: int = 500000,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0,
        timeout: float = 2.0,
    ):
        self.channel = channel
        self.bitrate = bitrate
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout

        self.bus: can.interface.Bus | None = None
        self._is_hardware: bool = False
        self._reconnect_count: int = 0
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    @property
    def is_hardware(self) -> bool:
        return self._is_hardware

    def connect(self) -> bool:
        """Connect to the CAN interface.

        Returns True on success, False on failure (with auto_reconnect)
        or raises on failure (without auto_reconnect).

        For socketCAN on Linux, tries 'socketcan_native' first (faster),
        then falls back to 'socketcan' (compatibility).
        """
        channel = self.channel
        if channel is None:
            try:
                channel = find_available_can_channel(prefer_hardware=True)
                logger.info("Auto-detected CAN channel: %s", channel)
            except CANHardwareNotFoundError as e:
                logger.warning("No CAN interface found: %s", e)
                if not self.auto_reconnect:
                    raise
                return False

        self.channel = channel
        self._is_hardware = not channel.startswith("vcan")

        # Try to connect
        last_error: Exception | None = None
        backends = []

        if platform.system() == "Linux":
            backends = ["socketcan_native", "socketcan"]
        else:
            backends = ["serial", "pcan", "ixxat", "neovi", "usb2can", "virtual"]

        for backend in backends:
            try:
                self.bus = can.interface.Bus(
                    channel=channel,
                    interface=backend,
                    bitrate=self.bitrate,
                    receive_own_messages=False,
                )
                logger.info(
                    "Connected to CAN interface %s via %s backend",
                    channel, backend,
                )
                self._reconnect_count = 0
                return True
            except (can.CanError, OSError, ValueError, IndexError) as e:
                last_error = e
                continue

        # All backends failed
        self.bus = None
        msg = f"Failed to connect to CAN interface {channel}: {last_error}"
        logger.error(msg)

        if self.auto_reconnect:
            return False

        raise CANConnectionError(msg)

    def disconnect(self):
        """Disconnect from the CAN interface."""
        with self._lock:
            if self.bus:
                try:
                    self.bus.shutdown()
                except Exception:
                    pass
                self.bus = None
            self._is_hardware = False

    def send(
        self,
        msg: can.Message,
        timeout: float | None = None,
    ) -> bool:
        """Send a CAN message with error handling.

        Returns True on success, False on transient failure.
        Raises CANBusOffError if the controller enters bus-off.
        """
        if not self._ensure_connected():
            return False

        t = timeout if timeout is not None else self.timeout
        try:
            self.bus.send(msg, timeout=t)
            return True
        except can.CanError as e:
            error_str = str(e).lower()
            if "bus off" in error_str or "bus-off" in error_str:
                raise CANBusOffError(
                    f"CAN controller entered bus-off on {self.channel}"
                ) from e
            logger.warning("CAN send error on %s: %s", self.channel, e)
            return self._handle_transient_error(e)
        except OSError as e:
            return self._handle_transient_error(e)

    def recv(
        self,
        timeout: float | None = None,
    ) -> can.Message | None:
        """Receive a CAN message with error handling.

        Returns the message on success, None on timeout or transient failure.
        Raises CANBusOffError if the controller enters bus-off.
        """
        if not self._ensure_connected():
            return None

        t = timeout if timeout is not None else self.timeout
        try:
            return self.bus.recv(timeout=t)
        except can.CanError as e:
            error_str = str(e).lower()
            if "bus off" in error_str or "bus-off" in error_str:
                raise CANBusOffError(
                    f"CAN controller entered bus-off on {self.channel}"
                ) from e
            logger.warning("CAN receive error on %s: %s", self.channel, e)
            self._handle_transient_error(e)
            return None
        except OSError as e:
            self._handle_transient_error(e)
            return None

    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the CAN interface.

        Returns a dict with status information.
        """
        result: dict[str, Any] = {
            "channel": self.channel,
            "connected": self.is_connected,
            "is_hardware": self.is_hardware,
            "reconnect_count": self._reconnect_count,
        }

        if not self.bus:
            result["status"] = "disconnected"
            return result

        # Try a simple send-receive test (only for virtual for safety)
        if not self._is_hardware:
            try:
                test_msg = can.Message(
                    arbitration_id=0x7FF,
                    data=[0],
                    is_extended_id=False,
                )
                self.bus.send(test_msg, timeout=0.5)
                result["send_ok"] = True
                result["status"] = "healthy"
            except can.CanError:
                result["send_ok"] = False
                result["status"] = "degraded"
            except Exception as e:
                result["send_ok"] = False
                result["status"] = "error"
                result["error"] = str(e)
        else:
            result["status"] = "connected"

        return result

    def _ensure_connected(self) -> bool:
        """Ensure the bus is connected before an operation."""
        if self.bus is not None:
            return True

        if not self.auto_reconnect:
            return False

        if (
            self.max_reconnect_attempts > 0
            and self._reconnect_count >= self.max_reconnect_attempts
        ):
            logger.error(
                "Max reconnect attempts (%d) reached for %s",
                self.max_reconnect_attempts,
                self.channel,
            )
            return False

        self._reconnect_count += 1
        logger.info(
            "Reconnecting to %s (attempt %d)...",
            self.channel,
            self._reconnect_count,
        )
        time.sleep(self.reconnect_delay)
        return self.connect()

    def _handle_transient_error(self, error: Exception) -> bool:
        """Handle a transient bus error, triggering reconnect if configured."""
        logger.warning("Transient CAN error on %s: %s", self.channel, error)

        if self.auto_reconnect:
            self.disconnect()
            return self._ensure_connected()

        return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ── OBD-II ELM327 Adapter ──────────────────────────────────────────────────


# Standard OBD-II PIDs relevant to EV battery monitoring
OBD2_PIDS: dict[str, dict[str, Any]] = {
    "battery_voltage": {
        "mode": "01",
        "pid": "42",
        "name": "Control module voltage",
        "unit": "V",
        "formula": lambda b: (b[0] * 256 + b[1]) / 1000.0,
    },
    "battery_current": {
        "mode": "01",
        "pid": "6C",
        "name": "High voltage battery current",
        "unit": "A",
        "formula": lambda b: ((b[0] * 256 + b[1]) / 1000.0) - 32768,
    },
    "battery_temperature": {
        "mode": "01",
        "pid": "6D",
        "name": "High voltage battery temperature",
        "unit": "°C",
        "formula": lambda b: b[0] - 40,
    },
    "soc": {
        "mode": "01",
        "pid": "6E",
        "name": "High voltage battery SOC",
        "unit": "%",
        "formula": lambda b: b[0] / 2.0,
    },
    "odometer": {
        "mode": "01",
        "pid": "A6",
        "name": "Odometer",
        "unit": "km",
        "formula": lambda b: (b[0] * 256 * 256 + b[1] * 256 + b[2]) / 10.0,
    },
}


class OBD2Adapter:
    """OBD-II ELM327 adapter for EV battery telemetry.

    Connects to an ELM327-compatible OBD-II adapter over a serial port
    (USB or Bluetooth RFCOMM) and queries standard OBD-II PIDs relevant
    to electric vehicle battery monitoring.

    Args:
        port: Serial port device (e.g. '/dev/ttyUSB0', '/dev/rfcomm0',
              'COM3' on Windows). Auto-discover if None.
        baudrate: Serial baud rate (ELM327 defaults: 38400 or 9600).
        timeout: Serial read timeout in seconds.
        auto_reconnect: If True, attempt reconnection on failure.

    Example:
        >>> obd = OBD2Adapter(port='/dev/ttyUSB0')
        >>> obd.connect()
        >>> telemetry = obd.get_telemetry()
        >>> print(telemetry['battery_voltage'])
        >>> obd.disconnect()
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 38400,
        timeout: float = 2.0,
        auto_reconnect: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect

        self._serial: Any = None
        self._connected = False
        self._lock = threading.Lock()
        self._version = ""
        self._protocol = ""

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Connect to the ELM327 adapter and initialize.

        Returns True on success, False on failure.
        """
        if self._connected:
            return True

        port = self.port or self._auto_detect_port()
        if not port:
            logger.error("No OBD-II adapter port found")
            if not self.auto_reconnect:
                raise OBD2ConnectionError("No OBD-II adapter port found")
            return False

        try:
            import serial  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "pyserial is required for OBD-II adapter support. "
                "Install with: pip install pyserial"
            )
            return False

        try:
            ser = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            self._serial = ser
        except (OSError, ValueError) as e:
            logger.error("Failed to open serial port %s: %s", port, e)
            return False

        # Initialize ELM327
        try:
            self._init_elm327()
            self._connected = True
            logger.info(
                "Connected to OBD-II adapter on %s (v%s, protocol: %s)",
                port, self._version, self._protocol,
            )
            return True
        except (OBD2ConnectionError, OBD2ProtocolError) as e:
            logger.error("ELM327 init failed on %s: %s", port, e)
            try:
                ser.close()
            except Exception:
                pass
            self._serial = None
            return False

    def disconnect(self):
        """Disconnect from the OBD-II adapter."""
        with self._lock:
            self._connected = False
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

    def send_command(self, command: str) -> str | None:
        """Send a raw AT or OBD-II command and return the response.

        Args:
            command: AT command (e.g. 'AT RV') or OBD-II command
                     (e.g. '01 42' for battery voltage).

        Returns:
            Response string (stripped) or None on failure.
        """
        if not self._ensure_connected():
            return None

        try:
            # Send command
            cmd_bytes = (command + "\r").encode("ascii", errors="replace")
            self._serial.write(cmd_bytes)  # type: ignore[union-attr]

            # Read response
            response = b""
            timeout_time = time.time() + self.timeout

            while time.time() < timeout_time:
                byte = self._serial.read(1)  # type: ignore[union-attr]
                if not byte:
                    break
                response += byte
                if response.endswith(b">"):
                    break

            result = response.decode("ascii", errors="replace").strip()
            # Strip the echo (command echo back) and prompt
            cmd_clean = command.strip()
            if result.startswith(cmd_clean):
                result = result[len(cmd_clean):].strip()
            result = result.rstrip(">").strip()

            if not result or result.upper() in ("NO DATA", "ERROR", "UNABLE TO CONNECT", "?"):
                logger.debug("OBD-II command %s returned: %s", command, result)
                return None

            return result

        except (OSError, serial.SerialException) as e:  # noqa: F821
            logger.warning("OBD-II serial error on command %s: %s", command, e)
            self._handle_serial_error()
            return None

    def query_pid(
        self,
        mode: str = "01",
        pid: str = "42",
    ) -> bytes | None:
        """Query an OBD-II PID and return raw bytes.

        Args:
            mode: OBD-II mode (e.g. '01' for current data).
            pid: Two-digit hex PID code (e.g. '42' for battery voltage).

        Returns:
            Raw response bytes after the mode+pid echo, or None on failure.
        """
        cmd = f"{mode} {pid}"
        response = self.send_command(cmd)
        if not response:
            return None

        # Parse hex bytes from response
        # Typical response: "41 42 0E E8" (mode+0x40, pid, data bytes)
        hex_values = re.findall(r'[0-9A-Fa-f]{2}', response)
        if not hex_values:
            return None

        data_bytes = bytes(int(h, 16) for h in hex_values)

        # First two bytes should be mode+0x40 and PID echo
        if len(data_bytes) >= 2:
            data_bytes = data_bytes[2:]  # Strip header

        return data_bytes if data_bytes else None

    def get_telemetry(self) -> dict[str, Any]:
        """Query all configured OBD-II PIDs and return telemetry.

        Returns a dict with keys matching OBD2_PIDS keys.
        Failed queries return None for that key.
        """
        telemetry: dict[str, Any] = {}

        for key, pid_info in OBD2_PIDS.items():
            raw = self.query_pid(pid_info["mode"], pid_info["pid"])
            if raw and len(raw) >= 2:
                try:
                    value = pid_info["formula"](raw)
                    telemetry[key] = round(value, 2)
                except (IndexError, ValueError, TypeError) as e:
                    logger.debug(
                        "Failed to decode PID %s: %s", key, e
                    )
                    telemetry[key] = None
            else:
                telemetry[key] = None

        # Also query battery voltage (alternate method: AT RV)
        if telemetry.get("battery_voltage") is None:
            at_rv = self.send_command("AT RV")
            if at_rv:
                try:
                    # AT RV returns voltage like "12.5V" or "14.2"
                    voltage_str = at_rv.replace("V", "").strip()
                    telemetry["battery_voltage"] = round(float(voltage_str), 2)
                except (ValueError, TypeError):
                    pass

        return telemetry

    def _auto_detect_port(self) -> str | None:
        """Auto-discover the OBD-II adapter serial port."""
        if platform.system() == "Windows":
            # Windows: scan COM ports
            candidates = [f"COM{i}" for i in range(1, 17)]
        elif platform.system() == "Linux":
            # Linux: check common serial ports
            candidates = [
                "/dev/ttyUSB0", "/dev/ttyUSB1",
                "/dev/ttyAMA0", "/dev/ttyS0",
                "/dev/rfcomm0", "/dev/rfcomm1",
            ]
        else:
            candidates = [
                "/dev/tty.usbserial*", "/dev/cu.usbserial*",
            ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return None

    def _init_elm327(self) -> None:
        """Initialize the ELM327 chip with AT commands."""
        if not self._serial:
            raise OBD2ConnectionError("No serial connection")

        # Flush any stale data
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        time.sleep(0.5)

        # Test communication
        for attempt in range(3):
            self._serial.write(b"AT Z\r")
            time.sleep(0.3)
            self._serial.reset_input_buffer()
            time.sleep(0.2)

        # Read ELM327 version
        self._serial.write(b"AT I\r")
        version_response = b""
        timeout_time = time.time() + self.timeout
        while time.time() < timeout_time:
            byte = self._serial.read(1)
            if not byte:
                break
            version_response += byte
            if b">" in version_response:
                break

        version_str = version_response.decode("ascii", errors="replace").strip()
        if "ELM327" in version_str or "OBD" in version_str:
            self._version = version_str
        else:
            logger.warning("Unexpected ELM327 version response: %s", version_str)

        # Set defaults
        for cmd in [b"AT E0\r", b"AT L0\r", b"AT S0\r", b"AT H0\r", b"AT SP 0\r"]:
            try:
                self._serial.write(cmd)
                time.sleep(0.1)
                self._serial.reset_input_buffer()
            except OSError:
                pass

        time.sleep(0.3)

        # Detect protocol
        self._serial.write(b"AT DP\r")
        proto_response = b""
        timeout_time = time.time() + self.timeout
        while time.time() < timeout_time:
            byte = self._serial.read(1)
            if not byte:
                break
            proto_response += byte
            if b">" in proto_response:
                break

        proto_str = proto_response.decode("ascii", errors="replace").strip()
        self._protocol = proto_str.replace("AT DP", "").replace(">", "").strip()

        if not self._protocol:
            self._protocol = "auto"

    def _ensure_connected(self) -> bool:
        """Ensure we're connected before an operation."""
        if self._connected and self._serial and self._serial.is_open:
            return True

        if not self.auto_reconnect:
            return False

        self.disconnect()
        return self.connect()

    def _handle_serial_error(self) -> None:
        """Handle a serial error, triggering reconnect if configured."""
        try:
            if self._serial:
                self._serial.close()
        except Exception:
            pass
        self._serial = None
        self._connected = False

        if self.auto_reconnect:
            logger.info("Attempting OBD-II reconnection...")
            self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ── Enhanced CAN Battery Simulator ──────────────────────────────────────────


class CANBatterySimulator:
    """
    Simulates battery telemetry messages on a CAN bus.

    Supports virtual CAN (vcan0) for development and real socketCAN
    interfaces (can0, can1) for hardware-in-the-loop testing.

    Messages:
    - 0x101: Voltage (uint16, 0.1V res) and Current (int16, 0.1A res)
    - 0x102: Temperature (int8, 1°C res) and SOC (uint8, 1% res)
    """

    def __init__(
        self,
        channel: str = "vcan0",
        hardware: bool = False,
        bitrate: int = 500000,
        auto_reconnect: bool = True,
    ):
        self.channel = channel
        self.hardware = hardware
        self.bitrate = bitrate
        self.auto_reconnect = auto_reconnect

        self._hw_interface: CANHardwareInterface | None = None
        self.bus: can.interface.Bus | None = None
        self.running = False
        self._thread: threading.Thread | None = None

    @property
    def is_hardware(self) -> bool:
        return self.hardware or (
            self._hw_interface is not None and self._hw_interface.is_hardware
        )

    def start(self):
        """Start simulation on the configured CAN interface."""
        if self.hardware:
            # Use hardware interface manager
            self._hw_interface = CANHardwareInterface(
                channel=self.channel,
                bitrate=self.bitrate,
                auto_reconnect=self.auto_reconnect,
                max_reconnect_attempts=3,
                reconnect_delay=2.0,
            )
            connected = self._hw_interface.connect()
            if not connected:
                logger.warning(
                    "Hardware CAN not available, falling back to log-only mode"
                )
                self._hw_interface = None
        else:
            # Use standard python-can Bus (virtual or socketcan)
            try:
                self.bus = can.interface.Bus(
                    channel=self.channel,
                    interface="socketcan" if not self.channel.startswith("vcan")
                    else "virtual",
                )
            except (can.CanError, OSError, ValueError) as e:
                logger.warning(
                    "CAN bus %s not available, using log-only mode: %s",
                    self.channel, e,
                )
                self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop simulation."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

        if self._hw_interface:
            self._hw_interface.disconnect()
            self._hw_interface = None
        elif self.bus:
            try:
                self.bus.shutdown()
            except Exception:
                pass
            self.bus = None

    def _run(self):
        """Internal simulation loop."""
        base_voltage = 396.0
        base_current = 50.0
        base_temp = 35
        base_soc = 80

        while self.running:
            voltage = base_voltage + random.uniform(-2, 2)
            current = base_current + random.uniform(-5, 5)
            temp = base_temp + random.randint(-1, 2)
            soc = base_soc

            # Pack 0x101: Voltage (2 bytes), Current (2 bytes)
            v_scaled = int(voltage * 10)
            c_scaled = int(current * 10)
            data1 = [
                (v_scaled >> 8) & 0xFF, v_scaled & 0xFF,
                (c_scaled >> 8) & 0xFF, c_scaled & 0xFF,
                0, 0, 0, 0,
            ]
            msg1 = can.Message(
                arbitration_id=0x101, data=data1, is_extended_id=False,
            )

            # Pack 0x102: Temperature (1 byte), SOC (1 byte)
            data2 = [temp & 0xFF, soc & 0xFF, 0, 0, 0, 0, 0, 0]
            msg2 = can.Message(
                arbitration_id=0x102, data=data2, is_extended_id=False,
            )

            if self._hw_interface:
                try:
                    self._hw_interface.send(msg1)
                    self._hw_interface.send(msg2)
                except HardwareCANError as e:
                    logger.error("Hardware CAN send failed: %s", e)
            elif self.bus:
                try:
                    self.bus.send(msg1)
                    self.bus.send(msg2)
                except can.CanError:
                    pass

            time.sleep(1.0)


class CANTelemetryReceiver:
    """
    Receives and decodes battery telemetry from a CAN bus.

    Supports both virtual CAN (vcan0) and real socketCAN interfaces.
    """

    def __init__(
        self,
        channel: str = "vcan0",
        hardware: bool = False,
        bitrate: int = 500000,
        auto_reconnect: bool = True,
    ):
        self.channel = channel
        self.hardware = hardware
        self.bitrate = bitrate
        self.auto_reconnect = auto_reconnect

        self.latest_data: dict[str, Any] = {
            "voltage": 0.0,
            "current": 0.0,
            "temperature": 0.0,
            "soc": 0.0,
        }
        self._hw_interface: CANHardwareInterface | None = None
        self.bus: can.interface.Bus | None = None
        self.running = False
        self._thread: threading.Thread | None = None

    @property
    def is_hardware(self) -> bool:
        return self.hardware or (
            self._hw_interface is not None and self._hw_interface.is_hardware
        )

    def start(self):
        """Start receiving from the configured CAN interface."""
        if self.hardware:
            self._hw_interface = CANHardwareInterface(
                channel=self.channel,
                bitrate=self.bitrate,
                auto_reconnect=self.auto_reconnect,
                max_reconnect_attempts=3,
                reconnect_delay=2.0,
            )
            connected = self._hw_interface.connect()
            if not connected:
                logger.warning(
                    "Hardware CAN not available for receiver, running idle"
                )
                self._hw_interface = None
        else:
            try:
                self.bus = can.interface.Bus(
                    channel=self.channel,
                    interface="socketcan" if not self.channel.startswith("vcan")
                    else "virtual",
                )
            except (can.CanError, OSError, ValueError):
                self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop receiver."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

        if self._hw_interface:
            self._hw_interface.disconnect()
            self._hw_interface = None
        elif self.bus:
            try:
                self.bus.shutdown()
            except Exception:
                pass
            self.bus = None

    def _run(self):
        """Internal receiver loop."""
        while self.running:
            msg: can.Message | None = None

            if self._hw_interface:
                try:
                    msg = self._hw_interface.recv(timeout=1.0)
                except HardwareCANError as e:
                    logger.error("Hardware CAN receive error: %s", e)
                    continue
                except Exception:
                    continue
            elif self.bus:
                try:
                    msg = self.bus.recv(timeout=1.0)
                except can.CanError:
                    continue
            else:
                time.sleep(0.1)
                continue

            if msg is None:
                continue

            if msg.arbitration_id == 0x101:
                v_scaled = (msg.data[0] << 8) | msg.data[1]
                c_scaled = (msg.data[2] << 8) | msg.data[3]
                # Handle signed current (2's complement)
                if c_scaled > 0x7FFF:
                    c_scaled -= 0x10000
                self.latest_data["voltage"] = v_scaled / 10.0
                self.latest_data["current"] = c_scaled / 10.0
            elif msg.arbitration_id == 0x102:
                temp = msg.data[0]
                if temp > 0x7F:
                    temp -= 0x100
                self.latest_data["temperature"] = float(temp)
                self.latest_data["soc"] = float(msg.data[1])

    def get_telemetry(self) -> dict[str, Any]:
        """Return latest telemetry."""
        return self.latest_data.copy()


class DBCFileSimulator:
    """
    CAN telemetry simulator driven by a DBC file definition.

    Reads signal definitions from a .dbc file and generates random
    data for all defined messages and signals. Supports both CAN 2.0B
    and J1939 (29-bit) IDs automatically.

    Args:
        dbc_path: Path to .dbc file, or None to use built-in battery DBC.
    """

    def __init__(self, dbc_path: str | None = None):
        if dbc_path:
            self.dbc = DBCParser(dbc_path)
        else:
            import tempfile

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".dbc", delete=False,
            )
            tmp.write(battery_dbc_content())
            tmp.close()
            self.dbc = DBCParser(tmp.name)
            os.unlink(tmp.name)

        self.running = False
        self._thread: threading.Thread | None = None
        self._hw_interface: CANHardwareInterface | None = None
        self._bus: can.interface.Bus | None = None

    def start(
        self,
        channel: str = "vcan0",
        hardware: bool = False,
        bitrate: int = 500000,
        auto_reconnect: bool = True,
    ):
        """Start simulation on the given CAN interface.

        Args:
            channel: CAN interface name (e.g. 'vcan0', 'can0').
            hardware: If True, use socketCAN hardware backend.
            bitrate: Bitrate for hardware CAN interfaces.
            auto_reconnect: Auto-reconnect on connection loss.
        """
        if hardware:
            self._hw_interface = CANHardwareInterface(
                channel=channel,
                bitrate=bitrate,
                auto_reconnect=auto_reconnect,
                max_reconnect_attempts=3,
                reconnect_delay=2.0,
            )
            connected = self._hw_interface.connect()
            if not connected:
                logger.warning(
                    "Hardware CAN not available for DBC simulator, "
                    "running in log-only mode"
                )
                self._hw_interface = None
        else:
            try:
                self._bus = can.interface.Bus(
                    channel=channel,
                    interface="socketcan" if not channel.startswith("vcan")
                    else "virtual",
                )
            except (can.CanError, OSError, ValueError) as e:
                logger.warning(
                    "CAN bus not available, running in log-only mode: %s", e
                )
                self._bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop simulation."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

        if self._hw_interface:
            self._hw_interface.disconnect()
            self._hw_interface = None
        elif self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None

    def _run(self):
        """Main simulation loop."""
        while self.running:
            for can_id, msg_def in self.dbc.messages.items():
                data = self._generate_frame(msg_def)
                msg = can.Message(
                    arbitration_id=can_id,
                    data=data,
                    is_extended_id=msg_def.is_extended,
                )

                if self._hw_interface:
                    try:
                        self._hw_interface.send(msg)
                    except HardwareCANError as e:
                        logger.error("DBC send failed: %s", e)
                elif self._bus:
                    try:
                        self._bus.send(msg)
                    except can.CanError:
                        pass

            time.sleep(1.0)

    def _generate_frame(self, msg_def) -> list[int]:
        """Generate random CAN data bytes for a message definition."""
        data = [0] * 8
        for sig_name, sig in msg_def.signals.items():
            raw = self._random_raw(sig)
            self._place_raw(data, sig, raw)
        return data

    @staticmethod
    def _random_raw(sig) -> int:
        """Generate a random raw value within the signal's range."""
        if sig.min_val < sig.max_val:
            phys = random.uniform(sig.min_val, sig.max_val)
        else:
            phys = random.uniform(0, 100)
        raw = sig.physical_to_raw(phys)
        max_raw = (1 << sig.length) - 1
        return max(0, min(raw, max_raw))

    @staticmethod
    def _place_raw(data: list[int], sig, raw: int):
        """Place a raw integer into the CAN data bytes."""
        for i in range(sig.length):
            bit_pos = (
                sig.start_bit + i
                if sig.byte_order == "Intel"
                else sig.start_bit - i
            )
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx >= len(data):
                continue
            if (raw >> i) & 1:
                data[byte_idx] |= 1 << bit_in_byte
            else:
                data[byte_idx] &= ~(1 << bit_in_byte)
