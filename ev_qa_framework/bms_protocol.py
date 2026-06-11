"""
BMS Communication Protocol Abstraction Layer.

Provides a unified interface for communicating with Battery Management
Systems over multiple transport protocols:

- CAN Bus (via python-can / socketCAN) — real-time cell-level data
- Modbus TCP — Ethernet-connected industrial BMS
- Modbus RTU — RS-485/RS-232 serial BMS

The abstraction layer offers:
- Protocol auto-detection (scan for available BMS connections)
- Unified telemetry data model (same dict structure regardless of protocol)
- Connection health monitoring
- Automatic failover between protocols

Architecture:
    BMSProtocolManager (auto-detect + unified interface)
    ├── BMSCANInterface     — wraps CANBatterySimulator / CANTelemetryReceiver
    ├── BMSModbusInterface  — wraps ModbusTCPClient / ModbusRTUClient
    └── BMSUnifiedTelemetry — common data model

Usage:
    # Auto-detect and connect
    manager = BMSProtocolManager()
    manager.auto_detect()
    telemetry = manager.read_telemetry()

    # Explicit protocol
    manager = BMSProtocolManager(protocol="can", channel="vcan0")
    manager.connect()
    telemetry = manager.read_telemetry()
    manager.disconnect()
"""

from __future__ import annotations

import logging
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Protocol Types ──────────────────────────────────────────────────────────


class ProtocolType(Enum):
    """Supported BMS communication protocols."""

    CAN = "can"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    AUTO = "auto"


# ── Unified Telemetry Data Model ────────────────────────────────────────────


@dataclass
class BMSTelemetry:
    """Unified BMS telemetry data model.

    All protocol interfaces normalize their data to this structure.
    Fields not supported by a particular protocol are set to None.
    """

    # Pack-level measurements
    pack_voltage: float | None = None  # V
    pack_current: float | None = None  # A
    soc: float | None = None  # %
    soh: float | None = None  # %

    # Temperature
    temperature_max: float | None = None  # °C
    temperature_min: float | None = None  # °C
    temperature_avg: float | None = None  # °C

    # Cell-level
    cell_voltage_min: float | None = None  # V
    cell_voltage_max: float | None = None  # V
    cell_voltage_delta: float | None = None  # V
    cell_voltages: list[float] = field(default_factory=list)  # Individual cell voltages

    # Cycle / lifetime
    charge_cycle_count: int | None = None

    # Status
    fault_flags: list[str] = field(default_factory=list)
    status_flags: int | None = None
    is_balancing: bool = False

    # Metadata
    protocol: str = ""
    timestamp: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict."""
        return {
            "pack_voltage": self.pack_voltage,
            "pack_current": self.pack_current,
            "soc": self.soc,
            "soh": self.soh,
            "temperature_max": self.temperature_max,
            "temperature_min": self.temperature_min,
            "temperature_avg": self.temperature_avg,
            "cell_voltage_min": self.cell_voltage_min,
            "cell_voltage_max": self.cell_voltage_max,
            "cell_voltage_delta": self.cell_voltage_delta,
            "cell_voltages": self.cell_voltages,
            "charge_cycle_count": self.charge_cycle_count,
            "fault_flags": self.fault_flags,
            "status_flags": self.status_flags,
            "is_balancing": self.is_balancing,
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @property
    def has_faults(self) -> bool:
        return len(self.fault_flags) > 0

    @property
    def is_healthy(self) -> bool:
        return not self.has_faults and self.soc is not None


# ── BMS Interface Base ──────────────────────────────────────────────────────


class BMSInterface(ABC):
    """Abstract base for BMS protocol interfaces.

    Each protocol implementation must provide connect/disconnect,
    read_telemetry, and health_check methods that return data in
    the unified BMSTelemetry format.
    """

    def __init__(self, protocol: ProtocolType):
        self.protocol = protocol
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the BMS."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    def read_telemetry(self) -> BMSTelemetry:
        """Read battery telemetry and return normalized data."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check connection health."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ── CAN Bus BMS Interface ──────────────────────────────────────────────────


class BMSCANInterface(BMSInterface):
    """BMS interface over CAN bus protocol.

    Wraps the existing CANBatterySimulator and CANTelemetryReceiver
    from the can_bus module, normalizing output to BMSTelemetry.

    Args:
        channel: CAN interface name (e.g. 'vcan0', 'can0').
        hardware: If True, use real CAN hardware. If False, simulation mode.
        bitrate: CAN bus bitrate (default 500000).
    """

    def __init__(
        self,
        channel: str = "vcan0",
        hardware: bool = False,
        bitrate: int = 500000,
    ):
        super().__init__(ProtocolType.CAN)
        self.channel = channel
        self.hardware = hardware
        self.bitrate = bitrate
        self._receiver = None

    def connect(self) -> bool:
        """Initialize CAN telemetry receiver."""
        try:
            from ev_qa_framework.can_bus import CANTelemetryReceiver

            self._receiver = CANTelemetryReceiver(
                channel=self.channel,
                hardware=self.hardware,
            )
            self._receiver.start()
            self._connected = True
            logger.info("BMS CAN interface connected on %s", self.channel)
            return True
        except Exception as e:
            logger.warning("BMS CAN connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Stop CAN telemetry receiver."""
        if self._receiver:
            try:
                self._receiver.stop()
            except Exception:
                pass
            self._receiver = None
        self._connected = False

    def read_telemetry(self) -> BMSTelemetry:
        """Read telemetry from CAN bus and normalize."""
        if not self._receiver or not self._connected:
            return BMSTelemetry(protocol="can", source=self.channel)

        raw = self._receiver.get_telemetry()

        return BMSTelemetry(
            pack_voltage=raw.get("voltage"),
            pack_current=raw.get("current"),
            soc=raw.get("soc"),
            soh=raw.get("soh"),
            temperature_avg=raw.get("temperature"),
            protocol="can",
            timestamp=time.time(),
            source=self.channel,
        )

    def health_check(self) -> dict[str, Any]:
        """Check CAN interface health."""
        return {
            "protocol": "can",
            "channel": self.channel,
            "connected": self._connected,
            "hardware": self.hardware,
            "status": "healthy" if self._connected else "disconnected",
        }


# ── Modbus TCP BMS Interface ───────────────────────────────────────────────


class BMSModbusTCPInterface(BMSInterface):
    """BMS interface over Modbus TCP protocol.

    Wraps ModbusTCPClient, normalizing output to BMSTelemetry.

    Args:
        host: IP address or hostname.
        port: TCP port (default 502).
        unit_id: Modbus unit ID (default 1).
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        timeout: float = 3.0,
    ):
        super().__init__(ProtocolType.MODBUS_TCP)
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout
        self._client = None

    def connect(self) -> bool:
        """Connect to Modbus TCP BMS."""
        try:
            from ev_qa_framework.modbus import ModbusTCPClient

            self._client = ModbusTCPClient(
                host=self.host,
                port=self.port,
                unit_id=self.unit_id,
                timeout=self.timeout,
            )
            self._client.connect()
            self._connected = True
            logger.info("BMS Modbus TCP connected to %s:%d", self.host, self.port)
            return True
        except Exception as e:
            logger.warning("BMS Modbus TCP connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Modbus TCP BMS."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self._connected = False

    def read_telemetry(self) -> BMSTelemetry:
        """Read telemetry via Modbus TCP and normalize."""
        if not self._client or not self._connected:
            return BMSTelemetry(
                protocol="modbus_tcp",
                source=f"{self.host}:{self.port}",
            )

        try:
            data = self._client.read_battery_telemetry()
            return BMSTelemetry(
                pack_voltage=data.get("pack_voltage"),
                pack_current=data.get("pack_current"),
                soc=data.get("soc"),
                soh=data.get("soh"),
                temperature_max=data.get("temperature_max"),
                temperature_min=data.get("temperature_min"),
                temperature_avg=data.get("temperature_avg"),
                cell_voltage_min=data.get("cell_voltage_min"),
                cell_voltage_max=data.get("cell_voltage_max"),
                cell_voltage_delta=data.get("cell_voltage_delta"),
                charge_cycle_count=data.get("charge_cycle_count"),
                fault_flags=data.get("fault_flags", []),
                status_flags=data.get("status_flags"),
                protocol="modbus_tcp",
                timestamp=time.time(),
                source=f"{self.host}:{self.port}",
            )
        except Exception as e:
            logger.error("Modbus TCP telemetry read failed: %s", e)
            return BMSTelemetry(
                protocol="modbus_tcp",
                source=f"{self.host}:{self.port}",
            )

    def health_check(self) -> dict[str, Any]:
        """Check Modbus TCP health."""
        if not self._client:
            return {
                "protocol": "modbus_tcp",
                "connected": False,
                "status": "disconnected",
            }
        try:
            result = self._client.health_check()
            result["protocol"] = "modbus_tcp"
            result["host"] = self.host
            result["port"] = self.port
            return result
        except Exception as e:
            return {
                "protocol": "modbus_tcp",
                "connected": self._connected,
                "status": "error",
                "error": str(e),
            }


# ── Modbus RTU BMS Interface ───────────────────────────────────────────────


class BMSModbusRTUInterface(BMSInterface):
    """BMS interface over Modbus RTU (RS-485) protocol.

    Wraps ModbusRTUClient, normalizing output to BMSTelemetry.

    Args:
        port: Serial port (e.g. '/dev/ttyUSB0', 'COM3').
        baudrate: Serial baud rate (default 9600).
        unit_id: Modbus unit ID (default 1).
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        unit_id: int = 1,
        timeout: float = 3.0,
    ):
        super().__init__(ProtocolType.MODBUS_RTU)
        self.port = port
        self.baudrate = baudrate
        self.unit_id = unit_id
        self.timeout = timeout
        self._client = None

    def connect(self) -> bool:
        """Connect to Modbus RTU BMS."""
        try:
            from ev_qa_framework.modbus import ModbusRTUClient

            self._client = ModbusRTUClient(
                port=self.port,
                baudrate=self.baudrate,
                unit_id=self.unit_id,
                timeout=self.timeout,
            )
            self._client.connect()
            self._connected = True
            logger.info(
                "BMS Modbus RTU connected to %s @ %d baud",
                self.port,
                self.baudrate,
            )
            return True
        except Exception as e:
            logger.warning("BMS Modbus RTU connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Modbus RTU BMS."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
        self._connected = False

    def read_telemetry(self) -> BMSTelemetry:
        """Read telemetry via Modbus RTU and normalize."""
        if not self._client or not self._connected:
            return BMSTelemetry(
                protocol="modbus_rtu",
                source=self.port,
            )

        try:
            data = self._client.read_battery_telemetry()
            return BMSTelemetry(
                pack_voltage=data.get("pack_voltage"),
                pack_current=data.get("pack_current"),
                soc=data.get("soc"),
                soh=data.get("soh"),
                temperature_max=data.get("temperature_max"),
                temperature_min=data.get("temperature_min"),
                temperature_avg=data.get("temperature_avg"),
                cell_voltage_min=data.get("cell_voltage_min"),
                cell_voltage_max=data.get("cell_voltage_max"),
                cell_voltage_delta=data.get("cell_voltage_delta"),
                charge_cycle_count=data.get("charge_cycle_count"),
                fault_flags=data.get("fault_flags", []),
                status_flags=data.get("status_flags"),
                protocol="modbus_rtu",
                timestamp=time.time(),
                source=self.port,
            )
        except Exception as e:
            logger.error("Modbus RTU telemetry read failed: %s", e)
            return BMSTelemetry(
                protocol="modbus_rtu",
                source=self.port,
            )

    def health_check(self) -> dict[str, Any]:
        """Check Modbus RTU health."""
        if not self._client:
            return {
                "protocol": "modbus_rtu",
                "connected": False,
                "status": "disconnected",
            }
        try:
            result = self._client.health_check()
            result["protocol"] = "modbus_rtu"
            result["port"] = self.port
            return result
        except Exception as e:
            return {
                "protocol": "modbus_rtu",
                "connected": self._connected,
                "status": "error",
                "error": str(e),
            }


# ── Protocol Auto-Detection ─────────────────────────────────────────────────


@dataclass
class DetectedBMS:
    """Information about a detected BMS connection."""

    protocol: ProtocolType
    description: str
    config: dict[str, Any]
    priority: int = 0  # Higher = preferred


def scan_can_interfaces() -> list[DetectedBMS]:
    """Scan for available CAN interfaces that might have a BMS."""
    results: list[DetectedBMS] = []

    if platform.system() != "Linux":
        return results

    try:
        from ev_qa_framework.can_bus import detect_can_interfaces

        interfaces = detect_can_interfaces()
        for iface in interfaces:
            results.append(
                DetectedBMS(
                    protocol=ProtocolType.CAN,
                    description=f"CAN interface {iface['name']} ({iface['type']})",
                    config={
                        "channel": iface["name"],
                        "hardware": iface["type"] == "hardware",
                    },
                    priority=10 if iface["type"] == "hardware" else 5,
                )
            )
    except Exception as e:
        logger.debug("CAN interface scan failed: %s", e)

    return results


def scan_modbus_tcp(
    hosts: list[str] | None = None,
    port: int = 502,
    timeout: float = 1.0,
) -> list[DetectedBMS]:
    """Scan for Modbus TCP devices on the network.

    Attempts TCP connection to each host:port to check if a Modbus
    device is listening.

    Args:
        hosts: List of IP addresses to scan. Defaults to common BMS addresses.
        port: Modbus TCP port (default 502).
        timeout: Connection timeout per host.
    """
    import socket

    if hosts is None:
        hosts = ["192.168.1.100", "192.168.0.100", "10.0.0.100"]

    results: list[DetectedBMS] = []
    for host in hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                results.append(
                    DetectedBMS(
                        protocol=ProtocolType.MODBUS_TCP,
                        description=f"Modbus TCP device at {host}:{port}",
                        config={"host": host, "port": port, "unit_id": 1},
                        priority=8,
                    )
                )
        except OSError:
            pass

    return results


def scan_modbus_rtu(
    ports: list[str] | None = None,
    baudrates: list[int] | None = None,
) -> list[DetectedBMS]:
    """Scan for Modbus RTU devices on serial ports.

    Attempts to open each serial port and send a Modbus probe
    (read holding register 0) to detect responding devices.

    Args:
        ports: List of serial port paths. Auto-detects if None.
        baudrates: List of baud rates to try. Defaults to common rates.
    """
    if baudrates is None:
        baudrates = [9600, 19200, 38400, 115200]

    results: list[DetectedBMS] = []

    if ports is None:
        # Auto-detect serial ports
        ports = _auto_detect_serial_ports()

    for port in ports:
        for baud in baudrates:
            try:
                from ev_qa_framework.modbus import ModbusRTUClient

                client = ModbusRTUClient(port=port, baudrate=baud, unit_id=1, timeout=1.0)
                client.connect()
                # Try to read register 0
                client.read_holding_registers(0, 1)
                client.disconnect()

                results.append(
                    DetectedBMS(
                        protocol=ProtocolType.MODBUS_RTU,
                        description=f"Modbus RTU device at {port} @ {baud} baud",
                        config={
                            "port": port,
                            "baudrate": baud,
                            "unit_id": 1,
                        },
                        priority=7,
                    )
                )
                break  # Found at this baudrate, no need to try others
            except Exception:
                continue

    return results


def _auto_detect_serial_ports() -> list[str]:
    """Auto-detect available serial ports on the system."""
    system = platform.system()
    if system == "Linux":
        import glob

        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        return ports
    elif system == "Windows":
        return [f"COM{i}" for i in range(1, 17)]
    elif system == "Darwin":
        import glob

        return glob.glob("/dev/tty.usbserial*") + glob.glob("/dev/cu.usbserial*")
    return []


# ── BMS Protocol Manager ────────────────────────────────────────────────────


class BMSProtocolManager:
    """Unified BMS communication manager with protocol auto-detection.

    Manages multiple BMS interfaces and provides a single entry point
    for reading telemetry regardless of the underlying protocol.

    Args:
        protocol: Protocol to use (ProtocolType.AUTO for auto-detection).
        config: Protocol-specific configuration dict.
        auto_fallback: If True, try other protocols on failure.

    Usage::

        # Auto-detect
        mgr = BMSProtocolManager()
        mgr.auto_detect()
        telemetry = mgr.read_telemetry()

        # Explicit CAN
        mgr = BMSProtocolManager(
            protocol=ProtocolType.CAN,
            config={"channel": "vcan0", "hardware": False}
        )
        mgr.connect()
        telemetry = mgr.read_telemetry()
    """

    def __init__(
        self,
        protocol: ProtocolType = ProtocolType.AUTO,
        config: dict[str, Any] | None = None,
        auto_fallback: bool = True,
    ):
        self.protocol = protocol
        self.config = config or {}
        self.auto_fallback = auto_fallback
        self._interfaces: list[BMSInterface] = []
        self._active_interface: BMSInterface | None = None
        self._detected: list[DetectedBMS] = []

    @property
    def is_connected(self) -> bool:
        return self._active_interface is not None and self._active_interface.is_connected

    @property
    def active_protocol(self) -> str:
        if self._active_interface:
            return self._active_interface.protocol.value
        return "none"

    def auto_detect(self) -> list[DetectedBMS]:
        """Scan for available BMS connections across all protocols.

        Returns:
            List of DetectedBMS objects sorted by priority (highest first).
        """
        self._detected = []

        # Scan CAN
        self._detected.extend(scan_can_interfaces())

        # Scan Modbus TCP
        tcp_hosts = self.config.get("modbus_tcp_hosts")
        self._detected.extend(scan_modbus_tcp(hosts=tcp_hosts))

        # Scan Modbus RTU
        rtu_ports = self.config.get("modbus_rtu_ports")
        rtu_baudrates = self.config.get("modbus_rtu_baudrates")
        self._detected.extend(scan_modbus_rtu(ports=rtu_ports, baudrates=rtu_baudrates))

        # Sort by priority (highest first)
        self._detected.sort(key=lambda d: d.priority, reverse=True)

        logger.info(
            "Auto-detect found %d BMS connection(s)",
            len(self._detected),
        )
        for d in self._detected:
            logger.info("  [%s] %s (priority=%d)", d.protocol.value, d.description, d.priority)

        return self._detected

    def connect(self) -> bool:
        """Connect to the BMS.

        If protocol is AUTO, auto-detects and tries connections in
        priority order. Otherwise, connects using the specified protocol
        and config.
        """
        if self.protocol == ProtocolType.AUTO:
            return self._connect_auto()
        else:
            return self._connect_single(self.protocol, self.config)

    def _connect_auto(self) -> bool:
        """Auto-detect and connect to the best available BMS."""
        if not self._detected:
            self.auto_detect()

        for detected in self._detected:
            try:
                iface = self._create_interface(detected)
                if iface.connect():
                    self._interfaces.append(iface)
                    self._active_interface = iface
                    logger.info(
                        "Auto-connected via %s: %s",
                        detected.protocol.value,
                        detected.description,
                    )
                    return True
            except Exception as e:
                logger.debug(
                    "Failed to connect via %s: %s",
                    detected.protocol.value,
                    e,
                )
                continue

        logger.warning("No BMS connection available")
        return False

    def _connect_single(self, protocol: ProtocolType, config: dict[str, Any]) -> bool:
        """Connect using a specific protocol."""
        detected = DetectedBMS(
            protocol=protocol,
            description=f"Manual {protocol.value}",
            config=config,
        )
        iface = self._create_interface(detected)
        if iface.connect():
            self._interfaces.append(iface)
            self._active_interface = iface
            return True
        return False

    def _create_interface(self, detected: DetectedBMS) -> BMSInterface:
        """Create a BMSInterface from a DetectedBMS."""
        cfg = detected.config

        if detected.protocol == ProtocolType.CAN:
            return BMSCANInterface(
                channel=cfg.get("channel", "vcan0"),
                hardware=cfg.get("hardware", False),
            )
        elif detected.protocol == ProtocolType.MODBUS_TCP:
            return BMSModbusTCPInterface(
                host=cfg.get("host", "192.168.1.100"),
                port=cfg.get("port", 502),
                unit_id=cfg.get("unit_id", 1),
            )
        elif detected.protocol == ProtocolType.MODBUS_RTU:
            return BMSModbusRTUInterface(
                port=cfg.get("port", "/dev/ttyUSB0"),
                baudrate=cfg.get("baudrate", 9600),
                unit_id=cfg.get("unit_id", 1),
            )
        else:
            raise ValueError(f"Unknown protocol: {detected.protocol}")

    def disconnect(self) -> None:
        """Disconnect all interfaces."""
        for iface in self._interfaces:
            try:
                iface.disconnect()
            except Exception:
                pass
        self._interfaces.clear()
        self._active_interface = None

    def read_telemetry(self) -> BMSTelemetry:
        """Read telemetry from the active BMS connection.

        If auto_fallback is enabled and the active connection fails,
        tries the next available interface.
        """
        if not self._active_interface:
            return BMSTelemetry(protocol="none", source="disconnected")

        try:
            telemetry = self._active_interface.read_telemetry()
            if telemetry.is_healthy or not self.auto_fallback:
                return telemetry
        except Exception as e:
            logger.warning(
                "Telemetry read failed on %s: %s",
                self._active_interface.protocol.value,
                e,
            )

        # Fallback: try other interfaces
        if self.auto_fallback:
            for iface in self._interfaces:
                if iface is self._active_interface:
                    continue
                if iface.is_connected:
                    try:
                        telemetry = iface.read_telemetry()
                        self._active_interface = iface
                        logger.info(
                            "Switched to fallback protocol: %s",
                            iface.protocol.value,
                        )
                        return telemetry
                    except Exception:
                        continue

        return BMSTelemetry(
            protocol=self._active_interface.protocol.value,
            source="error",
        )

    def health_check(self) -> dict[str, Any]:
        """Check health of all interfaces."""
        results = {
            "active_protocol": self.active_protocol,
            "connected": self.is_connected,
            "interfaces": [],
        }
        for iface in self._interfaces:
            results["interfaces"].append(iface.health_check())
        return results

    def get_detected(self) -> list[DetectedBMS]:
        """Return list of detected BMS connections from last auto_detect."""
        return list(self._detected)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
