"""Base class for manufacturer-specific BMS adapters.

Each adapter decodes raw CAN frames from a specific BMS manufacturer
into the unified BMSTelemetry data model. The decode functions work
with raw bytes so they can be tested without CAN hardware.
"""

from __future__ import annotations

import logging
import struct
import time
from abc import ABC, abstractmethod
from typing import Any

from ev_qa_framework.bms_protocol import BMSTelemetry

logger = logging.getLogger(__name__)


class BaseBMSAdapter(ABC):
    """Abstract base for manufacturer-specific BMS adapters.

    Subclasses implement connect/disconnect for CAN hardware access
    and static decode methods that parse raw CAN data bytes.

    CAN hardware (python-can) is lazy-imported only in connect(),
    so decode functions work without any hardware dependencies.
    """

    manufacturer: str = "generic"
    can_ids: dict[str, int] = {}

    def __init__(self, channel: str = "can0", bitrate: int = 500_000):
        self.channel = channel
        self.bitrate = bitrate
        self._bus = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Establish CAN bus connection. Lazy-imports python-can."""
        try:
            import can  # noqa: F401

            self._bus = can.interface.Bus(
                channel=self.channel,
                interface="socketcan",
                bitrate=self.bitrate,
            )
            self._connected = True
            logger.info("%s BMS connected on %s", self.manufacturer, self.channel)
            return True
        except ImportError:
            logger.warning("python-can not installed; cannot connect to CAN hardware")
            return False
        except Exception as e:
            logger.warning("%s BMS connection failed: %s", self.manufacturer, e)
            return False

    def disconnect(self) -> None:
        """Close CAN bus connection."""
        if self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
        self._connected = False

    @abstractmethod
    def read_telemetry(self) -> BMSTelemetry:
        """Read and decode all CAN messages into BMSTelemetry."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check adapter connection health."""

    @abstractmethod
    def get_manufacturer_info(self) -> dict[str, Any]:
        """Return manufacturer-specific metadata."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ── CAN Message Parsing Helpers ─────────────────────────────────────────────


def unpack_u8(data: bytes, offset: int = 0) -> int:
    """Unpack unsigned 8-bit integer from CAN data."""
    if len(data) < offset + 1:
        raise ValueError(f"Data too short: need {offset + 1} bytes, got {len(data)}")
    return data[offset]


def unpack_u16_be(data: bytes, offset: int = 0) -> int:
    """Unpack unsigned 16-bit big-endian from CAN data."""
    if len(data) < offset + 2:
        raise ValueError(f"Data too short: need {offset + 2} bytes, got {len(data)}")
    return struct.unpack_from(">H", data, offset)[0]


def unpack_u16_le(data: bytes, offset: int = 0) -> int:
    """Unpack unsigned 16-bit little-endian from CAN data."""
    if len(data) < offset + 2:
        raise ValueError(f"Data too short: need {offset + 2} bytes, got {len(data)}")
    return struct.unpack_from("<H", data, offset)[0]


def unpack_i16_be(data: bytes, offset: int = 0) -> int:
    """Unpack signed 16-bit big-endian from CAN data."""
    if len(data) < offset + 2:
        raise ValueError(f"Data too short: need {offset + 2} bytes, got {len(data)}")
    return struct.unpack_from(">h", data, offset)[0]


def unpack_i16_le(data: bytes, offset: int = 0) -> int:
    """Unpack signed 16-bit little-endian from CAN data."""
    if len(data) < offset + 2:
        raise ValueError(f"Data too short: need {offset + 2} bytes, got {len(data)}")
    return struct.unpack_from("<h", data, offset)[0]


def unpack_u32_be(data: bytes, offset: int = 0) -> int:
    """Unpack unsigned 32-bit big-endian from CAN data."""
    if len(data) < offset + 4:
        raise ValueError(f"Data too short: need {offset + 4} bytes, got {len(data)}")
    return struct.unpack_from(">I", data, offset)[0]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))


def now_timestamp() -> float:
    """Current monotonic timestamp for telemetry metadata."""
    return time.time()
