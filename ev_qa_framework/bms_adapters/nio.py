"""Nio BMS CAN adapter.

Decodes Nio battery management system CAN frames.

CAN ID mapping:
    0x1A0 — pack voltage (V)
    0x1A1 — pack current (A)
    0x1A2 — temperature sensors (°C)
    0x1A3 — state of health (%)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ev_qa_framework.bms_protocol import BMSTelemetry

from .base import (
    BaseBMSAdapter,
    clamp,
    now_timestamp,
    unpack_i16_le,
    unpack_u8,
    unpack_u16_le,
)

logger = logging.getLogger(__name__)

# Nio CAN IDs
CAN_ID_PACK_VOLTAGE = 0x1A0
CAN_ID_PACK_CURRENT = 0x1A1
CAN_ID_TEMPERATURE = 0x1A2
CAN_ID_SOH = 0x1A3


class NioBMSAdapter(BaseBMSAdapter):
    """Nio BMS adapter — decodes Nio-specific CAN frames.

    Args:
        channel: CAN interface name (e.g. 'can0', 'vcan0').
        bitrate: CAN bus bitrate (default 500000).
    """

    manufacturer = "nio"
    can_ids = {
        "pack_voltage": CAN_ID_PACK_VOLTAGE,
        "pack_current": CAN_ID_PACK_CURRENT,
        "temperature": CAN_ID_TEMPERATURE,
        "soh": CAN_ID_SOH,
    }

    def __init__(self, channel: str = "can0", bitrate: int = 500_000):
        super().__init__(channel=channel, bitrate=bitrate)
        self._latest: dict[int, bytes] = {}

    def connect(self) -> bool:
        """Connect to CAN bus. Lazy-imports python-can."""
        try:
            import can  # noqa: F401

            self._bus = can.interface.Bus(
                channel=self.channel,
                interface="socketcan",
                bitrate=self.bitrate,
            )
            self._connected = True
            logger.info("Nio BMS connected on %s", self.channel)
            return True
        except ImportError:
            logger.warning("python-can not installed; cannot connect to CAN hardware")
            return False
        except Exception as e:
            logger.warning("Nio BMS connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from CAN bus."""
        if self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
        self._connected = False

    def read_telemetry(self) -> BMSTelemetry:
        """Read CAN frames and decode into BMSTelemetry."""
        if not self._connected or not self._bus:
            return BMSTelemetry(protocol="nio_can", source=self.channel)

        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                msg = self._bus.recv(timeout=0.1)
                if msg is not None:
                    self._latest[msg.arbitration_id] = msg.data
        except Exception as e:
            logger.error("Nio CAN read error: %s", e)
            return BMSTelemetry(protocol="nio_can", source=self.channel)

        return self._decode_all()

    def _decode_all(self) -> BMSTelemetry:
        """Decode all cached CAN frames into BMSTelemetry."""
        telemetry = BMSTelemetry(
            protocol="nio_can",
            timestamp=now_timestamp(),
            source=self.channel,
        )

        if CAN_ID_PACK_VOLTAGE in self._latest:
            telemetry.pack_voltage = decode_voltage(
                self._latest[CAN_ID_PACK_VOLTAGE]
            )

        if CAN_ID_PACK_CURRENT in self._latest:
            telemetry.pack_current = decode_current(
                self._latest[CAN_ID_PACK_CURRENT]
            )

        if CAN_ID_TEMPERATURE in self._latest:
            t_max, t_min, t_avg = decode_temperature(
                self._latest[CAN_ID_TEMPERATURE]
            )
            telemetry.temperature_max = t_max
            telemetry.temperature_min = t_min
            telemetry.temperature_avg = t_avg

        if CAN_ID_SOH in self._latest:
            telemetry.soh = decode_soh(self._latest[CAN_ID_SOH])

        return telemetry

    def health_check(self) -> dict[str, Any]:
        """Check Nio BMS adapter health."""
        return {
            "manufacturer": "nio",
            "channel": self.channel,
            "connected": self._connected,
            "can_ids": self.can_ids,
            "status": "healthy" if self._connected else "disconnected",
        }

    def get_manufacturer_info(self) -> dict[str, Any]:
        """Return Nio-specific metadata."""
        return {
            "manufacturer": "nio",
            "protocol": "nio_can",
            "can_ids": self.can_ids,
            "description": "Nio Battery Swap BMS CAN decoder",
        }


# ── Nio CAN Decode Functions ────────────────────────────────────────────────


def decode_voltage(data: bytes) -> float:
    """Decode Nio CAN ID 0x1A0: pack voltage.

    Byte layout:
        [0:2] — pack voltage in 0.1V units (unsigned, little-endian)

    Returns:
        Pack voltage in volts.
    """
    raw = unpack_u16_le(data, 0)
    return raw * 0.1


def decode_current(data: bytes) -> float:
    """Decode Nio CAN ID 0x1A1: pack current.

    Byte layout:
        [0:2] — pack current in 0.1A units (signed, little-endian, positive = discharge)

    Returns:
        Pack current in amps.
    """
    raw = unpack_i16_le(data, 0)
    return raw * 0.1


def decode_temperature(data: bytes) -> tuple[float, float, float]:
    """Decode Nio CAN ID 0x1A2: temperature sensors.

    Byte layout:
        [0] — max temperature (unsigned, offset -40°C)
        [1] — min temperature (unsigned, offset -40°C)
        [2] — avg temperature (unsigned, offset -40°C)

    Returns:
        (t_max, t_min, t_avg) in °C.
    """
    t_max = unpack_u8(data, 0) - 40.0
    t_min = unpack_u8(data, 1) - 40.0
    t_avg = unpack_u8(data, 2) - 40.0
    return (t_max, t_min, t_avg)


def decode_soh(data: bytes) -> float:
    """Decode Nio CAN ID 0x1A3: state of health.

    Byte layout:
        [0] — SOH in 1% units (unsigned 8-bit)

    Returns:
        SOH as percentage (0.0–100.0).
    """
    raw = unpack_u8(data, 0)
    return clamp(float(raw), 0.0, 100.0)
