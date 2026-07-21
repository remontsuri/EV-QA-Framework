"""Tesla BMS CAN adapter.

Decodes Tesla Model S/X/3/Y battery management system CAN frames.

CAN ID mapping:
    0x352 — pack voltage (V) and current (A)
    0x353 — state of charge (%)
    0x355 — temperature sensors (°C)
    0x37A — cell voltage statistics (min/max/delta)
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
    unpack_i16_be,
    unpack_u8,
    unpack_u16_be,
)

logger = logging.getLogger(__name__)

# Tesla CAN IDs
CAN_ID_VOLTAGE_CURRENT = 0x352
CAN_ID_SOC = 0x353
CAN_ID_TEMPERATURE = 0x355
CAN_ID_CELL_STATS = 0x37A


class TeslaBMSAdapter(BaseBMSAdapter):
    """Tesla BMS adapter — decodes Tesla-specific CAN frames.

    Args:
        channel: CAN interface name (e.g. 'can0', 'vcan0').
        bitrate: CAN bus bitrate (default 500000).
    """

    manufacturer = "tesla"
    can_ids = {
        "voltage_current": CAN_ID_VOLTAGE_CURRENT,
        "soc": CAN_ID_SOC,
        "temperature": CAN_ID_TEMPERATURE,
        "cell_stats": CAN_ID_CELL_STATS,
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
            logger.info("Tesla BMS connected on %s", self.channel)
            return True
        except ImportError:
            logger.warning("python-can not installed; cannot connect to CAN hardware")
            return False
        except Exception as e:
            logger.warning("Tesla BMS connection failed: %s", e)
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
            return BMSTelemetry(protocol="tesla_can", source=self.channel)

        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                msg = self._bus.recv(timeout=0.1)
                if msg is not None:
                    self._latest[msg.arbitration_id] = msg.data
        except Exception as e:
            logger.error("Tesla CAN read error: %s", e)
            return BMSTelemetry(protocol="tesla_can", source=self.channel)

        return self._decode_all()

    def _decode_all(self) -> BMSTelemetry:
        """Decode all cached CAN frames into BMSTelemetry."""
        telemetry = BMSTelemetry(
            protocol="tesla_can",
            timestamp=now_timestamp(),
            source=self.channel,
        )

        if CAN_ID_VOLTAGE_CURRENT in self._latest:
            v, i = decode_voltage_current(self._latest[CAN_ID_VOLTAGE_CURRENT])
            telemetry.pack_voltage = v
            telemetry.pack_current = i

        if CAN_ID_SOC in self._latest:
            telemetry.soc = decode_soc(self._latest[CAN_ID_SOC])

        if CAN_ID_TEMPERATURE in self._latest:
            t_max, t_min, t_avg = decode_temperature(
                self._latest[CAN_ID_TEMPERATURE]
            )
            telemetry.temperature_max = t_max
            telemetry.temperature_min = t_min
            telemetry.temperature_avg = t_avg

        if CAN_ID_CELL_STATS in self._latest:
            v_min, v_max, delta = decode_cell_stats(
                self._latest[CAN_ID_CELL_STATS]
            )
            telemetry.cell_voltage_min = v_min
            telemetry.cell_voltage_max = v_max
            telemetry.cell_voltage_delta = delta

        return telemetry

    def health_check(self) -> dict[str, Any]:
        """Check Tesla BMS adapter health."""
        return {
            "manufacturer": "tesla",
            "channel": self.channel,
            "connected": self._connected,
            "can_ids": self.can_ids,
            "status": "healthy" if self._connected else "disconnected",
        }

    def get_manufacturer_info(self) -> dict[str, Any]:
        """Return Tesla-specific metadata."""
        return {
            "manufacturer": "tesla",
            "protocol": "tesla_can",
            "can_ids": self.can_ids,
            "description": "Tesla Model S/X/3/Y BMS CAN decoder",
        }


# ── Tesla CAN Decode Functions ──────────────────────────────────────────────


def decode_voltage_current(data: bytes) -> tuple[float, float]:
    """Decode Tesla CAN ID 0x352: pack voltage and current.

    Byte layout (8 bytes):
        [0:2] — pack voltage in 0.01V units (unsigned, big-endian)
        [2:4] — pack current in 0.01A units (signed, big-endian, positive = discharge)

    Returns:
        (voltage_v, current_a) tuple.
    """
    raw_voltage = unpack_u16_be(data, 0)
    raw_current = unpack_i16_be(data, 2)
    voltage = raw_voltage * 0.01
    current = raw_current * 0.01
    return (voltage, current)


def decode_soc(data: bytes) -> float:
    """Decode Tesla CAN ID 0x353: state of charge.

    Byte layout:
        [0] — SOC in 0.5% units (unsigned 8-bit)

    Returns:
        SOC as percentage (0.0–100.0).
    """
    raw_soc = unpack_u8(data, 0)
    soc = raw_soc * 0.5
    return clamp(soc, 0.0, 100.0)


def decode_temperature(data: bytes) -> tuple[float, float, float]:
    """Decode Tesla CAN ID 0x355: temperature sensors.

    Byte layout:
        [0] — max temperature in 1°C units
        [1] — min temperature in 1°C units
        [2] — avg temperature in 1°C units

    Temperature offset: 0 = -40°C (unsigned byte, range -40 to +215).

    Returns:
        (t_max, t_min, t_avg) in °C.
    """
    t_max = unpack_u8(data, 0) - 40.0
    t_min = unpack_u8(data, 1) - 40.0
    t_avg = unpack_u8(data, 2) - 40.0
    return (t_max, t_min, t_avg)


def decode_cell_stats(data: bytes) -> tuple[float, float, float]:
    """Decode Tesla CAN ID 0x37A: cell voltage statistics.

    Byte layout:
        [0:2] — min cell voltage in 0.001V units (unsigned, big-endian)
        [2:4] — max cell voltage in 0.001V units (unsigned, big-endian)
        [4:6] — voltage delta in 0.001V units (unsigned, big-endian)

    Returns:
        (v_min, v_max, delta) in volts.
    """
    v_min = unpack_u16_be(data, 0) * 0.001
    v_max = unpack_u16_be(data, 2) * 0.001
    delta = unpack_u16_be(data, 4) * 0.001
    return (v_min, v_max, delta)
