"""BYD BMS CAN adapter.

Decodes BYD battery management system CAN frames.

WARNING: CAN IDs are community reverse-engineered values, not manufacturer-documented.
Verify against your specific vehicle model and firmware version.

CAN ID mapping (extended 29-bit, community-sourced):
    0x1806E5F4 — pack voltage
    0x1806E5F5 — current and temperature
    0x1806E5F6 — individual cell voltages
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ev_qa_framework.bms_protocol import BMSTelemetry

from .base import (
    BaseBMSAdapter,
    now_timestamp,
    unpack_u8,
    unpack_u16_be,
)

logger = logging.getLogger(__name__)

# BYD CAN IDs (J1939-style extended IDs)
CAN_ID_VOLTAGE = 0x1806E5F4
CAN_ID_CURRENT_TEMP = 0x1806E5F5
CAN_ID_CELLS = 0x1806E5F6

# BYD cell voltage message: up to 4 cells per 8-byte frame
BYD_CELLS_PER_FRAME = 4
# BYD uses 0.001V resolution for cell voltages
BYD_CELL_VOLTAGE_SCALE = 0.001
# BYD pack voltage scale: 0.1V
BYD_PACK_VOLTAGE_SCALE = 0.1
# BYD current scale: 0.1A (signed)
BYD_CURRENT_SCALE = 0.1
# BYD temperature: 1°C offset -40
BYD_TEMP_OFFSET = -40.0


class BYDBMSAdapter(BaseBMSAdapter):
    """BYD BMS adapter — decodes BYD-specific CAN frames.

    Args:
        channel: CAN interface name (e.g. 'can0', 'vcan0').
        bitrate: CAN bus bitrate (default 500000).
    """

    manufacturer = "byd"
    can_ids = {
        "voltage": CAN_ID_VOLTAGE,
        "current_temp": CAN_ID_CURRENT_TEMP,
        "cells": CAN_ID_CELLS,
    }

    def __init__(self, channel: str = "can0", bitrate: int = 500_000):
        super().__init__(channel=channel, bitrate=bitrate)
        self._latest: dict[int, bytes] = {}

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
            return BMSTelemetry(protocol="byd_can", source=self.channel)

        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                msg = self._bus.recv(timeout=0.1)
                if msg is not None:
                    self._latest[msg.arbitration_id] = msg.data
        except Exception as e:
            logger.error("BYD CAN read error: %s", e)
            return BMSTelemetry(protocol="byd_can", source=self.channel)

        return self._decode_all()

    def _decode_all(self) -> BMSTelemetry:
        """Decode all cached CAN frames into BMSTelemetry."""
        telemetry = BMSTelemetry(
            protocol="byd_can",
            timestamp=now_timestamp(),
            source=self.channel,
        )

        if CAN_ID_VOLTAGE in self._latest:
            telemetry.pack_voltage = decode_voltage(self._latest[CAN_ID_VOLTAGE])

        if CAN_ID_CURRENT_TEMP in self._latest:
            current, t_max, t_min, t_avg = decode_current_temp(
                self._latest[CAN_ID_CURRENT_TEMP]
            )
            telemetry.pack_current = current
            telemetry.temperature_max = t_max
            telemetry.temperature_min = t_min
            telemetry.temperature_avg = t_avg

        if CAN_ID_CELLS in self._latest:
            cells = decode_cells(self._latest[CAN_ID_CELLS])
            telemetry.cell_voltages = cells
            if cells:
                telemetry.cell_voltage_min = min(cells)
                telemetry.cell_voltage_max = max(cells)
                telemetry.cell_voltage_delta = max(cells) - min(cells)

        return telemetry

    def health_check(self) -> dict[str, Any]:
        """Check BYD BMS adapter health."""
        return {
            "manufacturer": "byd",
            "channel": self.channel,
            "connected": self._connected,
            "can_ids": self.can_ids,
            "status": "healthy" if self._connected else "disconnected",
        }

    def get_manufacturer_info(self) -> dict[str, Any]:
        """Return BYD-specific metadata."""
        return {
            "manufacturer": "byd",
            "protocol": "byd_can",
            "can_ids": self.can_ids,
            "description": "BYD Blade Battery BMS CAN decoder",
        }


# ── BYD CAN Decode Functions ────────────────────────────────────────────────


def decode_voltage(data: bytes) -> float:
    """Decode BYD CAN ID 0x1806E5F4: pack voltage.

    Byte layout:
        [0:2] — total voltage in 0.1V units (unsigned, big-endian)

    Returns:
        Pack voltage in volts.
    """
    raw = unpack_u16_be(data, 0)
    return raw * BYD_PACK_VOLTAGE_SCALE


def decode_current_temp(data: bytes) -> tuple[float, float, float, float]:
    """Decode BYD CAN ID 0x1806E5F5: current and temperatures.

    Byte layout:
        [0:2] — current in 0.1A units (signed, big-endian, positive = discharge)
        [2]   — max temperature (unsigned, offset -40°C)
        [3]   — min temperature (unsigned, offset -40°C)
        [4]   — avg temperature (unsigned, offset -40°C)

    Returns:
        (current_a, t_max, t_min, t_avg).
    """
    raw_current = unpack_u16_be(data, 0)
    # Treat as signed 16-bit
    if raw_current > 0x7FFF:
        raw_current -= 0x10000
    current = raw_current * BYD_CURRENT_SCALE

    t_max = unpack_u8(data, 2) + BYD_TEMP_OFFSET
    t_min = unpack_u8(data, 3) + BYD_TEMP_OFFSET
    t_avg = unpack_u8(data, 4) + BYD_TEMP_OFFSET
    return (current, t_max, t_min, t_avg)


def decode_cells(data: bytes) -> list[float]:
    """Decode BYD CAN ID 0x1806E5F6: individual cell voltages.

    Byte layout (up to 4 cells per frame):
        [0:2] — cell 1 voltage in 0.001V units (unsigned, big-endian)
        [2:4] — cell 2 voltage
        [4:6] — cell 3 voltage
        [6:8] — cell 4 voltage

    Returns:
        List of cell voltages in volts.
    """
    cells = []
    for i in range(BYD_CELLS_PER_FRAME):
        offset = i * 2
        if offset + 2 > len(data):
            break
        raw = unpack_u16_be(data, offset)
        cells.append(raw * BYD_CELL_VOLTAGE_SCALE)
    return cells
