"""Manufacturer-specific BMS telemetry adapters.

Provides Tesla, BYD, and Nio BMS adapters that implement
BaseBMSAdapter and decode real CAN message frames into
the unified BMSTelemetry data model.
"""

from __future__ import annotations

from .base import BaseBMSAdapter
from .byd import BYDBMSAdapter
from .nio import NioBMSAdapter
from .tesla import TeslaBMSAdapter

__all__ = [
    "BaseBMSAdapter",
    "BYDBMSAdapter",
    "NioBMSAdapter",
    "TeslaBMSAdapter",
]
