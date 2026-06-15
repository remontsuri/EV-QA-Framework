"""
Hardware-in-the-Loop (HIL) interface for EV-QA-Framework.

Provides CAN bus communication for real hardware testing.
Gracefully degrades to simulation mode if python-can is not installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import FrameworkConfig

logger = logging.getLogger(__name__)

# Try to import python-can
try:
    import can

    HAS_CAN = True
except ImportError:
    HAS_CAN = False
    logger.warning("python-can not installed. HIL running in simulation mode.")


@dataclass
class CANMessage:
    """Represents a CAN message."""

    arbitration_id: int
    data: bytes
    timestamp: float = 0.0
    is_extended: bool = False
    dlc: int = 8

    @classmethod
    def from_can_msg(cls, msg) -> CANMessage:
        return cls(
            arbitration_id=msg.arbitration_id,
            data=bytes(msg.data),
            timestamp=msg.timestamp if hasattr(msg, "timestamp") else time.time(),
            is_extended=msg.is_extended_id if hasattr(msg, "is_extended_id") else False,
            dlc=msg.dlc if hasattr(msg, "dlc") else len(msg.data),
        )

    def to_can_msg(self):
        """Convert to python-can Message."""
        if not HAS_CAN:
            raise RuntimeError("python-can not installed")
        msg = can.Message(
            arbitration_id=self.arbitration_id,
            data=list(self.data),
            is_extended_id=self.is_extended,
        )
        return msg


@dataclass
class HILTestResult:
    """Result of a HIL test."""

    test_name: str
    passed: bool
    duration_s: float
    messages_sent: int = 0
    messages_received: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_s": self.duration_s,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "errors": self.errors,
            "warnings": self.warnings,
            "data": self.data,
        }


class HILInterface:
    """
    Hardware-in-the-Loop interface for CAN bus communication.

    Supports both real CAN hardware (via python-can) and simulation mode.
    """

    def __init__(
        self,
        channel: str = "vcan0",
        bustype: str = "virtual",
        bitrate: int = 500000,
        simulation: bool = False,
    ):
        self.channel = channel
        self.bustype = bustype
        self.bitrate = bitrate
        self.simulation = simulation or not HAS_CAN
        self.bus = None
        self._sim_messages: list[CANMessage] = []

        if not self.simulation:
            try:
                self.bus = can.interface.Bus(channel=channel, bustype=bustype, bitrate=bitrate)
                logger.info(f"CAN bus opened: {channel}@{bitrate}bps")
            except Exception as e:
                logger.warning(f"Failed to open CAN bus: {e}. Using simulation mode.")
                self.simulation = True

    def send(self, msg: CANMessage):
        """Send a CAN message."""
        if self.simulation:
            self._sim_messages.append(msg)
        elif self.bus:
            try:
                self.bus.send(msg.to_can_msg())
            except Exception as e:
                logger.error(f"CAN send error: {e}")

    def receive(self, timeout: float = 1.0) -> Optional[CANMessage]:
        """Receive a CAN message."""
        if self.simulation:
            if self._sim_messages:
                return self._sim_messages.pop(0)
            return None
        elif self.bus:
            try:
                msg = self.bus.recv(timeout)
                if msg:
                    return CANMessage.from_can_msg(msg)
            except Exception as e:
                logger.error(f"CAN receive error: {e}")
        return None

    def send_telemetry(
        self,
        voltage: float,
        current: float,
        temperature: float,
        soc: float,
        msg_id: int = 0x100,
    ):
        """Send telemetry data as CAN message."""
        # Pack data: voltage (2 bytes), current (2 bytes), temp (1 byte), soc (1 byte)
        v_raw = int(voltage * 10)  # 0.1V resolution
        i_raw = int((current + 500) * 10)  # offset for negative current
        t_raw = int(temperature + 50)  # offset for negative temp
        soc_raw = int(soc * 2)  # 0.5% resolution

        data = [
            (v_raw >> 8) & 0xFF,
            v_raw & 0xFF,
            (i_raw >> 8) & 0xFF,
            i_raw & 0xFF,
            t_raw & 0xFF,
            soc_raw & 0xFF,
            0,
            0,
        ]

        msg = CANMessage(arbitration_id=msg_id, data=bytes(data))
        self.send(msg)

    def close(self):
        """Close CAN bus."""
        if self.bus:
            try:
                self.bus.shutdown()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BMSHardwareEmulator:
    """
    Emulates BMS hardware for HIL testing.

    Generates realistic CAN messages with battery telemetry.
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()

    def generate_telemetry_message(
        self,
        msg_id: int = 0x100,
        voltage_range: tuple[float, float] | None = None,
        current_profile: str | None = None,
        temperature_range: tuple[float, float] | None = None,
    ) -> CANMessage:
        """Generate a single telemetry CAN message."""
        # Generate realistic telemetry values
        if voltage_range:
            voltage = np.random.uniform(*voltage_range)
        else:
            voltage = np.random.normal(400, 5)
        if current_profile == "charge":
            current = abs(np.random.normal(50, 10))
        elif current_profile == "discharge":
            current = -abs(np.random.normal(50, 10))
        elif current_profile == "cycle":
            current = np.random.uniform(-50, 50)
        else:
            current = np.random.normal(50, 10)
        if temperature_range:
            temperature = np.random.uniform(*temperature_range)
        else:
            temperature = np.random.normal(30, 3)
        soc = np.random.uniform(20, 95)

        v_raw = int(voltage * 10)
        i_raw = int((current + 500) * 10)
        t_raw = int(temperature + 50)
        soc_raw = int(soc * 2)

        msg_data = [
            (v_raw >> 8) & 0xFF,
            v_raw & 0xFF,
            (i_raw >> 8) & 0xFF,
            i_raw & 0xFF,
            t_raw & 0xFF,
            soc_raw & 0xFF,
            0,
            0,
        ]

        return CANMessage(arbitration_id=msg_id, data=bytes(msg_data))

    def generate_cycle(
        self,
        n_messages: int = 100,
        msg_id: int = 0x100,
        voltage_range: tuple[float, float] | None = None,
        current_profile: str | None = None,
        temperature_range: tuple[float, float] | None = None,
    ) -> list[CANMessage]:
        """Generate a cycle of telemetry messages."""
        messages = []
        for _ in range(n_messages):
            msg = self.generate_telemetry_message(
                msg_id=msg_id,
                voltage_range=voltage_range,
                current_profile=current_profile,
                temperature_range=temperature_range,
            )
            messages.append(msg)
        return messages


class HILTestRunner:
    """
    Run HIL tests with real or simulated CAN hardware.
    """

    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        channel: str = "vcan0",
        simulation: bool = False,
    ):
        self.config = config or FrameworkConfig()
        self.hil = HILInterface(channel=channel, simulation=simulation)
        self.emulator = BMSHardwareEmulator(self.config)

    def run_hil_test(
        self,
        test_profile: dict,
        duration: float = 10.0,
    ) -> HILTestResult:
        """
        Run a HIL test.

        Args:
            test_profile: dict with test parameters:
                - name: test name (str)
                - voltage_range: (min_v, max_v) tuple
                - current_profile: 'charge' | 'discharge' | 'cycle'
                - temperature_range: (min_c, max_c) tuple
            duration: test duration in seconds

        Returns:
            HILTestResult
        """
        test_name = test_profile.get("name", "unnamed_test")
        voltage_range = test_profile.get("voltage_range")
        current_profile = test_profile.get("current_profile")
        temperature_range = test_profile.get("temperature_range")
        start_time = time.time()
        messages_sent = 0
        messages_received = 0
        errors = []
        warnings = []

        try:
            # Generate and send messages
            n_messages = int(duration * 10)  # 10 messages per second
            messages = self.emulator.generate_cycle(n_messages)

            for msg in messages:
                self.hil.send(msg)
                messages_sent += 1

                # Try to receive response
                response = self.hil.receive(timeout=0.1)
                if response:
                    messages_received += 1

            elapsed = time.time() - start_time

            return HILTestResult(
                test_name=test_name,
                passed=len(errors) == 0,
                duration_s=elapsed,
                messages_sent=messages_sent,
                messages_received=messages_received,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            errors.append(str(e))
            return HILTestResult(
                test_name=test_name,
                passed=False,
                duration_s=elapsed,
                messages_sent=messages_sent,
                messages_received=messages_received,
                errors=errors,
            )

    def compare_expected_vs_actual(
        self,
        expected: pd.DataFrame,
        actual: pd.DataFrame,
    ) -> dict:
        """Compare expected vs actual telemetry."""
        diff = {}
        for col in expected.columns:
            if col in actual.columns:
                exp_vals = expected[col].values
                act_vals = actual[col].values
                min_len = min(len(exp_vals), len(act_vals))
                if min_len > 0:
                    errors = exp_vals[:min_len] - act_vals[:min_len]
                    diff[col] = {
                        "mae": float(np.mean(np.abs(errors))),
                        "max_error": float(np.max(np.abs(errors))),
                        "mean_error": float(np.mean(errors)),
                    }
        return diff

    def generate_hil_report(self, results: list[HILTestResult]) -> dict:
        """Generate a HIL test report."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "total_duration_s": sum(r.duration_s for r in results),
            "total_messages_sent": sum(r.messages_sent for r in results),
            "total_messages_received": sum(r.messages_received for r in results),
            "results": [r.to_dict() for r in results],
        }
