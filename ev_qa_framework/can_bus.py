"""
CAN Bus Module: Hardware-level battery telemetry simulation and reception.
"""
import random
import threading
import time
from typing import Any

import can

from .dbc_parser import DBCParser, battery_dbc_content


class CANBatterySimulator:
    """
    Simulates battery telemetry messages on a virtual CAN bus.

    Messages:
    - 0x101: Voltage (uint16, 0.1V res) and Current (int16, 0.1A res)
    - 0x102: Temperature (int8, 1°C res) and SOC (uint8, 1% res)
    """

    def __init__(self, interface: str = "virtual", channel: str = "vcan0"):
        self.interface = interface
        self.channel = channel
        self.bus: can.interface.Bus | None = None
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start simulation"""
        try:
            self.bus = can.interface.Bus(channel=self.channel,
                                         interface=self.interface)
        except (can.CanError, OSError, ValueError) as e:
            # Fallback for systems without virtual CAN support in kernel
            print(f"Virtual CAN not supported, using simple emulation: {e}")
            self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop simulation"""
        self.running = False
        if self._thread:
            self._thread.join()
        if self.bus:
            self.bus.shutdown()

    def _run(self):
        """Internal simulation loop"""
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
                0, 0, 0, 0
            ]
            msg1 = can.Message(arbitration_id=0x101, data=data1,
                               is_extended_id=False)

            # Pack 0x102: Temperature (1 byte), SOC (1 byte)
            data2 = [temp & 0xFF, soc & 0xFF, 0, 0, 0, 0, 0, 0]
            msg2 = can.Message(arbitration_id=0x102, data=data2,
                               is_extended_id=False)

            if self.bus:
                try:
                    self.bus.send(msg1)
                    self.bus.send(msg2)
                except can.CanError:
                    pass

            time.sleep(1.0)


class CANTelemetryReceiver:
    """
    Receives and decodes battery telemetry from CAN bus.
    """

    def __init__(self, interface: str = "virtual", channel: str = "vcan0"):
        self.interface = interface
        self.channel = channel
        self.latest_data: dict[str, Any] = {
            "voltage": 0.0,
            "current": 0.0,
            "temperature": 0.0,
            "soc": 0.0
        }
        self.bus: can.interface.Bus | None = None
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start receiver"""
        try:
            self.bus = can.interface.Bus(channel=self.channel,
                                         interface=self.interface)
        except (can.CanError, OSError, ValueError):
            self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop receiver"""
        self.running = False
        if self._thread:
            self._thread.join()
        if self.bus:
            self.bus.shutdown()

    def _run(self):
        """Internal receiver loop"""
        if not self.bus:
            return

        while self.running:
            msg = self.bus.recv(timeout=1.0)
            if msg:
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
        """Return latest telemetry"""
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
            import os
            import tempfile
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False)
            tmp.write(battery_dbc_content())
            tmp.close()
            self.dbc = DBCParser(tmp.name)
            os.unlink(tmp.name)

        self.running = False
        self._thread: threading.Thread | None = None
        self._bus: can.interface.Bus | None = None

    def start(self, interface: str = "virtual", channel: str = "vcan0"):
        """Start simulation on the given CAN interface."""
        try:
            self._bus = can.interface.Bus(channel=channel, interface=interface)
        except (can.CanError, OSError, ValueError) as e:
            print(f"CAN bus not available, running in log-only mode: {e}")
            self._bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop simulation."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._bus:
            self._bus.shutdown()

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
                if self._bus:
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
        # Generate a value in physical range, convert to raw
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
            bit_pos = sig.start_bit + i if sig.byte_order == "Intel" else sig.start_bit - i
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx >= len(data):
                continue
            if (raw >> i) & 1:
                data[byte_idx] |= 1 << bit_in_byte
            else:
                data[byte_idx] &= ~(1 << bit_in_byte)
