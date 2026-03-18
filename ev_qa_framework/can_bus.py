import can
import time
import random
import threading
from typing import Dict, Any, Optional

class CANBatterySimulator:
    """
    Simulates battery telemetry messages on a virtual CAN bus.

    Messages:
    - 0x101: Voltage (uint16, 0.1V res) and Current (int16, 0.1A res)
    - 0x102: Temperature (int8, 1°C res) and SOC (uint8, 1% res)
    """

    def __init__(self, interface: str = 'virtual', channel: str = 'vcan0'):
        self.interface = interface
        self.channel = channel
        self.bus: Optional[can.interface.Bus] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            # Fallback for systems without virtual CAN support in kernel
            print(f"Virtual CAN not supported, using simple emulation: {e}")
            self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        if self.bus:
            self.bus.shutdown()

    def _run(self):
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
            msg1 = can.Message(arbitration_id=0x101, data=data1, is_extended_id=False)

            # Pack 0x102: Temperature (1 byte), SOC (1 byte)
            data2 = [temp & 0xFF, soc & 0xFF, 0, 0, 0, 0, 0, 0]
            msg2 = can.Message(arbitration_id=0x102, data=data2, is_extended_id=False)

            if self.bus:
                try:
                    self.bus.send(msg1)
                    self.bus.send(msg2)
                except:
                    pass

            time.sleep(1.0)

class CANTelemetryReceiver:
    """
    Receives and decodes battery telemetry from CAN bus.
    """

    def __init__(self, interface: str = 'virtual', channel: str = 'vcan0'):
        self.interface = interface
        self.channel = channel
        self.latest_data: Dict[str, Any] = {
            'voltage': 0.0,
            'current': 0.0,
            'temperature': 0.0,
            'soc': 0.0
        }
        self.bus: Optional[can.interface.Bus] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except:
            self.bus = None

        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        if self.bus:
            self.bus.shutdown()

    def _run(self):
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
                    self.latest_data['voltage'] = v_scaled / 10.0
                    self.latest_data['current'] = c_scaled / 10.0
                elif msg.arbitration_id == 0x102:
                    temp = msg.data[0]
                    if temp > 0x7F:
                        temp -= 0x100
                    self.latest_data['temperature'] = float(temp)
                    self.latest_data['soc'] = float(msg.data[1])

    def get_telemetry(self) -> Dict[str, Any]:
        return self.latest_data.copy()
