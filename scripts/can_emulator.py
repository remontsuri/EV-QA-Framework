import can
import time
import random
import json


# Virtual CAN bus setup
# In Windows, we can use 'virtual' or 'serial' if drivers are installed, 
# but for a demo, we'll just simulate the CAN protocol logic.

class CANEmulator:
    def __init__(self, channel='vcan0', bustype='virtual'):
        self.channel = channel
        self.bustype = bustype
        # Note: 'virtual' might require specific setup on Windows.
        # For this demo, we'll simulate the "capturing" part.
        
    def generate_telemetry(self):
        """Generates realistic battery telemetry"""
        return {
            "vin": "EV" + "".join([str(random.randint(0,9)) for _ in range(15)]),
            "voltage": round(random.uniform(360, 420), 2),
            "current": round(random.uniform(10, 200), 2),
            "temperature": round(random.uniform(25, 55), 1),
            "soc": round(random.uniform(10, 95), 1),
            "soh": 98.5
        }

    def run(self):
        print(f"⚡ Starting CAN Emulation on {self.channel}...")
        while True:
            data = self.generate_telemetry()
            
            # Simulate a 10% chance of a "spike" (anomaly)
            if random.random() > 0.90:
                data["temperature"] += 15
                print(f"🚨 ANOMALY GENERATED: Temp spike to {data['temperature']}°C")
            
            # In a real CAN system, we'd do:
            # msg = can.Message(arbitration_id=0x123, data=[...])
            # bus.send(msg)
            
            print(f"📡 Sending CAN Frame: ID=0x501 [V={data['voltage']}V, T={data['temperature']}°C]")
            
            # Send to our dashboard API (for demonstration/integration)
            try:
                # This could be a webhook or we could just write to a shared pipe/socket
                pass
            except Exception as e:
                print(f"Error sending data: {e}")
                
            time.sleep(1)

if __name__ == "__main__":
    emulator = CANEmulator()
    emulator.run()
