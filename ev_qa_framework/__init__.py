"""EV-QA-Framework package"""
from .framework import EVQAFramework
from .models import BatteryTelemetryModel
from .analysis import EVBatteryAnalyzer, AnomalyDetector
from .soh_predictor import SOHPredictor
from .can_bus import CANBatterySimulator, CANTelemetryReceiver
from .config import FrameworkConfig, SafetyThresholds, MLConfig

__version__ = "1.0.0"
__all__ = [
    "EVQAFramework", 
    "BatteryTelemetryModel", 
    "EVBatteryAnalyzer", 
    "AnomalyDetector",
    "SOHPredictor",
    "CANBatterySimulator",
    "CANTelemetryReceiver",
    "FrameworkConfig",
    "SafetyThresholds",
    "MLConfig"
]