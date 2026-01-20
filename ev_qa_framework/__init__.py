"""EV-QA-Framework package"""
from .framework import EVQAFramework, BatteryTelemetry
from .analysis import EVBatteryAnalyzer, AnomalyDetector

__version__ = "1.0.0"
__all__ = ["EVQAFramework", "BatteryTelemetry", "EVBatteryAnalyzer", "AnomalyDetector"]