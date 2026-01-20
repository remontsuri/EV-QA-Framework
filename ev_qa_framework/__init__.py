"""EV-QA-Framework package"""
from .framework import EVQAFramework
from .models import BatteryTelemetryModel
from .analysis import EVBatteryAnalyzer, AnomalyDetector

__version__ = "1.0.0"
__all__ = ["EVQAFramework", "BatteryTelemetryModel", "EVBatteryAnalyzer", "AnomalyDetector"]