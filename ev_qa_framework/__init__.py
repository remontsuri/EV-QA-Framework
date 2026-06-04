"""EV-QA-Framework package"""
from .framework import EVQAFramework
from .cell_balance import CellBalanceAnalyzer
from .dbc_parser import DBCParser, builtin_dbc
from .metrics import (
    battery_temperature_celsius,
    battery_voltage_volts,
    battery_current_amps,
    battery_soc_percent,
    battery_soh_percent,
    battery_anomaly_total,
    battery_cell_imbalance_max,
)
from .models import BatteryTelemetryModel
from .analysis import EVBatteryAnalyzer, AnomalyDetector
from .can_bus import CANBatterySimulator, CANTelemetryReceiver, DBCFileSimulator
from .soh_predictor import SOHPredictor
from .config import FrameworkConfig, SafetyThresholds, MLConfig

__version__ = "1.0.0"
__all__ = [
    "EVQAFramework",
    "BatteryTelemetryModel",
    "BatteryCellDataModel",
    "EVBatteryAnalyzer",
    "AnomalyDetector",
    "SOHPredictor",
    "CANBatterySimulator",
    "CANTelemetryReceiver",
    "DBCFileSimulator",
    "DBCParser",
    "builtin_dbc",
    "CellBalanceAnalyzer",
    "FrameworkConfig",
    "SafetyThresholds",
    "MLConfig",
]
