"""EV-QA-Framework package"""
from .analysis import AnomalyDetector, EVBatteryAnalyzer
from .can_bus import CANBatterySimulator, CANTelemetryReceiver, DBCFileSimulator
from .cell_balance import CellBalanceAnalyzer
from .config import FrameworkConfig, MLConfig, SafetyThresholds
from .dbc_parser import DBCParser, builtin_dbc
from .framework import EVQAFramework
from .metrics import (
    battery_anomaly_total,
    battery_cell_imbalance_max,
    battery_current_amps,
    battery_soc_percent,
    battery_soh_percent,
    battery_temperature_celsius,
    battery_voltage_volts,
)
from .models import BatteryCellDataModel, BatteryTelemetryModel
from .soh_predictor import SOHPredictor
from .thermal_runaway import ThermalRunawayPredictor

__version__ = "1.0.0"
__all__ = [
    "EVQAFramework",
    "BatteryTelemetryModel",
    "BatteryCellDataModel",
    "EVBatteryAnalyzer",
    "AnomalyDetector",
    "SOHPredictor",
    "ThermalRunawayPredictor",
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
