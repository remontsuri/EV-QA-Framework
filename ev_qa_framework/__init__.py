"""EV-QA-Framework package"""
from .analysis import AnomalyDetector, EVBatteryAnalyzer
from .can_bus import (
    CANBatterySimulator,
    CANBusOffError,
    CANConnectionError,
    CANHardwareInterface,
    CANHardwareNotFoundError,
    CANTelemetryReceiver,
    CANTimeoutError,
    DBCFileSimulator,
    HardwareCANError,
    OBD2Adapter,
    OBD2ConnectionError,
    OBD2ProtocolError,
    detect_can_interfaces,
    find_available_can_channel,
    find_hardware_can_interfaces,
)
from .cell_balance import CellBalanceAnalyzer
from .chemistries import (
    ALL_CHEMISTRIES,
    BatteryChemistryProfile,
    CellImbalanceThresholds,
    ChemistryKey,
    SOHDegradationParams,
    get_profile,
    list_profiles,
    load_custom_profile_from_file,
    register_custom_profile,
)
from .config import DEFAULT_CONFIG, TESLA_CONFIG, FrameworkConfig, MLConfig, SafetyThresholds
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

__version__ = "1.1.0"
__all__ = [
    "EVQAFramework",
    "BatteryTelemetryModel",
    "BatteryCellDataModel",
    "EVBatteryAnalyzer",
    "AnomalyDetector",
    "SOHPredictor",
    "ThermalRunawayPredictor",
    "CANBatterySimulator",
    "CANHardwareInterface",
    "CANTelemetryReceiver",
    "DBCFileSimulator",
    "HardwareCANError",
    "CANConnectionError",
    "CANBusOffError",
    "CANTimeoutError",
    "CANHardwareNotFoundError",
    "OBD2Adapter",
    "OBD2ConnectionError",
    "OBD2ProtocolError",
    "detect_can_interfaces",
    "find_hardware_can_interfaces",
    "find_available_can_channel",
    "DBCParser",
    "builtin_dbc",
    "CellBalanceAnalyzer",
    # Chemistry profiles
    "BatteryChemistryProfile",
    "SOHDegradationParams",
    "CellImbalanceThresholds",
    "ChemistryKey",
    "ALL_CHEMISTRIES",
    "get_profile",
    "list_profiles",
    "register_custom_profile",
    "load_custom_profile_from_file",
    # Configuration
    "FrameworkConfig",
    "SafetyThresholds",
    "MLConfig",
    "DEFAULT_CONFIG",
    "TESLA_CONFIG",
]
