"""EV-QA-Framework — ML-powered QA Framework for Electric Vehicle & IoT Battery Testing."""

# Module exports
from . import (
    automl,
    battery_scoring,
    bms_protocol,
    digital_twin,
    fleet_analytics,
    hil,
    modbus,
    physics_features,
    soh_transformer,
    v2g_scenarios,
)
from .analysis import AnomalyDetector, EVBatteryAnalyzer
from .automl import AutoMLAnomaly, AutoMLSOH

# v2.0 module classes
from .battery_scoring import BatteryScorer

# BMS protocol abstraction layer
from .bms_protocol import (
    BMSCANInterface,
    BMSModbusRTUInterface,
    BMSModbusTCPInterface,
    BMSProtocolManager,
    BMSTelemetry,
    ProtocolType,
)
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
    AGING_LFP,
    AGING_NCA,
    AGING_NMC,
    ALL_CHEMISTRIES,
    OCV_LFP,
    OCV_NCA,
    OCV_NMC,
    THERMAL_LFP,
    THERMAL_NCA,
    THERMAL_NMC,
    AgingModel,
    BatteryChemistryProfile,
    CellImbalanceThresholds,
    ChemistryKey,
    OCVCurve,
    SOHDegradationParams,
    ThermalModel,
    ThermalParams,
    get_profile,
    list_profiles,
    load_custom_profile_from_file,
    register_custom_profile,
)
from .config import get_default_config, get_tesla_config, FrameworkConfig, MLConfig, SafetyThresholds
from .dbc_parser import DBCParser, builtin_dbc
from .digital_twin import BatteryDigitalTwin, BatteryState
from .fleet_analytics import FleetAlert, FleetAnalytics
from .framework import EVQAFramework
from .hil import BMSHardwareEmulator, CANMessage, HILInterface, HILTestResult, HILTestRunner
from .metrics import (
    battery_anomaly_total,
    battery_cell_imbalance_max,
    battery_current_amps,
    battery_soc_percent,
    battery_soh_percent,
    battery_temperature_celsius,
    battery_voltage_volts,
)

# Modbus protocol
from .modbus import (
    BMS_REGISTER_MAP,
    ModbusRTUClient,
    ModbusTCPClient,
)
from .models import BatteryCellDataModel, BatteryTelemetryModel
from .physics_features import PhysicsFeatureExtractor
from .soh_predictor import SOHPredictor
from .soh_transformer import SOHTransformer
from .thermal_runaway import ThermalRunawayPredictor
from .v2g_scenarios import V2GHealthAnalyzer, V2GScenarioGenerator

__version__ = "2.3.1"

__all__ = [
    # Core framework
    "EVQAFramework",
    # Models
    "BatteryTelemetryModel",
    "BatteryCellDataModel",
    # Analysis
    "EVBatteryAnalyzer",
    "AnomalyDetector",
    "SOHPredictor",
    "ThermalRunawayPredictor",
    # CAN bus
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
    # DBC
    "DBCParser",
    "builtin_dbc",
    # Cell balance
    "CellBalanceAnalyzer",
    # Chemistry profiles
    "BatteryChemistryProfile",
    "SOHDegradationParams",
    "CellImbalanceThresholds",
    "ChemistryKey",
    "ALL_CHEMISTRIES",
    # New chemistry models (v2.1)
    "OCVCurve",
    "OCV_LFP",
    "OCV_NMC",
    "OCV_NCA",
    "AgingModel",
    "AGING_LFP",
    "AGING_NMC",
    "AGING_NCA",
    "ThermalParams",
    "ThermalModel",
    "THERMAL_LFP",
    "THERMAL_NMC",
    "THERMAL_NCA",
    "get_profile",
    "list_profiles",
    "register_custom_profile",
    "load_custom_profile_from_file",
    # Configuration
    "FrameworkConfig",
    "SafetyThresholds",
    "MLConfig",
    "get_default_config",
    "get_tesla_config",
    # v2.0 modules
    "battery_scoring",
    "physics_features",
    "fleet_analytics",
    "digital_twin",
    "v2g_scenarios",
    "automl",
    "soh_transformer",
    "hil",
    "bms_protocol",
    "modbus",
    # v2.0 classes
    "BatteryScorer",
    "PhysicsFeatureExtractor",
    "FleetAnalytics",
    "FleetAlert",
    "BatteryDigitalTwin",
    "BatteryState",
    "V2GScenarioGenerator",
    "V2GHealthAnalyzer",
    "AutoMLSOH",
    "AutoMLAnomaly",
    "SOHTransformer",
    "HILInterface",
    "HILTestRunner",
    "HILTestResult",
    "CANMessage",
    "BMSHardwareEmulator",
    # BMS protocol
    "BMSProtocolManager",
    "BMSTelemetry",
    "BMSCANInterface",
    "BMSModbusTCPInterface",
    "BMSModbusRTUInterface",
    "ProtocolType",
    # Modbus
    "ModbusTCPClient",
    "ModbusRTUClient",
    "BMS_REGISTER_MAP",
]
