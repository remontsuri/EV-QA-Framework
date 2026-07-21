"""EV-QA-Framework — ML-powered QA Framework for Electric Vehicle & IoT Battery Testing."""

__version__ = "2.5.0"

# Core imports — always available (lightweight)
from .analysis import AnomalyDetector, EVBatteryAnalyzer
from .config import (
    FrameworkConfig,
    MLConfig,
    SafetyThresholds,
    get_default_config,
    get_tesla_config,
)
from .framework import EVQAFramework
from .models import BatteryCellDataModel, BatteryTelemetryModel
from .thermal_runaway import ThermalRunawayPredictor

# Lazy-loaded modules — imported on first access
_LAZY_IMPORTS: dict[str, str] = {
    # CAN bus
    "CANBatterySimulator": ".can_bus",
    "CANHardwareInterface": ".can_bus",
    "CANTelemetryReceiver": ".can_bus",
    "DBCFileSimulator": ".can_bus",
    "HardwareCANError": ".can_bus",
    "CANConnectionError": ".can_bus",
    "CANBusOffError": ".can_bus",
    "CANTimeoutError": ".can_bus",
    "CANHardwareNotFoundError": ".can_bus",
    "OBD2Adapter": ".can_bus",
    "OBD2ConnectionError": ".can_bus",
    "OBD2ProtocolError": ".can_bus",
    "detect_can_interfaces": ".can_bus",
    "find_hardware_can_interfaces": ".can_bus",
    "find_available_can_channel": ".can_bus",
    # DBC
    "DBCParser": ".dbc_parser",
    "builtin_dbc": ".dbc_parser",
    # Cell balance
    "CellBalanceAnalyzer": ".cell_balance",
    # Chemistry
    "BatteryChemistryProfile": ".chemistries",
    "SOHDegradationParams": ".chemistries",
    "CellImbalanceThresholds": ".chemistries",
    "ChemistryKey": ".chemistries",
    "ALL_CHEMISTRIES": ".chemistries",
    "OCVCurve": ".chemistries",
    "OCV_LFP": ".chemistries",
    "OCV_NMC": ".chemistries",
    "OCV_NCA": ".chemistries",
    "AgingModel": ".chemistries",
    "AGING_LFP": ".chemistries",
    "AGING_NMC": ".chemistries",
    "AGING_NCA": ".chemistries",
    "ThermalParams": ".chemistries",
    "ThermalModel": ".chemistries",
    "THERMAL_LFP": ".chemistries",
    "THERMAL_NMC": ".chemistries",
    "THERMAL_NCA": ".chemistries",
    "get_profile": ".chemistries",
    "list_profiles": ".chemistries",
    "register_custom_profile": ".chemistries",
    "load_custom_profile_from_file": ".chemistries",
    # v2.0 modules
    "battery_scoring": ".battery_scoring",
    "physics_features": ".physics_features",
    "fleet_analytics": ".fleet_analytics",
    "digital_twin": ".digital_twin",
    "v2g_scenarios": ".v2g_scenarios",
    "automl": ".automl",
    "soh_transformer": ".soh_transformer",
    "hil": ".hil",
    "bms_protocol": ".bms_protocol",
    "modbus": ".modbus",
    # v2.0 classes
    "BatteryScorer": ".battery_scoring",
    "PhysicsFeatureExtractor": ".physics_features",
    "FleetAnalytics": ".fleet_analytics",
    "FleetAlert": ".fleet_analytics",
    "BatteryDigitalTwin": ".digital_twin",
    "BatteryState": ".digital_twin",
    "V2GScenarioGenerator": ".v2g_scenarios",
    "V2GHealthAnalyzer": ".v2g_scenarios",
    "V2SScenarioGenerator": ".v2g_scenarios",
    "ChargingStationSimulator": ".v2g_scenarios",
    "AutoMLSOH": ".automl",
    "AutoMLAnomaly": ".automl",
    "SOHTransformer": ".soh_transformer",
    "SOHPredictor": ".soh_predictor",
    "HILInterface": ".hil",
    "HILTestRunner": ".hil",
    "HILTestResult": ".hil",
    "CANMessage": ".hil",
    "BMSHardwareEmulator": ".hil",
    "BMSProtocolManager": ".bms_protocol",
    "BMSTelemetry": ".bms_protocol",
    "BMSCANInterface": ".bms_protocol",
    "BMSModbusTCPInterface": ".bms_protocol",
    "BMSModbusRTUInterface": ".bms_protocol",
    "ProtocolType": ".bms_protocol",
    "ModbusTCPClient": ".modbus",
    "ModbusRTUClient": ".modbus",
    "BMS_REGISTER_MAP": ".modbus",
    # BMS adapters
    "BaseBMSAdapter": ".bms_adapters",
    "TeslaBMSAdapter": ".bms_adapters",
    "BYDBMSAdapter": ".bms_adapters",
    "NioBMSAdapter": ".bms_adapters",
    # Metrics
    "battery_anomaly_total": ".metrics",
    "battery_cell_imbalance_max": ".metrics",
    "battery_current_amps": ".metrics",
    "battery_soc_percent": ".metrics",
    "battery_soh_percent": ".metrics",
    "battery_temperature_celsius": ".metrics",
    "battery_voltage_volts": ".metrics",
    # Vector export
    "VectorExporter": ".vector_export",
}

__all__ = list(_LAZY_IMPORTS.keys()) + [
    "EVQAFramework",
    "BatteryTelemetryModel",
    "BatteryCellDataModel",
    "EVBatteryAnalyzer",
    "AnomalyDetector",
    "ThermalRunawayPredictor",
    "FrameworkConfig",
    "SafetyThresholds",
    "MLConfig",
    "get_default_config",
    "get_tesla_config",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache for future access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
