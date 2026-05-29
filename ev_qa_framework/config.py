from __future__ import annotations

"""
EV-QA-Framework Configuration Module
Safety threshold settings and analysis parameters
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with timestamp and severity."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = True,
) -> logging.Logger:
    """Configure structured logging for the EV-QA-Framework.

    Sets up root logger with JSON-formatted output to stdout and
    an optional file handler.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.
        json_format: If True (default), uses JSON formatting; otherwise
                     uses a plain-text format.

    Returns:
        The root logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any pre-existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={"level": level, "log_file": log_file, "json_format": json_format},
    )
    return root_logger


@dataclass
class SafetyThresholds:
    """
    Safety thresholds for battery telemetry validation.

    Attributes:
        max_temperature: Maximum safe temperature (°C)
        min_voltage: Minimum safe voltage (V)
        max_voltage: Maximum safe voltage (V)
        max_temperature_jump: Maximum allowable temperature jump (°C)
        min_soc: Minimum state of charge for warning (%)
        critical_soh: Critical battery state of health (%)
    """
    
    # Temperature thresholds
    max_temperature: float = 60.0
    min_temperature: float = -40.0
    max_temperature_jump: float = 5.0
    
    # Voltage thresholds
    min_voltage: float = 200.0
    max_voltage: float = 900.0
    
    # Charge and health thresholds
    min_soc: float = 10.0  # Low charge warning
    critical_soh: float = 70.0  # Critical battery health
    
    # Current thresholds (optional)
    max_current: Optional[float] = 500.0  # Maximum safe current
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'max_temperature': self.max_temperature,
            'min_temperature': self.min_temperature,
            'max_temperature_jump': self.max_temperature_jump,
            'min_voltage': self.min_voltage,
            'max_voltage': self.max_voltage,
            'min_soc': self.min_soc,
            'critical_soh': self.critical_soh,
            'max_current': self.max_current
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SafetyThresholds':
        """Create from dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SafetyThresholds':
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class MLConfig:
    """
    Configuration for the ML anomaly analyzer.

    Attributes:
        contamination: Expected proportion of anomalies (0.0 - 1.0)
        n_estimators: Number of trees in Isolation Forest
        random_state: Seed for reproducibility
        severity_thresholds: Thresholds for anomaly severity assessment
    """
    
    contamination: float = 0.1
    n_estimators: int = 200
    random_state: int = 42
    
    # Severity assessment thresholds (anomaly scores)
    critical_score_threshold: float = -0.8
    warning_score_threshold: float = -0.5
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'critical_score_threshold': self.critical_score_threshold,
            'warning_score_threshold': self.warning_score_threshold
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MLConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class FrameworkConfig:
    """
    Main EV-QA-Framework configuration.

    Combines all settings: safety thresholds, ML configuration.
    """
    
    safety_thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    ml_config: MLConfig = field(default_factory=MLConfig)
    default_vin: str = "TESTVEHCLE0123456"
    # if True, any rule-based anomalies (temperature spikes etc.) are considered
    # test failures and increment the failed counter; default is False.
    fail_on_anomaly: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'safety_thresholds': self.safety_thresholds.to_dict(),
            'ml_config': self.ml_config.to_dict(),
            'default_vin': self.default_vin,
            'fail_on_anomaly': self.fail_on_anomaly
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FrameworkConfig':
        """Create from dictionary"""
        return cls(
            safety_thresholds=SafetyThresholds.from_dict(data.get('safety_thresholds', {})),
            ml_config=MLConfig.from_dict(data.get('ml_config', {})),
            default_vin=data.get('default_vin', "TESTVEHCLE0123456"),
            fail_on_anomaly=data.get('fail_on_anomaly', False)
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FrameworkConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(filepath):
            # If file does not exist, return default configuration
            return cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Global default configuration
DEFAULT_CONFIG = FrameworkConfig()

# Premium profile for Tesla
# Threshold values tuned to actual Tesla battery parameters
TESLA_CONFIG = FrameworkConfig(
    safety_thresholds=SafetyThresholds(
        max_temperature=55.0,
        min_temperature=-20.0,
        max_temperature_jump=8.0,
        min_voltage=250.0,
        max_voltage=450.0,
        min_soc=20.0,
        critical_soh=75.0
    ),
    default_vin="5YJSA1E26HF000337",  # example of a valid 17-character VIN
    fail_on_anomaly=True
)


# Usage example
if __name__ == '__main__':
    # Create configuration
    config = FrameworkConfig()
    
    # Custom thresholds for Tesla
    tesla_thresholds = SafetyThresholds(
        max_temperature=55.0,  # Tesla is more conservative
        min_voltage=250.0,
        max_voltage=450.0
    )
    config.safety_thresholds = tesla_thresholds
    
    # Save
    config.save_to_file('tesla_config.json')
    print("✅ Configuration saved to tesla_config.json")
    
    # Load
    loaded_config = FrameworkConfig.load_from_file('tesla_config.json')
    print(f"📖 Loaded: max_temp = {loaded_config.safety_thresholds.max_temperature}°C")
    
    # Display default configuration
    print("\n🔧 Default configuration:")
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
