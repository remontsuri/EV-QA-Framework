from __future__ import annotations

"""
EV-QA-Framework Configuration Module
Safety thresholds and analysis parameters.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .chemistries import ChemistryKey
else:
    # Runtime alias — str is accepted; ChemistryKey is only for static checkers
    ChemistryKey = str


@dataclass
class SafetyThresholds:
    """
    Safety thresholds for battery telemetry validation.

    Attributes:
        max_temperature: Maximum safe temperature (°C)
        min_temperature: Minimum safe temperature (°C)
        max_temperature_jump: Maximum allowed temperature jump (°C)
        min_voltage: Minimum safe voltage (V)
        max_voltage: Maximum safe voltage (V)
        min_soc: Minimum charge level for warning (%)
        critical_soh: Critical battery health level (%)
        max_current: Maximum safe current (A), optional
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

    # Current threshold (optional)
    max_current: float | None = 500.0  # Maximum safe current

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
            "max_temperature_jump": self.max_temperature_jump,
            "min_voltage": self.min_voltage,
            "max_voltage": self.max_voltage,
            "min_soc": self.min_soc,
            "critical_soh": self.critical_soh,
            "max_current": self.max_current,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyThresholds":
        """Create from dictionary."""
        return cls(**data)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "SafetyThresholds":
        """Load configuration from JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class MLConfig:
    """
    Configuration for ML anomaly detector.

    Attributes:
        contamination: Expected proportion of anomalies (0.0 - 1.0)
        n_estimators: Number of trees in Isolation Forest
        random_state: Seed for reproducibility
        severity_thresholds: Thresholds for anomaly severity assessment
    """

    contamination: float = 0.1
    n_estimators: int = 200
    random_state: int = 42

    # Severity thresholds (anomaly scores)
    critical_score_threshold: float = -0.8
    warning_score_threshold: float = -0.5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "critical_score_threshold": self.critical_score_threshold,
            "warning_score_threshold": self.warning_score_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MLConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FrameworkConfig:
    """
    Main EV-QA-Framework configuration.

    Combines all settings: safety thresholds, ML config.
    Supports battery chemistry selection via *chemistry* — when specified,
    safety thresholds can be auto-populated from the profile.
    """

    safety_thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    ml_config: MLConfig = field(default_factory=MLConfig)
    default_vin: str = "TESTVEHCLE0123456"
    # If True, any rule-based anomalies (temperature jumps, etc.) are treated
    # as test failures and increment the failed counter; default False.
    fail_on_anomaly: bool = False

    # --- Chemistry profile integration ---
    # Chemistry identifier: "lfp", "nmc", "nca" (or None for manual config).
    chemistry: ChemistryKey | None = None
    # Number of cells in series (for pack voltage calculation).
    cells_in_series: int = 96

    def __post_init__(self) -> None:
        """Auto-populate safety_thresholds from chemistry profile if specified."""
        FrameworkConfig._apply_chemistry(self)

    @staticmethod
    def _apply_chemistry(cfg: FrameworkConfig) -> None:
        """Internal helper to populate safety thresholds from a chemistry profile."""
        if cfg.chemistry is None:
            return
        # Lazy import to avoid circular dependency at module level
        from .chemistries import get_profile  # fmt: skip

        profile = get_profile(cfg.chemistry)
        safe = profile.to_safety_thresholds_dict(cells_in_series=cfg.cells_in_series)
        cfg.safety_thresholds = SafetyThresholds.from_dict(safe)

    def get_chemistry_profile(self) -> Any | None:
        """Return the ``BatteryChemistryProfile`` for the selected *chemistry*, or ``None``."""
        if self.chemistry is None:
            return None
        from .chemistries import get_profile  # fmt: skip

        return get_profile(self.chemistry)

    def configure_from_chemistry(self) -> FrameworkConfig:
        """Explicitly populate *safety_thresholds* from the selected chemistry profile.

        Useful when you constructed ``FrameworkConfig(chemistry="lfp")`` without
        thresholds being auto-applied (e.g. after JSON deserialisation that
        included explicit thresholds in the file).
        """
        FrameworkConfig._apply_chemistry(self)
        return self

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d: dict = {
            "safety_thresholds": self.safety_thresholds.to_dict(),
            "ml_config": self.ml_config.to_dict(),
            "default_vin": self.default_vin,
            "fail_on_anomaly": self.fail_on_anomaly,
        }
        if self.chemistry is not None:
            d["chemistry"] = self.chemistry
            d["cells_in_series"] = self.cells_in_series
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FrameworkConfig":
        """Create from dictionary."""
        cfg = cls(
            safety_thresholds=SafetyThresholds.from_dict(data.get("safety_thresholds", {})),
            ml_config=MLConfig.from_dict(data.get("ml_config", {})),
            default_vin=data.get("default_vin", "TESTVEHCLE0123456"),
            fail_on_anomaly=data.get("fail_on_anomaly", False),
            chemistry=data.get("chemistry"),
            cells_in_series=data.get("cells_in_series", 96),
        )
        return cfg

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "FrameworkConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            # If file doesn't exist, return default configuration
            return cls()

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_yaml(cls, filepath: str, profile: str | None = None) -> "FrameworkConfig":
        """Load configuration from a unified YAML file.

        Args:
            filepath: Path to YAML file (e.g. ``config/settings.yaml``).
            profile:  Profile name from the ``profiles`` section.
                      If ``None``, the ``default`` profile is used.
                      If the profile contains a ``chemistry`` key, thresholds
                      are auto-populated from the built-in chemistry profile
                      via the ``__post_init__`` mechanism.
        """
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # Determine the profile name
        profile_name = profile or "default"
        profiles = raw.get("profiles", {})

        if profiles and profile_name in profiles:
            data = profiles[profile_name]
        elif profile_name != "default":
            # Fall back to default if named profile not found
            data = profiles.get("default", {})
        else:
            data = {}

        # If the profile section is just a dict (not a full FrameworkConfig dict),
        # normalise it via from_dict which handles missing keys gracefully.
        cfg = cls.from_dict(data)
        return cfg


# Global default configuration (NMC, 96s)
DEFAULT_CONFIG = FrameworkConfig(chemistry="nmc")

# Special profile for Tesla Potesti
# Thresholds tuned to real Tesla battery parameters
TESLA_CONFIG = FrameworkConfig(
    chemistry="nca",
    cells_in_series=108,  # Model S 108s (~400 V nominal)
    default_vin="5YJSA1E26HF000337",  # example valid 17-character VIN
    fail_on_anomaly=True,
)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = FrameworkConfig()

    # Custom thresholds for Tesla
    tesla_thresholds = SafetyThresholds(
        max_temperature=55.0,  # Tesla is more conservative
        min_voltage=250.0,
        max_voltage=450.0,
    )
    config.safety_thresholds = tesla_thresholds

    # Save
    config.save_to_file("tesla_config.json")
    print("Configuration saved to tesla_config.json")

    # Load
    loaded_config = FrameworkConfig.load_from_file("tesla_config.json")
    print(f"Loaded: max_temp = {loaded_config.safety_thresholds.max_temperature}°C")

    # Print default configuration
    print("\nDefault configuration:")
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
