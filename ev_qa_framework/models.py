"""Pydantic models for strict battery telemetry validation."""

import logging
import warnings
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)


class BatteryCellDataModel(BaseModel):
    """Model for detailed battery cell analysis."""

    vin: str = Field(..., min_length=17, max_length=17)
    cell_voltages: list[float] = Field(
        ..., description="Cell group voltages (usually 96 for Tesla)"
    )
    timestamp: datetime | None = Field(default_factory=datetime.now)

    @field_validator("cell_voltages")
    @classmethod
    def check_voltages(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Cell voltage list cannot be empty")
        if any(volt < 0 or volt > 5.0 for volt in v):
            raise ValueError("Cell voltage must be in range 0-5.0V")
        return v


class BatteryTelemetryModel(BaseModel):
    """Strict EV battery telemetry model with automatic validation."""

    vin: str = Field(..., min_length=17, max_length=17, description="Vehicle VIN (17 characters)")
    voltage: float = Field(..., ge=0.0, le=1000.0, description="Voltage (0-1000V)")
    current: float = Field(..., description="Current in amperes")
    temperature: float = Field(..., ge=-50.0, le=150.0, description="Temperature (-50 to +150°C)")
    soc: float = Field(..., ge=0.0, le=100.0, description="State of Charge (0-100%)")
    soh: float = Field(..., ge=0.0, le=100.0, description="Battery health (0-100%)")
    timestamp: datetime | None = Field(default_factory=datetime.now, description="Timestamp")

    @field_validator("vin")
    @classmethod
    def validate_vin_format(cls, v: str) -> str:
        """VIN format validation (letters and digits only, no I, O, Q)."""
        if not v.isalnum():
            raise ValueError("VIN must contain only letters and digits")
        forbidden = {"I", "O", "Q"}
        if any(char in forbidden for char in v.upper()):
            raise ValueError("VIN cannot contain letters I, O, Q")
        return v.upper()

    @field_validator("temperature")
    @classmethod
    def check_temperature_safety(cls, v: float) -> float:
        """Warning for critical temperatures (logged, not blocking)."""
        if v > 60:
            LOGGER.warning("High temperature %s°C", v)
        if v < 0:
            LOGGER.warning("Negative temperature %s°C", v)
        return v

    @field_validator("soc", "soh")
    @classmethod
    def check_percentage_range(cls, v: float) -> float:
        """Additional validation for percentage values."""
        if not (0 <= v <= 100):
            raise ValueError("Value must be in range 0-100%")
        return v

    @model_validator(mode="after")
    def check_soc_soh_plausibility(self) -> "BatteryTelemetryModel":
        """Enforce physically plausible SOC and SOH relationships."""
        if self.soh < 30.0 and self.soc > 80.0:
            raise ValueError(
                "Battery SOH too low to hold high SOC: "
                f"SOH={self.soh}% cannot support SOC={self.soc}%"
            )
        if self.soh > 80.0 and self.soc < 10.0:
            warnings.warn(
                f"Healthy battery at critically low charge: SOH={self.soh}% SOC={self.soc}%",
                UserWarning,
            )
        return self

    model_config = {
        "validate_assignment": True,
        "json_schema_extra": {
            "example": {
                "vin": "1HGBH41JXMN109186",
                "voltage": 396.5,
                "current": 125.3,
                "temperature": 35.2,
                "soc": 78.5,
                "soh": 96.2,
                "timestamp": "2026-01-19T23:00:00",
            }
        },
    }


def validate_telemetry(data: dict) -> BatteryTelemetryModel:
    """Validate battery telemetry using Pydantic."""
    return BatteryTelemetryModel(**data)
