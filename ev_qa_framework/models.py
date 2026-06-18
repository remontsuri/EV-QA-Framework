"""
Pydantic models for strict battery telemetry validation.
Author: Remontsuri
"""

import logging
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

LOGGER = logging.getLogger(__name__)


class BatteryCellDataModel(BaseModel):
    """
    Model for detailed battery cell analysis.
    Used for detecting cell imbalance.
    """

    vin: str = Field(..., min_length=17, max_length=17)
    cell_voltages: list[float] = Field(
        ..., description="Cell group voltages (usually 96 for Tesla)"
    )
    timestamp: datetime | None = Field(default_factory=datetime.now)

    @field_validator("cell_voltages")
    @classmethod
    def check_voltages(cls, v):
        if not v:
            raise ValueError("Cell voltage list cannot be empty")
        if any(volt < 0 or volt > 5.0 for volt in v):
            raise ValueError("Cell voltage must be in range 0-5.0V")
        return v


class BatteryTelemetryModel(BaseModel):
    """
    Strict EV battery telemetry model with automatic validation.

    Fields:
        vin: Vehicle Identification Number (17 characters)
        voltage: Battery voltage in volts (0-1000V)
        current: Current in amperes (can be negative during discharge)
        temperature: Battery temperature in degrees Celsius
        soc: State of Charge — charge level (0-100%)
        soh: State of Health — battery health (0-100%)
        timestamp: Timestamp (Unix timestamp or datetime)
    """

    vin: str = Field(..., min_length=17, max_length=17, description="Vehicle VIN (17 characters)")
    voltage: float = Field(..., ge=0.0, le=1000.0, description="Voltage (0-1000V)")
    current: float = Field(..., description="Current in amperes")
    temperature: float = Field(..., ge=-50.0, le=150.0, description="Temperature (-50 to +150°C)")
    soc: float = Field(..., ge=0.0, le=100.0, description="State of Charge (0-100%)")
    soh: float = Field(..., ge=0.0, le=100.0, description="Battery health (0-100%)")
    timestamp: datetime | None = Field(default_factory=datetime.now, description="Timestamp")

    @field_validator("vin")
    @classmethod
    def validate_vin_format(cls, v):
        """VIN format validation (letters and digits only, no I, O, Q)"""
        if not v.isalnum():
            raise ValueError("VIN must contain only letters and digits")
        forbidden = set("IOQ")
        if any(c in forbidden for c in v.upper()):
            raise ValueError("VIN cannot contain letters I, O, Q")
        return v.upper()

    @field_validator("temperature")
    @classmethod
    def check_temperature_safety(cls, v):
        """Warning for critical temperatures (logged, not blocking)."""
        if v > 60:
            LOGGER.warning("High temperature %s°C", v)
        if v < 0:
            LOGGER.warning("Negative temperature %s°C", v)
        return v

    @field_validator("soc", "soh")
    @classmethod
    def check_percentage_range(cls, v):
        """Additional validation for percentage values."""
        if not (0 <= v <= 100):
            raise ValueError("Value must be in range 0-100%")
        return v

    model_config = {
        "validate_assignment": True,  # Validate on field change
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
    """
    Validate battery telemetry using Pydantic.
    """
    return BatteryTelemetryModel(**data)
