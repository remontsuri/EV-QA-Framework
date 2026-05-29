"""
Pydantic models for strict battery telemetry validation
Author: Remontsuri
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class BatteryCellDataModel(BaseModel):
    """
    Model for detailed battery cell analysis.
    Used for detecting Cell Imbalance.
    """
    vin: str = Field(..., min_length=17, max_length=17)
    cell_voltages: List[float] = Field(..., description="Cell group voltages (typically 96 for Tesla)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    @field_validator('cell_voltages')
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
        vin: Vehicle Identification Number
        voltage: Battery voltage in volts (0-1000V)
        current: Current in amperes (can be negative during discharge)
        temperature: Battery temperature in degrees Celsius
        soc: State of Charge (0-100%)
        soh: State of Health (0-100%)
        timestamp: Timestamp (Unix timestamp or datetime)
    """
    
    vin: str = Field(..., min_length=17, max_length=17, description="VIN (17 characters)")
    voltage: float = Field(..., ge=0.0, le=1000.0, description="Voltage (0-1000V)")
    current: float = Field(..., description="Current in amperes")
    temperature: float = Field(..., ge=-50.0, le=150.0, description="Temperature (-50 to +150°C)")
    soc: float = Field(..., ge=0.0, le=100.0, description="State of Charge (0-100%)")
    soh: float = Field(..., ge=0.0, le=100.0, description="State of Health (0-100%)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp")
    
    @field_validator('vin')
    @classmethod
    def validate_vin_format(cls, v):
        """Validate VIN format (alphanumeric only, excluding I, O, Q)"""
        if not v.isalnum():
            raise ValueError('VIN must contain only letters and digits')
        forbidden = set('IOQ')
        if any(c in forbidden for c in v.upper()):
            raise ValueError('VIN cannot contain letters I, O, Q')
        return v.upper()
    
    @field_validator('temperature')
    @classmethod
    def check_temperature_safety(cls, v):
        """Warning for critical temperatures"""
        if v > 60:
            # Log warning but do not block
            logger.warning("High temperature %s°C", v)
        if v < 0:
            logger.warning("Negative temperature %s°C", v)
        return v
    
    @field_validator('soc', 'soh')
    @classmethod
    def check_percentage_range(cls, v):
        """Additional check for percentage values"""
        if not (0 <= v <= 100):
            raise ValueError('Value must be in range 0-100%')
        return v
    
    model_config = {
        "validate_assignment": True,  # Validate on field assignment
        "json_schema_extra": {
            "example": {
                "vin": "1HGBH41JXMN109186",
                "voltage": 396.5,
                "current": 125.3,
                "temperature": 35.2,
                "soc": 78.5,
                "soh": 96.2,
                "timestamp": "2026-01-19T23:00:00"
            }
        }
    }


def validate_telemetry(data: dict) -> BatteryTelemetryModel:
    """
    Telemetry validation function using Pydantic.
    """
    return BatteryTelemetryModel(**data)
