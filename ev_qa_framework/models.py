"""
Pydantic модели для строгой валидации телеметрии батареи
Автор: Remontsuri
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional


class BatteryTelemetryModel(BaseModel):
    """
    Строгая модель телеметрии батареи EV с автоматической валидацией.
    
    Поля:
        vin: VIN автомобиля (Vehicle Identification Number)
        voltage: Напряжение батареи в вольтах (0-1000V)
        current: Ток в амперах (может быть отрицательным при разряде)
        temperature: Температура батареи в градусах Цельсия
        soc: State of Charge - уровень заряда (0-100%)
        soh: State of Health - состояние здоровья батареи (0-100%)
        timestamp: Временная метка (Unix timestamp или datetime)
    """
    
    vin: str = Field(..., min_length=17, max_length=17, description="VIN автомобиля (17 символов)")
    voltage: float = Field(..., ge=0.0, le=1000.0, description="Напряжение (0-1000V)")
    current: float = Field(..., description="Ток в амперах")
    temperature: float = Field(..., ge=-50.0, le=150.0, description="Температура (-50 до +150°C)")
    soc: float = Field(..., ge=0.0, le=100.0, description="Уровень заряда (0-100%)")
    soh: float = Field(..., ge=0.0, le=100.0, description="Состояние батареи (0-100%)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Временная метка")
    
    @field_validator('vin')
    @classmethod
    def validate_vin_format(cls, v):
        """Проверка формата VIN (только буквы и цифры, без I, O, Q)"""
        if not v.isalnum():
            raise ValueError('VIN должен содержать только буквы и цифры')
        forbidden = set('IOQ')
        if any(c in forbidden for c in v.upper()):
            raise ValueError('VIN не может содержать буквы I, O, Q')
        return v.upper()
    
    @field_validator('temperature')
    @classmethod
    def check_temperature_safety(cls, v):
        """Предупреждение о критических температурах"""
        if v > 60:
            # Логирование предупреждения, но не блокирование
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Высокая температура {v}°C")
        if v < 0:
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Отрицательная температура {v}°C")
        return v
    
    @field_validator('soc', 'soh')
    @classmethod
    def check_percentage_range(cls, v):
        """Дополнительная проверка процентных значений"""
        if not (0 <= v <= 100):
            raise ValueError('Значение должно быть в диапазоне 0-100%')
        return v
    
    model_config = {
        "validate_assignment": True,  # Валидация при изменении полей
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
    Функция валидации телеметрии с использованием Pydantic.
    """
    return BatteryTelemetryModel(**data)
