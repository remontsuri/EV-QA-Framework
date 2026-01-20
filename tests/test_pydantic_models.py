"""
Параметризованные тесты для валидации SOC (State of Charge) и Pydantic моделей
"""

import pytest
from pydantic import ValidationError
from ev_qa_framework.models import BatteryTelemetryModel, validate_telemetry


class TestSOCValidation:
    """Тесты для проверки валидации SOC (State of Charge)"""
    
    @pytest.mark.parametrize("soc,expected_valid", [
        # Граничные значения
        (-1, False),      # Меньше минимума
        (0, True),        # Минимальное валидное значение
        (0.1, True),      # Чуть больше минимума
        (1, True),        # Малое значение
        (50, True),       # Середина диапазона
        (99, True),       # Почти максимум
        (99.9, True),     # Чуть меньше максимума
        (100, True),      # Максимальное валидное значение
        (100.1, False),   # Чуть больше максимума
        (101, False),     # Больше максимума
    ])
    def test_soc_boundary_values(self, soc, expected_valid):
        """Граничные значения SOC: -1, 0, 1, 99, 100, 101"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": 100,
            "temperature": 35,
            "soc": soc,
            "soh": 95
        }
        
        if expected_valid:
            telemetry = validate_telemetry(data)
            assert telemetry.soc == soc
        else:
            with pytest.raises(ValidationError) as exc_info:
                validate_telemetry(data)
            assert "soc" in str(exc_info.value).lower()
    
    @pytest.mark.parametrize("soc_value", [
        "строка",          # Строка вместо числа
        "50",              # Число в виде строки (Pydantic может конвертировать)
        None,              # Пустое значение
        [],                # Список
        {},                # Словарь
        True,              # Булево значение (может быть приведено к 1)
    ])
    def test_soc_invalid_types(self, soc_value):
        """Проверка типов: строка вместо числа, None, другие типы"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": 100,
            "temperature": 35,
            "soc": soc_value,
            "soh": 95
        }
        
        # Pydantic может конвертировать некоторые типы (например, "50" -> 50.0)
        # Но строки типа "строка", None, [], {} выбросят ошибку
        if soc_value in [None, [], {}]:
            with pytest.raises(ValidationError):
                validate_telemetry(data)
        elif soc_value == "50":
            # Pydantic автоматически конвертирует строку "50" в float
            telemetry = validate_telemetry(data)
            assert telemetry.soc == 50.0
        elif soc_value is True:
            # True конвертируется в 1.0
            telemetry = validate_telemetry(data)
            assert telemetry.soc == 1.0
        else:
            # "строка" и другие не конвертируемые типы
            with pytest.raises(ValidationError):
                validate_telemetry(data)


class TestPydanticTelemetryModel:
    """Тесты для Pydantic модели BatteryTelemetryModel"""
    
    def test_valid_telemetry_creation(self):
        """Создание валидной телеметрии"""
        data = {
            "vin": "1HGBH41JXMN109186",
            "voltage": 396.5,
            "current": 125.3,
            "temperature": 35.2,
            "soc": 78.5,
            "soh": 96.2
        }
        telemetry = validate_telemetry(data)
        assert telemetry.vin == "1HGBH41JXMN109186"
        assert telemetry.voltage == 396.5
        assert telemetry.soc == 78.5
    
    def test_vin_validation_length(self):
        """VIN должен быть ровно 17 символов"""
        # Короткий VIN
        with pytest.raises(ValidationError) as exc_info:
            validate_telemetry({
                "vin": "SHORT",
                "voltage": 400,
                "current": 100,
                "temperature": 35,
                "soc": 80,
                "soh": 95
            })
        assert "vin" in str(exc_info.value).lower()
        
        # Длинный VIN
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "TOOLONGVINCODE123456",
                "voltage": 400,
                "current": 100,
                "temperature": 35,
                "soc": 80,
                "soh": 95
            })
    
    def test_vin_forbidden_characters(self):
        """VIN не должен содержать I, O, Q"""
        for forbidden_char in ['I', 'O', 'Q']:
            vin = f"1HGBH41JXMN10918{forbidden_char}"
            with pytest.raises(ValidationError) as exc_info:
                validate_telemetry({
                    "vin": vin,
                    "voltage": 400,
                    "current": 100,
                    "temperature": 35,
                    "soc": 80,
                    "soh": 95
                })
            assert "VIN" in str(exc_info.value) or "vin" in str(exc_info.value)
    
    def test_voltage_range_validation(self):
        """Напряжение должно быть 0-1000V"""
        # Отрицательное напряжение
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "1HGBH41JXMN109186",
                "voltage": -10,
                "current": 100,
                "temperature": 35,
                "soc": 80,
                "soh": 95
            })
        
        # Слишком высокое напряжение
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "1HGBH41JXMN109186",
                "voltage": 1500,
                "current": 100,
                "temperature": 35,
                "soc": 80,
                "soh": 95
            })
    
    def test_temperature_range_validation(self):
        """Температура должна быть в диапазоне -50 до +150°C"""
        # Слишком низкая температура
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "1HGBH41JXMN109186",
                "voltage": 400,
                "current": 100,
                "temperature": -100,
                "soc": 80,
                "soh": 95
            })
        
        # Слишком высокая температура
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "1HGBH41JXMN109186",
                "voltage": 400,
                "current": 100,
                "temperature": 200,
                "soc": 80,
                "soh": 95
            })
    
    def test_negative_current_allowed(self):
        """Отрицательный ток разрешен (разряд батареи)"""
        telemetry = validate_telemetry({
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": -50,  # Разряд
            "temperature": 35,
            "soc": 80,
            "soh": 95
        })
        assert telemetry.current == -50
    
    def test_timestamp_auto_generation(self):
        """Timestamp автоматически генерируется, если не указан"""
        telemetry = validate_telemetry({
            "vin": "1HGBH41JXMN109186",
            "voltage": 400,
            "current": 100,
            "temperature": 35,
            "soc": 80,
            "soh": 95
        })
        assert telemetry.timestamp is not None
    
    def test_missing_required_fields(self):
        """Отсутствие обязательных полей выбрасывает ошибку"""
        with pytest.raises(ValidationError):
            validate_telemetry({
                "vin": "1HGBH41JXMN109186",
                "voltage": 400,
                # current отсутствует
                "temperature": 35,
                "soc": 80,
                "soh": 95
            })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
