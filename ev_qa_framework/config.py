from __future__ import annotations

"""
EV-QA-Framework Configuration Module
Настройки порогов безопасности и параметров анализа
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import json


@dataclass
class SafetyThresholds:
    """
    Пороги безопасности для валидации телеметрии батареи.
    
    Attributes:
        max_temperature: Максимальная безопасная температура (°C)
        min_voltage: Минимальное безопасное напряжение (V)
        max_voltage: Максимальное безопасное напряжение (V)
        max_temperature_jump: Максимальный допустимый скачок температуры (°C)
        min_soc: Минимальный уровень заряда для предупреждения (%)
        critical_soh: Критический уровень здоровья батареи (%)
    """
    
    # Температурные пороги
    max_temperature: float = 60.0
    min_temperature: float = -40.0
    max_temperature_jump: float = 5.0
    
    # Пороги напряжения
    min_voltage: float = 200.0
    max_voltage: float = 900.0
    
    # Пороги заряда и здоровья
    min_soc: float = 10.0  # Предупреждение о низком заряде
    critical_soh: float = 70.0  # Критическое здоровье батареи
    
    # Пороги тока (опционально)
    max_current: Optional[float] = 500.0  # Максимальный безопасный ток
    
    def to_dict(self) -> dict:
        """Конвертация в словарь для сериализации"""
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
        """Создание из словаря"""
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Сохранение конфигурации в JSON файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SafetyThresholds':
        """Загрузка конфигурации из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class MLConfig:
    """
    Конфигурация для ML-анализатора аномалий.
    
    Attributes:
        contamination: Ожидаемая доля аномалий (0.0 - 1.0)
        n_estimators: Количество деревьев в Isolation Forest
        random_state: Seed для воспроизводимости
        severity_thresholds: Пороги для оценки серьезности аномалий
    """
    
    contamination: float = 0.1
    n_estimators: int = 200
    random_state: int = 42
    
    # Пороги для оценки серьезности (anomaly scores)
    critical_score_threshold: float = -0.8
    warning_score_threshold: float = -0.5
    
    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'critical_score_threshold': self.critical_score_threshold,
            'warning_score_threshold': self.warning_score_threshold
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MLConfig':
        """Создание из словаря"""
        return cls(**data)


@dataclass
class FrameworkConfig:
    """
    Главная конфигурация EV-QA-Framework.
    
    Объединяет все настройки: пороги безопасности, ML-конфиг.
    """
    
    safety_thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    ml_config: MLConfig = field(default_factory=MLConfig)
    default_vin: str = "TESTVEHCLE0123456"
    # если True, любые rule-based аномалии (скачки температуры и пр.) считаются
    # за отказ теста и увеличивают счетчик failed; по умолчанию False.
    fail_on_anomaly: bool = False
    
    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            'safety_thresholds': self.safety_thresholds.to_dict(),
            'ml_config': self.ml_config.to_dict(),
            'default_vin': self.default_vin,
            'fail_on_anomaly': self.fail_on_anomaly
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FrameworkConfig':
        """Создание из словаря"""
        return cls(
            safety_thresholds=SafetyThresholds.from_dict(data.get('safety_thresholds', {})),
            ml_config=MLConfig.from_dict(data.get('ml_config', {})),
            default_vin=data.get('default_vin', "TESTVEHCLE0123456"),
            fail_on_anomaly=data.get('fail_on_anomaly', False)
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Сохранение конфигурации в JSON файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FrameworkConfig':
        """Загрузка конфигурации из JSON файла"""
        if not os.path.exists(filepath):
            # Если файла нет, возвращаем дефолтную конфигурацию
            return cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Глобальная дефолтная конфигурация
DEFAULT_CONFIG = FrameworkConfig()

# Специальный профиль для Tesla Potesti
# Пороговые значения подобраны под реальные параметры батарей Tesla
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
    default_vin="5YJSA1E26HF000337",  # пример валидного 17-символьного VIN
    fail_on_anomaly=True
)


# Пример использования
if __name__ == '__main__':
    # Создание конфигурации
    config = FrameworkConfig()
    
    # Кастомные пороги для Tesla
    tesla_thresholds = SafetyThresholds(
        max_temperature=55.0,  # Tesla более консервативна
        min_voltage=250.0,
        max_voltage=450.0
    )
    config.safety_thresholds = tesla_thresholds
    
    # Сохранение
    config.save_to_file('tesla_config.json')
    print("✅ Конфигурация сохранена в tesla_config.json")
    
    # Загрузка
    loaded_config = FrameworkConfig.load_from_file('tesla_config.json')
    print(f"📖 Загружено: max_temp = {loaded_config.safety_thresholds.max_temperature}°C")
    
    # Вывод дефолтной конфигурации
    print("\n🔧 Дефолтная конфигурация:")
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
