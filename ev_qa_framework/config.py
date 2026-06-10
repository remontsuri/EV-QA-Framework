from __future__ import annotations

"""
EV-QA-Framework Configuration Module
Настройки порогов безопасности и параметров анализа
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
    max_current: float | None = 500.0  # Максимальный безопасный ток

    def to_dict(self) -> dict:
        """Конвертация в словарь для сериализации"""
        return {
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
            "max_temperature_jump": self.max_temperature_jump,
            "min_voltage": self.min_voltage,
            "max_voltage": self.max_voltage,
            "min_soc": self.min_soc,
            "critical_soh": self.critical_soh,
            "max_current": self.max_current
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SafetyThresholds':
        """Создание из словаря"""
        return cls(**data)

    def save_to_file(self, filepath: str) -> None:
        """Сохранение конфигурации в JSON файл"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'SafetyThresholds':
        """Загрузка конфигурации из JSON файла"""
        with open(filepath, encoding="utf-8") as f:
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
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "critical_score_threshold": self.critical_score_threshold,
            "warning_score_threshold": self.warning_score_threshold
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
    Поддерживает выбор химии батареи через *chemistry* — при указании
    пороги безопасности могут быть автоматически заполнены из профиля.
    """

    safety_thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    ml_config: MLConfig = field(default_factory=MLConfig)
    default_vin: str = "TESTVEHCLE0123456"
    # если True, любые rule-based аномалии (скачки температуры и пр.) считаются
    # за отказ теста и увеличивают счетчик failed; по умолчанию False.
    fail_on_anomaly: bool = False

    # --- Chemistry profile integration ---
    # Идентификатор химии: "lfp", "nmc", "nca" (или None для ручной настройки).
    chemistry: ChemistryKey | None = None
    # Количество ячеек в последовательной сборке (для расчёта pack-напряжения).
    cells_in_series: int = 96

    def __post_init__(self) -> None:
        """Авто-заполнение safety_thresholds из профиля химии, если указана."""
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

        Useful when you constructed ``FrameworkConfig(chemistry=\"lfp\")`` without
        thresholds being auto-applied (e.g. after JSON deserialisation that
        included explicit thresholds in the file).
        """
        FrameworkConfig._apply_chemistry(self)
        return self

    def to_dict(self) -> dict:
        """Конвертация в словарь"""
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
    def from_dict(cls, data: dict) -> 'FrameworkConfig':
        """Создание из словаря"""
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
        """Сохранение конфигурации в JSON файл"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'FrameworkConfig':
        """Загрузка конфигурации из JSON файла"""
        if not os.path.exists(filepath):
            # Если файла нет, возвращаем дефолтную конфигурацию
            return cls()

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_yaml(cls, filepath: str, profile: str | None = None) -> 'FrameworkConfig':
        """Загрузка конфигурации из единого YAML-файла.

        Args:
            filepath: Путь к YAML-файлу (например ``config/settings.yaml``).
            profile:  Имя профиля из секции ``profiles``.
                      Если ``None`` — используется профиль ``default``.
                      Если профиль содержит ключ ``chemistry``, пороги
                      автоматически заполняются из встроенного профиля химии
                      через механизм ``__post_init__``.
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


# Глобальная дефолтная конфигурация (NMC, 96s)
DEFAULT_CONFIG = FrameworkConfig(chemistry="nmc")

# Специальный профиль для Tesla Potesti
# Пороговые значения подобраны под реальные параметры батарей Tesla
TESLA_CONFIG = FrameworkConfig(
    chemistry="nca",
    cells_in_series=108,  # Model S 108s (~400 V nominal)
    default_vin="5YJSA1E26HF000337",  # пример валидного 17-символьного VIN
    fail_on_anomaly=True,
)


# Пример использования
if __name__ == "__main__":
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
    config.save_to_file("tesla_config.json")
    print("✅ Конфигурация сохранена в tesla_config.json")

    # Загрузка
    loaded_config = FrameworkConfig.load_from_file("tesla_config.json")
    print(f"📖 Загружено: max_temp = {loaded_config.safety_thresholds.max_temperature}°C")

    # Вывод дефолтной конфигурации
    print("\n🔧 Дефолтная конфигурация:")
    print(json.dumps(DEFAULT_CONFIG.to_dict(), indent=2))
