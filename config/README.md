# 🔧 Configuration Guide - EV-QA-Framework

## Обзор

Модуль конфигурации позволяет настраивать пороги безопасности и параметры ML-анализа без изменения кода. Все настройки вынесены в структурированные классы и могут сохраняться/загружаться из JSON файлов.

## 📝 Основные классы

### 1. `SafetyThresholds` - Пороги безопасности

Определяет пороговые значения для валидации телеметрии батареи.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `max_temperature` | float | 60.0 | Максимальная безопасная температура (°C) |
| `min_temperature` | float | -40.0 | Минимальная безопасная температура (°C) |
| `max_temperature_jump` | float | 5.0 | Максимальный допустимый скачок температуры (°C) |
| `min_voltage` | float | 200.0 | Минимальное безопасное напряжение (V) |
| `max_voltage` | float | 900.0 | Максимальное безопасное напряжение (V) |
| `min_soc` | float | 10.0 | Минимальный уровень заряда для предупреждения (%) |
| `critical_soh` | float | 70.0 | Критический уровень здоровья батареи (%) |
| `max_current` | float | 500.0 | Максимальный безопасный ток (A) |

### 2. `MLConfig` - Конфигурация ML

Параметры для ML-анализатора аномалий (Isolation Forest).

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `contamination` | float | 0.1 | Ожидаемая доля аномалий (0.0 - 1.0) |
| `n_estimators` | int | 200 | Количество деревьев в ансамбле |
| `random_state` | int | 42 | Seed для воспроизводимости |
| `critical_score_threshold` | float | -0.8 | Порог для CRITICAL severity |
| `warning_score_threshold` | float | -0.5 | Порог для WARNING severity |

### 3. `FrameworkConfig` - Главная конфигурация

Объединяет все настройки фреймворка.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `safety_thresholds` | SafetyThresholds | Пороги безопасности |
| `ml_config` | MLConfig | Конфигурация ML |
| `default_vin` | str | VIN по умолчанию для тестов |

## 🚀 Использование

### Базовый пример

```python
from ev_qa_framework import EVQAFramework, FrameworkConfig

# Использование дефолтной конфигурации
qa = EVQAFramework("My-QA")

# Или создание кастомной конфигурации
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0  # Более строгий порог
qa = EVQAFramework("My-QA", config=config)
```

### Создание кастомной конфигурации

```python
from ev_qa_framework import FrameworkConfig, SafetyThresholds, MLConfig

# Строгие пороги для Tesla
tesla_thresholds = SafetyThresholds(
    max_temperature=55.0,
    min_voltage=250.0,
    max_voltage=450.0,
    max_temperature_jump=3.0
)

# ML с низким contamination для высокой точности
ml_config = MLConfig(
    contamination=0.05,
    n_estimators=250,
    critical_score_threshold=-0.9
)

# Создание конфигурации
config = FrameworkConfig(
    safety_thresholds=tesla_thresholds,
    ml_config=ml_config,
    default_vin="5YJ3E1EA8KF000001"
)

# Использование
qa = EVQAFramework("Tesla-QA", config=config)
```

### Сохранение и загрузка конфигурации

```python
from ev_qa_framework import FrameworkConfig

# Создание конфигурации
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0

# Сохранение в JSON
config.save_to_file("my_config.json")

# Загрузка из JSON
loaded_config = FrameworkConfig.load_from_file("my_config.json")

# Использование загруженной конфигурации
qa = EVQAFramework("My-QA", config=loaded_config)
```

## 📁 Пример JSON конфигурации

```json
{
  "safety_thresholds": {
    "max_temperature": 55.0,
    "min_temperature": -30.0,
    "max_temperature_jump": 3.0,
    "min_voltage": 250.0,
    "max_voltage": 450.0,
    "min_soc": 15.0,
    "critical_soh": 75.0,
    "max_current": 600.0
  },
  "ml_config": {
    "contamination": 0.05,
    "n_estimators": 250,
    "random_state": 42,
    "critical_score_threshold": -0.9,
    "warning_score_threshold": -0.6
  },
  "default_vin": "5YJ3E1EA8KF000001"
}
```

## 🎯 Готовые конфигурации

В директории `config/` доступны готовые конфигурации:

- **`default_config.json`** - Дефолтная конфигурация для общего использования
- **`tesla_config.json`** - Строгая конфигурация для Tesla EV

### Использование готовых конфигураций

```python
from ev_qa_framework import FrameworkConfig, EVQAFramework

# Загрузка Tesla конфигурации
tesla_config = FrameworkConfig.load_from_file("config/tesla_config.json")
qa = EVQAFramework("Tesla-QA", config=tesla_config)
```

## 🧪 Тестирование

Все конфигурационные классы покрыты тестами в `tests/test_config.py`:

```bash
pytest tests/test_config.py -v
```

Тесты включают:
- ✅ Инициализацию классов
- ✅ Сериализацию/десериализацию
- ✅ Сохранение/загрузку из файлов
- ✅ Интеграцию с EVQAFramework
- ✅ Валидацию с кастомными порогами

## 💡 Лучшие практики

1. **Используйте конфигурационные файлы** для разных типов батарей (Tesla, Nissan, etc.)
2. **Настраивайте contamination** в зависимости от ожидаемого процента аномалий
3. **Адаптируйте пороги температуры** под климатические условия
4. **Логируйте конфигурацию** при старте тестов для воспроизводимости

## 🔄 Миграция с hardcoded значений

**Старый код (v0.x):**
```python
qa = EVQAFramework("Test")
# Пороги были hardcoded в коде
```

**Новый код (v1.0+):**
```python
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0  # Настраиваемо!
qa = EVQAFramework("Test", config=config)
```

## 📚 Дополнительные ресурсы

- См. также: `ev_qa_framework/config.py` - полная документация в docstrings
- Примеры использования: `examples/` директория
- Тесты: `tests/test_config.py`
