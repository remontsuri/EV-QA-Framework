# 💾 ML Model Persistence Guide

## Обзор

Модуль персистентности позволяет сохранять обученные ML-модели и использовать их в production без переобучения. Это критично для **train-once, deploy-many** паттерна.

## 🎯 Зачем это нужно?

**Проблема:** Isolation Forest обучается каждый раз заново при вызове `analyze_telemetry()`, что:
- ❌ Медленно для production (переобучение занимает время)
- ❌ Нестабильно (разные результаты при каждом запуске без random_state)
- ❌ Невозможно версионирование моделей

**Решение:** Сохранение обученной модели:
- ✅ Обучаем один раз offline
- ✅ Сохраняем модель + scaler + metadata
- ✅ Загружаем в production для быстрого inference
- ✅ Версионируем модели (A/B тестирование)

---

## 📝 API Reference

### `save_model(filepath, metadata=None)`

Сохраняет обученную модель в файл.

**Параметры:**
- `filepath` (str): Путь для сохранения (автоматически добавляется `.joblib`)
- `metadata` (dict, optional): Метаданные модели

**Что сохраняется:**
- Обученная модель `IsolationForest`
- Обученный `StandardScaler`
- Параметры: `contamination`, `critical_threshold`, `warning_threshold`
- Timestamp сохранения
- Пользовательские метаданные

**Пример:**
```python
analyzer = EVBatteryAnalyzer(contamination=0.1)
analyzer.analyze_telemetry(train_data)

analyzer.save_model(
    'models/battery_v1',
    metadata={'version': '1.0', 'dataset': 'Tesla_2024'}
)
```

**Raises:**
- `ValueError` - если модель не обучена

---

### `load_model(filepath)` (classmethod)

Загружает сохраненную модель из файла.

**Параметры:**
- `filepath` (str): Путь к файлу модели

**Возвращает:**
- `EVBatteryAnalyzer` - новый экземпляр с загруженной моделью

**Пример:**
```python
analyzer = EVBatteryAnalyzer.load_model('models/battery_v1.joblib')
results = analyzer.analyze_telemetry(new_data)
```

**Raises:**
- `FileNotFoundError` - если файл не найден
- `ValueError` - если файл поврежден

---

### `get_model_info()`

Получает информацию о текущей модели.

**Возвращает:**
- `dict` с параметрами модели

**Пример:**
```python
info = analyzer.get_model_info()
print(info)
# {
#     'contamination': 0.1,
#     'n_estimators': 200,
#     'critical_threshold': -0.8,
#     'warning_threshold': -0.5,
#     'is_fitted': True
# }
```

---

## 🚀 Production Workflow

### 1. Offline Training

```python
from ev_qa_framework import EVBatteryAnalyzer
import pandas as pd

# Загрузка исторических "нормальных" данных
historical_data = pd.read_csv('historical_telemetry.csv')

# Обучение модели
analyzer = EVBatteryAnalyzer(
    contamination=0.05,
    n_estimators=300,
    critical_threshold=-0.9
)

analyzer.analyze_telemetry(historical_data)

# Сохранение с метаданными
analyzer.save_model(
    'models/production/baseline_v2.0',
    metadata={
        'version': '2.0',
        'dataset_size': len(historical_data),
        'trained_on': '2024-01-28',
        'performance_metrics': {...}
    }
)
```

### 2. Production Deployment

```python
# В production сервисе
from ev_qa_framework import EVBatteryAnalyzer

# Загрузка один раз при старте сервиса
model = EVBatteryAnalyzer.load_model('models/production/baseline_v2.0')

# Использование для real-time inference
def process_telemetry(incoming_batch):
    results = model.analyze_telemetry(incoming_batch)
    
    if results['severity'] == 'CRITICAL':
        send_alert(results)
    
    return results
```

### 3. Monitoring & Versioning

```python
# A/B тестирование моделей
model_a = EVBatteryAnalyzer.load_model('models/model_a_conservative')
model_b = EVBatteryAnalyzer.load_model('models/model_b_tolerant')

results_a = model_a.analyze_telemetry(test_data)
results_b = model_b.analyze_telemetry(test_data)

# Сравнение метрик
compare_models(results_a, results_b)
```

---

## 📁 Структура сохраненной модели

Файл `.joblib` содержит словарь:

```python
{
    'model': IsolationForest object,      # Обученная модель
    'scaler': StandardScaler object,      # Обученный scaler
    'contamination': 0.1,                 # Параметры модели
    'critical_threshold': -0.8,
    'warning_threshold': -0.5,
    'save_timestamp': '2024-01-28T12:00:00',
    'metadata': {                         # Пользовательские метаданные
        'version': '1.0',
        'dataset': 'Tesla_2024',
        ...
    }
}
```

---

## 🔄 Версионирование моделей

### Рекомендуемая схема именования

```
models/
├── production/
│   ├── baseline_v1.0.joblib       # Продакшн версия
│   ├── baseline_v1.1.joblib       # Обновленная версия
│   └── champion_v2.0.joblib       # Новая best модель
├── experiments/
│   ├── exp_conservative.joblib    # Эксперименты
│   ├── exp_tolerant.joblib
│   └── exp_highres.joblib
└── archive/
    └── deprecated_v0.9.joblib     # Старые версии
```

### Semantic Versioning для моделей

```
[MAJOR].[MINOR].[PATCH]

MAJOR: Изменение алгоритма или breaking changes
MINOR: Изменение параметров (contamination, thresholds)
PATCH: Переобучение на новых данных с теми же параметрами
```

**Примеры:**
- `v1.0.0` → `v1.0.1` - переобучение на обновленном датасете
- `v1.0.1` → `v1.1.0` - изменение contamination с 0.1 на 0.05
- `v1.1.0` → `v2.0.0` - переход с IsolationForest на другой алгоритм

---

## ⚡ Performance Tips

### 1. Используйте сжатие

```python
# По умолчанию compress=3 (хороший баланс)
analyzer.save_model('model.joblib')  # ~500 KB

# Без сжатия (быстрее, но больше размер)
import joblib
joblib.dump(model_data, 'model.joblib', compress=0)  # ~2 MB

# Максимальное сжатие (медленнее, но меньше размер)
joblib.dump(model_data, 'model.joblib', compress=9)  # ~300 KB
```

### 2. Lazy Loading в production

```python
# Плохо: загружаем при каждом запросе
def handle_request(data):
    model = EVBatteryAnalyzer.load_model('model.joblib')  # Медленно!
    return model.analyze_telemetry(data)

# Хорошо: загружаем один раз
class ModelService:
    def __init__(self):
        self.model = EVBatteryAnalyzer.load_model('model.joblib')
    
    def predict(self, data):
        return self.model.analyze_telemetry(data)
```

---

## 🧪 Testing

Тесты для персистентности находятся в `tests/test_model_persistence.py`:

```bash
pytest tests/test_model_persistence.py -v
```

**Покрываемые сценарии:**
- ✅ Базовое save/load
- ✅ Save/load с метаданными
- ✅ Ошибка при сохранении необученной модели
- ✅ Inference после загрузки
- ✅ Load несуществующего файла
- ✅ Автоматическое создание директорий
- ✅ Версионирование

---

## 🐛 Troubleshooting

### Ошибка: "Модель не обучена"

```python
analyzer = EVBatteryAnalyzer()
analyzer.save_model('model.joblib')  # ValueError!

# Решение: сначала обучите
analyzer.analyze_telemetry(data)
analyzer.save_model('model.joblib')  # ✅
```

### Ошибка: "Файл модели не найден"

```python
model = EVBatteryAnalyzer.load_model('nonexistent.joblib')  # FileNotFoundError!

# Решение: проверьте путь
import os
print(os.path.exists('model.joblib'))
```

### Несовместимость версий scikit-learn

Если модель сохранена на scikit-learn 1.2.0, а загружается на 1.3.0:

```python
# Решение: используйте виртуальное окружение с фиксированными версиями
# requirements.txt:
scikit-learn==1.2.0
joblib==1.3.0
```

---

## 📚 Best Practices

1. **Всегда сохраняйте метаданные**
   ```python
   metadata = {
       'version': '1.0',
       'trained_on': str(datetime.now()),
       'dataset_size': len(data),
       'contamination': 0.1
   }
   analyzer.save_model('model', metadata=metadata)
   ```

2. **Используйте версионирование**
   - Сохраняйте несколько версий модели
   - Ведите changelog изменений

3. **Тестируйте перед деплоем**
   - Сохраните модель
   - Загрузите в тестовом окружении
   - Проверьте на валидационных данных

4. **Документируйте эксперименты**
   - Какие данные использовались
   - Какие параметры настроены
   - Какие метрики получены

---

## 🔗 См. также

- [Configuration Guide](../config/README.md) - настройка параметров ML
- [examples/model_persistence_example.py](../examples/model_persistence_example.py) - примеры использования
- [tests/test_model_persistence.py](../tests/test_model_persistence.py) - тесты
