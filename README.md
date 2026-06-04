# EV-QA-Framework

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/releases)

Паранойя батарей — штука дорогая. Электромобили генерируют терабайты телеметрии, а BMS обычно смотрит только на базовые пороги. Этот фреймворк забирает сырые данные с CAN шины или CSV, прогоняет через ML (Isolation Forest, LSTM), и говорит: «вот тут у тебя ячейка просела», «тут перегрев на подходе», «SOH упадёт ниже 80% через 600 циклов».

Написан на Python, не требует коммерческих лицензий, всё под MIT.

## Что внутри

**Валидация телеметрии.** Pydantic схемы для voltage, current, temperature, SOC, SOH. Если VIN неправильный или SOC за 100% — отсекается на входе.

**ML детекция аномалий.** Isolation Forest на voltage/current/temperature. Конфигурируется contamination (доля аномалий), severity пороги, количество деревьев. На выходе — список аномальных точек с severity (INFO / WARNING / CRITICAL). 

**SOH prediction.** LSTM модель, предсказывает State of Health на исторических данных. TensorFlow опционален — пакет работает и без него.

**Cell imbalance.** Статистический анализ напряжений групп ячеек: среднее, медиана, std, max-min дисбаланс. Автоматически находит выбросы (по std * 2 и absolute deviation), классифицирует severity (NORMAL / WARNING / CRITICAL), строит тренд через линейную регрессию последних N замеров. Рисует график.

**Thermal runaway prediction.** Два режима: rule-based (настраиваемые веса для dT/dt, макс температуры, аномалий) и ML (IsolationForest на темповых признаках). Пороги: CRITICAL при температуре >65°C или скорости нагрева >5°C/min.

**CAN bus.** Поддержка CAN 2.0B (11-bit ID) и J1939 (29-bit extended, PGN 0xFEF6-0xFEF9). Симуляция батарейных сообщений, приём и декада данных. DBC-парсер: читает Vector CANdb файлы, декодирует сигналы (Intel/Motorola byte order, signed/unsigned), умеет работать с SavvyCAN экспортом.

**Dashboard.** FastAPI + WebSocket + Chart.js. Показывает телеметрию в реальном времени: напряжение, ток, температура, SOC/SOH, аномалии. 

**Prometheus метрики.** Endpoint `/metrics` с temperature, voltage, current, SOC, SOH, anomaly counter (by severity), cell imbalance max. Готовый dashboard для Grafana в `dashboard/grafana/dashboard.json`.

**CI/CD.** GitHub Actions: тесты на 4 версиях Python, линтинг (ruff), релизный pipeline для PyPI.

## Быстрый старт

```bash
# поставить зависимости (без TensorFlow — он нужен только для SOH)
pip install -r requirements.txt

# запустить дашборд
python -m ev_qa_framework.cli dashboard
# → http://localhost:8000
# → http://localhost:8000/metrics (Prometheus)

# прогнать анализ CSV
python -m ev_qa_framework.cli analyze -i examples/tesla_model_s_defective.csv -o report.json

# симуляция CAN шины из DBC файла
python -m ev_qa_framework.cli emulate --dbc my_battery.dbc --duration 60

# тесты
python -m pytest -v
```

## Примеры

**Валидация телеметрии:**
```python
from ev_qa_framework.models import validate_telemetry

data = {
    "vin": "1HGBH41JXMN109186",
    "voltage": 396.5,
    "current": 125.3,
    "temperature": 35.2,
    "soc": 78.5,
    "soh": 96.2
}
telemetry = validate_telemetry(data)  # Pydantic, с field_validator на VIN и SOC
```

**Аномалии на CSV:**
```python
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd

df = pd.read_csv("battery_telemetry.csv")
detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(df[["voltage", "current", "temperature"]])
predictions, scores = detector.detect(new_data)
# predictions: 1 = норма, -1 = аномалия
```

**Cell imbalance:**
```python
from ev_qa_framework.cell_balance import CellBalanceAnalyzer

analyzer = CellBalanceAnalyzer(warning_threshold=0.02, critical_threshold=0.05)
cell_v = [3.30, 3.31, 3.305, 3.312, 3.29]

print(analyzer.compute_statistics(cell_v))
# {'mean': 3.30, 'max_min_imbalance': 0.022, ...}
print(analyzer.classify_severity(cell_v))
# WARNING
```

**DBC парсинг:**
```python
from ev_qa_framework.dbc_parser import DBCParser

dbc = DBCParser("tesla_battery.dbc")
msg = dbc.get_message(0x101)
print(msg.signals.keys())
# dict_keys(['Voltage', 'Current'])

# Декодировать сырые CAN данные
data = bytes([0x7D, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
vals = dbc.decode(0x101, data)
# {'Voltage': 396.5}
```

**Thermal runaway:**
```python
from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor
import pandas as pd

predictor = ThermalRunawayPredictor(mode="rule")
df = pd.DataFrame({"temperature": [35, 37, 42, 58, 62]})
risk = predictor.predict_risk(df)
# {'risk_level': 'HIGH', 'risk_score': 8.3, ...}
```

## Структура

```
ev_qa_framework/
  framework.py         # основной QA engine
  models.py            # Pydantic модели
  config.py            # настройки порогов и ML
  analysis.py          # Isolation Forest, EVBatteryAnalyzer
  soh_predictor.py     # LSTM для SOH (TF optional)
  can_bus.py           # CAN 2.0B + J1939 симуляция
  dbc_parser.py        # парсер .dbc файлов (Vector CANdb + SavvyCAN)
  cell_balance.py      # анализ дисбаланса ячеек
  thermal_runaway.py   # прогноз теплового разгона (rule + ML)
  metrics.py           # Prometheus метрики
  cli.py               # точка входа
dashboard/
  app.py               # FastAPI
  grafana/             # дашборд для Grafana
tests/                 # 95+ тестов
```

## Деплой

```bash
# через Docker
docker compose -f docker-compose.prod.yml up -d

# из сорцов (сборка образа)
docker build -t ev-qa-framework .
```

## Совместимость

- CAN 2.0B (11-bit) и J1939 (29-bit extended)
- SavvyCAN / BUSMASTER экспорт DBC
- Prometheus + Grafana
- TensorFlow — опционально (SOH prediction)
- python-can — только для реальной CAN шины, без него работает в режиме эмуляции

## Лицензия

MIT
