# EV-QA-Framework

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/releases)

ML-powered QA framework for electric vehicle battery systems. Validates BMS telemetry, detects anomalies, predicts SOH degradation, and emulates CAN bus traffic — no commercial licenses, MIT licensed.

## What it does

**Telemetry validation.** Pydantic schemas for voltage, current, temperature, SOC, SOH. Catches bad VINs, out-of-range values at the input layer.

**ML anomaly detection.** Isolation Forest on voltage/current/temperature streams. Configurable contamination, severity thresholds, and number of estimators. Outputs anomaly list with severity (INFO / WARNING / CRITICAL).

**SOH prediction.** LSTM-based State of Health forecasting from historical telemetry. TensorFlow is optional — the package works without it.

**Cell imbalance analysis.** Statistical analysis of cell group voltages: mean, median, std, max-min imbalance. Auto-detects outliers by std deviation and absolute deviation. Classifies severity, builds trend via linear regression, exports plots.

**Thermal runaway prediction.** Two modes: rule-based with adjustable weights (dT/dt, max temperature, anomaly score) and ML (Isolation Forest on thermal features). Triggers CRITICAL at >65°C or heating rate >5°C/min.

**CAN bus.** CAN 2.0B (11-bit ID) and J1939 (29-bit extended, PGN 0xFEF6-0xFEF9) simulation and reception. DBC parser: reads Vector CANdb format, decodes signals (Intel/Motorola byte order, signed/unsigned), compatible with SavvyCAN exports.

**Dashboard.** FastAPI + WebSocket + Chart.js. Real-time telemetry: voltage, current, temperature, SOC/SOH, anomalies, cell voltage heatmap.

**Prometheus metrics.** `/metrics` endpoint with temperature, voltage, current, SOC, SOH, anomaly counter (by severity), cell imbalance max. Ready-to-import Grafana dashboard in `dashboard/grafana/dashboard.json`.

**CI/CD.** GitHub Actions: tests on 4 Python versions, ruff linting, release pipeline for PyPI.

## Quick start

```bash
pip install -r requirements.txt

# launch dashboard
python -m ev_qa_framework.cli dashboard
# → http://localhost:8000
# → http://localhost:8000/metrics (Prometheus)

# analyze a CSV
python -m ev_qa_framework.cli analyze -i examples/tesla_model_s_defective.csv -o report.json

# CAN simulation from DBC file
python -m ev_qa_framework.cli emulate --dbc my_battery.dbc --duration 60

# run tests
python -m pytest -v
```

## Examples

**Telemetry validation:**
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
telemetry = validate_telemetry(data)  # Pydantic, field_validator for VIN and SOC
```

**Anomaly detection:**
```python
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd

df = pd.read_csv("battery_telemetry.csv")
detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(df[["voltage", "current", "temperature"]])
predictions, scores = detector.detect(new_data)
# predictions: 1 = normal, -1 = anomaly
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

**DBC parsing:**
```python
from ev_qa_framework.dbc_parser import DBCParser

dbc = DBCParser("tesla_battery.dbc")
msg = dbc.get_message(0x101)
print(msg.signals.keys())
# dict_keys(['Voltage', 'Current'])

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

## Project structure

```
ev_qa_framework/
  framework.py         # core QA engine
  models.py            # Pydantic models
  config.py            # thresholds and ML config
  analysis.py          # Isolation Forest, EVBatteryAnalyzer
  soh_predictor.py     # LSTM for SOH (TF optional)
  can_bus.py           # CAN 2.0B + J1939 simulation
  dbc_parser.py        # .dbc file parser (Vector CANdb + SavvyCAN)
  cell_balance.py      # cell voltage imbalance analysis
  thermal_runaway.py   # thermal runaway prediction (rule + ML)
  metrics.py           # Prometheus metrics
  cli.py               # CLI entry point
dashboard/
  app.py               # FastAPI
  grafana/             # Grafana dashboard JSON
tests/                 # 95+ tests
```

## Deploy

```bash
# Docker
docker compose -f docker-compose.prod.yml up -d

# or build from source
docker build -t ev-qa-framework .
```

## Compatibility

- CAN 2.0B (11-bit) and J1939 (29-bit extended)
- SavvyCAN / BUSMASTER DBC exports
- Prometheus + Grafana
- TensorFlow — optional (SOH prediction only)
- python-can — only needed for physical CAN hardware; simulation works without it

## License

MIT
