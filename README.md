# EV-QA-Framework

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml/badge.svg)
[![GitHub Release](https://img.shields.io/github/v/release/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/releases)

ML-powered QA framework for electric vehicle battery systems. Validates BMS telemetry, detects anomalies, predicts SOH degradation, emulates CAN bus traffic, and evaluates thermal runaway risk — MIT licensed.

## What it does

**Telemetry validation.** Pydantic schemas for voltage, current, temperature, SOC, SOH. Catches bad VINs, out-of-range values at the input layer.

**ML anomaly detection.** Isolation Forest on voltage/current/temperature streams. Configurable contamination, severity thresholds, and number of estimators.

**SOH prediction.** LSTM-based State of Health forecasting from historical telemetry (TensorFlow optional).

**Cell imbalance analysis.** Statistical analysis of cell group voltages with configurable thresholds, outlier detection, linear regression trend, and plot export.

**Thermal runaway prediction.** Standalone `ThermalRunawayPredictor` with two modes:
  - **rule** — configurable heuristic with adjustable weights (dT/dt, temperature, anomaly score)
  - **ml** — Isolation Forest on thermal features
  CRITICAL trigger at >65°C or heating rate >5°C/min.

**CAN bus.** CAN 2.0B (11-bit ID) and J1939 (29-bit extended) simulation and reception. DBC parser supports Vector CANdb format, SavvyCAN exports, Intel/Motorola byte order, signed/unsigned signals.

**Dashboard.** FastAPI + WebSocket + Chart.js. Real-time telemetry and Prometheus `/metrics` endpoint with ready-to-import Grafana dashboard.

**CLI.** Analyze CSV telemetry, run CAN emulation, train SOH models, start dashboard.

## Quick start

```bash
# Install from GitHub
pip install git+https://github.com/remontsuri/EV-QA-Framework.git

# Launch dashboard
python -m ev_qa_framework.cli dashboard
# → http://localhost:8000
# → http://localhost:8000/metrics (Prometheus)

# Analyze a CSV
python -m ev_qa_framework.cli analyze -i examples/tesla_model_s_defective.csv -o report.json

# CAN simulation from DBC
python -m ev_qa_framework.cli emulate --dbc my_battery.dbc --duration 60

# Run tests
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
telemetry = validate_telemetry(data)
```

**Anomaly detection:**
```python
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd

df = pd.read_csv("battery_telemetry.csv")
detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(df[["voltage", "current", "temperature"]])
predictions, scores = detector.detect(new_data)
```

**Cell imbalance:**
```python
from ev_qa_framework.cell_balance import CellBalanceAnalyzer

analyzer = CellBalanceAnalyzer(warning_threshold=0.02, critical_threshold=0.05)
cell_v = [3.30, 3.31, 3.305, 3.312, 3.29]
print(analyzer.compute_statistics(cell_v))
print(analyzer.classify_severity(cell_v))
```

**Thermal runaway (recommended API):**
```python
from ev_qa_framework.thermal_runaway import ThermalRunawayPredictor
import pandas as pd

predictor = ThermalRunawayPredictor(mode="rule")
df = pd.DataFrame({"temperature": [35, 37, 42, 58, 62]})
risk = predictor.predict_risk(df)
# {'risk_level': 'HIGH', 'risk_score': 8.3, 'confidence': 0.85, ...}
```

**DBC parsing:**
```python
from ev_qa_framework.dbc_parser import DBCParser

dbc = DBCParser("tesla_battery.dbc")
msg = dbc.get_message(0x101)
vals = dbc.decode(0x101, bytes([0x7D, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
# {'Voltage': 396.5}
```

## Project structure

```
ev_qa_framework/
  framework.py         # core QA engine
  models.py            # Pydantic models
  config.py            # thresholds and ML config
  analysis.py          # Isolation Forest, EVBatteryAnalyzer
  soh_predictor.py     # LSTM for SOH (TensorFlow optional)
  can_bus.py           # CAN 2.0B + J1939 simulation
  dbc_parser.py        # .dbc file parser (Vector CANdb + SavvyCAN)
  cell_balance.py      # cell voltage imbalance analysis
  thermal_runaway.py   # thermal runaway prediction (rule + ML)
  metrics.py           # Prometheus metrics
  cli.py               # CLI entry point
dashboard/
  app.py               # FastAPI
  grafana/             # Grafana dashboard JSON
tests/                 # 160+ tests
```

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework
pip install -e .[dev,ml]

# Run linting
ruff check .

# Run tests
pytest -v
```

## Changelog

### v1.1.0
- Thermal runaway deduplicated — `ThermalRunawayPredictor` is the single API (removed duplicate from `EVBatteryAnalyzer`)
- Fixed risk score calculation: temperature contribution uses deviation from 50°C, not absolute value
- CLI `analyze` now handles both `temperature` and `temp` column names
- Migrated `setup.py` → `pyproject.toml`, added `uv.lock`
- Applied ruff auto-fixes across the codebase
- Fixed `BatteryCellDataModel` import in package `__init__.py`
- Fixed SOHPredictor type hint (`Sequential` → `Any`)
- Fixed example in `framework.py` (`__main__`) — uses pack voltage (396V) instead of cell voltage (3.9V)
- Removed stale `build/` artifacts

## Compatibility

- CAN 2.0B (11-bit) and J1939 (29-bit extended)
- SavvyCAN / BUSMASTER DBC exports
- Prometheus + Grafana
- TensorFlow — optional (SOH prediction only)
- python-can — only needed for physical CAN hardware; simulation works without it

## License

MIT
