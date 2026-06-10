# EV-QA-Framework

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-592%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)
![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
[![GitHub Release](https://img.shields.io/github/v/release/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/releases)

QA framework for electric vehicle battery systems. Validates BMS telemetry, detects anomalies, predicts SOH degradation, emulates CAN bus traffic, evaluates thermal runaway risk, scores battery health, runs fleet analytics, simulates V2G scenarios, and provides a battery digital twin.

22 modules. 592 tests. 86% coverage. MIT licensed.

## What it does

**Telemetry validation.** Pydantic schemas for voltage, current, temperature, SOC, SOH. Catches bad VINs and out-of-range values at the input layer.

**ML anomaly detection.** Isolation Forest on voltage/current/temperature streams. Configurable contamination, severity thresholds, and number of estimators.

**SOH prediction.** LSTM-based State of Health forecasting from historical telemetry (TensorFlow optional). Transformer-based prediction via `soh_transformer` for longer sequences.

**Cell imbalance analysis.** Statistical analysis of cell group voltages with configurable thresholds, outlier detection, and linear regression trend.

**Thermal runaway prediction.** `ThermalRunawayPredictor` with two modes: rule-based heuristic (dT/dt, temperature, anomaly score) and ML (Isolation Forest). CRITICAL trigger at >65°C or heating rate >5°C/min.

**CAN bus.** CAN 2.0B (11-bit) and J1939 (29-bit extended) simulation and reception. DBC parser supports Vector CANdb format, SavvyCAN exports, Intel/Motorola byte order.

**Battery scoring.** Composite health score (0–100) with letter grades (A+ through F). Combines SOH, internal resistance, cell balance, and thermal history.

**Physics-based features.** Electrochemical and thermal feature extraction — diffusion rates, heat generation estimates, equivalent circuit model parameters.

**Fleet analytics.** Aggregate analysis across vehicle fleets: degradation curves, anomaly distribution, comparative benchmarking, SOH histograms.

**Digital twin.** Real-time battery simulation mirroring physical pack behavior. Supports what-if scenarios for charge/discharge profiles and aging projections.

**V2G scenarios.** Vehicle-to-Grid simulation: bidirectional energy flow, grid demand response, cycling impact on battery health, revenue estimation.

**AutoML.** Automated model selection and hyperparameter optimization for SOH prediction and anomaly detection.

**HIL integration.** Hardware-in-the-Loop interface for connecting to physical BMS hardware and test stands via TCP/Serial.

**Test standards.** Compliance testing against UN 38.3, IEC 62660, SAE J2464, ISO 12405, GB/T 31484, GB/T 31486, GB 38031.

**Dashboard.** FastAPI + WebSocket + Chart.js. Real-time telemetry and Prometheus `/metrics` endpoint with Grafana dashboard.

**CLI.** Analyze CSV telemetry, run CAN emulation, train SOH models, start dashboard, run fleet reports.

## Quick start

```bash
# Install from GitHub
uv pip install git+https://github.com/remontsuri/EV-QA-Framework.git

# Launch dashboard
python -m ev_qa_framework.cli dashboard
# → http://localhost:8000
# → http://localhost:8000/metrics (Prometheus)

# Analyze a CSV
python -m ev_qa_framework.cli analyze -i examples/tesla_model_s_defective.csv -o report.json

# CAN simulation from DBC
python -m ev_qa_framework.cli emulate --dbc my_battery.dbc --duration 60

# Run tests
uv run pytest -v
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

**Thermal runaway:**
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

**Battery scoring:**
```python
from ev_qa_framework.battery_scoring import BatteryScorer

scorer = BatteryScorer()
telemetry = pd.DataFrame({
    "voltage": [396.5], "current": [125.3], "temperature": [42.0],
    "soh": [92.5], "internal_resistance": [0.15]
})
result = scorer.compute_score(telemetry_df=telemetry, cell_voltages=[3.30, 3.31, 3.305, 3.312, 3.29])
# {"score": 91.0, "grade": "A", ...}
```

**Fleet analytics:**
```python
from ev_qa_framework.fleet_analytics import FleetAnalytics

fa = FleetAnalytics()
fa.add_battery("V1", telemetry_df_v1)
fa.add_battery("V2", telemetry_df_v2)
summary = fa.get_fleet_summary()
# {"fleet_size": 2, "avg_soh": 91.2, ...}
```

**Digital twin:**
```python
from ev_qa_framework.digital_twin import BatteryDigitalTwin

twin = BatteryDigitalTwin()
cycle = pd.DataFrame({"current": np.random.normal(50, 20, 3600)})
result = twin.simulate_drive_cycle(cycle_profile=cycle, dt=1.0)
state = twin.get_state()
# {"soh": 92.8, "voltage": 395.7, ...}
```

**V2G scenarios:**
```python
from ev_qa_framework.v2g_scenarios import V2GScenarioGenerator

gen = V2GScenarioGenerator(battery_capacity_ah=220, nominal_voltage=400)
cycle = gen.generate_v2g_cycle(duration_hours=24, grid_demand_profile="typical")
# DataFrame with columns: grid_demand, battery_power
```

## Project structure

```
ev_qa_framework/
  framework.py          # core QA engine
  models.py             # Pydantic models
  config.py             # thresholds and ML config
  analysis.py           # Isolation Forest, EVBatteryAnalyzer
  soh_predictor.py      # LSTM for SOH (TensorFlow optional)
  can_bus.py            # CAN 2.0B + J1939 simulation
  dbc_parser.py         # .dbc file parser (Vector CANdb + SavvyCAN)
  cell_balance.py       # cell voltage imbalance analysis
  thermal_runaway.py    # thermal runaway prediction (rule + ML)
  metrics.py            # Prometheus metrics
  cli.py                # CLI entry point
  chemistries.py        # battery chemistry definitions (LFP, NMC, NCA, etc.)
  battery_scoring.py    # composite battery health scoring
  physics_features.py   # electrochemical/thermal feature extraction
  fleet_analytics.py    # fleet-wide analytics and benchmarking
  digital_twin.py       # real-time battery digital twin
  v2g_scenarios.py      # Vehicle-to-Grid simulation
  automl.py             # automated model selection and HPO
  soh_transformer.py    # Transformer-based SOH prediction
  hil.py                # Hardware-in-the-Loop interface
dashboard/
  app.py                # FastAPI
  grafana/              # Grafana dashboard JSON
tests/                  # 592 tests
```

## Module reference

| Module | Description |
|---|---|
| `framework.py` | Core QA engine — orchestrates validation, analysis, and reporting |
| `models.py` | Pydantic data models for telemetry, BMS messages, and VIN validation |
| `config.py` | Centralized configuration — thresholds, ML params, CAN settings |
| `analysis.py` | Anomaly detection (Isolation Forest), statistical analysis |
| `soh_predictor.py` | LSTM-based SOH forecasting from historical telemetry |
| `can_bus.py` | CAN 2.0B / J1939 simulation, transmission, and reception |
| `dbc_parser.py` | DBC file parser — Vector CANdb, SavvyCAN, Intel/Motorola byte order |
| `cell_balance.py` | Cell voltage imbalance detection and trend analysis |
| `thermal_runaway.py` | Thermal runaway prediction — rule-based and ML modes |
| `metrics.py` | Prometheus metrics for dashboard and monitoring |
| `cli.py` | CLI entry point — analyze, emulate, train, dashboard, fleet |
| `chemistries.py` | Battery chemistry definitions — LFP, NMC, NCA, LMO parameters |
| `battery_scoring.py` | Composite health scoring (0–100) with letter grades |
| `physics_features.py` | Electrochemical/thermal feature extraction from telemetry |
| `fleet_analytics.py` | Fleet-wide degradation, anomaly distribution, benchmarking |
| `digital_twin.py` | Real-time battery simulation with what-if scenarios |
| `v2g_scenarios.py` | V2G simulation — energy flow, revenue, cycling impact |
| `automl.py` | AutoML — model selection, hyperparameter optimization |
| `soh_transformer.py` | Transformer architecture for SOH prediction |
| `hil.py` | Hardware-in-the-Loop interface for physical BMS integration |

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework
uv sync --all-extras

# Run linting
uv run ruff check .

# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=ev_qa_framework --cov-report=term-missing
```

## Changelog

### v2.0.0
- Added 10 new modules: `battery_scoring`, `physics_features`, `fleet_analytics`, `digital_twin`, `v2g_scenarios`, `automl`, `soh_transformer`, `hil`, `test_standards`, `test_standards_gb`
- Expanded test suite from 235 to 592 tests (86% coverage)
- Fixed SOH scaler serialization/deserialization
- Fixed Dockerfile multi-stage build
- Fixed config merge logic for nested dictionaries
- Migrated all tooling from pip to uv

### v1.1.0
- Thermal runaway deduplicated — `ThermalRunawayPredictor` is the single API
- Fixed risk score calculation: temperature contribution uses deviation from 50°C
- CLI `analyze` now handles both `temperature` and `temp` column names
- Migrated `setup.py` to `pyproject.toml`, added `uv.lock`

## Compatibility

- CAN 2.0B (11-bit) and J1939 (29-bit extended)
- SavvyCAN / BUSMASTER DBC exports
- Prometheus + Grafana
- TensorFlow — optional (SOH prediction only)
- python-can — only needed for physical CAN hardware; simulation works without it
- Python 3.9+

## License

MIT
