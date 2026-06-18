# EV-QA-Framework

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-955%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)
![Version](https://img.shields.io/badge/version-2.3.1-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/ev-qa-framework.svg)](https://pypi.org/project/ev-qa-framework/)
[![GitHub Release](https://img.shields.io/github/v/release/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/releases)


**EV Battery QA Framework — detect thermal runaway, validate BMS telemetry, comply with UN 38.3 / IEC 62660 / GB 38031, and ship with 955 passing tests and a Docker-ready pipeline.**

22 modules. MIT licensed. Python 3.10+.

---

## Installation

```bash
pip install ev-qa-framework
```

---

## 30-second value

```bash
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework
docker compose up -d
open http://localhost:8081
```

Done. You have a running battery QA workstation:
- telemetry validation
- ML anomaly detection
- thermal runaway early warning
- cell imbalance analysis
- SOH prediction
- compliance testing against 6 international standards
- live dashboard with Prometheus metrics

No cloud account required. No external dependencies. Just a CSV and a terminal.

---

## What you get

**Input safety layer.** Pydantic schemas for voltage, current, temperature, SOC, SOH. Bad VINs, out-of-range values, and malformed rows are rejected before they reach your models.

**Anomaly detection.** Isolation Forest on voltage/current/temperature streams. Configurable contamination, severity thresholds, estimator count.

**Thermal runaway prediction.** Rule-based heuristics (temperature, delta-temp, anomaly score, chemistry runaway point). CRITICAL trigger defaults at >=130 C, rapid-rise trigger at >10 C/min. Catches overheating before cascade onset. Confidence score clamped to [0, 1].

**SOH prediction.** LSTM-based State of Health forecasting from historical telemetry. Transformer-based prediction via soh_transformer for longer sequences.

**Cell imbalance analysis.** Statistical analysis of cell group voltages with configurable thresholds, outlier detection, linear regression trend.

**Battery scoring.** Composite health score (0-100) with letter grades (A+ through F). Combines SOH, internal resistance, cell balance, and thermal history.

**CAN bus and DBC.** CAN 2.0B and J1939 simulation and reception. DBC parser supports Vector CANdb, SavvyCAN exports, Intel/Motorola byte order.

**Fleet analytics.** Aggregate analysis across vehicle fleets: degradation curves, anomaly distribution, SOH histograms.

**Digital twin.** Real-time battery simulation mirroring physical pack behavior. Charge/discharge what-if scenarios and aging projections.

**V2G scenarios.** Vehicle-to-Grid simulation: bidirectional energy flow, grid demand response, cycling impact on battery health, revenue estimation.

**AutoML.** Automated model selection and hyperparameter optimization for SOH prediction and anomaly detection.

**HIL integration.** Hardware-in-the-Loop interface for physical BMS hardware and test stands via TCP/Serial.

**Compliance testing.** UN 38.3, IEC 62660, SAE J2464, ISO 12405, GB/T 31484, GB/T 31486, GB 38031.

**Observability.** Prometheus /metrics endpoint, Grafana dashboard, HTML coverage reports, JUnit XML.

---

## Quick start

```bash
# Python CLI (direct)
uv run pytest -v
uv run python run_factory_inspection.py

# Docker Compose (recommended for fresh environments)
docker compose up --build
```

- Tests + HTML coverage: http://localhost:8081/coverage/
- Prometheus metrics: http://localhost:8081/metrics

---

## One-liners

Analyze a CSV:

```bash
uv run python -m ev_qa_framework.cli analyze -i examples/tesla_model_s_defective.csv -o report.json
```

Emulate CAN traffic:

```bash
uv run python -m ev_qa_framework.cli emulate --dbc my_battery.dbc --duration 60
```

Train SOH model:

```bash
uv run python -m ev_qa_framework.cli train-soh -d examples/tesla_battery_qa_test.py
```

---

## Project structure

`
ev_qa_framework/
  framework.py         # core QA engine
  models.py            # Pydantic models + telemetry validation
  config.py            # thresholds and ML config
  analysis.py          # Isolation Forest, EVBatteryAnalyzer
  soh_predictor.py     # LSTM for SOH (TensorFlow optional)
  soh_transformer.py   # Transformer SOH predictor
  can_bus.py           # CAN 2.0B + J1939 simulation
  dbc_parser.py        # .dbc file parser (Vector CANdb + SavvyCAN)
  cell_balance.py      # cell voltage imbalance analysis
  thermal_runaway.py   # thermal runaway prediction (rule + ML)
  battery_scoring.py   # composite battery health scoring
  physics_features.py  # electrochemical/thermal feature extraction
  fleet_analytics.py   # fleet-wide analytics and benchmarking
  digital_twin.py      # real-time battery digital twin
  v2g_scenarios.py     # Vehicle-to-Grid simulation
  automl.py            # automated model selection and HPO
  hil.py               # Hardware-in-the-Loop interface
  metrics.py           # Prometheus metrics
  cli.py               # CLI entry point
  chemistries.py       # battery chemistry definitions (LFP, NMC, NCA)
tests/                  # 955 tests
examples/               # sample telemetry and demos
run_factory_inspection.py  # end-to-end factory QA demo
`

---

## Status

| Artifact | Value |
|---|---|
| Tests | 955 passing |
| Coverage | 93% |
| CI | Lint + Test + Coverage |
| License | MIT |
| Python | 3.10+ |
| PyPI | ev-qa-framework 2.1.3 |

Regression risk is tracked in tests/. Coverage artifacts (coverage/, junit.xml) are present in the release pipeline.

---

## Roadmap

- [x] GitHub Actions CI badge + nightly coverage job
- [x] Grafana dashboard import JSON + provisioning
- [x] public PyPI release
- [ ] real BMS telemetry adapters (Tesla, BYD, Nio)
- [ ] V2S + charging-station scenarios
- [ ] integration with Vector CANoe / CANalyzer
