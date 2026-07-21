# EV-QA-Framework — Project Structure

> **Version:** 2.5.0  
> **Python:** ≥3.10  
> **License:** MIT  
> **Generated:** 2026-07-21

---

## Overview

EV-QA-Framework is an ML-powered Quality Assurance framework for Electric Vehicle and IoT battery testing. It provides:

- **Telemetry validation** — Safety threshold checks for voltage, current, temperature, SOC, SOH
- **Anomaly detection** — Rule-based + Isolation Forest ML models
- **SOH prediction** — LSTM-based State of Health forecasting
- **CAN bus emulation** — Virtual & hardware socketCAN, OBD-II ELM327, DBC-driven simulation
- **BMS protocol abstraction** — Unified interface for CAN, Modbus TCP/RTU
- **Chemistry profiles** — LFP / NMC / NCA parameter libraries with OCV, aging, thermal models
- **Digital twin** — Battery state simulation for HIL testing
- **Fleet analytics** — Multi-vehicle monitoring, alerting, trend analysis
- **V2G scenarios** — Vehicle-to-Grid health impact assessment
- **AutoML** — Automated hyperparameter optimization for anomaly/SOH models
- **Dashboard** — FastAPI + Grafana observability stack

---

## Directory Tree

```
EV-QA-Framework/
├── .github/
│   ├── ISSUE_TEMPLATE/          # GitHub issue templates
│   └── workflows/               # CI/CD pipelines
├── .hermes/
│   └── plans/                   # Internal planning docs
├── config/
│   ├── README.md
│   └── settings.yaml            # Unified configuration (YAML profiles)
├── dashboard/
│   ├── app.py                   # FastAPI dashboard server
│   └── grafana/                 # Grafana provisioning & dashboards
├── docs/
│   ├── source/                  # Sphinx source
│   └── build/                   # Generated HTML docs
├── ev_qa_framework/             # Main package (22 modules)
├── examples/                    # Usage examples & demos
├── notebooks/                   # Jupyter notebooks
├── research/                    # ML audit & research notes
├── scripts/                     # Utility scripts
├── tests/                       # Test suite (38 test files)
├── utils/
│   └── logger.py                # Shared logging utilities
├── .dockerignore
├── .gitlab-ci.yml
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml               # Project metadata, deps, tool config
├── README.md
├── requirements.txt             # Locked dependencies
├── SECURITY.md
└── STATUS.md
```

---

## Core Modules (`ev_qa_framework/`)

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `analysis.py` | ML anomaly detection (Isolation Forest) + telemetry analysis | `EVBatteryAnalyzer`, `AnomalyDetector` |
| `automl.py` | AutoML for anomaly detection & SOH prediction | `AutoMLAnomaly`, `AutoMLSOH` |
| `battery_scoring.py` | Battery health scoring & grading | `BatteryScorer` |
| `bms_protocol.py` | Unified BMS protocol abstraction (CAN + Modbus) | `BMSProtocolManager`, `BMSTelemetry`, `BMSCANInterface`, `BMSModbusTCPInterface`, `BMSModbusRTUInterface`, `ProtocolType` |
| `can_bus.py` | CAN bus simulation & hardware interface | `CANBatterySimulator`, `CANTelemetryReceiver`, `CANHardwareInterface`, `OBD2Adapter`, `DBCFileSimulator`, exceptions |
| `cell_balance.py` | Cell voltage imbalance analysis | `CellBalanceAnalyzer` |
| `chemistries.py` | Battery chemistry profiles (LFP/NMC/NCA) | `BatteryChemistryProfile`, `OCVCurve`, `AgingModel`, `ThermalModel`, `get_profile`, `list_profiles`, constants |
| `cli.py` | Command-line interface (`ev-qa`) | `main()`, subcommands: analyze, can-demo, emulate, train-soh, dashboard |
| `config.py` | Configuration dataclasses & YAML loading | `FrameworkConfig`, `SafetyThresholds`, `MLConfig`, `DEFAULT_CONFIG`, `TESLA_CONFIG` |
| `dbc_parser.py` | DBC file parsing for CAN message definitions | `DBCParser`, `builtin_dbc` |
| `digital_twin.py` | Battery digital twin for HIL simulation | `BatteryDigitalTwin`, `BatteryState` |
| `fleet_analytics.py` | Multi-vehicle fleet monitoring | `FleetAnalytics`, `FleetAlert` |
| `framework.py` | Main facade class orchestrating all components | `EVQAFramework` |
| `hil.py` | Hardware-in-the-loop test runner | `HILInterface`, `HILTestRunner`, `HILTestResult`, `BMSHardwareEmulator`, `CANMessage` |
| `metrics.py` | Prometheus metrics definitions | `battery_*` metric gauges |
| `modbus.py` | Modbus TCP/RTU client implementations | `ModbusTCPClient`, `ModbusRTUClient`, `BMS_REGISTER_MAP` |
| `models.py` | Pydantic data models | `BatteryTelemetryModel`, `BatteryCellDataModel` |
| `physics_features.py` | Physics-informed feature engineering | `PhysicsFeatureExtractor` |
| `soh_predictor.py` | LSTM-based SOH prediction | `SOHPredictor` |
| `soh_transformer.py` | Transformer-based SOH prediction | `SOHTransformer` |
| `thermal_runaway.py` | Thermal runaway risk prediction | `ThermalRunawayPredictor` |
| `utils.py` | Shared utilities | `normalize_columns` |
| `v2g_scenarios.py` | V2G scenario generation & health analysis | `V2GScenarioGenerator`, `V2GHealthAnalyzer` |

**Total: 22 modules, ~220KB source code**

---

## Test Suite (`tests/`)

| Test File | Target Module | Description |
|-----------|---------------|-------------|
| `test_advanced_analysis.py` | analysis | Advanced ML analysis scenarios |
| `test_automl.py` | automl | AutoML optimization tests |
| `test_battery_scoring.py` | battery_scoring | Health scoring & grading |
| `test_bms_protocol.py` | bms_protocol | BMS abstraction layer (core) |
| `test_bms_protocol_edge.py` | bms_protocol | Edge cases & error handling |
| `test_can_bus.py` | can_bus | CAN bus simulation & hardware |
| `test_cell_balance.py` | cell_balance | Cell imbalance detection |
| `test_chemistries.py` | chemistries | Chemistry profile validation |
| `test_chemistry_models.py` | chemistries | OCV/Aging/Thermal model tests |
| `test_cli.py` | cli | CLI subcommand tests |
| `test_config.py` | config | Configuration loading & validation |
| `test_config_edge_cases.py` | config | Edge cases in config system |
| `test_dbc_parser.py` | dbc_parser | DBC parsing correctness |
| `test_digital_twin.py` | digital_twin | Digital twin simulation |
| `test_ev_qa_anomalies.py` | analysis | Anomaly detection integration |
| `test_ev_qa_limits.py` | framework | Limit/boundary testing |
| `test_fleet_analytics.py` | fleet_analytics | Fleet monitoring & alerts |
| `test_hardware_can.py` | can_bus | Hardware CAN interface tests |
| `test_hil.py` | hil | HIL test runner core |
| `test_hil_edge_cases.py` | hil | HIL edge cases |
| `test_integration.py` | framework | End-to-end integration |
| `test_integration_extended.py` | framework | Extended integration scenarios |
| `test_modbus.py` | modbus | Modbus client tests |
| `test_model_persistence.py` | soh_predictor/transformer | Model save/load |
| `test_models_edge_cases.py` | models | Pydantic model validation |
| `test_physics_features.py` | physics_features | Feature extraction |
| `test_pydantic_models.py` | models | Model serialization |
| `test_soh_predictor.py` | soh_predictor | LSTM SOH prediction |
| `test_soh_transformer.py` | soh_transformer | Transformer SOH prediction |
| `test_standards.py` | — | Industry standard compliance |
| `test_standards_gb.py` | — | GB/T standard tests |
| `test_tesla_battery.py` | — | Tesla-specific test cases |
| `test_tesla_potesti.py` | — | Tesla Potesti validation |
| `test_tesla_script.py` | — | Tesla script scenarios |
| `test_v2g_scenarios.py` | v2g_scenarios | V2G scenario tests |

**Total: 38 test files, ~592 tests, 86% coverage**

---

## Configuration Files

### `pyproject.toml`
- **Build system:** setuptools
- **Dependencies:** pytest, scikit-learn, pandas, numpy, python-can, pydantic, fastapi, uvicorn, websockets, jinja2, pyyaml, aiohttp, aiofiles, matplotlib, prometheus-client, pyserial
- **Optional deps:** `ml` (tensorflow), `hardware` (python-can, pyserial), `dev` (ruff, mypy, pylint, pre-commit), `docs` (sphinx)
- **Entry point:** `ev-qa = ev_qa_framework.cli:main`
- **Tools:** ruff (lint/format), pytest (asyncio=auto)

### `requirements.txt`
Pinned versions for reproducible installs (includes dev deps: black, flake8, isort, docker, sphinx, python-dotenv)

### `config/settings.yaml`
Unified YAML configuration with **named profiles**:
- `default` — Baseline NMC 96s
- `tesla` — NCA 108s, strict thresholds, fail_on_anomaly=true
- `lfp` — LFP (CATL/BYD) 96s, thermal stability focus
- `nca` — NCA (Tesla 4680/2170) 96s
- `nmc` — Standard NMC 96s

Each profile defines: `safety_thresholds`, `ml_config`, `chemistry`, `cells_in_series`, `default_vin`, `fail_on_anomaly`

### `.pre-commit-config.yaml`
Hooks: ruff (fix + format), trailing-whitespace, end-of-file-fixer, check-yaml/json, merge-conflict, debug-statements, check-ast, pytest (pre-push)

### `Dockerfile`
- Base: `python:3.12-slim`
- uv 0.5 for dependency management
- Single-stage: `uv venv && uv pip install -e .`
- Healthcheck: version verification
- Entrypoint: `ev-qa`

### `docker-compose.yml`
Services:
- **tests** — pytest with coverage (HTML + XML + JUnit)
- **dashboard** — HTTP server for test artifacts (port 8081)
- **grafana** — Battery metrics visualization (port 3000)

### GitHub Workflows (`.github/workflows/`)
| Workflow | Trigger | Jobs |
|----------|---------|------|
| `test.yml` | push/PR to main/develop | lint → test (3.10/3.11/3.12) → coverage (Codecov) |
| `release.yml` | tag push `v*` | test → build (uv build) → publish PyPI (trusted) → GitHub Release |
| `nightly-coverage.yml` | cron 03:00 UTC / manual | Full coverage run, artifact upload |

---

## Examples (`examples/`)

| File | Purpose |
|------|---------|
| `simple_demo.py` | Minimal framework usage |
| `config_usage_example.py` | Configuration profiles demo |
| `tesla_battery_qa_test.py` | Tesla pack QA workflow |
| `tesla_advanced_analysis.py` | Advanced Tesla telemetry analysis |
| `advanced_integration_demo.py` | Multi-module integration |
| `model_persistence_example.py` | SOH model save/load |
| `tesla_qa_report.json` | Sample output report |

---

## Key Design Patterns

1. **Dataclass-based config** — `SafetyThresholds`, `MLConfig`, `FrameworkConfig` with `to_dict/from_dict` serialization
2. **Profile-driven YAML** — Single `settings.yaml` with named profiles, loaded via `FrameworkConfig.load_from_yaml()`
3. **Chemistry auto-application** — `FrameworkConfig(chemistry="lfp")` auto-populates thresholds from `chemistries.py`
4. **Protocol abstraction** — `BMSProtocolManager` unifies CAN/Modbus behind `BMSTelemetry` dataclass
5. **Pydantic models** — `BatteryTelemetryModel` validates all telemetry input
6. **Async test runner** — `EVQAFramework.run_test_suite()` is async for HIL/fleet workflows
7. **CLI subcommands** — argparse with dedicated functions per command

---

## Quick Start

```bash
# Install
uv pip install -e .

# Run tests
pytest tests/ -v --cov=ev_qa_framework

# CLI usage
ev-qa analyze -i data.csv -o report.json
ev-qa can-demo -d 10
ev-qa train-soh -i history.csv -m ./model_dir

# Docker
docker-compose up tests
```

---

## Related Documentation

- `README.md` — Project overview & installation
- `CHANGELOG.md` — Version history
- `CONTRIBUTING.md` — Development workflow
- `docs/` — Sphinx API reference (run `pip install -e .[docs] && sphinx-build docs/source docs/build`)
- `config/README.md` — Configuration guide