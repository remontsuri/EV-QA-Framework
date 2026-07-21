# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-07-21

### Fixed
- **framework.py**: Removed `asyncio.run()` on synchronous method (TypeError at runtime)
- **analysis.py**: `analyze_telemetry()` returns error dict instead of None for small datasets
- **analysis.py**: `load_model()` correctly marks model as fitted after restoration
- **v2g_scenarios.py**: Fixed key mismatch (`components.soh` → `soh_score`)
- **release.yml**: Fixed version variable case (`Version` → `VERSION`)

### Changed
- Version synced to 2.5.0 across pyproject.toml, __init__.py, Dockerfile, README
- ruff target-version updated from py39 to py310
- Removed Python 3.9 classifier (project requires 3.10+)
- Removed unused imports (asyncio, json, warnings)
- CI workflow: `uv sync --no-dev --frozen` → `uv sync --frozen` for test job
- docker-compose.yml: Grafana password env var syntax fixed, version label updated
- .dockerignore: Added exclusions for notebooks/, research/, examples/, docs/
- pyproject.toml: Moved fastapi/uvicorn/websockets/jinja2 to optional [web] extra
- pyproject.toml: Removed F401 from global ruff ignore
- Deleted requirements.txt (contradicted pyproject.toml)
- Deleted .gitlab-ci.yml (GitLab CI on GitHub project)
- Deleted redundant tests_soc_soh_cross/ directory
- Deleted Russian-language config/README.md and examples/config_usage_example.py
- Updated SECURITY.md supported versions to 2.5.x
- Updated CONTRIBUTING.md test count to 967
- Updated PROJECT_STRUCTURE.md version to 2.5.0

## [2.4.0] - 2026-06-18

### Fixed
- Round 2 roast findings: version drift, streaming perf, error swallowing, min samples
- README threshold, settings.yaml ref, demo English

### Added
- vehicle_id metrics, health endpoint, factory configs
- GradientBoosting AutoML, StreamingAnomalyDetector, uncertainty quantification
- Optional extras (ml, hardware, web, can, monitoring)
- SIGTERM/SIGINT handler with correct logger name

## [2.3.1] - 2026-06-18

### Fixed
- All 27 roast findings verified and fixed

## [2.3.0] - 2026-06-18

### Fixed
- Complete roast: all HIGH + MEDIUM + LOW findings

## [2.2.0] - 2026-06-18

### Fixed
- Full roast completion: all findings fixed

## [2.1.5] - 2026-06-18

### Fixed
- Roast fixes: security, docs, code quality

## [2.1.4] - 2026-06-18

### Fixed
- Kanban audit fixes + README cleanup

## [2.1.3] - 2026-06-15

### Fixed
- Docs & CI alignment

## [2.1.0] - 2026-06-12

### Added
- Chemistry data fix & security hardening

## [2.0.0] - 2026-06-10

### Added — New Modules (10)

- **`battery_scoring`** — Composite battery health scoring. Combines SOH, internal resistance, cell balance, and thermal history into a 0–100 score with letter grades (A+ through F). Configurable weights per chemistry type.
- **`physics_features`** — Electrochemical and thermal feature extraction from raw telemetry. Computes diffusion rates, heat generation estimates (Joule + entropic), and equivalent circuit model parameters (R0, R1, C1).
- **`fleet_analytics`** — Fleet-wide aggregate analysis. Degradation curve fitting, anomaly distribution heatmaps, comparative benchmarking across vehicle groups, fleet-wide SOH histograms. Supports CSV and Parquet input.
- **`digital_twin`** — Real-time battery digital twin simulation. Mirrors physical pack behavior using electrochemical models. Supports what-if scenarios for arbitrary charge/discharge profiles and long-term aging projections.
- **`v2g_scenarios`** — Vehicle-to-Grid simulation. Models bidirectional energy flow, grid demand response events, cycling impact on battery health, revenue estimation. Pre-built scenarios: peak_shaving, frequency_regulation, solar_buffering.
- **`automl`** — Automated model selection and hyperparameter optimization for SOH prediction and anomaly detection. Supports scikit-learn and TensorFlow backends. Bayesian optimization via Optuna (optional).
- **`soh_transformer`** — Transformer-based SOH prediction. Multi-head attention over temporal telemetry sequences. Outperforms LSTM on sequences >500 steps. Compatible with the AutoML pipeline.
- **`hil`** — Hardware-in-the-Loop interface. Connects the framework to physical BMS hardware and test stands via TCP/Serial. Supports real-time data exchange, closed-loop testing, and automated test sequence execution.
- **`test_standards`** — Compliance testing against UN 38.3, IEC 62660, UL 1973, UL 2054. Automated test report generation with pass/fail criteria.
- **`test_standards_gb`** — Compliance testing against Chinese GB/T standards (GB/T 31484, GB/T 31485, GB/T 31486, GB 38031). GB-specific test profiles and report templates.

### Added — Tests

- Expanded test suite from 235 to 592 tests (+153%).
- Coverage improved from ~60% to 86%.
- Unit tests for all 10 new modules.
- Integration tests for CAN bus + DBC parser pipeline.
- End-to-end tests for CLI commands (analyze, emulate, dashboard, fleet-report).
- Property-based tests for telemetry validation models (Hypothesis).
- Regression tests for SOH predictor serialization round-trips.

### Added — Infrastructure

- `pyproject.toml` `[dependency-groups]` for dev, ml, and docs extras.
- `uv.lock` for reproducible dependency resolution.
- GitHub Actions workflow for multi-version Python testing (3.10, 3.11, 3.12).
- Pre-commit hooks: ruff, mypy, conventional-commit linting.
- Dockerfile with multi-stage build (builder + runtime).

### Changed

- Migrated all tooling from pip to uv.
- Updated minimum Python version from 3.8 to 3.9.
- Refactored `config.py` — nested dictionary merge now uses deep merge strategy.
- Refactored `cli.py` — added `fleet-report` and `hil-test` subcommands.
- Refactored `metrics.py` — added fleet-level Prometheus gauges.
- Improved error messages across all modules with structured error codes.

### Fixed

- **SOH scaler serialization** — `StandardScaler` state was not correctly serialized/deserialized when saving and loading SOH predictor models. Fixed by implementing custom `get_params()` / `set_params()` round-trip.
- **Dockerfile** — multi-stage build was failing due to missing build dependencies in the runtime stage. Fixed by properly separating build and runtime layers.
- **Config merge** — `Config.merge()` was performing shallow merge on nested dictionaries, causing nested keys to be overwritten. Fixed with recursive deep merge.
- **DBC parser** — Motorola byte order signals with offset=0 were off by one bit. Fixed bit indexing in `_decode_motorola()`.
- **Thermal runaway** — `predict_risk()` returned incorrect confidence for single-row DataFrames. Fixed confidence calculation edge case.
- **CAN bus** — J1939 extended frame IDs > 0x1FFFFFFF were not rejected. Added validation for 29-bit ID range.

### Deprecated

- `EVBatteryAnalyzer.thermal_runaway()` — use `ThermalRunawayPredictor.predict_risk()` instead (deprecated since v1.1.0, will be removed in v3.0.0).
- `soh_predictor.SOHPredictor(use_gpu=True)` — GPU support is now handled via the `automl` module configuration.

## [1.1.0] - 2026-01-20

### Changed

- Refactored project structure: moved core models to `ev_qa_framework.models` and consolidated ML analysis.
- Cleaned up redundant scripts and moved utility tools to `scripts/`.
- Simplified documentation and removed AI-generated reports.
- Improved test organization by moving all tests to the `tests/` directory.

### Fixed

- Thermal runaway deduplicated — `ThermalRunawayPredictor` is the single API (removed duplicate from `EVBatteryAnalyzer`).
- Fixed risk score calculation: temperature contribution uses deviation from 50°C, not absolute value.
- CLI `analyze` now handles both `temperature` and `temp` column names.
- Fixed `BatteryCellDataModel` import in package `__init__.py`.
- Fixed SOHPredictor type hint (`Sequential` to `Any`).
- Fixed example in `framework.py` (`__main__`) — uses pack voltage (396V) instead of cell voltage (3.9V).
- Removed stale `build/` artifacts.

### Infrastructure

- Migrated `setup.py` to `pyproject.toml`, added `uv.lock`.
- Applied ruff auto-fixes across the codebase.

## [1.0.0] - 2026-01-20

### Added

- Initial release of EV-QA-Framework.
- Pydantic models for strict telemetry validation.
- ML Anomaly Detection using Isolation Forest (200 estimators).
- LSTM-based SOH prediction.
- CAN Bus emulation support.
- Interactive Dashboard using FastAPI and Chart.js.
- Comprehensive test suite with 85+ automated tests.
- Docker support and CI/CD configurations.

## [0.1.0] - Pre-release

### Added

- Basic `EVQAFramework` class.
- Simple temperature/voltage validation.
- Initial test suite.
