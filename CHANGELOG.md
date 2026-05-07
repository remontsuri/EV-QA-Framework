# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-20

### Changed
- Refactored project structure: moved core models to `ev_qa_framework.models` and consolidated ML analysis.
- Cleaned up redundant scripts and moved utility tools to `scripts/`.
- Simplified documentation and removed AI-generated reports for a more developer-focused experience.
- Improved test organization by moving all tests to the `tests/` directory.

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
