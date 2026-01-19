# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Examples directory with `simple_demo.py` and `sample_telemetry.csv`
- GitHub issue templates (bug report, feature request)
- Comprehensive CONTRIBUTING.md guide
- Comparison table in README.md vs competitors
- Additional badges (pytest, ML, Docker)

## [1.0.0] - 2026-01-20

### Added
- Initial release of EV-QA-Framework
- **Pydantic Models** (`ev_qa_models.py`)
  - `BatteryTelemetryModel` with strict validation
  - VIN validation (17 chars, no I/O/Q)
  - Voltage range: 0-1000V
  - Temperature range: -50 to +150°C
  - SOC/SOH: 0-100%
  
- **ML Anomaly Detection** (`ev_qa_analysis.py`)
  - `EVBatteryAnalyzer` with Isolation Forest (200 estimators)
  - `AnomalyDetector` with train/detect separation
  - Severity classification (CRITICAL/WARNING/INFO)
  - Comprehensive docstrings in Russian
  
- **Test Suite** (64+ tests)
  - `test_ev_qa.py` - Original 14 tests
  - `test_ev_qa_limits.py` - 23+ boundary tests
  - `test_ev_qa_anomalies.py` - 15+ anomaly detection tests
  - `test_ml_analysis.py` - 12+ ML component tests
  - `test_pydantic_models.py` - 14+ validation tests
  
- **CI/CD Infrastructure**
  - Dockerfile for containerization
  - docker-compose.yml for multi-service setup
  - .gitlab-ci.yml pipeline
  - pytest coverage reporting
  
- **Documentation**
  - Professional README.md with badges
  - CONTRIBUTING.md guide
  - IMPROVEMENTS_REPORT.md
  - OUTREACH_STRATEGY.md
  - PERPLEXITY_COMET_PROMPT.md

### Changed
- Updated `requirements.txt` with pydantic>=2.0.0
- Enhanced `ev_qa_analysis.py` with type hints
- Improved IsolationForest parameters (n_estimators=200, n_jobs=-1)

### Fixed
- N/A (initial release)

## [0.1.0] - Pre-release

### Added
- Basic `EVQAFramework` class
- Simple temperature/voltage validation
- 14 initial tests

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
