<div align="center">

# EV-QA-Framework

ML-Powered Quality Assurance Framework for Electric Vehicle Battery Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml/badge.svg)](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml)
[![Docker](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/docker.yml/badge.svg)](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/docker.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## Why

Electric vehicle battery systems produce a lot of telemetry data - thousands of readings per minute. EV-QA-Framework helps QA engineers and battery researchers catch anomalies, predict degradation, and validate battery performance without needing expensive test rigs.

The framework combines rule-based validation with machine learning (Isolation Forest for anomaly detection, LSTM for State of Health prediction), plus a CAN bus emulator so you can test without physical hardware.

---

## Features

- ML anomaly detection via Isolation Forest with adjustable sensitivity
- State of Health prediction using an LSTM neural network
- CAN bus emulation (CAN 2.0B and J1939 protocols)
- Interactive real-time dashboard (FastAPI + WebSocket + Chart.js)
- Configurable safety thresholds per vehicle profile
- Save/load trained models as JSON or joblib
- 100+ pytest tests with ML validation edge cases

---

## Quick Start

```bash
pip install -r requirements.txt

# Run analysis on sample data
python -m ev_qa_framework.cli analyze examples/tesla_model_s_defective.csv

# Launch dashboard
python -m ev_qa_framework.cli dashboard

# Generate synthetic CAN data
python -m ev_qa_framework.cli emulate --duration 60
```

With Docker:

```bash
docker compose -f docker-compose.prod.yml up -d
# then open http://localhost:8080
```

---

## Architecture

The framework has four main components:

- **Core QA Engine** - Pydantic-based validation with configurable safety thresholds
- **ML Analyzer** - Isolation Forest for anomaly detection, LSTM for SOH prediction
- **CAN Emulator** - Generates CAN 2.0B and J1939 data streams for offline testing
- **Dashboard** - FastAPI server with WebSocket-powered real-time charts

Data flows: CSV/API input -> Pydantic validation -> ML analysis -> dashboard visualization.

---

## CLI Reference

```bash
python -m ev_qa_framework.cli analyze examples/tesla_model_s_defective.csv

python -m ev_qa_framework.cli dashboard

python -m ev_qa_framework.cli emulate --duration 120 --protocol j1939

python -m ev_qa_framework.cli analyze examples/tesla_model_s_defective.csv \
  --config config/tesla_config.json \
  --output report.json \
  --save-model
```

---

## Docker

```bash
docker compose -f docker-compose.prod.yml up -d

# With custom configuration
cp .env.example .env
docker compose -f docker-compose.prod.yml --env-file .env up -d
```

Images are published to GitHub Container Registry:
`ghcr.io/remontsuri/ev-qa-framework:latest`

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v --cov=ev_qa_framework

# Specific test suites
pytest tests/test_ml_analysis.py -v
pytest tests/test_integration.py -v
```

---

## Project Structure

```
EV-QA-Framework/
  ev_qa_framework/         # Core package
    analysis.py            # ML anomaly detection
    cli.py                 # CLI entry point
    config.py              # Thresholds and logging setup
    framework.py           # Main QA engine
    models.py              # Pydantic validation models
    soh_predictor.py       # LSTM SOH predictor
    can_bus.py             # CAN bus simulator
  dashboard/               # Web dashboard
    app.py                 # FastAPI + WebSocket server
    templates/             # Jinja2 frontend
  api/                     # REST API
    routes.py              # API endpoints
  config/                  # Configuration profiles
  examples/                # Usage examples
  tests/                   # Test suite
```

---

## Roadmap

- [x] Core QA engine
- [x] ML anomaly detection
- [x] SOH prediction
- [x] CAN bus emulation
- [x] Interactive dashboard
- [x] Docker deployment
- [ ] PyPI package
- [ ] Automated release pipeline
- [ ] Cell imbalance detection
- [ ] Thermal runaway prediction
- [ ] Grafana datasource plugin

---

## Contributing

Bug reports, feature requests, and pull requests are welcome. See CONTRIBUTING.md for the workflow.

---

## License

MIT. See LICENSE for details.
