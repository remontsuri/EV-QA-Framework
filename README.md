# EV-QA-Framework

**ML-powered Quality Assurance Framework for Electric Vehicle Battery Systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml/badge.svg)](https://github.com/remontsuri/EV-QA-Framework/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub issues](https://img.shields.io/github/issues/remontsuri/EV-QA-Framework)](https://github.com/remontsuri/EV-QA-Framework/issues)

A comprehensive **Python framework** for automated quality assurance of EV battery telemetry. Provides ML-powered anomaly detection, State-of-Health (SOH) prediction, CAN bus emulation, and real-time monitoring — purpose-built for battery engineers, QA teams, and automotive developers.

---

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Battery Telemetry Validation** | Schema-based validation for voltage, current, temperature, SOC, SOH using Pydantic |
| **ML Anomaly Detection** | Isolation Forest-based detection of irregular battery patterns (cell imbalance, thermal runaway precursors) |
| **SOH Prediction** | LSTM-based model for battery degradation forecasting and remaining useful life estimation |
| **CAN Bus Emulation** | Tools to simulate and test physical vehicle network communication (CAN 2.0, OBD-II protocols) |
| **Real-time Dashboard** | FastAPI web dashboard for live telemetry monitoring and visualization |
| **Test Suite** | 85+ automated tests covering telemetry validation, anomaly detection, and integration |

---

## Quick Start

### Installation

```bash
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Dashboard

```bash
python quickstart.py
```

Opens a real-time battery telemetry dashboard at `http://localhost:8000`.

### Running Tests

```bash
python -m pytest -v
```

---

## Usage Examples

### 1. Battery Telemetry Validation

Validate incoming telemetry from BMS (Battery Management System) or CAN bus:

```python
from ev_qa_framework.models import validate_telemetry

data = {
    "vin": "1HGBH41JXMN109186",
    "voltage": 396.5,       # Pack voltage (V)
    "current": 125.3,       # Current (A)
    "temperature": 35.2,    # Cell temperature (°C)
    "soc": 78.5,            # State of Charge (%)
    "soh": 96.2             # State of Health (%)
}
telemetry = validate_telemetry(data)
```

### 2. Anomaly Detection on Battery Data

Detect outliers in battery telemetry streams — useful for identifying cell degradation, sensor faults, or thermal events:

```python
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd

# Load battery telemetry CSV (voltage, current, temp per cell group)
df = pd.read_csv("battery_telemetry.csv")

detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(df[["voltage", "current", "temperature", "soc"]])

predictions, scores = detector.detect(new_data)
# predictions: 1 = normal, -1 = anomaly
```

### 3. State of Health (SOH) Prediction

Forecast battery degradation using LSTM deep learning:

```python
from ev_qa_framework.soh_predictor import SOHPredictor

model = SOHPredictor(seq_length=50, n_features=4)
model.train(soh_training_data, epochs=50)

# Predict future SOH
future_soh = model.predict_soh(sequence_data)
print(f"Predicted SOH: {future_soh:.1f}%")
```

### 4. CAN Bus Emulation

Simulate battery telemetry over CAN bus for HIL (Hardware-in-the-Loop) testing:

```python
from ev_qa_framework.can_bus import CANEmulator

emulator = CANEmulator(interface="virtual", channel="vcan0")
emulator.start()

# Simulate battery cell voltage broadcast
emulator.send_battery_status(voltage=400.0, current=120.0, soc=75.0)
```

---

## Architecture

```
ev_qa_framework/
├── models.py          # Pydantic schemas for battery telemetry
├── analysis.py        # ML anomaly detection (Isolation Forest)
├── soh_predictor.py   # LSTM-based SOH degradation forecasting
├── can_bus.py         # CAN bus emulation and simulation
├── config.py          # Configuration management
├── framework.py       # Core orchestration
└── cli.py             # CLI entry point

dashboard/             # FastAPI web dashboard
tests/                 # 85+ automated tests
examples/              # Usage demos and notebooks
scripts/               # Utility scripts
```

---

## Project Structure

- `ev_qa_framework/` — Core Python package with battery QA models and ML
- `dashboard/` — FastAPI web dashboard for real-time telemetry visualization
- `scripts/` — Utility scripts for CAN emulation and debugging
- `tests/` — Comprehensive test suite (pytest)
- `examples/` — Usage demos and Jupyter notebooks
- `notebooks/` — Interactive analysis notebooks

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — Version history and release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [LICENSE](LICENSE) — MIT License

---

## Who Is This For?

- **Battery QA Engineers** — Validate telemetry data and detect anomalies in battery packs
- **EV System Developers** — Integrate battery monitoring into vehicle software pipelines
- **BMS Algorithm Engineers** — Test SOH prediction and anomaly detection models against real data
- **Test & Validation Teams** — Automate battery telemetry validation in CI/CD with CAN emulation
- **Automotive Researchers** — Analyze battery degradation patterns with ML models

## Technologies

- **Python 3.8+** with type hints and Pydantic validation
- **scikit-learn** Isolation Forest for anomaly detection
- **TensorFlow/Keras** LSTM for SOH forecasting
- **python-can** for CAN bus protocol simulation
- **FastAPI** for the real-time dashboard
- **pytest** (85+ tests) for QA automation

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
Battery engineers and EV developers — your domain expertise is especially valued.

## License

MIT License — see [LICENSE](LICENSE) for details.

