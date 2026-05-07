# EV-QA-Framework

**Python framework for Electric Vehicle battery quality assurance and anomaly detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EV-QA-Framework provides tools for automated quality assurance of battery telemetry (voltage, current, temperature, SOC). It includes ML-powered anomaly detection and SOH prediction models specifically tuned for EV battery datasets.

## Key Features

- **Automated QA**: Validation for battery telemetry data using Pydantic.
- **ML Anomaly Detection**: Isolation Forest based detection of irregular patterns.
- **SOH Prediction**: LSTM-based model for battery degradation forecasting.
- **CAN Bus Emulation**: Tools to simulate and test physical vehicle network communication.
- **Dashboard**: Real-time visualization for telemetry monitoring.
- **Test Suite**: 85+ automated tests for safety and integration.

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

### Running Tests

```bash
python -m pytest
```

## Usage Example

```python
from ev_qa_framework.models import validate_telemetry
from ev_qa_framework.analysis import AnomalyDetector
import pandas as pd

# 1. Validate telemetry data
data = {
    "vin": "1HGBH41JXMN109186",
    "voltage": 396.5,
    "current": 125.3,
    "temperature": 35.2,
    "soc": 78.5,
    "soh": 96.2
}
telemetry = validate_telemetry(data)

# 2. ML Anomaly Detection
detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(training_df)

predictions, scores = detector.detect(new_telemetry_df)
# predictions: 1 = normal, -1 = anomaly
```

## Project Structure

- `ev_qa_framework/`: Core logic and models.
- `dashboard/`: FastAPI web dashboard and visualization.
- `scripts/`: Utility scripts for emulation and debugging.
- `tests/`: Extensive test suite.
- `examples/`: Usage demos.

## Contributing

Contributions are welcome. Please check the `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
