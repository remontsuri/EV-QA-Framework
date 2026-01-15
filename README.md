# EV-QA-Framework

[![GitHub license](https://img.shields.io/github/license/remontsuri/EV-QA-Framework)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://pytest.org/)

Mini QA Framework for Electric Vehicle & IoT Testing - AI-powered battery management system testing framework with pytest and CAN protocol support. Includes telemetry monitoring, ML-based anomaly detection, and GitLab CI/CD integration.

## Features

- **Battery Telemetry Management** - Real-time monitoring of EV battery parameters (voltage, current, temperature, SOC, SOH)
- **Automated Validation** - Safety threshold checks for battery parameters
- **ML-Based Anomaly Detection** - Statistical analysis for detecting unusual patterns in telemetry data
- **Async Testing Framework** - Concurrent test execution with asyncio support
- **CAN Protocol Support** - Ready for CAN bus integration (python-can)
- **GitLab CI/CD** - Automated testing, linting, and Docker builds
- **Comprehensive Test Suite** - 14+ pytest test cases covering all major functionality

## Installation

```bash
# Clone repository
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from ev_qa_framework import EVQAFramework, BatteryTelemetry
import asyncio

# Initialize framework
qa = EVQAFramework("ChargePoint-QA")

# Prepare telemetry data
test_data = [
    {'voltage': 3.9, 'current': 50, 'temperature': 35, 'soc': 80, 'soh': 98},
    {'voltage': 3.95, 'current': 45, 'temperature': 36, 'soc': 85, 'soh': 98},
    {'voltage': 3.85, 'current': 60, 'temperature': 45, 'soc': 75, 'soh': 97},
]

# Run test suite
result = asyncio.run(qa.run_test_suite(test_data))
print(result)
# Output: {'total_tests': 3, 'passed': 3, 'failed': 0, 'anomalies': []}
```

## Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=ev_qa_framework --cov-report=html

# Run specific test class
pytest tests/test_ev_qa.py::TestEVQAFramework -v
```

## Project Structure

```
EV-QA-Framework/
├── ev_qa_framework.py       # Main QA framework module
├── requirements.txt         # Python dependencies
├── .gitlab-ci.yml          # CI/CD pipeline configuration
├── tests/
│   └── test_ev_qa.py       # Comprehensive test suite (14 tests)
└── README.md               # This file
```

## Architecture

### BatteryTelemetry
Represents real-time EV battery parameters with timestamp.

### EVQAFramework  
Core testing framework providing:
- `validate_telemetry()` - Validates against safety thresholds
- `detect_anomalies()` - ML-based anomaly detection
- `run_test_suite()` - Async test execution with results

## Safety Thresholds

- **Temperature**: < 60°C
- **Voltage**: 3.0V - 4.3V
- **SOC (State of Charge)**: 0% - 100%
- **Temperature Jump**: Alert on >5°C sudden change

## Use Cases

- **ChargePoint Infrastructure** - QA for intelligent charging stations
- **Battery Management Systems** - Validation of BMS firmware and algorithms
- **IoT Fleet Monitoring** - Telemetry analysis for EV fleets
- **Energy Optimization** - Testing of AI-driven charging strategies

## Technologies Used

- **Python 3.8+** - Core framework
- **pytest** - Testing framework
- **asyncio** - Async test execution
- **scikit-learn** - ML anomaly detection
- **python-can** - CAN protocol support
- **Docker** - Containerization
- **GitLab CI/CD** - Automated pipelines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Author

Created by Remontsuri - AI/QA Engineer

## Related Projects

- **ChargePoint** - EV charging infrastructure
- **Verwall** - Multi-domain QA for EV systems
- **Intangles** - EV telemetry platforms

## Roadmap

- [ ] CAN bus protocol full integration
- [ ] Real-time streaming telemetry
- [ ] Advanced ML models (LSTM for prediction)
- [ ] REST API for remote monitoring
- [ ] Web dashboard for visualization
- [ ] Kafka integration for big data pipelines

---

**Last Updated**: January 2026
**Status**: Active Development
