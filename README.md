# 🔋 EV-QA-Framework
**Open-source Python framework for Electric Vehicle battery quality assurance and anomaly detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-64%2B-brightgreen.svg)](./tests/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)]()

---

## 🚀 Why This Matters for EV Industry

Modern electric vehicles generate **millions of telemetry points daily** from battery management systems. This framework provides:

- ✅ **Automated quality assurance** for battery telemetry (voltage, current, temperature, SOC)
- ✅ **ML-powered anomaly detection** using Isolation Forest (scikit-learn)
- ✅ **64+ comprehensive tests** covering safety boundaries and edge cases
- ✅ **Pydantic data validation** ensuring data integrity
- ✅ **CI/CD ready** with Docker and GitLab CI

**Target Audience**: QA engineers at Tesla, Rivian, Lucid Motors, BYD, and automotive suppliers working on BMS (Battery Management Systems).

---

## 📊 Real-World Problem

**Battery failures cost the EV industry billions annually.** Early detection of anomalies in telemetry can:

- 🔥 Prevent thermal runaway events
- 📉 Reduce warranty claims
- ⚡ Extend battery lifespan
- 🛡️ Improve vehicle safety

This framework automates detection of:
- Temperature spikes (>5°C jumps)
- Voltage anomalies (out of 3.0-4.3V range)
- Invalid SOC readings
- ML-detected outliers in multidimensional space

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         EV Battery Telemetry Data           │
│    (CAN bus / OBD-II / Cloud API)          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       Pydantic Validation Layer             │
│  (VIN, Voltage, Current, Temp, SOC, SOH)   │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌─────────────────┐
│ Rule-based   │    │  ML Anomaly     │
│ Validation   │    │  Detection      │
│ (Thresholds) │    │ (IsolationForest)│
└──────┬───────┘    └────────┬────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
       ┌──────────────────┐
       │  Test Results    │
       │  + Anomaly Report│
       └──────────────────┘
```

---

## 🔧 Quick Start

### Installation

```bash
git clone https://github.com/remontsuri/EV-QA-Framework.git
cd EV-QA-Framework
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests

```bash
# All tests (64+)
pytest -v

# With coverage report
pytest --cov=. --cov-report=html
```

### Usage Example

```python
from ev_qa_models import validate_telemetry
from ev_qa_analysis import AnomalyDetector
import pandas as pd

# 1. Validate incoming telemetry
data = {
    "vin": "1HGBH41JXMN109186",
    "voltage": 396.5,
    "current": 125.3,
    "temperature": 35.2,
    "soc": 78.5,
    "soh": 96.2
}
telemetry = validate_telemetry(data)  # Pydantic auto-validation

# 2. ML Anomaly Detection
detector = AnomalyDetector(contamination=0.01, n_estimators=200)
detector.train(normal_telemetry_df)  # Train on historical "good" data

# Real-time detection
predictions, scores = detector.detect(new_telemetry_df)
# predictions: 1 = normal, -1 = anomaly
```

---

## 📈 Test Coverage

| Category | Tests | Coverage |
|----------|-------|----------|
| Boundary Tests (Voltage/Temp/SOC) | 23+ | Temperature: >60°C, Voltage: 3.0-4.3V, SOC: 0-100% |
| Anomaly Detection | 15+ | Temperature jumps, multiple anomalies |
| ML Analyzer | 12+ | Isolation Forest, severity classification |
| Pydantic Models | 14+ | VIN validation, type checking, edge cases |
| **TOTAL** | **64+** | **~85% code coverage** |

---

## 🧪 Technologies

- **Python 3.8+**
- **pytest** (testing framework)
- **Pydantic** (data validation)
- **scikit-learn** (Isolation Forest ML)
- **pandas/numpy** (data processing)
- **Docker** (containerization)
- **GitLab CI/CD** (automation)

---

## 🌟 Key Features

### 1. Pydantic Data Models
```python
class BatteryTelemetryModel(BaseModel):
    vin: str = Field(min_length=17, max_length=17)
    voltage: float = Field(ge=0.0, le=1000.0)
    temperature: float = Field(ge=-50.0, le=150.0)
    soc: float = Field(ge=0.0, le=100.0)
    soh: float = Field(ge=0.0, le=100.0)
    # ... with automatic validation
```

### 2. ML Anomaly Detection
- **Isolation Forest** with 200 estimators
- Separates `train()` and `detect()` for production use
- Severity classification: CRITICAL / WARNING / INFO
- Detailed Russian documentation explaining algorithm

### 3. Comprehensive Testing
- Parametrized tests for boundary values
- Negative test cases (invalid types, extreme values)
- Async test suite support
- Docker-based CI/CD pipeline

---

## 🐳 Docker Support

```bash
# Build
docker build -t ev-qa-framework .

# Run tests in container
docker run --rm ev-qa-framework pytest -v

# Docker Compose
docker-compose up --build
```

---

## 🤝 Contributing

We welcome contributions from the EV and QA community!

**Areas for collaboration:**
- Integration with real CAN bus data (python-can)
- MQTT/OBD-II protocol support
- Integration with Tesla API / other EV APIs
- Enhanced ML models (LSTM for time-series)
- Web dashboard for real-time monitoring

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📝 Industry Contact

**For EV manufacturers, BMS suppliers, or automotive QA teams interested in collaboration:**

This is an **open-source educational project**, but we're open to:
- Enterprise consulting on battery QA systems
- Custom ML models for specific battery chemistries
- Integration with proprietary BMS systems
- Training workshops for QA teams

📧 Contact: [Your Email]  
💼 LinkedIn: [Your Profile]  
🐙 GitHub: [@remontsuri](https://github.com/remontsuri)

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

**Free for commercial use**, including Tesla, Rivian, Lucid, BYD, and other EV manufacturers.

---

## 🙏 Acknowledgments

Inspired by real-world challenges in EV battery safety and quality assurance. Built with ❤️ for the sustainable transportation revolution.

**If this helps your EV project, consider:**
- ⭐ Starring this repo
- 🍴 Forking and contributing
- 📢 Sharing with your QA/automotive network
- 💼 Hiring the author for EV QA roles 😊

---

## 🔗 Related Projects

- [python-can](https://github.com/hardbyte/python-can) - CAN bus library
- [Tesla API](https://github.com/timdorr/tesla-api) - Unofficial Tesla API
- [Battery-Optimization](https://github.com/rdbraatz/Battery-Optimization) - Battery modeling

---

**Built for the future of electric mobility** 🌍⚡🚗
