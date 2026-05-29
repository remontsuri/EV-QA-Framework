---
title: EV-QA-Framework
description: ML-Powered QA Framework for Electric Vehicle Battery Systems
---

# ⚡ EV-QA-Framework

**ML-Powered Quality Assurance Framework for Electric Vehicle Battery Systems**

🔍 Detect anomalies · 📈 Predict SOH · 🚗 Emulate CAN · 📊 Real-time Dashboard

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m ev_qa_framework.cli analyze examples/tesla_model_s_defective.csv
```

## 📊 Dashboard

Start the real-time dashboard:

```bash
python -m ev_qa_framework.cli dashboard
```

Then open **[http://localhost:8000](http://localhost:8000)**.

## 🐳 Docker

```bash
docker compose -f docker-compose.prod.yml up -d
# Open http://localhost:8080
```

## 📚 Documentation

- [Model Persistence Guide](MODEL_PERSISTENCE.md) — saving/loading trained models
- [Configuration Profiles](https://github.com/remontsuri/EV-QA-Framework/tree/main/config) — Tesla, default, custom
- [Examples](https://github.com/remontsuri/EV-QA-Framework/tree/main/examples) — usage examples
- [API Reference](https://github.com/remontsuri/EV-QA-Framework/tree/main/api) — REST API endpoints

## 📈 Features

| Feature | Stack |
|---------|-------|
| Anomaly Detection | Isolation Forest (scikit-learn) |
| SOH Prediction | LSTM (TensorFlow) |
| CAN Bus Emulation | python-can, J1939/CAN 2.0B |
| Dashboard | FastAPI + Chart.js + WebSocket |
| Validation | Pydantic v2 |
| CLI | Python argparse |
| Deployment | Docker + nginx |

## 🤝 Contributing

PRs, issues, and feature requests are welcome! See [CONTRIBUTING.md](https://github.com/remontsuri/EV-QA-Framework/blob/main/CONTRIBUTING.md).

---

*Built with ❤️ for the EV community* — [remontsuri](https://github.com/remontsuri)
