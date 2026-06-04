---
title: "Show HN: EV-QA-Framework — Open-source battery QA with ML anomaly detection and SOH prediction"
---

I built an open-source Python framework for Electric Vehicle battery quality assurance. It's designed for battery engineers, BMS algorithm teams, and automotive QA.

**Core features:**
- Pydantic-based validation of battery telemetry (voltage, current, temp, SOC, SOH)
- Isolation Forest anomaly detection tuned for battery data
- LSTM model for State of Health degradation forecasting
- CAN bus emulation (python-can) for HIL testing
- Real-time FastAPI dashboard
- 85+ automated tests

**Stack:** Python 3.8+, scikit-learn, TensorFlow/Keras, FastAPI, pytest

The goal is to provide a solid open-source alternative to the proprietary/ MATLAB tooling that dominates battery QA right now. Everything from validating BMS telemetry to running degradation models on historical cycler data.

MIT licensed. Contributions from battery engineers especially welcome.

https://github.com/remontsuri/EV-QA-Framework

Would appreciate any feedback, issues, or PRs.
