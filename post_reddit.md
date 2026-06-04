---
subreddit: r/batteries
title: "EV-QA-Framework — open-source Python framework for battery telemetry QA, anomaly detection, and SOH prediction"
---

Hey battery engineers,

I've been working on an open-source Python framework for Electric Vehicle battery quality assurance and wanted to share it with folks who might find it useful on the bench or in production.

**What it does:**
- Validates battery telemetry (voltage, current, temp, SOC, SOH) using Pydantic schemas
- Isolation Forest-based anomaly detection — finds irregular cell behavior, thermal precursors
- LSTM-based State of Health (SOH) degradation forecasting
- CAN bus emulation for HIL testing
- Real-time FastAPI dashboard for monitoring

**Tech stack:** Python 3.8+, scikit-learn, TensorFlow/Keras, FastAPI, pytest (85+ tests)

**Why I built it:** Most battery QA tooling is proprietary or scattered across MATLAB scripts. This is meant to be a solid, open foundation that teams can build on — whether you're validating BMS telemetry or running degradation models on cycler data.

https://github.com/remontsuri/EV-QA-Framework

Would love feedback, issues, or PRs — especially from people working on BMS, CAN bus telemetry, or battery degradation IRL.

---

*Also posted to r/electricvehicles*
