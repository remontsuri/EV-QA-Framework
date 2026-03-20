# 🔍 Case Study: Detecting High-Temperature Spikes

**Scenario:** Monitoring a fleet of 1,000 EVs operating in high-temperature environments (e.g., Phoenix, Arizona).

## 📉 The Problem

Battery cells degrade rapidly when operated above 60°C. A "hidden" cell defect can cause localized heating that traditional average-based sensors might miss.

## 🧪 Testing Approach

Using the **EV-QA-Framework**, we simulated a telemetry stream where one module exhibited a 5.2°C jump within 10 seconds—while still remaining below the 60°C absolute threshold.

### 1. Rule-Based Detection
The framework's `detect_anomalies` function flagged the event immediately:
```python
# Threshold configured at 5.0°C jump
anomalies = qa.detect_anomalies(telemetry_data)
# Output: ["Резкий скачок температуры: 5.2°C (порог: 5.0°C)"]
```

### 2. ML Analysis
The **Isolation Forest** analyzer detected the event as an "outlier" in the multidimensional space of voltage and temperature, assigning it a `CRITICAL` severity score.

## 📊 Results & Impact

- **Detection Time**: <100ms from telemetry ingestion.
- **Outcome**: The vehicle was automatically flagged for a cooling system inspection before any permanent cell damage occurred.
- **Estimated Savings**: Preventing one module replacement saves approximately **$2,500 in warranty costs**.

---

[Back to Main README](../README.md)
