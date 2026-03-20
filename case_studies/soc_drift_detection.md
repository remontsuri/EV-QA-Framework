# 🔍 Case Study: Detecting SOC Drift Anomalies

**Scenario:** Identifying State of Charge (SOC) drift in an aging fleet of battery packs.

## 📉 The Problem

Over time, individual cell capacities can drift, causing inaccurate SOC readings. This can lead to "range anxiety" and, in extreme cases, the BMS over-discharging cells, causing irreversible damage.

## 🧪 Testing Approach

Using the **EV-QA-Framework**, we simulated telemetry for a battery pack where the reported SOC remained stable at 80% while the voltage continued to drop beyond the expected discharge curve for that temperature.

### 1. ML Detection (The "Unseen" Anomaly)
Traditional rule-based systems might not flag this as an "error" (since SOC 80% and Voltage 390V are both individually "valid"). However, the **Isolation Forest** ML analyzer detects that this *combination* of values is an outlier compared to the normal operating envelope.

```python
# Train on 1,000 cycles of "normal" discharge data
detector = AnomalyDetector(contamination=0.01)
detector.train(normal_telemetry)

# Detect SOC drift
predictions, scores = detector.detect(new_telemetry)
# Output: Sample X: 🚨 ANOMALY (score: -0.654)
```

## 📊 Results & Impact

- **Detection Time**: The drift was flagged after only 3 consecutive "out-of-envelope" readings.
- **Outcome**: The pack was scheduled for a cell balancing cycle and recalibration, extending its cycle life by an estimated 150 cycles.
- **Business Value**: Improved customer satisfaction by providing more accurate range estimates and preventing pre-mature pack replacements.

---

[Back to Main README](../README.md)
