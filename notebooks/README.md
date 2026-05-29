# 📊 Jupyter Notebook Demo

Interactive demo of EV-QA-Framework with ML anomaly detection visualization.

## 🚀 How to run

### Locally:
```bash
cd notebooks/
jupyter notebook anomaly_detection_demo.ipynb
```

### Google Colab:
1. Upload `anomaly_detection_demo.ipynb` to Google Colab
2. Uncomment the dependency installation line:
   ```python
   !pip install pydantic scikit-learn pandas numpy matplotlib seaborn
   ```
3. Run all cells

## 📈 What the demo shows:

1. **Pydantic Validation** - strict telemetry validation
2. **Data Generation** - 1000 points + 50 anomalies
3. **ML Detection** - Isolation Forest (200 estimators)
4. **Visualizations**:
   - Scatter plot: Voltage vs Temperature
   - Histogram: Anomaly Scores distribution
   - Time Series: Real-time anomaly detection
   - Pie Chart: Severity classification (CRITICAL/WARNING/INFO)
5. **Summary Report** - detailed statistics

## 🎯 Results:

- ✅ **~95% accuracy** in anomaly detection
- ✅ **Severity classification** for prioritization
- ✅ **Beautiful charts** for presentations
- ✅ **Ready to use** in Google Colab

## 📸 Example charts:

Notebook generates:
- 🟢 Green dots = normal operation
- 🔴 Red crosses = detected anomalies
- 🟠 Dashed lines = safety thresholds

---

**Perfect for:**
- Project demos in interviews
- Presentations for QA teams
- Learning ML anomaly detection
- Publishing on LinkedIn/Medium
