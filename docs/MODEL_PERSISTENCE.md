# 💾 ML Model Persistence Guide

## Overview

The persistence module allows saving trained ML models and using them in production without retraining. This is critical for the **train-once, deploy-many** pattern.

## 🎯 Why is this needed?

**Problem:** Isolation Forest retrains every time `analyze_telemetry()` is called, which:
- ❌ Is slow for production (retraining takes time)
- ❌ Is unstable (different results each run without random_state)
- ❌ Makes model versioning impossible

**Solution:** Save the trained model:
- ✅ Train once offline
- ✅ Save model + scaler + metadata
- ✅ Load in production for fast inference
- ✅ Version models (A/B testing)

---

## 📝 API Reference

### `save_model(filepath, metadata=None)`

Saves the trained model to a file.

**Parameters:**
- `filepath` (str): Path for saving (automatically adds `.joblib`)
- `metadata` (dict, optional): Model metadata

**What is saved:**
- Trained `IsolationForest` model
- Trained `StandardScaler`
- Parameters: `contamination`, `critical_threshold`, `warning_threshold`
- Save timestamp
- Custom metadata

**Example:**
```python
analyzer = EVBatteryAnalyzer(contamination=0.1)
analyzer.analyze_telemetry(train_data)

analyzer.save_model(
    'models/battery_v1',
    metadata={'version': '1.0', 'dataset': 'Tesla_2024'}
)
```

**Raises:**
- `ValueError` - if model is not trained

---

### `load_model(filepath)` (classmethod)

Loads a saved model from a file.

**Parameters:**
- `filepath` (str): Path to the model file

**Returns:**
- `EVBatteryAnalyzer` - new instance with the loaded model

**Example:**
```python
analyzer = EVBatteryAnalyzer.load_model('models/battery_v1.joblib')
results = analyzer.analyze_telemetry(new_data)
```

**Raises:**
- `FileNotFoundError` - if file is not found
- `ValueError` - if file is corrupted

---

### `get_model_info()`

Gets information about the current model.

**Returns:**
- `dict` with model parameters

**Example:**
```python
info = analyzer.get_model_info()
print(info)
# {
#     'contamination': 0.1,
#     'n_estimators': 200,
#     'critical_threshold': -0.8,
#     'warning_threshold': -0.5,
#     'is_fitted': True
# }
```

---

## 🚀 Production Workflow

### 1. Offline Training

```python
from ev_qa_framework import EVBatteryAnalyzer
import pandas as pd

# Load historical "normal" data
historical_data = pd.read_csv('historical_telemetry.csv')

# Train the model
analyzer = EVBatteryAnalyzer(
    contamination=0.05,
    n_estimators=300,
    critical_threshold=-0.9
)

analyzer.analyze_telemetry(historical_data)

# Save with metadata
analyzer.save_model(
    'models/production/baseline_v2.0',
    metadata={
        'version': '2.0',
        'dataset_size': len(historical_data),
        'trained_on': '2024-01-28',
        'performance_metrics': {...}
    }
)
```

### 2. Production Deployment

```python
# In production service
from ev_qa_framework import EVBatteryAnalyzer

# Load once at service startup
model = EVBatteryAnalyzer.load_model('models/production/baseline_v2.0')

# Use for real-time inference
def process_telemetry(incoming_batch):
    results = model.analyze_telemetry(incoming_batch)
    
    if results['severity'] == 'CRITICAL':
        send_alert(results)
    
    return results
```

### 3. Monitoring & Versioning

```python
# A/B testing of models
model_a = EVBatteryAnalyzer.load_model('models/model_a_conservative')
model_b = EVBatteryAnalyzer.load_model('models/model_b_tolerant')

results_a = model_a.analyze_telemetry(test_data)
results_b = model_b.analyze_telemetry(test_data)

# Compare metrics
compare_models(results_a, results_b)
```

---

## 📁 Saved model structure

The `.joblib` file contains a dictionary:

```python
{
    'model': IsolationForest object,      # Trained model
    'scaler': StandardScaler object,      # Trained scaler
    'contamination': 0.1,                 # Model parameters
    'critical_threshold': -0.8,
    'warning_threshold': -0.5,
    'save_timestamp': '2024-01-28T12:00:00',
    'metadata': {                         # Custom metadata
        'version': '1.0',
        'dataset': 'Tesla_2024',
        ...
    }
}
```

---

## 🔄 Model versioning

### Recommended naming scheme

```
models/
├── production/
│   ├── baseline_v1.0.joblib       # Production version
│   ├── baseline_v1.1.joblib       # Updated version
│   └── champion_v2.0.joblib       # New best model
├── experiments/
│   ├── exp_conservative.joblib    # Experiments
│   ├── exp_tolerant.joblib
│   └── exp_highres.joblib
└── archive/
    └── deprecated_v0.9.joblib     # Old versions
```

### Semantic Versioning for models

```
[MAJOR].[MINOR].[PATCH]

MAJOR: Algorithm change or breaking changes
MINOR: Parameter changes (contamination, thresholds)
PATCH: Retraining on new data with the same parameters
```

**Examples:**
- `v1.0.0` → `v1.0.1` - retraining on an updated dataset
- `v1.0.1` → `v1.1.0` - changing contamination from 0.1 to 0.05
- `v1.1.0` → `v2.0.0` - switching from IsolationForest to another algorithm

---

## ⚡ Performance Tips

### 1. Use compression

```python
# Default compress=3 (good balance)
analyzer.save_model('model.joblib')  # ~500 KB

# No compression (faster but larger)
import joblib
joblib.dump(model_data, 'model.joblib', compress=0)  # ~2 MB

# Maximum compression (slower but smaller)
joblib.dump(model_data, 'model.joblib', compress=9)  # ~300 KB
```

### 2. Lazy Loading in production

```python
# Bad: load on every request
def handle_request(data):
    model = EVBatteryAnalyzer.load_model('model.joblib')  # Slow!
    return model.analyze_telemetry(data)

# Good: load once
class ModelService:
    def __init__(self):
        self.model = EVBatteryAnalyzer.load_model('model.joblib')
    
    def predict(self, data):
        return self.model.analyze_telemetry(data)
```

---

## 🧪 Testing

Persistence tests are in `tests/test_model_persistence.py`:

```bash
pytest tests/test_model_persistence.py -v
```

**Covered scenarios:**
- ✅ Basic save/load
- ✅ Save/load with metadata
- ✅ Error when saving an untrained model
- ✅ Inference after loading
- ✅ Loading a non-existent file
- ✅ Automatic directory creation
- ✅ Versioning

---

## 🐛 Troubleshooting

### Error: "Model not trained"

```python
analyzer = EVBatteryAnalyzer()
analyzer.save_model('model.joblib')  # ValueError!

# Solution: train first
analyzer.analyze_telemetry(data)
analyzer.save_model('model.joblib')  # ✅
```

### Error: "Model file not found"

```python
model = EVBatteryAnalyzer.load_model('nonexistent.joblib')  # FileNotFoundError!

# Solution: check the path
import os
print(os.path.exists('model.joblib'))
```

### scikit-learn version incompatibility

If the model was saved on scikit-learn 1.2.0 and loaded on 1.3.0:

```python
# Solution: use a virtual environment with fixed versions
# requirements.txt:
scikit-learn==1.2.0
joblib==1.3.0
```

---

## 📚 Best Practices

1. **Always save metadata**
   ```python
   metadata = {
       'version': '1.0',
       'trained_on': str(datetime.now()),
       'dataset_size': len(data),
       'contamination': 0.1
   }
   analyzer.save_model('model', metadata=metadata)
   ```

2. **Use versioning**
   - Save multiple model versions
   - Maintain a changelog of changes

3. **Test before deployment**
   - Save the model
   - Load in a test environment
   - Validate on validation data

4. **Document experiments**
   - What data was used
   - What parameters were tuned
   - What metrics were obtained

---

## 🔗 See also

- [Configuration Guide](../config/README.md) - ML parameter configuration
- [examples/model_persistence_example.py](../examples/model_persistence_example.py) - usage examples
- [tests/test_model_persistence.py](../tests/test_model_persistence.py) - tests
