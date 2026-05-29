# 🔧 Configuration Guide - EV-QA-Framework

## Overview

The configuration module allows you to customize safety thresholds and ML analysis parameters without changing code. All settings are organized into structured classes and can be saved/loaded from JSON files.

## 📝 Main classes

### 1. `SafetyThresholds` - Safety thresholds

Defines threshold values for battery telemetry validation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_temperature` | float | 60.0 | Maximum safe temperature (°C) |
| `min_temperature` | float | -40.0 | Minimum safe temperature (°C) |
| `max_temperature_jump` | float | 5.0 | Maximum allowed temperature jump (°C) |
| `min_voltage` | float | 200.0 | Minimum safe voltage (V) |
| `max_voltage` | float | 900.0 | Maximum safe voltage (V) |
| `min_soc` | float | 10.0 | Minimum charge level for warning (%) |
| `critical_soh` | float | 70.0 | Critical battery health level (%) |
| `max_current` | float | 500.0 | Maximum safe current (A) |

### 2. `MLConfig` - ML configuration

Parameters for the ML anomaly detector (Isolation Forest).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contamination` | float | 0.1 | Expected proportion of anomalies (0.0 - 1.0) |
| `n_estimators` | int | 200 | Number of trees in the ensemble |
| `random_state` | int | 42 | Seed for reproducibility |
| `critical_score_threshold` | float | -0.8 | Threshold for CRITICAL severity |
| `warning_score_threshold` | float | -0.5 | Threshold for WARNING severity |

### 3. `FrameworkConfig` - Main configuration

Combines all framework settings.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `safety_thresholds` | SafetyThresholds | Safety thresholds |
| `ml_config` | MLConfig | ML configuration |
| `default_vin` | str | Default VIN for tests |

## 🚀 Usage

### Basic example

```python
from ev_qa_framework import EVQAFramework, FrameworkConfig

# Use default configuration
qa = EVQAFramework("My-QA")

# Or create a custom configuration
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0  # Stricter threshold
qa = EVQAFramework("My-QA", config=config)
```

### Creating a custom configuration

```python
from ev_qa_framework import FrameworkConfig, SafetyThresholds, MLConfig

# Strict thresholds for Tesla
tesla_thresholds = SafetyThresholds(
    max_temperature=55.0,
    min_voltage=250.0,
    max_voltage=450.0,
    max_temperature_jump=3.0
)

# ML with low contamination for high accuracy
ml_config = MLConfig(
    contamination=0.05,
    n_estimators=250,
    critical_score_threshold=-0.9
)

# Create configuration
config = FrameworkConfig(
    safety_thresholds=tesla_thresholds,
    ml_config=ml_config,
    default_vin="5YJ3E1EA8KF000001"
)

# Use it
qa = EVQAFramework("Tesla-QA", config=config)
```

### Saving and loading configuration

```python
from ev_qa_framework import FrameworkConfig

# Create configuration
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0

# Save to JSON
config.save_to_file("my_config.json")

# Load from JSON
loaded_config = FrameworkConfig.load_from_file("my_config.json")

# Use loaded configuration
qa = EVQAFramework("My-QA", config=loaded_config)
```

## 📁 Example JSON configuration

```json
{
  "safety_thresholds": {
    "max_temperature": 55.0,
    "min_temperature": -30.0,
    "max_temperature_jump": 3.0,
    "min_voltage": 250.0,
    "max_voltage": 450.0,
    "min_soc": 15.0,
    "critical_soh": 75.0,
    "max_current": 600.0
  },
  "ml_config": {
    "contamination": 0.05,
    "n_estimators": 250,
    "random_state": 42,
    "critical_score_threshold": -0.9,
    "warning_score_threshold": -0.6
  },
  "default_vin": "5YJ3E1EA8KF000001"
}
```

## 🎯 Pre-built configurations

The `config/` directory contains ready-to-use configurations:

- **`default_config.json`** - Default configuration for general use
- **`tesla_config.json`** - Strict configuration for Tesla EV

### Using pre-built configurations

```python
from ev_qa_framework import FrameworkConfig, EVQAFramework

# Load Tesla configuration
tesla_config = FrameworkConfig.load_from_file("config/tesla_config.json")
qa = EVQAFramework("Tesla-QA", config=tesla_config)
```

## 🧪 Testing

All configuration classes are covered by tests in `tests/test_config.py`:

```bash
pytest tests/test_config.py -v
```

Tests include:
- ✅ Class initialization
- ✅ Serialization/deserialization
- ✅ Save/load from files
- ✅ Integration with EVQAFramework
- ✅ Validation with custom thresholds

## 💡 Best practices

1. **Use configuration files** for different battery types (Tesla, Nissan, etc.)
2. **Tune contamination** based on expected anomaly percentage
3. **Adapt temperature thresholds** to climate conditions
4. **Log the configuration** at test start for reproducibility

## 🔄 Migration from hardcoded values

**Old code (v0.x):**
```python
qa = EVQAFramework("Test")
# Thresholds were hardcoded in code
```

**New code (v1.0+):**
```python
config = FrameworkConfig()
config.safety_thresholds.max_temperature = 55.0  # Configurable!
qa = EVQAFramework("Test", config=config)
```

## 📚 Additional resources

- See also: `ev_qa_framework/config.py` - full documentation in docstrings
- Usage examples: `examples/` directory
- Tests: `tests/test_config.py`
