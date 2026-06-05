"""
Prometheus metrics for EV-QA-Framework.

Exposes battery telemetry readings and anomaly counters in Prometheus
text format so they can be scraped by a Prometheus server or consumed
directly by Grafana via the built-in Prometheus datasource.
"""

from prometheus_client import Counter, Gauge

# -- Battery state gauges (instantaneous readings) --

battery_temperature_celsius = Gauge(
    "battery_temperature_celsius",
    "Current battery temperature in Celsius",
)

battery_voltage_volts = Gauge(
    "battery_voltage_volts",
    "Current battery voltage in Volts",
)

battery_current_amps = Gauge(
    "battery_current_amps",
    "Current battery current in Amperes",
)

battery_soc_percent = Gauge(
    "battery_soc_percent",
    "Current state of charge (0-100 %)",
)

battery_soh_percent = Gauge(
    "battery_soh_percent",
    "Current state of health (0-100 %)",
)

# -- Anomaly counters --

battery_anomaly_total = Counter(
    "battery_anomaly_total",
    "Total number of detected battery anomalies",
    ["severity"],
)

battery_cell_imbalance_max = Gauge(
    "battery_cell_imbalance_max",
    "Maximum voltage imbalance among cells (V)",
)
