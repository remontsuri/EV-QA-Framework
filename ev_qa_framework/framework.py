from __future__ import annotations

"""
EV-QA-Framework: Mini QA Framework for EV & IoT Testing

Author: Remontsuri
License: MIT

AI-powered battery management system testing framework with pytest,
CAN protocol support, telemetry monitoring, and ML-based anomaly detection.
"""

import asyncio
import json
import logging
import signal
import atexit
from typing import Any

import pandas as pd

from .analysis import EVBatteryAnalyzer
from .config import FrameworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _ShutdownHandler:
    _cleanups = []

    @classmethod
    def register(cls, fn):
        cls._cleanups.append(fn)

    @classmethod
    def shutdown(cls, signum=None, frame=None):
        import logging
        _log = logging.getLogger(__name__)
        for fn in cls._cleanups:
            try:
                fn()
            except Exception as e:
                _log.warning("Cleanup handler failed: %s", e)
        if signum is not None:
            raise SystemExit(0)


def _setup_signal_handlers():
    signal.signal(signal.SIGTERM, _ShutdownHandler.shutdown)
    signal.signal(signal.SIGINT, _ShutdownHandler.shutdown)


_setup_signal_handlers()



from .models import BatteryTelemetryModel


class EVQAFramework:
    """Main QA Framework for EV & IoT testing"""

    # Default VIN for testing legacy data without VINs (deprecated, use config.default_vin)
    DEFAULT_TEST_VIN = "TESTVEHCLE0123456"

    def __init__(self, name: str = "EV-QA-Tester", config: FrameworkConfig | None = None):
        """
        Initialize QA Framework.

        Args:
            name: Framework instance name
            config: Custom configuration (if None, default is used)
        """
        self.name = name
        self.telemetry_data: list[BatteryTelemetryModel] = []
        # generic results dictionary with mixed values
        self.test_results: dict[str, Any] = {}

        # Load configuration
        self.config = config if config is not None else FrameworkConfig()
        # Verify that default_vin passes Pydantic validation.
        # Use simple dummy telemetry to leverage model checks.
        try:
            BatteryTelemetryModel(
                vin=self.config.default_vin,
                voltage=0.0,
                current=0.0,
                temperature=0.0,
                soc=0.0,
                soh=0.0,
            )
        except Exception as e:
            logger.warning(
                f"default_vin '{self.config.default_vin}' is invalid ({e}), "
                f"replacing with DEFAULT_TEST_VIN ({self.DEFAULT_TEST_VIN})"
            )
            self.config.default_vin = self.DEFAULT_TEST_VIN

        # Initialize ML analyzer with config parameters
        self.ml_analyzer = EVBatteryAnalyzer(
            contamination=self.config.ml_config.contamination,
            n_estimators=self.config.ml_config.n_estimators,
            random_state=self.config.ml_config.random_state,
        )
        logger.info(
            f"Initialized {self.name} with ML analyzer (contamination={self.config.ml_config.contamination})"
        )


    def health_check(self) -> dict:
        """Return health status for HTTP /health endpoint."""
        status = {
            "status": "healthy",
            "ml_model_trained": hasattr(self.ml_analyzer.model, "estimators_"),
            "chemistry": self.config.chemistry,
            "thresholds": {
                "max_temperature": self.config.safety_thresholds.max_temperature,
                "min_voltage": self.config.safety_thresholds.min_voltage,
                "max_voltage": self.config.safety_thresholds.max_voltage,
            },
        }
        return status
    def validate_telemetry(self, telemetry: BatteryTelemetryModel) -> tuple[bool, list[str]]:
        """
        Validate battery telemetry against safety thresholds.

        Returns:
            Tuple of (is_valid, warnings) where warnings is a list of
            human-readable warning messages.
        """
        thresholds = self.config.safety_thresholds
        warnings: list[str] = []

        # Temperature check
        if telemetry.temperature > thresholds.max_temperature:
            msg = (
                f"Temperature warning: {telemetry.temperature}°C "
                f"(threshold: {thresholds.max_temperature}°C)"
            )
            logger.warning(msg)
            warnings.append(msg)
            return False, warnings

        if telemetry.temperature < thresholds.min_temperature:
            msg = (
                f"Temperature warning: {telemetry.temperature}°C "
                f"(minimum: {thresholds.min_temperature}°C)"
            )
            logger.warning(msg)
            warnings.append(msg)
            return False, warnings

        # Voltage check
        if telemetry.voltage < thresholds.min_voltage or telemetry.voltage > thresholds.max_voltage:
            msg = (
                f"Voltage warning: {telemetry.voltage}V "
                f"(range: {thresholds.min_voltage}-{thresholds.max_voltage}V)"
            )
            logger.warning(msg)
            warnings.append(msg)
            return False, warnings

        # Additional checks (non-critical warnings)
        if telemetry.soc < thresholds.min_soc:
            msg = f"Low charge level: {telemetry.soc}%"
            logger.warning(msg)
            warnings.append(msg)

        if telemetry.soh < thresholds.critical_soh:
            msg = f"Critical battery health: {telemetry.soh}%"
            logger.warning(msg)
            warnings.append(msg)

        return True, warnings

    def detect_anomalies(self, telemetry_list: list[BatteryTelemetryModel]) -> list[str]:
        """
        Rule-based anomaly detection in telemetry.

        Detects two types of events:
        1. Overheating above max_temperature (threshold from safety_thresholds)
        2. Sharp temperature jump between adjacent points exceeding max_temperature_jump

        Returns a list of anomaly messages (strings).
        """
        anomalies: list[str] = []
        if not telemetry_list:
            return anomalies

        thresholds = self.config.safety_thresholds
        # 1. Check temperature values at each step
        for t in telemetry_list:
            if t.temperature > thresholds.max_temperature:
                anomalies.append(
                    f"Temperature: {t.temperature}°C (threshold: {thresholds.max_temperature}°C)"
                )

        # 2. Temperature jumps
        if len(telemetry_list) >= 2:
            temp_jump_threshold = thresholds.max_temperature_jump
            for i in range(1, len(telemetry_list)):
                temp_change = abs(telemetry_list[i].temperature - telemetry_list[i - 1].temperature)
                if temp_change > temp_jump_threshold:
                    anomalies.append(
                        f"Sharp temperature jump: {temp_change}°C "
                        f"(threshold: {temp_jump_threshold}°C)"
                    )
        return anomalies

    def run_test_suite(self, telemetry_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full QA test suite with ML analysis"""
        results: dict[str, Any] = {
            "total_tests": len(telemetry_data),
            "passed": 0,
            "failed": 0,
            "anomalies": [],
            "ml_analysis": None,
            "critical_issues": [],  # collect validation errors and critical findings
        }

        telemetries: list[BatteryTelemetryModel] = []
        status: list[bool] = []  # True=passed, False=failed
        for data in telemetry_data:
            # Compatibility layer: Inject VIN if missing (copy to avoid mutating input)
            _data = dict(data)  # shallow copy to avoid mutating caller's data
            if "vin" not in _data:
                _data["vin"] = self.config.default_vin

            try:
                telemetry = BatteryTelemetryModel(**_data)
                telemetries.append(telemetry)

                # initial validation
                is_valid, warnings = self.validate_telemetry(telemetry)
                if is_valid:
                    status.append(True)
                else:
                    status.append(False)
                    results["critical_issues"].extend(warnings)
            except Exception as e:
                msg = f"Validation failed - {e}"
                logger.error(msg)
                results["critical_issues"].append(msg)
                status.append(False)
                # also append telemetry so anomalies logic can inspect
                try:
                    telemetries.append(BatteryTelemetryModel(**data))
                except Exception as e:
                    logger.warning("Failed to parse telemetry for anomaly inspection: %s", e)
                    pass
                continue

        # compute initial counts
        results["passed"] = sum(1 for s in status if s)
        results["failed"] = len(status) - results["passed"]

        # Rule-based anomaly detection
        anomalies = self.detect_anomalies(telemetries)
        results["anomalies"] = anomalies
        # Optionally treat jumps as failures (configurable)
        if self.config.fail_on_anomaly:
            jump_threshold = self.config.safety_thresholds.max_temperature_jump
            jump_indices: set[int] = set()
            for i in range(1, len(telemetries)):
                # only consider jump if previous telemetry was not already failed
                if i - 1 < len(status) and not status[i - 1]:
                    continue
                if (
                    abs(telemetries[i].temperature - telemetries[i - 1].temperature)
                    > jump_threshold
                ):
                    jump_indices.add(i)
            # adjust status counts
            for idx in jump_indices:
                if idx < len(status) and status[idx]:
                    status[idx] = False
                    results["passed"] -= 1
                    results["failed"] += 1

        # ML-based analysis
        if telemetries:
            # Convert Pydantic models to dicts for DataFrame
            df = pd.DataFrame([t.model_dump() for t in telemetries])
            df.rename(columns={"temperature": "temp"}, inplace=True)
            ml_results = self.ml_analyzer.analyze_telemetry(df)
            results["ml_analysis"] = ml_results

        self.test_results = results
        logger.info(f"Test Results: {results}")
        return results


# Example usage
if __name__ == "__main__":
    # Create QA framework instance
    qa = EVQAFramework("ChargePoint-QA")

    # Sample telemetry data
    test_data: list[dict[str, Any]] = [
        {"voltage": 396.5, "current": 50, "temperature": 35, "soc": 80, "soh": 98},
        {"voltage": 398.0, "current": 45, "temperature": 36, "soc": 85, "soh": 98},
        {"voltage": 395.0, "current": 60, "temperature": 45, "soc": 75, "soh": 97},
    ]

    # Run tests
    result = asyncio.run(qa.run_test_suite(test_data))
    print(json.dumps(result, indent=2))
