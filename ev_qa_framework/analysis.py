from __future__ import annotations

"""EV QA Analysis: ML-based battery telemetry and quality assurance.

Machine learning module for anomaly detection in battery telemetry.
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Any

import joblib  # type: ignore  # no stub available
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .physics_features import PhysicsFeatureExtractor
from .utils import normalize_columns

logger = logging.getLogger(__name__)

# Cell imbalance thresholds (volts)
CELL_IMBALANCE_WARNING: float = 0.05
CELL_IMBALANCE_CRITICAL: float = 0.1



class EVBatteryAnalyzer:
    """
    ML-based EV battery telemetry analyzer using the Isolation Forest algorithm.

    Isolation Forest is an anomaly detection algorithm that isolates outliers
    by randomly selecting a feature and then randomly selecting a split value
    between the maximum and minimum of the selected feature. Anomalies are
    isolated faster than normal data points.

    Attributes:
        model: IsolationForest model from scikit-learn
        scaler: StandardScaler for data normalization
        anomalies: DataFrame with detected anomalies
        contamination: Expected proportion of anomalies in the dataset (default 0.1 = 10%)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 200,
        random_state: int = 42,
        critical_threshold: float = -0.8,
        warning_threshold: float = -0.5,
    ):
        """
        Initialize the telemetry analyzer.

        Args:
            contamination: Expected proportion of anomalies in the data (0.0 - 1.0).
                          For example, 0.1 means ~10% of data may be anomalous.
            n_estimators: Number of trees in the ensemble (more = more accurate, but slower).
                         Recommended 100-200 for a balance of accuracy and speed.
            random_state: Seed for reproducible results.
            critical_threshold: Threshold for CRITICAL severity (default -0.8)
            warning_threshold: Threshold for WARNING severity (default -0.5)

        Note:
            - contamination affects sensitivity: lower value = fewer false positives
            - n_estimators recommended 100+ for stable results
        """
        # Create the Isolation Forest model with configured parameters
        self.model = IsolationForest(
            contamination=contamination,  # Expected proportion of anomalies
            n_estimators=n_estimators,  # Number of trees (more = more stable)
            max_samples="auto",  # Auto-select subsample size
            random_state=random_state,  # For reproducibility
            n_jobs=-1,  # Use all CPU cores
        )

        # StandardScaler normalizes data: (x - mean) / std
        # This matters because IsolationForest is sensitive to feature scale
        self.scaler = StandardScaler()

        # Storage for detected anomalies (populated after analyze_telemetry)
        # anomalies stored as DataFrame; start empty
        self.anomalies: pd.DataFrame = pd.DataFrame()

        # Store parameters for external access
        self.contamination = contamination
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.physics_extractor = PhysicsFeatureExtractor()

    def analyze_telemetry(self, df_telemetry: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze battery telemetry for anomalies.

        Algorithm:
        1. Normalize data via StandardScaler (bring to a single scale)
        2. Train IsolationForest on normalized data
        3. Predict anomalies (-1 = anomaly, 1 = normal)
        4. Compute anomaly scores (lower = more anomalous)
        5. Assess severity based on minimum score

        Args:
            df_telemetry: DataFrame with columns ['voltage', 'current', 'temp', 'soc'].
                         Each row represents a single point in time.

        Returns:
            Dictionary with analysis results:
                - total_samples: Total number of data points
                - anomalies_detected: Number of detected anomalies
                - anomaly_percentage: Percentage of anomalies
                - severity: Severity level ('CRITICAL', 'WARNING', 'INFO')

        Example:
            >>> df = pd.DataFrame({
            ...     'voltage': [48, 48, 200],  # 200 — anomaly
            ...     'current': [100, 100, 100],
            ...     'temp': [35, 35, 35],
            ...     'soc': [85, 85, 85]
            ... })
            >>> analyzer = EVBatteryAnalyzer()
            >>> results = analyzer.analyze_telemetry(df)
            >>> print(results['anomalies_detected'])
            1
        """
        # Step 1: Data preparation — normalize column names
        from .utils import normalize_columns, require_columns

        df: pd.DataFrame = normalize_columns(df_telemetry)
        require_columns(df, ["voltage", "current", "temp"])

        # Step 1: Select only numeric features for analysis
        # SOC is not used for detection as it is a dependent variable
        features: list[str] = ["voltage", "current", "temp"]
        X: pd.DataFrame = df[features]

        # Step 2: Normalize data (mean=0, std=1)
        # Use existing scaler if already fitted, otherwise fit_transform
        if hasattr(self.scaler, "mean_"):
            X_scaled = self.scaler.transform(X)  # type: ignore
        else:
            X_scaled = self.scaler.fit_transform(X)  # type: ignore

        # Step 3: Train model and predict anomalies
        if hasattr(self.model, "estimators_"):
            predictions = self.model.predict(X_scaled)  # type: ignore[arg-type]
        else:
            self.model.fit(X_scaled)
            predictions = self.model.predict(X_scaled)  # type: ignore[arg-type]

        # Step 4: Compute anomaly scores (lower = more anomalous point)
        # score_samples works for an already-fitted model
        anomaly_scores: np.ndarray = self.model.score_samples(X_scaled)  # type: ignore

        # Step 5: Filter anomalies
        # In addition to the standard prediction (-1), also account for cases
        # where score_samples dropped below warning_threshold. This helps avoid
        # missing rare outliers on small samples (e.g. when the model was trained
        # on identical points).
        mask: np.ndarray = (predictions == -1) | (anomaly_scores < self.warning_threshold)

        # Apply mask to the same DataFrame used to build X to ensure index alignment
        self.anomalies = df_telemetry.iloc[mask].copy()  # type: ignore

        # Add anomaly scores to results for further analysis
        if not self.anomalies.empty:
            self.anomalies["anomaly_score"] = anomaly_scores[mask]

        # Step 6: Build analysis result
        total = len(df_telemetry)
        count = len(self.anomalies)
        # Step 7: Detect gradient attacks
        gradient = self._detect_gradient_attack(df)
        if gradient["gradient_detected"]:
            # Boost severity if gradient detected
            current_severity = self._assess_severity(anomaly_scores)
            if current_severity == "INFO":
                severity = "WARNING"
            else:
                severity = "CRITICAL"
        else:
            severity = self._assess_severity(anomaly_scores)

        return {
            "total_samples": total,
            "anomalies_detected": count,
            "anomaly_percentage": (count / total) * 100 if total else 0.0,
            "severity": severity,
            "gradient_attack": gradient,
        }


    def _detect_gradient_attack(self, df: pd.DataFrame) -> dict:
        """Detect slow monotonic drifts that evade IsolationForest."""
        result = {"gradient_detected": False, "details": {}}
        for col in ["voltage", "current", "temp"]:
            if col not in df.columns:
                continue
            values = df[col].values
            if len(values) < 10:
                continue
            # Compute monotonicity: how many consecutive pairs increase/decrease
            diffs = np.diff(values)
            if len(diffs) == 0:
                continue
            # Check for sustained monotonic trend (>80% same direction)
            pos_ratio = np.sum(diffs > 0) / len(diffs)
            neg_ratio = np.sum(diffs < 0) / len(diffs)
            monotonicity = max(pos_ratio, neg_ratio)
            if monotonicity > 0.80:
                total_drift = values[-1] - values[0]
                drift_per_sample = total_drift / len(values)
                result["gradient_detected"] = True
                result["details"][col] = {
                    "monotonicity": round(monotonicity, 3),
                    "total_drift": round(total_drift, 4),
                    "drift_per_sample": round(drift_per_sample, 6),
                    "direction": "increasing" if pos_ratio > neg_ratio else "decreasing"
                }
        return result
    def _assess_severity(self, scores: np.ndarray) -> str:
        """
        Assess the severity level of detected anomalies.

        Assessment logic:
        - CRITICAL: Extreme outliers present (score < critical_threshold)
                    Immediate attention required — possible critical failure
        - WARNING: Moderate anomalies (score < warning_threshold)
                  Inspection required — possible system degradation
        - INFO: Weak anomalies or none (score >= warning_threshold)
               System is normal, anomalies are insignificant

        Args:
            scores: Array of anomaly scores from IsolationForest

        Returns:
            Severity level string: 'CRITICAL', 'WARNING', or 'INFO'

        Note:
            Thresholds are configurable via constructor parameters and can be tuned
            for a specific system based on historical data.
        """
        min_score = np.min(scores)

        if min_score < self.critical_threshold:
            return "CRITICAL"  # Extreme anomaly — critical level
        elif min_score < self.warning_threshold:
            return "WARNING"  # Moderate anomaly — warning
        return "INFO"  # Weak anomaly or normal

    def save_model(self, filepath: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Save the trained model and scaler to a file.

        Saves:
        - Trained IsolationForest model
        - Trained StandardScaler
        - Model parameters (contamination, thresholds, etc.)
        - Metadata (training date, version, comments)

        Args:
            filepath: Path to save the model (without extension, .joblib will be appended)
            metadata: Optional metadata (comments, version, dataset info)

        Example:
            >>> analyzer = EVBatteryAnalyzer()
            >>> analyzer.analyze_telemetry(df)
            >>> analyzer.save_model('models/battery_analyzer_v1',
            ...                     metadata={'version': '1.0', 'dataset': 'Tesla_2024'})

        Raises:
            ValueError: If model is not trained (analyze_telemetry or train was not called)
        """
        # Check that the model is trained (scaler must be fitted)
        if not hasattr(self.scaler, "mean_"):
            raise ValueError("Model not trained! Call analyze_telemetry() or train() first")

        # Prepare data for saving
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "contamination": self.contamination,
            "critical_threshold": self.critical_threshold,
            "warning_threshold": self.warning_threshold,
            "save_timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Add .joblib extension if not present
        if not filepath.endswith(".joblib"):
            filepath = filepath + ".joblib"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        # Save
        joblib.dump(model_data, filepath, compress=3)
        logger.info("Model saved: %s", filepath)

        if metadata:
            logger.info("Metadata: %s", metadata)

    @classmethod
    def load_model(cls, filepath: str) -> "EVBatteryAnalyzer":
        """
        Load a saved model from a file.

        Loads all model components and creates a new EVBatteryAnalyzer instance
        with restored state.

        Args:
            filepath: Path to the saved model (.joblib)

        Returns:
            New EVBatteryAnalyzer instance with the loaded model

        Example:
            >>> analyzer = EVBatteryAnalyzer.load_model('models/battery_analyzer_v1.joblib')
            >>> results = analyzer.analyze_telemetry(new_data)

        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file is corrupted or has an invalid format
        """
        # Add extension if not present
        if not filepath.endswith(".joblib"):
            filepath = filepath + ".joblib"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # FIX: limit file size to prevent DoS via joblib deserialization
        _max_size = 100 * 1024 * 1024  # 100 MB
        _file_size = os.path.getsize(filepath)
        if _file_size > _max_size:
            raise ValueError(f"Model file too large: {_file_size} bytes (max {_max_size})")

        # NOTE: joblib.load is used for internal model serialization only.
        # Models are saved/loaded within the same trusted application context.
        # For untrusted sources, use JSON/msgspec with schema validation instead.
        try:
            model_data = joblib.load(filepath)

            # Create new instance
            analyzer = cls(
                contamination=model_data["contamination"],
                critical_threshold=model_data.get("critical_threshold", -0.8),
                warning_threshold=model_data.get("warning_threshold", -0.5),
            )

            # Restore model and scaler
            analyzer.model = model_data["model"]
            analyzer.scaler = model_data["scaler"]

            # Print info about the loaded model
            save_time = model_data.get("save_timestamp", "Unknown")
            metadata = model_data.get("metadata", {})

            logger.info("Model loaded: %s", filepath)
            logger.info("Saved: %s", save_time)
            if metadata:
                logger.info("Metadata: %s", metadata)

            return analyzer

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model parameters
        """
        return {
            "contamination": self.contamination,
            "n_estimators": getattr(self.model, "n_estimators", None),
            "critical_threshold": self.critical_threshold,
            "warning_threshold": self.warning_threshold,
            "is_fitted": hasattr(self.scaler, "mean_"),
        }

    def detect_cell_imbalance(self, cell_voltages: list[float]) -> dict[str, Any]:
        """
        Analyze cell imbalance (Cell Imbalance).
        Critical for Tesla batteries and other electric vehicles.

        Args:
            cell_voltages: List of cell group voltages.

        Returns:
            Dictionary with imbalance metrics.
        """
        if not cell_voltages:
            return {"status": "error", "message": "No data"}

        avg_v = np.mean(cell_voltages)
        imbalance = np.max(cell_voltages) - np.min(cell_voltages)
        std_v = np.std(cell_voltages)

        # Tesla thresholds
        severity = "NORMAL"
        if imbalance > CELL_IMBALANCE_CRITICAL:
            severity = "CRITICAL"
        elif imbalance > CELL_IMBALANCE_WARNING:
            severity = "WARNING"

        return {
            "average_voltage": float(avg_v),
            "max_imbalance": float(imbalance),
            "std_dev": float(std_v),
            "severity": severity,
            "outliers_count": int(np.sum(np.abs(np.array(cell_voltages) - avg_v) > 0.05)),
        }

    def get_physics_features(self, df: pd.DataFrame) -> dict[str, Any]:
        """Extract all physics-informed features from a telemetry DataFrame.

        Expects columns: 'voltage', 'current', 'temp', 'soc'.
        Optional columns: 'capacity', 'time', 'charge_capacity', 'discharge_capacity', 'cycle_number'.

        Args:
            df: DataFrame with battery telemetry data.

        Returns:
            Dictionary with all physics features:
                - ic_curve: IC curve analysis results
                - delta_q: Capacity fade analysis results
                - resistance: Internal resistance estimation
                - thermal_diffusivity: Thermal diffusivity estimation
                - coulombic_efficiency: Coulombic efficiency results
        """
        result: dict[str, Any] = {}

        # IC curve: dQ/dV
        if "capacity" in df.columns and "voltage" in df.columns:
            result["ic_curve"] = self.physics_extractor.extract_ic_curve(
                df["voltage"].values,
                df["capacity"].values,
            )
        else:
            result["ic_curve"] = None

        # Delta Q analysis
        if "capacity" in df.columns:
            cycle_col = "cycle_number" if "cycle_number" in df.columns else None
            cycles = df[cycle_col].values if cycle_col else None
            result["delta_q"] = self.physics_extractor.compute_delta_q(
                df["capacity"].values,
                cycles,
            )
        else:
            result["delta_q"] = None

        # Internal resistance estimation
        if "voltage" in df.columns and "current" in df.columns:
            voltage = df["voltage"].values
            voltage_drop = np.abs(np.diff(voltage, prepend=voltage[0]))
            current = df["current"].values
            result["resistance"] = self.physics_extractor.estimate_resistance(voltage_drop, current)
        else:
            result["resistance"] = None

        # Thermal diffusivity
        if "temp" in df.columns:
            time_col = None
            for candidate in ("time", "timestamp", "elapsed_time"):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col:
                result["thermal_diffusivity"] = self.physics_extractor.compute_thermal_diffusivity(
                    df["temp"].values,
                    df[time_col].values,
                )
            else:
                # Use index as time proxy
                result["thermal_diffusivity"] = self.physics_extractor.compute_thermal_diffusivity(
                    df["temp"].values,
                    np.arange(len(df), dtype=float),
                )
        else:
            result["thermal_diffusivity"] = None

        # Coulombic efficiency
        if "charge_capacity" in df.columns and "discharge_capacity" in df.columns:
            result["coulombic_efficiency"] = self.physics_extractor.compute_coulombic_efficiency(
                df["discharge_capacity"].values,
                df["charge_capacity"].values,
            )
        else:
            result["coulombic_efficiency"] = None

        return result


class AnomalyDetector(EVBatteryAnalyzer):
    """
    Extended anomaly detector class with separate train/detect methods.

    This class allows:
    1. Train the model on "normal" data (train)
    2. Use the trained model to detect anomalies on new data (detect)

    This is useful in production when the model is trained once on historical
    data and then used for real-time detection.
    """

    def __init__(
        self, contamination: float = 0.01, n_estimators: int = 200, random_state: int = 42
    ):
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (default 0.01 = 1%).
                          For training on "clean" data, use a small value.
            n_estimators: Number of trees (recommended 200 for stability).
            random_state: Seed for reproducibility.
        """
        super().__init__(contamination, n_estimators, random_state)
        self._is_trained = False  # Model training flag

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model on "normal" data.

        It is recommended to use anomaly-free data for training so the model
        learns to recognize normal battery behavior.

        Args:
            data: DataFrame with columns ['voltage', 'current', 'temp', 'soc'].
                  Data should contain predominantly normal values.

        Example:
            >>> normal_data = pd.DataFrame({
            ...     'voltage': np.random.normal(48, 1, 1000),
            ...     'current': np.random.normal(100, 5, 1000),
            ...     'temp': np.random.normal(35, 2, 1000),
            ...     'soc': np.random.normal(85, 5, 1000)
            ... })
            >>> detector = AnomalyDetector()
            >>> detector.train(normal_data)
        """
        features = ["voltage", "current", "temp"]
        df = normalize_columns(data)
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)

        # Train IsolationForest
        self.model.fit(X_scaled)
        self._is_trained = True
        logger.info("Model trained on %d data points", len(data))

    def detect(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies on new data using the trained model.

        Args:
            data: DataFrame with new telemetry for analysis.

        Returns:
            Tuple of (predictions, scores):
                - predictions: Prediction array (-1 = anomaly, 1 = normal)
                - scores: Anomaly scores array

        Raises:
            ValueError: If model is not trained (train must be called first)

        Example:
            >>> new_data = pd.DataFrame({
            ...     'voltage': [48, 200],  # 200 — anomaly
            ...     'current': [100, 100],
            ...     'temp': [35, 35],
            ...     'soc': [85, 85]
            ... })
            >>> predictions, scores = detector.detect(new_data)
            >>> print(predictions)  # [1, -1]
        """
        if not self._is_trained:
            raise ValueError("Model not trained! Call the train() method first")

        features = ["voltage", "current", "temp"]
        df = normalize_columns(data)
        X = df[features]
        X_scaled = self.scaler.transform(X)

        # Predict on new data
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        anomaly_count = np.sum(predictions == -1)
        logger.info("Anomalies detected: %d/%d", anomaly_count, len(data))

        return predictions, scores

    def save_detector(self, path: str) -> None:
        """Save trained model and scaler to disk."""
        if not self._is_trained:
            raise ValueError("Model not trained! Call train() first")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "isolation_forest.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))

    def load_detector(self, path: str) -> None:
        """Load trained model and scaler from disk."""
        self.model = joblib.load(os.path.join(path, "isolation_forest.joblib"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        self._is_trained = True


if __name__ == "__main__":
    # EVBatteryAnalyzer usage example
    print("=== EVBatteryAnalyzer Test ===")
    analyzer = EVBatteryAnalyzer()

    # Generate test telemetry
    np.random.seed(42)
    data = {
        "voltage": np.random.normal(48, 2, 1000),
        "current": np.random.normal(100, 15, 1000),
        "temp": np.random.normal(35, 5, 1000),
        "soc": np.random.normal(85, 10, 1000),
    }
    df = pd.DataFrame(data)

    # Analysis
    results = analyzer.analyze_telemetry(df)
    print(f"Analysis complete: {results}")
    print(f"Anomalies: {results['anomalies_detected']}/{results['total_samples']}")
    print(f"Severity: {results['severity']}")

    # AnomalyDetector usage example
    print("\n=== AnomalyDetector Test (train/detect) ===")
    detector = AnomalyDetector(contamination=0.01, n_estimators=200)

    # Train on normal data
    normal_data = pd.DataFrame(
        {
            "voltage": np.random.normal(48, 1, 500),
            "current": np.random.normal(100, 5, 500),
            "temp": np.random.normal(35, 2, 500),
            "soc": np.random.normal(85, 5, 500),
        }
    )
    detector.train(normal_data)

    # Detect on new data with anomaly
    test_data = pd.DataFrame(
        {
            "voltage": [48, 48, 200, 48],  # 200V — obvious anomaly
            "current": [100, 100, 100, 100],
            "temp": [35, 35, 35, 35],
            "soc": [85, 85, 85, 85],
        }
    )
    predictions, scores = detector.detect(test_data)
    print(f"Predictions: {predictions}")
    print(f"Scores: {scores}")
