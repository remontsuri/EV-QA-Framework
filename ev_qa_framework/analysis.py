from __future__ import annotations

"""EV QA Analysis: ML-based battery telemetry and quality assurance.

Machine learning module for detecting anomalies in battery telemetry.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Tuple, List
import warnings
import joblib  # type: ignore  # no stub available
import os
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EVBatteryAnalyzer:
    """
    ML analyzer for EV battery telemetry based on the Isolation Forest algorithm.
    
    Isolation Forest is an anomaly detection algorithm that isolates outliers
    by randomly selecting a feature and then randomly selecting a split value
    between the maximum and minimum of the selected feature. Anomalies are isolated
    faster than normal data points.
    
    Attributes:
        model: IsolationForest model from scikit-learn
        scaler: StandardScaler for data normalization
        anomalies: DataFrame with detected anomalies
        contamination: Proportion of anomalies in the dataset (default 0.1 = 10%)
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 200, random_state: int = 42,
                 critical_threshold: float = -0.8, warning_threshold: float = -0.5):
        """
        Initialize the telemetry analyzer.
        
        Args:
            contamination: Expected proportion of anomalies in the data (0.0 - 1.0).
                          For example, 0.1 means ~10% of data may be anomalous.
            n_estimators: Number of trees in the ensemble (more = more accurate, but slower).
                         Recommended 100-200 for balance of accuracy and speed.
            random_state: Seed for result reproducibility.
            critical_threshold: Threshold for CRITICAL severity (default -0.8)
            warning_threshold: Threshold for WARNING severity (default -0.5)
        
        Note:
            - contamination affects sensitivity: lower value = fewer false positives
            - n_estimators is recommended to be 100+ for stable results
        """
        # Create Isolation Forest model with configured parameters
        self.model = IsolationForest(
            contamination=contamination,    # Expected proportion of anomalies
            n_estimators=n_estimators,      # Number of trees (more = more stable)
            max_samples='auto',             # Auto-select subsample size
            random_state=random_state,      # For reproducibility
            n_jobs=-1                       # Use all CPU cores
        )
        
        # StandardScaler normalizes data: (x - mean) / std
        # This is important because IsolationForest is sensitive to feature scale
        self.scaler = StandardScaler()
        
        # Storage for detected anomalies (populated after analyze_telemetry)
        # anomalies stored as DataFrame; start empty
        self.anomalies: pd.DataFrame = pd.DataFrame()
        
        # Save parameters for external access
        self.contamination = contamination
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        
    def analyze_telemetry(self, df_telemetry: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze battery telemetry for anomalies.
        
        Algorithm:
        1. Data normalization via StandardScaler (bringing to the same scale)
        2. Train IsolationForest on normalized data
        3. Predict anomalies (-1 = anomaly, 1 = normal)
        4. Calculate anomaly scores (lower = more anomalous point)
        5. Assess severity based on minimum score
        
        Args:
            df_telemetry: DataFrame with columns ['voltage', 'current', 'temp', 'soc'].
                         Each row represents a single point in time.
        
        Returns:
            Dictionary with analysis results:
                - total_samples: Total number of data points
                - anomalies_detected: Number of detected anomalies
                - anomaly_percentage: Percentage of anomalies relative to total
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
        # Step 1: Data preparation
        # Accept either 'temp' or the more readable 'temperature' and convert to 'temp'
        df: pd.DataFrame = df_telemetry.copy()
        if 'temperature' in df.columns and 'temp' not in df.columns:
            df = df.rename(columns={'temperature': 'temp'})
        
        # Step 2: Select only numeric features for analysis
        # SOC is not used for detection since it is a dependent variable
        features: List[str] = ['voltage', 'current', 'temp']
        X: pd.DataFrame = df[features]
        
        # Step 2: Data normalization (mean=0, std=1)
        # Use existing scaler if already fitted, otherwise fit_transform
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)  # type: ignore
        else:
            X_scaled = self.scaler.fit_transform(X)  # type: ignore
        
        # Step 3: Train model and predict anomalies
        if hasattr(self.model, 'estimators_'):
            predictions = self.model.predict(X_scaled)  # type: ignore
        else:
            predictions = self.model.fit_predict(X_scaled)  # type: ignore
        
        # Step 4: Calculate anomaly scores (lower = more anomalous point)
        # score_samples works even for a previously fitted model
        anomaly_scores: np.ndarray = self.model.score_samples(X_scaled)  # type: ignore
        
        # Step 5: Filter anomalies
        # In addition to the standard prediction (-1), also consider cases where
        # score_samples fell below warning_threshold. This helps avoid missing
        # rare outliers on small samples (e.g., when the model is trained
        # on identical points).
        mask: np.ndarray = (predictions == -1) | (anomaly_scores < self.warning_threshold)
        self.anomalies = df_telemetry[mask].copy()  # type: ignore
        
        # Add anomaly scores to results for further analysis
        if not self.anomalies.empty:
            self.anomalies['anomaly_score'] = anomaly_scores[mask]
        
        # Step 6: Compile the analysis result
        total = len(df_telemetry)
        count = len(self.anomalies)
        return {
            'total_samples': total,
            'anomalies_detected': count,
            'anomaly_percentage': (count / total) * 100 if total else 0.0,
            'severity': self._assess_severity(anomaly_scores)
        }
    
    def _assess_severity(self, scores: np.ndarray) -> str:
        """
        Assess the severity level of detected anomalies.
        
        Assessment logic:
        - CRITICAL: Extreme outliers present (score < critical_threshold)
                   Requires immediate attention — possible critical failure
        - WARNING: Moderate anomalies (score < warning_threshold)
                  Requires inspection — possible system degradation
        - INFO: Minor anomalies or none (score >= warning_threshold)
               System is normal, anomalies are insignificant
        
        Args:
            scores: Array of anomaly scores from IsolationForest
        
        Returns:
            String with severity level: 'CRITICAL', 'WARNING', or 'INFO'
        
        Note:
            Thresholds are configured via constructor parameters and can be adjusted
            for a specific system based on historical data.
        """
        min_score = np.min(scores)
        
        if min_score < self.critical_threshold:
            return 'CRITICAL'  # Extreme anomaly — critical level
        elif min_score < self.warning_threshold:
            return 'WARNING'   # Moderate anomaly — warning
        return 'INFO'          # Minor anomaly or normal
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] | None = None) -> None:
        """
        Save the trained model and scaler to a file.
        
        Saves:
        - Trained IsolationForest model
        - Trained StandardScaler
        - Model parameters (contamination, thresholds, etc.)
        - Metadata (training date, version, comments)
        
        Args:
            filepath: Path to save the model (without extension, .joblib will be added)
            metadata: Optional metadata (comments, version, dataset info)
        
        Example:
            >>> analyzer = EVBatteryAnalyzer()
            >>> analyzer.analyze_telemetry(df)
            >>> analyzer.save_model('models/battery_analyzer_v1', 
            ...                     metadata={'version': '1.0', 'dataset': 'Tesla_2024'})
        
        Raises:
            ValueError: If model is not trained (analyze_telemetry was not called)
        """
        # Check that the model is trained (scaler must be fitted)
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError(
                "Model not trained! Call analyze_telemetry() or train() first"
            )
        
        # Prepare data for saving
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'critical_threshold': self.critical_threshold,
            'warning_threshold': self.warning_threshold,
            'save_timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Add .joblib extension if not present
        if not filepath.endswith('.joblib'):
            filepath = filepath + '.joblib'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        
        # Save
        joblib.dump(model_data, filepath, compress=3)
        logger.info("Model saved: %s", filepath)

        if metadata:
            logger.info("Model metadata: %s", metadata)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EVBatteryAnalyzer':
        """
        Load a saved model from a file.
        
        Loads all model components and creates a new EVBatteryAnalyzer instance
        with restored state.
        
        Args:
            filepath: Path to the saved model (.joblib)
        
        Returns:
            New EVBatteryAnalyzer instance with loaded model
        
        Example:
            >>> analyzer = EVBatteryAnalyzer.load_model('models/battery_analyzer_v1.joblib')
            >>> results = analyzer.analyze_telemetry(new_data)
        
        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file is corrupted or has an invalid format
        """
        # Add extension if not present
        if not filepath.endswith('.joblib'):
            filepath = filepath + '.joblib'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Load data
            model_data = joblib.load(filepath)
            
            # Create new instance
            analyzer = cls(
                contamination=model_data['contamination'],
                critical_threshold=model_data.get('critical_threshold', -0.8),
                warning_threshold=model_data.get('warning_threshold', -0.5)
            )
            
            # Restore model and scaler
            analyzer.model = model_data['model']
            analyzer.scaler = model_data['scaler']
            
            # Output information about the loaded model
            save_time = model_data.get('save_timestamp', 'Unknown')
            metadata = model_data.get('metadata', {})
            
            logger.info("Model loaded: %s", filepath)
            logger.info("Saved at: %s", save_time)
            if metadata:
                logger.info("Metadata: %s", metadata)
            
            return analyzer
            
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model parameters
        """
        return {
            'contamination': self.contamination,
            'n_estimators': getattr(self.model, 'n_estimators', None),
            'critical_threshold': self.critical_threshold,
            'warning_threshold': self.warning_threshold,
            'is_fitted': hasattr(self.scaler, 'mean_')
        }

    def detect_cell_imbalance(self, cell_voltages: List[float]) -> Dict[str, Any]:
        """
        Analyze cell imbalance.
        Critically important for Tesla and other electric vehicle batteries.

        Args:
            cell_voltages: List of cell group voltages.

        Returns:
            Dictionary with imbalance metrics.
        """
        if not cell_voltages:
            return {"status": "error", "message": "No data"}

        avg_v = np.mean(cell_voltages)
        max_v = np.max(cell_voltages)
        min_v = np.min(cell_voltages)
        imbalance = max_v - min_v
        std_v = np.std(cell_voltages)

        # Thresholds for Tesla are typically around 0.05V - 0.1V
        severity = "NORMAL"
        if imbalance > 0.1:
            severity = "CRITICAL"
        elif imbalance > 0.05:
            severity = "WARNING"

        return {
            "average_voltage": float(avg_v),
            "max_imbalance": float(imbalance),
            "std_dev": float(std_v),
            "severity": severity,
            "outliers_count": int(np.sum(np.abs(np.array(cell_voltages) - avg_v) > 0.05))
        }

    def predict_thermal_runaway_risk(self, df_recent: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict thermal runaway risk.
        Uses a combination of temperature rise rate and ML scores.

        Args:
            df_recent: Latest telemetry points (minimum 5-10 points).
        """
        if len(df_recent) < 2:
            return {"risk_level": "LOW", "score": 0.0}

        # 1. Temperature trend analysis
        temp_diffs = np.diff(df_recent['temp'].values if 'temp' in df_recent else df_recent['temperature'].values)
        avg_temp_rise = np.mean(temp_diffs)
        max_temp_rise = np.max(temp_diffs)
        current_temp = df_recent['temp'].iloc[-1] if 'temp' in df_recent else df_recent['temperature'].iloc[-1]

        # 2. Integration with ML (Isolation Forest)
        ml_results = self.analyze_telemetry(df_recent)
        anomaly_score = ml_results['anomaly_percentage'] / 100.0

        # 3. Combined risk score
        risk_score = (avg_temp_rise * 2.0) + (max_temp_rise * 1.5) + (anomaly_score * 5.0)
        if current_temp > 50:
            risk_score += (current_temp - 50) * 0.5

        risk_level = "LOW"
        if risk_score > 10 or current_temp > 65:
            risk_level = "CRITICAL"
        elif risk_score > 5 or current_temp > 55:
            risk_level = "HIGH"
        elif risk_score > 2:
            risk_level = "MEDIUM"

        return {
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "avg_temp_rise_rate": float(avg_temp_rise),
            "current_temp": float(current_temp)
        }


class AnomalyDetector(EVBatteryAnalyzer):
    """
    Extended anomaly detector class with separate train/detect methods.
    
    This class allows:
    1. Train the model on "normal" data (train)
    2. Use the trained model for detection on new data (detect)
    
    This is useful in production, where the model is trained once on historical
    data and then used for real-time detection.
    """
    
    def __init__(self, contamination: float = 0.01, n_estimators: int = 200, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 0.01 = 1%).
                          Use a small value for training on "clean" data.
            n_estimators: Number of trees (recommended 200 for stability).
            random_state: Seed for reproducibility.
        """
        super().__init__(contamination, n_estimators, random_state)
        self._is_trained = False  # Model training flag
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model on "normal" data.
        
        It is recommended to use data without anomalies for training,
        so the model learns to recognize normal battery behavior.
        
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
        features = ['voltage', 'current', 'temp']
        X = data[features]
        
        # Train scaler on normal data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train IsolationForest
        self.model.fit(X_scaled)
        self._is_trained = True
        logger.info("Model trained on %d data points", len(data))
    
    def detect(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies on new data using the trained model.
        
        Args:
            data: DataFrame with new telemetry data for analysis.
        
        Returns:
            Tuple (predictions, scores):
                - predictions: Array of predictions (-1 = anomaly, 1 = normal)
                - scores: Array of anomaly scores
        
        Raises:
            ValueError: If model is not trained (call train() first)
        
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
        
        features = ['voltage', 'current', 'temp']
        X = data[features]
        
        # Apply the already fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Predict on new data
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        anomaly_count = np.sum(predictions == -1)
        logger.info("Anomalies detected: %d/%d", anomaly_count, len(data))
        
        return predictions, scores


if __name__ == '__main__':
    # Example usage of EVBatteryAnalyzer
    print("=== EVBatteryAnalyzer Test ===")
    analyzer = EVBatteryAnalyzer()
    
    # Generate test telemetry
    np.random.seed(42)
    data = {
        'voltage': np.random.normal(48, 2, 1000),
        'current': np.random.normal(100, 15, 1000),
        'temp': np.random.normal(35, 5, 1000),
        'soc': np.random.normal(85, 10, 1000)
    }
    df = pd.DataFrame(data)
    
    # Analysis
    results = analyzer.analyze_telemetry(df)
    print(f"Analysis complete: {results}")
    print(f"Anomalies: {results['anomalies_detected']}/{results['total_samples']}")
    print(f"Severity: {results['severity']}")
    
    # Example usage of AnomalyDetector
    print("\n=== AnomalyDetector Test (train/detect) ===")
    detector = AnomalyDetector(contamination=0.01, n_estimators=200)
    
    # Train on normal data
    normal_data = pd.DataFrame({
        'voltage': np.random.normal(48, 1, 500),
        'current': np.random.normal(100, 5, 500),
        'temp': np.random.normal(35, 2, 500),
        'soc': np.random.normal(85, 5, 500)
    })
    detector.train(normal_data)
    
    # Detect on new data with anomaly
    test_data = pd.DataFrame({
        'voltage': [48, 48, 200, 48],  # 200V — obvious anomaly
        'current': [100, 100, 100, 100],
        'temp': [35, 35, 35, 35],
        'soc': [85, 85, 85, 85]
    })
    predictions, scores = detector.detect(test_data)
    print(f"Predictions: {predictions}")
    print(f"Scores: {scores}")
