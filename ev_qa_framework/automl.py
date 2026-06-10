"""
AutoML pipeline for automatic SOH model selection and hyperparameter tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from .config import FrameworkConfig


class AutoMLSOH:
    """Automatic ML pipeline for SOH prediction.

    Tries multiple algorithms and selects the best one.
    """

    def __init__(self, config: FrameworkConfig | None = None):
        self.config = config or FrameworkConfig()
        self.best_model = None
        self.best_score = float("-inf")
        self.results: list[dict] = []
        self.scaler = MinMaxScaler()

    def fit(self, df: pd.DataFrame, target_col: str = "soh") -> dict:
        """
        Try multiple models and select the best.

        Args:
            df: DataFrame with features and target
            target_col: name of target column

        Returns:
            dict with best model info
        """
        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]

        if not available_features or target_col not in df.columns:
            return {"error": "Missing required columns"}

        X = df[available_features].values
        y = df[target_col].values

        X_scaled = self.scaler.fit_transform(X)

        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42
            ),
        }

        self.results = []
        best_name = None
        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X_scaled, y, cv=3, scoring="r2"
                )
                mean_score = scores.mean()
                std_score = scores.std()

                result = {
                    "model_name": name,
                    "mean_r2": mean_score,
                    "std_r2": std_score,
                    "n_samples": len(y),
                }
                self.results.append(result)

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model
                    best_name = name

            except Exception as e:
                self.results.append({
                    "model_name": name,
                    "error": str(e),
                })

        if self.best_model is not None:
            self.best_model.fit(X_scaled, y)

        return {
            "best_model": best_name if self.best_model else None,
            "best_r2": self.best_score,
            "n_models_tried": len(models),
            "results": self.results,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict SOH using the best model."""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")

        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def evaluate(self, df: pd.DataFrame, target_col: str = "soh") -> dict:
        """Evaluate the best model on test data."""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")

        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        y = df[target_col].values
        X_scaled = self.scaler.transform(X)

        predictions = self.best_model.predict(X_scaled)

        return {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "n_samples": len(y),
        }


class AutoMLAnomaly:
    """Automatic ML pipeline for anomaly detection.

    Tries multiple algorithms and selects the best.
    """

    def __init__(self, config: FrameworkConfig | None = None):
        self.config = config or FrameworkConfig()
        self.best_model = None
        self.best_score = float("-inf")

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Try multiple anomaly detection algorithms.

        Returns:
            dict with best model info
        """
        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]

        if not available_features:
            return {"error": "Missing required columns"}

        X = df[available_features].values

        contamination = self.config.ml_config.contamination

        models = {
            "isolation_forest": IsolationForest(
                contamination=contamination, random_state=42
            ),
        }

        results = []
        for name, model in models.items():
            try:
                model.fit(X)
                scores = model.decision_function(X)
                mean_score = scores.mean()

                results.append({
                    "model_name": name,
                    "mean_anomaly_score": mean_score,
                    "n_anomalies": int((model.predict(X) == -1).sum()),
                })

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    self.best_model = model

            except Exception as e:
                results.append({
                    "model_name": name,
                    "error": str(e),
                })

        return {
            "best_model": self.best_model.__class__.__name__ if self.best_model else None,
            "results": results,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using the best model."""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")

        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        return self.best_model.predict(X)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores."""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")

        features = ["voltage", "current", "temperature"]
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        return self.best_model.decision_function(X)
