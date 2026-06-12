"""
Thermal Runaway Prediction Module.

Two modes:
- rule: enhanced heuristic with configurable weights
- ml: Isolation Forest on temperature features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .utils import normalize_columns


class ThermalRunawayPredictor:
    """
    Predictor for thermal runaway risk in EV batteries.

    Parameters
    ----------
    mode : str, default='rule'
        'rule' for rule-based heuristic, 'ml' for Machine Learning based.
    rule_weights : dict, optional
        Weights for rule-based scoring: rise_rate, max_temp, anomaly, dt_dt.
    thresholds : dict, optional
        Custom thresholds: critical_temp, high_temp, critical_dtdt, etc.
    contamination : float, default=0.1
        Expected contamination for IsolationForest (only in ML mode).
    """

    def __init__(
        self,
        mode: str = "rule",
        rule_weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.mode = mode.lower()
        if self.mode not in ("rule", "ml"):
            raise ValueError("mode must be 'rule' or 'ml'")

        self.rule_weights = {"rise_rate": 2.0, "max_temp": 1.5, "anomaly": 5.0, "dt_dt": 3.0}
        if rule_weights:
            self.rule_weights.update(rule_weights)

        self.thresholds = {
            "critical_temp": 85.0,      # FIX: was 65.0 — too low, causes false CRITICAL
            "critical_risk": 10.0,
            "critical_dtdt": 10.0,      # FIX: was 5.0 — too sensitive
            "high_temp": 65.0,          # FIX: was 55.0 — too low, causes false HIGH
            "high_risk": 5.0,
            "medium_risk": 2.0,
        }
        if thresholds:
            self.thresholds.update(thresholds)

        self._isolation_forest = None
        self._is_fitted = False

        if self.mode == "ml":
            self._isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
            )

    def analyze_temperature_trend(self, df_recent: pd.DataFrame) -> dict[str, float]:
        """
        Extract temperature features from DataFrame.

        Returns dict: temp_rise_rate, max_temp, volatility, dt_dt
        """
        df_recent = normalize_columns(df_recent)
        if df_recent.empty or "temp" not in df_recent.columns:
            return {"temp_rise_rate": 0.0, "max_temp": 0.0, "volatility": 0.0, "dt_dt": 0.0}
        temps = df_recent["temp"].values
        n = len(temps)

        if n < 2:
            return {
                "temp_rise_rate": 0.0,
                "max_temp": float(temps[0]) if n == 1 else 0.0,
                "volatility": 0.0,
                "dt_dt": 0.0,
            }

        slope = float(np.polyfit(np.arange(n), temps, 1)[0])
        gradients = np.diff(temps)
        max_dt = float(np.max(gradients)) if len(gradients) > 0 else 0.0

        return {
            "temp_rise_rate": slope,
            "max_temp": float(np.max(temps)),
            "volatility": float(np.std(temps, ddof=1)),
            "dt_dt": max_dt,
        }

    def predict_risk(self, df_recent: pd.DataFrame) -> dict[str, object]:
        """
        Predict thermal runaway risk.

        Returns dict: risk_level (LOW/MEDIUM/HIGH/CRITICAL), risk_score, confidence
        """
        df_recent = normalize_columns(df_recent)
        if df_recent.empty or "temp" not in df_recent.columns:
            return {"risk_level": "LOW", "risk_score": 0.0, "confidence": 0.0}
        if len(df_recent) < 2:
            return {"risk_level": "LOW", "risk_score": 0.0, "confidence": 1.0}

        features = self.analyze_temperature_trend(df_recent)
        current_temp = features["max_temp"]

        if self.mode == "ml" and self._isolation_forest is not None:
            # ML mode, need to fit first
            X = df_recent[["temp"]].values
            if not self._is_fitted:
                self._isolation_forest.fit(X)
                self._is_fitted = True

            scores = self._isolation_forest.score_samples(X)
            anomaly_score = float(np.mean(scores < -0.5))
        else:
            temps = df_recent["temp"].values
            mean, std = float(np.mean(temps)), float(np.std(temps))
            anomaly_score = float(np.sum(np.abs(temps - mean) > 2 * std)) / len(temps)

        risk_score = (
            features["temp_rise_rate"] * self.rule_weights["rise_rate"]
            + max(0, features["max_temp"] - 50) * self.rule_weights["max_temp"]
            + anomaly_score * self.rule_weights["anomaly"]
            + features["dt_dt"] * self.rule_weights["dt_dt"]
        )
        # FIX: removed duplicate temperature penalty (current_temp - 50) * 0.5
        # The rule_weights["max_temp"] already handles temperature scoring above

        risk_level = "LOW"
        if (
            risk_score > self.thresholds["critical_risk"]
            or current_temp > self.thresholds["critical_temp"]
        ):
            risk_level = "CRITICAL"
        elif features["dt_dt"] > self.thresholds["critical_dtdt"]:
            risk_level = "CRITICAL"
        elif (
            risk_score > self.thresholds["high_risk"] or current_temp > self.thresholds["high_temp"]
        ):
            risk_level = "HIGH"
        elif risk_score > self.thresholds["medium_risk"]:
            risk_level = "MEDIUM"

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "confidence": round(max(0.0, 1.0 - anomaly_score), 2),  # FIX: clamp to >=0
            **features,
        }
