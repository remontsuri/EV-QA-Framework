"""
Thermal Runaway Prediction Module.

Два режима:
- rule: улучшенная эвристика с настраиваемыми весами
- ml: Isolation Forest на темповых признаках
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class ThermalRunawayPredictor:
    """
    Predictor for thermal runaway risk in EV batteries.

    Parameters
    ----------
    mode : str, default='rule'
        'rule' for rule-based heuristic, 'ml' for Machine Learning based.
    rule_weights : dict, optional
        Веса для rule-based scoring: rise_rate, max_temp, anomaly, dt_dt.
    thresholds : dict, optional
        Кастомные пороги: critical_temp, high_temp, critical_dtdt и т.д.
    contamination : float, default=0.1
        Expected contamination for IsolationForest (only in ML mode).
    """

    def __init__(
        self,
        mode: str = "rule",
        rule_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
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
            "critical_temp": 65.0,
            "critical_risk": 10.0,
            "critical_dtdt": 5.0,
            "high_temp": 55.0,
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

    def analyze_temperature_trend(self, df_recent: pd.DataFrame) -> Dict[str, float]:
        """
        Извлечение темповых признаков из DataFrame.

        Returns dict: temp_rise_rate, max_temp, volatility, dt_dt
        """
        temp_col = "temp" if "temp" in df_recent.columns else "temperature"
        temps = df_recent[temp_col].values
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

    def predict_risk(self, df_recent: pd.DataFrame) -> Dict:
        """
        Прогноз риска thermal runaway.

        Returns dict: risk_level (LOW/MEDIUM/HIGH/CRITICAL), risk_score, confidence
        """
        if len(df_recent) < 2:
            return {"risk_level": "LOW", "risk_score": 0.0, "confidence": 1.0}

        features = self.analyze_temperature_trend(df_recent)
        current_temp = features["max_temp"]

        if self.mode == "ml" and self._isolation_forest is not None:
            # ML mode, need to fit first
            temp_col = "temp" if "temp" in df_recent.columns else "temperature"
            X = df_recent[[temp_col]].values
            if not self._is_fitted:
                self._isolation_forest.fit(X)
                self._is_fitted = True

            scores = self._isolation_forest.score_samples(X)
            anomaly_score = float(np.mean(scores < -0.5))
        else:
            temp_col = "temp" if "temp" in df_recent.columns else "temperature"
            temps = df_recent[temp_col].values
            mean, std = float(np.mean(temps)), float(np.std(temps))
            anomaly_score = float(np.sum(np.abs(temps - mean) > 2 * std)) / len(temps)

        risk_score = (
            features["temp_rise_rate"] * self.rule_weights["rise_rate"]
            + features["max_temp"] * self.rule_weights["max_temp"]
            + anomaly_score * self.rule_weights["anomaly"]
            + features["dt_dt"] * self.rule_weights["dt_dt"]
        )

        if current_temp > 50:
            risk_score += (current_temp - 50) * 0.5

        risk_level = "LOW"
        if risk_score > self.thresholds["critical_risk"] or current_temp > self.thresholds["critical_temp"]:
            risk_level = "CRITICAL"
        elif features["dt_dt"] > self.thresholds["critical_dtdt"]:
            risk_level = "CRITICAL"
        elif risk_score > self.thresholds["high_risk"] or current_temp > self.thresholds["high_temp"]:
            risk_level = "HIGH"
        elif risk_score > self.thresholds["medium_risk"]:
            risk_level = "MEDIUM"

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "confidence": round(1.0 - anomaly_score, 2),
            **features,
        }
