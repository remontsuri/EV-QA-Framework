"""Battery Scoring System: Composite health score for EV battery packs.

Combines SOH, anomaly detection, cell balance, and thermal risk into a
single 0–100 health score with letter grades and actionable recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import EVBatteryAnalyzer
from .cell_balance import CellBalanceAnalyzer
from .thermal_runaway import ThermalRunawayPredictor

# ---------------------------------------------------------------------------
# Default weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_SOH = 0.40
WEIGHT_ANOMALY = 0.15  # FIX: was 0.25 — rebalanced for thermal weight increase
WEIGHT_CELL_BALANCE = 0.20
WEIGHT_THERMAL = 0.25  # FIX: was 0.15 — thermal is critical for EV safety



# ---------------------------------------------------------------------------
# Grade boundaries
# ---------------------------------------------------------------------------
GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (40, "D"),
]


# ---------------------------------------------------------------------------
# BatteryScorer
# ---------------------------------------------------------------------------
class BatteryScorer:
    """Composite battery health scorer.

    Parameters
    ----------
    soh_weight : float
        Weight for SOH component (default 0.40).
    anomaly_weight : float
        Weight for anomaly score component (default 0.15).
    cell_balance_weight : float
        Weight for cell balance component (default 0.20).
    thermal_weight : float
        Weight for thermal risk component (default 0.25).
    soh_baseline : float
        Baseline SOH value (%). If None, derived from telemetry or
        defaults to 100.
    cell_voltages : list[float] | None
        Per-cell voltages for balance analysis. If None, cell balance
        score defaults to 100.
    """

    def __init__(
        self,
        soh_weight: float = WEIGHT_SOH,
        anomaly_weight: float = WEIGHT_ANOMALY,
        cell_balance_weight: float = WEIGHT_CELL_BALANCE,
        thermal_weight: float = WEIGHT_THERMAL,
        soh_baseline: float | None = None,
        cell_voltages: list[float] | None = None,
    ):
        total = soh_weight + anomaly_weight + cell_balance_weight + thermal_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.soh_weight = soh_weight
        self.anomaly_weight = anomaly_weight
        self.cell_balance_weight = cell_balance_weight
        self.thermal_weight = thermal_weight
        self.soh_baseline = soh_baseline
        self.cell_voltages = cell_voltages

        # Sub-analyzers (lightweight; created once)
        self._anomaly_analyzer = EVBatteryAnalyzer()
        self._cell_analyzer = CellBalanceAnalyzer()
        self._thermal_predictor = ThermalRunawayPredictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_score(
        self,
        telemetry_df: pd.DataFrame,
        cell_voltages: list[float] | None = None,
    ) -> dict:
        """Compute composite battery health score.

        Parameters
        ----------
        telemetry_df : pd.DataFrame
            Must contain columns: voltage, current, temp (or temperature),
            soc. Optionally soh.
        cell_voltages : list[float] | None
            Overrides voltages provided at init. Used for cell balance.

        Returns
        -------
        dict with keys:
            score, grade, soh_score, anomaly_score, cell_balance_score,
            thermal_score, details
        """
        df = telemetry_df.copy()
        # Normalise column name
        if "temperature" in df.columns and "temp" not in df.columns:
            df = df.rename(columns={"temperature": "temp"})

        # --- SOH component ---
        soh_score = self._compute_soh(df)

        # --- Anomaly component ---
        anomaly_score = self._compute_anomaly(df)

        # --- Cell balance component ---
        voltages = cell_voltages if cell_voltages is not None else self.cell_voltages
        cell_balance_score = self._compute_cell_balance(voltages)

        # --- Thermal component ---
        thermal_score = self._compute_thermal(df)

        # --- Weighted composite ---
        composite = (
            soh_score * self.soh_weight
            + anomaly_score * self.anomaly_weight
            + cell_balance_score * self.cell_balance_weight
            + thermal_score * self.thermal_weight
        )
        composite = float(np.clip(round(composite, 2), 0.0, 100.0))

        return {
            "score": composite,
            "grade": self.get_grade(composite),
            "soh_score": round(soh_score, 2),
            "anomaly_score": round(anomaly_score, 2),
            "cell_balance_score": round(cell_balance_score, 2),
            "thermal_score": round(thermal_score, 2),
            "details": {
                "weights": {
                    "soh": self.soh_weight,
                    "anomaly": self.anomaly_weight,
                    "cell_balance": self.cell_balance_weight,
                    "thermal": self.thermal_weight,
                },
                "recommendations": self.get_recommendations(
                    {
                        "score": composite,
                        "soh_score": soh_score,
                        "anomaly_score": anomaly_score,
                        "cell_balance_score": cell_balance_score,
                        "thermal_score": thermal_score,
                    }
                ),
            },
        }

    @staticmethod
    def get_grade(score: float) -> str:
        """Map numeric score to letter grade.

        A: 90–100  (Excellent)
        B: 75–89   (Good)
        C: 60–74   (Fair)
        D: 40–59   (Poor)
        F: 0–39    (Critical)
        """
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"

    @staticmethod
    def get_recommendations(score_dict: dict) -> list[str]:
        """Generate actionable recommendations from component scores.

        Parameters
        ----------
        score_dict : dict
            Must contain keys: score, soh_score, anomaly_score,
            cell_balance_score, thermal_score.
        """
        recs: list[str] = []

        overall = score_dict.get("score", 100)
        soh = score_dict.get("soh_score", 100)
        anomaly = score_dict.get("anomaly_score", 100)
        balance = score_dict.get("cell_balance_score", 100)
        thermal = score_dict.get("thermal_score", 100)

        if overall >= 90:
            recs.append("Battery health is excellent. Continue routine monitoring.")
            return recs

        if soh < 60:
            recs.append(
                "CRITICAL: SOH is very low. Consider battery replacement or " "deep diagnostic."
            )
        elif soh < 75:
            recs.append(
                "WARNING: SOH is degraded. Schedule capacity test and "
                "reduce fast-charge frequency."
            )

        if anomaly < 60:
            recs.append(
                "CRITICAL: High anomaly rate detected. Inspect voltage, "
                "current, and temperature sensors."
            )
        elif anomaly < 80:
            recs.append(
                "WARNING: Moderate anomalies detected. Review recent "
                "telemetry for irregular patterns."
            )

        if balance < 60:
            recs.append(
                "CRITICAL: Severe cell imbalance. Perform cell balancing " "procedure immediately."
            )
        elif balance < 80:
            recs.append(
                "WARNING: Cell imbalance detected. Schedule balancing "
                "service at next maintenance."
            )

        if thermal < 60:
            recs.append(
                "CRITICAL: High thermal risk. Stop charging, allow pack "
                "to cool, inspect cooling system."
            )
        elif thermal < 80:
            recs.append(
                "WARNING: Elevated thermal risk. Avoid fast charging and "
                "high-load driving until inspected."
            )

        if not recs:
            recs.append("All parameters within acceptable limits.")

        return recs

    # ------------------------------------------------------------------
    # Internal component scorers (each returns 0–100)
    # ------------------------------------------------------------------
    def _compute_soh(self, df: pd.DataFrame) -> float:
        """SOH score: directly from 'soh' column or baseline."""
        if "soh" in df.columns:
            return float(np.clip(df["soh"].iloc[-1], 0.0, 100.0))
        if self.soh_baseline is not None:
            return float(np.clip(self.soh_baseline, 0.0, 100.0))
        # No SOH data available — assume perfect
        return 100.0

    def _compute_anomaly(self, df: pd.DataFrame) -> float:
        """Anomaly score: 100 – penalty based on anomaly percentage."""
        try:
            result = self._anomaly_analyzer.analyze_telemetry(df)
            pct = result.get("anomaly_percentage", 0.0)
            # Map: 0% anomalies -> 100, 100% anomalies -> 0
            score = max(0.0, 100.0 - pct * 2)
            return score
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Scoring component failed: %s", e)
            return 50.0

    def _compute_cell_balance(self, voltages: list[float] | None) -> float:
        """Cell balance score derived from imbalance severity."""
        if voltages is None or len(voltages) == 0:
            return 100.0
        try:
            severity = self._cell_analyzer.classify_severity(voltages)
            if severity == "NORMAL":
                return 100.0
            if severity == "WARNING":
                return 70.0
            return 30.0
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Scoring component failed: %s", e)
            return 50.0

    def _compute_thermal(self, df: pd.DataFrame) -> float:
        """Thermal risk score: 100 for LOW, decreasing for higher risk."""
        try:
            result = self._thermal_predictor.predict_risk(df)
            level = result.get("risk_level", "LOW")
            mapping = {"LOW": 100.0, "MEDIUM": 70.0, "HIGH": 40.0, "CRITICAL": 10.0}
            return mapping.get(level, 100.0)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Scoring component failed: %s", e)
            return 50.0
