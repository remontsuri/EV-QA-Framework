"""Fleet Analytics: Multi-battery aggregation and analysis for EV fleets.

Provides fleet-wide health monitoring, cross-battery comparison,
degradation trend analysis, and anomaly detection across a fleet
of electric vehicles.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .analysis import EVBatteryAnalyzer
from .battery_scoring import BatteryScorer
from .physics_features import PhysicsFeatureExtractor

warnings.filterwarnings("ignore", category=UserWarning, module="ev_qa_framework\\.fleet_analytics")


# ---------------------------------------------------------------------------
# FleetAlert
# ---------------------------------------------------------------------------
@dataclass
class FleetAlert:
    """Represents a fleet-level anomaly alert."""

    battery_id: str
    alert_type: str
    severity: str  # CRITICAL, WARNING, INFO
    message: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# FleetAnalytics
# ---------------------------------------------------------------------------
class FleetAnalytics:
    """Aggregate analytics across a fleet of EV batteries.

    Manages multiple batteries, each identified by a unique string ID
    and associated with a telemetry DataFrame. Provides fleet-wide
    summary statistics, cross-battery comparison, degradation trend
    analysis, and anomaly detection.

    Parameters
    ----------
    scorer : BatteryScorer | None
        Pre-configured scorer instance. If None, a default BatteryScorer
        is created.
    analyzer : EVBatteryAnalyzer | None
        Pre-configured anomaly analyzer. If None, a default EVBatteryAnalyzer
        is created.
    physics : PhysicsFeatureExtractor | None
        Pre-configured physics feature extractor. If None, a default
        PhysicsFeatureExtractor is created.
    """

    def __init__(
        self,
        scorer: BatteryScorer | None = None,
        analyzer: EVBatteryAnalyzer | None = None,
        physics: PhysicsFeatureExtractor | None = None,
    ):
        self._batteries: dict[str, pd.DataFrame] = {}
        self._scores: dict[str, dict[str, Any]] = {}
        self._anomalies: dict[str, dict[str, Any]] = {}

        self.scorer = scorer or BatteryScorer()
        self.analyzer = analyzer or EVBatteryAnalyzer()
        self.physics = physics or PhysicsFeatureExtractor()

    # ------------------------------------------------------------------
    # Battery management
    # ------------------------------------------------------------------
    def add_battery(self, battery_id: str, telemetry_df: pd.DataFrame) -> None:
        """Register a battery and its telemetry data in the fleet.

        Parameters
        ----------
        battery_id : str
            Unique identifier for the battery / vehicle.
        telemetry_df : pd.DataFrame
            Telemetry data with columns: voltage, current, temp (or
            temperature), soc. Optionally soh, capacity, cycle_number.

        Raises
        ------
        ValueError
            If battery_id is empty or telemetry_df is not a DataFrame.
        """
        if not battery_id or not isinstance(battery_id, str):
            raise ValueError("battery_id must be a non-empty string")
        if not isinstance(telemetry_df, pd.DataFrame):
            raise ValueError("telemetry_df must be a pandas DataFrame")
        if telemetry_df.empty:
            raise ValueError("telemetry_df must not be empty")

        self._batteries[battery_id] = telemetry_df.copy()
        # Invalidate cached results for this battery
        self._scores.pop(battery_id, None)
        self._anomalies.pop(battery_id, None)

    def remove_battery(self, battery_id: str) -> None:
        """Remove a battery from the fleet."""
        self._batteries.pop(battery_id, None)
        self._scores.pop(battery_id, None)
        self._anomalies.pop(battery_id, None)

    @property
    def battery_ids(self) -> list[str]:
        """Return list of registered battery IDs."""
        return list(self._batteries.keys())

    @property
    def fleet_size(self) -> int:
        """Return number of batteries in the fleet."""
        return len(self._batteries)

    def get_telemetry(self, battery_id: str) -> pd.DataFrame:
        """Return telemetry DataFrame for a given battery."""
        if battery_id not in self._batteries:
            raise KeyError(f"Battery '{battery_id}' not found in fleet")
        return self._batteries[battery_id].copy()

    # ------------------------------------------------------------------
    # Scoring (per-battery, cached)
    # ------------------------------------------------------------------
    def score_battery(self, battery_id: str) -> dict[str, Any]:
        """Compute composite health score for a single battery.

        Results are cached so repeated calls with unchanged telemetry
        do not recompute.

        Returns
        -------
        dict with keys: score, grade, soh_score, anomaly_score,
            cell_balance_score, thermal_score, details
        """
        if battery_id in self._scores:
            return self._scores[battery_id]
        df = self.get_telemetry(battery_id)
        result = self.scorer.compute_score(df)
        self._scores[battery_id] = result
        return result

    # ------------------------------------------------------------------
    # Fleet summary
    # ------------------------------------------------------------------
    def get_fleet_summary(self) -> dict[str, Any]:
        """Compute aggregate statistics across all registered batteries.

        Returns
        -------
        dict with keys:
            fleet_size: int
            avg_score: float
            min_score: float
            max_score: float
            std_score: float
            avg_soh: float
            avg_anomaly_pct: float
            avg_thermal_risk: float
            grade_distribution: dict[str, int]
            batteries: list[dict]  (per-battery summary)
        """
        if not self._batteries:
            return {
                "fleet_size": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "std_score": 0.0,
                "avg_soh": 0.0,
                "avg_anomaly_pct": 0.0,
                "avg_thermal_risk": 0.0,
                "grade_distribution": {},
                "batteries": [],
            }

        battery_summaries: list[dict[str, Any]] = []
        scores: list[float] = []
        sohs: list[float] = []
        anomaly_pcts: list[float] = []
        thermal_risks: list[float] = []

        for bid in self._batteries:
            score_result = self.score_battery(bid)
            anomaly_result = self._analyze_anomaly(bid)

            s = score_result["score"]
            scores.append(s)
            sohs.append(score_result.get("soh_score", 0.0))
            anomaly_pcts.append(anomaly_result.get("anomaly_percentage", 0.0))
            thermal_risks.append(score_result.get("thermal_score", 0.0))

            battery_summaries.append(
                {
                    "battery_id": bid,
                    "score": s,
                    "grade": score_result["grade"],
                    "soh_score": score_result.get("soh_score", 0.0),
                    "anomaly_score": score_result.get("anomaly_score", 0.0),
                    "cell_balance_score": score_result.get("cell_balance_score", 0.0),
                    "thermal_score": score_result.get("thermal_score", 0.0),
                    "anomaly_percentage": anomaly_result.get("anomaly_percentage", 0.0),
                    "num_samples": len(self._batteries[bid]),
                }
            )

        scores_arr = np.array(scores)
        grade_dist: dict[str, int] = {}
        for bs in battery_summaries:
            g = bs["grade"]
            grade_dist[g] = grade_dist.get(g, 0) + 1

        return {
            "fleet_size": len(self._batteries),
            "avg_score": round(float(np.mean(scores_arr)), 2),
            "min_score": round(float(np.min(scores_arr)), 2),
            "max_score": round(float(np.max(scores_arr)), 2),
            "std_score": round(float(np.std(scores_arr)), 2),
            "avg_soh": round(float(np.mean(sohs)), 2),
            "avg_anomaly_pct": round(float(np.mean(anomaly_pcts)), 2),
            "avg_thermal_risk": round(float(np.mean(thermal_risks)), 2),
            "grade_distribution": grade_dist,
            "batteries": battery_summaries,
        }

    # ------------------------------------------------------------------
    # Battery comparison
    # ------------------------------------------------------------------
    def compare_batteries(self, battery_ids: list[str] | None = None) -> pd.DataFrame:
        """Build a comparison table for the specified (or all) batteries.

        Parameters
        ----------
        battery_ids : list[str] | None
            Subset of battery IDs to compare. If None, compares all.

        Returns
        -------
        pd.DataFrame with one row per battery and columns:
            battery_id, score, grade, soh_score, anomaly_score,
            cell_balance_score, thermal_score, anomaly_percentage,
            num_samples
        """
        ids = battery_ids or self.battery_ids
        rows: list[dict[str, Any]] = []
        for bid in ids:
            if bid not in self._batteries:
                raise KeyError(f"Battery '{bid}' not found in fleet")
            sr = self.score_battery(bid)
            ar = self._analyze_anomaly(bid)
            rows.append(
                {
                    "battery_id": bid,
                    "score": sr["score"],
                    "grade": sr["grade"],
                    "soh_score": sr.get("soh_score", 0.0),
                    "anomaly_score": sr.get("anomaly_score", 0.0),
                    "cell_balance_score": sr.get("cell_balance_score", 0.0),
                    "thermal_score": sr.get("thermal_score", 0.0),
                    "anomaly_percentage": ar.get("anomaly_percentage", 0.0),
                    "num_samples": len(self._batteries[bid]),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Fleet degradation
    # ------------------------------------------------------------------
    def compute_fleet_degradation(self) -> dict[str, Any]:
        """Analyze degradation trends across the fleet.

        For each battery that has a 'soh' column with multiple distinct
        values, fits a linear trend (SOH over row index as a proxy for
        time/cycles). Aggregates per-battery trends into fleet-level
        statistics.

        Returns
        -------
        dict with keys:
            per_battery: dict[str, dict]  (battery_id -> trend info)
            fleet_avg_slope: float  (average SOH change per sample)
            fleet_avg_initial_soh: float
            fleet_avg_final_soh: float
            batteries_with_data: int
        """
        per_battery: dict[str, dict[str, Any]] = {}
        slopes: list[float] = []
        initials: list[float] = []
        finals: list[float] = []

        for bid, df in self._batteries.items():
            soh_col = None
            if "soh" in df.columns:
                soh_col = "soh"
            elif "SOH" in df.columns:
                soh_col = "SOH"

            if soh_col is None:
                per_battery[bid] = {"trend_available": False}
                continue

            soh_values = df[soh_col].dropna().values
            if len(soh_values) < 2:
                per_battery[bid] = {"trend_available": False}
                continue

            x = np.arange(len(soh_values), dtype=float)
            coeffs = np.polyfit(x, soh_values, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])

            # R²
            pred = np.polyval(coeffs, x)
            ss_res = np.sum((soh_values - pred) ** 2)
            ss_tot = np.sum((soh_values - np.mean(soh_values)) ** 2)
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            per_battery[bid] = {
                "trend_available": True,
                "slope": round(slope, 6),
                "intercept": round(intercept, 4),
                "initial_soh": round(float(soh_values[0]), 2),
                "final_soh": round(float(soh_values[-1]), 2),
                "total_fade": round(float(soh_values[-1] - soh_values[0]), 2),
                "r_squared": round(r_squared, 4),
                "n_points": len(soh_values),
            }
            slopes.append(slope)
            initials.append(float(soh_values[0]))
            finals.append(float(soh_values[-1]))

        n_data = len(slopes)
        return {
            "per_battery": per_battery,
            "fleet_avg_slope": round(float(np.mean(slopes)), 6) if slopes else 0.0,
            "fleet_avg_initial_soh": round(float(np.mean(initials)), 2) if initials else 0.0,
            "fleet_avg_final_soh": round(float(np.mean(finals)), 2) if finals else 0.0,
            "batteries_with_data": n_data,
        }

    # ------------------------------------------------------------------
    # Fleet anomaly detection
    # ------------------------------------------------------------------
    def detect_fleet_anomalies(
        self,
        score_threshold: float = 60.0,
        anomaly_pct_threshold: float = 20.0,
    ) -> list[FleetAlert]:
        """Scan all batteries and generate alerts for unhealthy units.

        Two categories of alerts are produced:
        1. Low composite health score (score < score_threshold).
        2. High anomaly percentage (anomaly_pct > anomaly_pct_threshold).

        Parameters
        ----------
        score_threshold : float
            Batteries with a composite score below this trigger a WARNING
            (or CRITICAL if below 40).
        anomaly_pct_threshold : float
            Batteries with anomaly percentage above this trigger a WARNING
            (or CRITICAL if above 50).

        Returns
        -------
        list[FleetAlert]
        """
        alerts: list[FleetAlert] = []

        for bid in self._batteries:
            # --- Score-based alerts ---
            sr = self.score_battery(bid)
            score = sr["score"]
            if score < 40.0:
                alerts.append(
                    FleetAlert(
                        battery_id=bid,
                        alert_type="LOW_HEALTH_SCORE",
                        severity="CRITICAL",
                        message=f"Battery {bid} health score is critical: {score:.1f}/100",
                        value=score,
                        threshold=40.0,
                    )
                )
            elif score < score_threshold:
                alerts.append(
                    FleetAlert(
                        battery_id=bid,
                        alert_type="LOW_HEALTH_SCORE",
                        severity="WARNING",
                        message=f"Battery {bid} health score is low: {score:.1f}/100",
                        value=score,
                        threshold=score_threshold,
                    )
                )

            # --- Anomaly-based alerts ---
            ar = self._analyze_anomaly(bid)
            pct = ar.get("anomaly_percentage", 0.0)
            if pct > 50.0:
                alerts.append(
                    FleetAlert(
                        battery_id=bid,
                        alert_type="HIGH_ANOMALY_RATE",
                        severity="CRITICAL",
                        message=f"Battery {bid} anomaly rate is critical: {pct:.1f}%",
                        value=pct,
                        threshold=50.0,
                    )
                )
            elif pct > anomaly_pct_threshold:
                alerts.append(
                    FleetAlert(
                        battery_id=bid,
                        alert_type="HIGH_ANOMALY_RATE",
                        severity="WARNING",
                        message=f"Battery {bid} anomaly rate is elevated: {pct:.1f}%",
                        value=pct,
                        threshold=anomaly_pct_threshold,
                    )
                )

        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _analyze_anomaly(self, battery_id: str) -> dict[str, Any]:
        """Run anomaly detection for a battery (cached)."""
        if battery_id in self._anomalies:
            return self._anomalies[battery_id]
        df = self.get_telemetry(battery_id)
        result = self.analyzer.analyze_telemetry(df)
        self._anomalies[battery_id] = result
        return result
