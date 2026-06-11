"""
Cell Balance Analyzer: Detection and analysis of cell voltage imbalance in EV battery packs.

Uses statistical methods, configurable thresholds, and linear regression trend
prediction to identify cell imbalance conditions before they become critical.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no-display backend for headless/server use

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class CellBalanceAnalyzer:
    """
    Analyzes cell voltage imbalance in battery packs.

    Detects outlier cells, classifies severity, predicts trends via linear
    regression, and generates visualisation plots.

    Parameters
    ----------
    warning_threshold : float
        Max voltage difference (V) for NORMAL status. Below this is normal.
    critical_threshold : float
        Above this voltage difference (V) the state is CRITICAL.
    outlier_std_factor : float
        Multiplier for standard deviation to flag outliers (mean ± factor*std).
    outlier_abs_deviation : float
        Absolute deviation from mean (V) to flag outliers.
    trend_window : int
        Number of most recent measurements to use for linear regression trend.
    """

    def __init__(
        self,
        warning_threshold: float = 0.02,
        critical_threshold: float = 0.05,
        outlier_std_factor: float = 2.0,
        outlier_abs_deviation: float = 0.05,
        trend_window: int = 10,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.outlier_std_factor = outlier_std_factor
        self.outlier_abs_deviation = outlier_abs_deviation
        self.trend_window = trend_window

    def compute_statistics(self, voltages: list[float]) -> dict[str, float]:
        """
        Compute basic statistics of cell voltages.

        Parameters
        ----------
        voltages : list of float
            Cell voltage readings (V).

        Returns
        -------
        dict with keys: mean, median, std, max, min, max_min_imbalance
        """
        if not voltages:
            raise ValueError("Voltage list is empty.")
        arr = np.array(voltages)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "max_min_imbalance": float(np.max(arr) - np.min(arr)),
        }

    def detect_outliers(self, voltages: list[float]) -> list[int]:
        """
        Identify indices of outlier cells based on voltage thresholds.

        Outliers are cells whose voltage deviates more than
        ``outlier_std_factor * std`` from the mean, or whose absolute
        deviation exceeds ``outlier_abs_deviation``.

        Returns
        -------
        list of int
            Sorted 0-based indices of outlier cells.
        """
        if not voltages:
            return []
        arr = np.array(voltages)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)

        lower = mean - self.outlier_std_factor * std
        upper = mean + self.outlier_std_factor * std
        outliers: list[int] = []
        for idx, v in enumerate(arr):
            if v < lower or v > upper:
                outliers.append(idx)
            elif abs(v - mean) > self.outlier_abs_deviation:
                outliers.append(idx)
        return sorted(outliers)

    def classify_severity(self, voltages: list[float]) -> str:
        """
        Classify overall imbalance severity based on max-min difference.

        Returns 'NORMAL', 'WARNING', or 'CRITICAL'.
        """
        if not voltages:
            return "NORMAL"
        imbalance = max(voltages) - min(voltages)
        if imbalance < self.warning_threshold:
            return "NORMAL"
        if imbalance < self.critical_threshold:
            return "WARNING"
        return "CRITICAL"

    def predict_trend(self, timeline_measurements: list[list[float]]) -> tuple[float, float]:
        """
        Fit linear regression to max-min imbalance over recent snapshots.

        Parameters
        ----------
        timeline_measurements : list of list of float
            Each inner list is a snapshot of cell voltages at one point in time.

        Returns
        -------
        (slope, intercept)
            Linear regression coefficients. (0.0, 0.0) if insufficient data.
        """
        if len(timeline_measurements) < 2:
            return (0.0, 0.0)

        imbalances = [max(snap) - min(snap) for snap in timeline_measurements]
        window = (
            imbalances[-self.trend_window :] if len(imbalances) > self.trend_window else imbalances
        )
        n = len(window)
        if n < 2:
            return (0.0, 0.0)

        x = np.arange(n).reshape(-1, 1)
        y = np.array(window)
        model = LinearRegression()
        model.fit(x, y)
        return (float(model.coef_[0]), float(model.intercept_))

    def plot_imbalance(
        self,
        timeline_voltages: list[list[float]],
        save_path: str | None = "imbalance_plot.png",
    ) -> None:
        """
        Plot max-min imbalance over time and save to file.

        Parameters
        ----------
        timeline_voltages : list of list of float
            Each inner list is a snapshot of cell voltages.
        save_path : str, optional
            Path to save the figure.
        """
        if not timeline_voltages:
            raise ValueError("No data to plot.")

        imbalances = [max(snap) - min(snap) for snap in timeline_voltages]
        time_steps = np.arange(len(imbalances))

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, imbalances, "b-o", label="Max-Min Imbalance")
        plt.axhline(
            y=self.warning_threshold,
            color="orange",
            linestyle="--",
            label="Warning Threshold",
        )
        plt.axhline(
            y=self.critical_threshold,
            color="red",
            linestyle="--",
            label="Critical Threshold",
        )
        plt.xlabel("Measurement Index")
        plt.ylabel("Voltage Imbalance (V)")
        plt.title("Cell Voltage Imbalance Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
