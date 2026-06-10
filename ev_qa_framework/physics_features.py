from __future__ import annotations

"""Physics-informed feature extraction for EV battery telemetry.

This module provides physically meaningful feature extraction methods
for battery quality assurance, including IC curves, capacity fade tracking,
internal resistance estimation, thermal diffusivity, and coulombic efficiency.
"""

from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter


class PhysicsFeatureExtractor:
    """Extractor for physics-informed battery features.

    Provides methods to compute physically meaningful features from
    battery telemetry data that are commonly used in battery research
    and industry for SOH estimation, degradation analysis, and QA.
    """

    def __init__(self, smoothing_window: int = 5, polyorder: int = 2):
        """Initialize the extractor.

        Args:
            smoothing_window: Window size for Savitzky-Golay filter (must be odd).
            polyorder: Polynomial order for Savitzky-Golay filter.
        """
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder

    @staticmethod
    def _safe_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute dy/dx with protection against division by zero.

        Args:
            y: Dependent variable values.
            x: Independent variable values.

        Returns:
            Derivative dy/dx, with zeros where dx == 0.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        # Avoid division by zero
        safe_dx = np.where(np.abs(dx) < 1e-12, np.nan, dx)
        deriv = dy / safe_dx
        # Replace NaN/Inf with 0
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)
        return deriv

    def extract_ic_curve(
        self,
        voltage: np.ndarray,
        capacity: np.ndarray,
        min_peak_height: float | None = None,
        min_peak_distance: int = 5,
    ) -> dict[str, Any]:
        """Extract Incremental Capacity (IC) curve features: dQ/dV.

        The IC curve is a key diagnostic tool in battery analysis. Peaks in
        the IC curve correspond to phase transitions in the electrode
        materials, and their positions/shifts indicate degradation modes.

        Args:
            voltage: Array of voltage measurements (V).
            capacity: Array of capacity measurements (Ah).
            min_peak_height: Minimum peak height for detection. If None,
                             uses 10% of the max IC value.
            min_peak_distance: Minimum distance between peaks (in samples).

        Returns:
            Dictionary with:
                - ic_values: dQ/dV values (array of length N-1)
                - voltage_mid: Mid-point voltage array (length N-1)
                - peaks: Indices of detected peaks
                - valleys: Indices of detected valleys
                - peak_voltages: Voltage values at peaks
                - peak_heights: IC values at peaks
                - valley_voltages: Voltage values at valleys
                - valley_depths: IC values at valleys
                - num_peaks: Number of detected peaks
                - num_valleys: Number of detected valleys
        """
        v = np.asarray(voltage, dtype=float)
        q = np.asarray(capacity, dtype=float)

        if len(v) < 3 or len(q) < 3:
            return {
                "ic_values": np.array([]),
                "voltage_mid": np.array([]),
                "peaks": np.array([], dtype=int),
                "valleys": np.array([], dtype=int),
                "peak_voltages": np.array([]),
                "peak_heights": np.array([]),
                "valley_voltages": np.array([]),
                "valley_depths": np.array([]),
                "num_peaks": 0,
                "num_valleys": 0,
            }

        # Compute dQ/dV
        ic_values = self._safe_derivative(q, v)

        # Mid-point voltages
        voltage_mid = (v[:-1] + v[1:]) / 2.0

        # Smooth IC curve for peak detection
        if len(ic_values) >= self.smoothing_window:
            ic_smooth = savgol_filter(ic_values, self.smoothing_window, self.polyorder)
        else:
            ic_smooth = ic_values.copy()

        # Detect peaks
        if min_peak_height is None:
            min_peak_height = 0.1 * np.max(np.abs(ic_smooth))

        peaks, peak_props = find_peaks(
            ic_smooth,
            height=min_peak_height,
            distance=min_peak_distance,
        )

        # Detect valleys (peaks in negated signal)
        neg_smooth = -ic_smooth
        valleys, valley_props = find_peaks(
            neg_smooth,
            height=min_peak_height,
            distance=min_peak_distance,
        )

        return {
            "ic_values": ic_values,
            "voltage_mid": voltage_mid,
            "peaks": peaks,
            "valleys": valleys,
            "peak_voltages": voltage_mid[peaks] if len(peaks) > 0 else np.array([]),
            "peak_heights": ic_smooth[peaks] if len(peaks) > 0 else np.array([]),
            "valley_voltages": voltage_mid[valleys] if len(valleys) > 0 else np.array([]),
            "valley_depths": ic_smooth[valleys] if len(valleys) > 0 else np.array([]),
            "num_peaks": len(peaks),
            "num_valleys": len(valleys),
        }

    @staticmethod
    def compute_delta_q(
        capacity_series: np.ndarray,
        cycle_numbers: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute Delta Q analysis for capacity fade tracking.

        Tracks capacity fade over cycles. The slope of the linear fit
        to capacity vs. cycle number gives the fade rate (Ah/cycle).

        Args:
            capacity_series: Array of capacity values per cycle (Ah).
            cycle_numbers: Array of cycle numbers. If None, uses
                           np.arange(len(capacity_series)).

        Returns:
            Dictionary with:
                - delta_q: Array of capacity differences (Q[i] - Q[0])
                - fade_rate: Linear fade rate (Ah/cycle), slope of linear fit
                - initial_capacity: First capacity value
                - final_capacity: Last capacity value
                - total_fade: Total capacity fade (Ah)
                - fade_percentage: Total fade as percentage of initial
                - r_squared: R² of the linear fit
        """
        q = np.asarray(capacity_series, dtype=float)
        n = len(q)

        if n < 2:
            return {
                "delta_q": np.array([]),
                "fade_rate": 0.0,
                "initial_capacity": float(q[0]) if n > 0 else 0.0,
                "final_capacity": float(q[-1]) if n > 0 else 0.0,
                "total_fade": 0.0,
                "fade_percentage": 0.0,
                "r_squared": 0.0,
            }

        if cycle_numbers is None:
            cycles = np.arange(n, dtype=float)
        else:
            cycles = np.asarray(cycle_numbers, dtype=float)

        # Delta Q relative to first cycle
        delta_q = q - q[0]

        # Linear fit for fade rate
        # y = a*x + b, where a is fade rate
        coeffs = np.polyfit(cycles, q, 1)
        fade_rate = float(coeffs[0])  # Ah/cycle (negative = fading)

        # R² calculation
        q_pred = np.polyval(coeffs, cycles)
        ss_res = np.sum((q - q_pred) ** 2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        total_fade = float(q[-1] - q[0])
        fade_pct = (total_fade / q[0] * 100.0) if q[0] != 0 else 0.0

        return {
            "delta_q": delta_q,
            "fade_rate": fade_rate,
            "initial_capacity": float(q[0]),
            "final_capacity": float(q[-1]),
            "total_fade": total_fade,
            "fade_percentage": fade_pct,
            "r_squared": r_squared,
        }

    @staticmethod
    def estimate_resistance(
        voltage_drop: np.ndarray | float,
        current: np.ndarray | float,
    ) -> dict[str, Any]:
        """Estimate internal resistance from voltage drop and current.

        Uses Ohm's law: R = ΔV / I. Supports both single-point and
        array-based estimation (returns statistics).

        Args:
            voltage_drop: Voltage drop(s) in Volts. Positive values expected.
            current: Current(s) in Amps. Positive values expected.

        Returns:
            Dictionary with:
                - resistance: Estimated resistance(s) in Ohms
                - mean_resistance: Mean resistance (for array input)
                - std_resistance: Std of resistance (for array input)
                - min_resistance: Minimum resistance
                - max_resistance: Maximum resistance
        """
        dv = np.asarray(voltage_drop, dtype=float)
        i_arr = np.asarray(current, dtype=float)

        # Avoid division by zero
        safe_i = np.where(np.abs(i_arr) < 1e-12, np.nan, i_arr)
        resistance = np.abs(dv / safe_i)
        resistance = np.nan_to_num(resistance, nan=0.0, posinf=0.0, neginf=0.0)

        result: dict[str, Any] = {
            "resistance": resistance,
        }

        if resistance.size > 1:
            result["mean_resistance"] = float(np.mean(resistance))
            result["std_resistance"] = float(np.std(resistance))
            result["min_resistance"] = float(np.min(resistance))
            result["max_resistance"] = float(np.max(resistance))
        else:
            result["mean_resistance"] = float(resistance.flat[0])
            result["std_resistance"] = 0.0
            result["min_resistance"] = float(resistance.flat[0])
            result["max_resistance"] = float(resistance.flat[0])

        return result

    @staticmethod
    def compute_thermal_diffusivity(
        temp_series: np.ndarray,
        time_series: np.ndarray,
    ) -> dict[str, Any]:
        """Estimate thermal diffusivity from temperature-time data.

        Thermal diffusivity α characterizes how quickly heat spreads
        through a material. Estimated from the rate of temperature change
        using a simplified 1D model: α ≈ (dT/dt) * L² / ΔT

        where L is a characteristic length (normalized to 1.0 for relative
        comparison), dT/dt is the temperature rate of change, and ΔT is
        the temperature difference from ambient.

        Args:
            temp_series: Array of temperature measurements (°C).
            time_series: Array of time values (seconds).

        Returns:
            Dictionary with:
                - thermal_diffusivity: Array of estimated α values (m²/s, normalized)
                - mean_diffusivity: Mean thermal diffusivity
                - max_temp_rate: Maximum rate of temperature change (°C/s)
                - temp_gradient: Temperature gradient array
                - time_gradient: Time gradient array
        """
        t = np.asarray(temp_series, dtype=float)
        time_s = np.asarray(time_series, dtype=float)

        if len(t) < 2 or len(time_s) < 2:
            return {
                "thermal_diffusivity": np.array([]),
                "mean_diffusivity": 0.0,
                "max_temp_rate": 0.0,
                "temp_gradient": np.array([]),
                "time_gradient": np.array([]),
            }

        # Temperature gradient (dT/dt)
        dt_arr = np.diff(t)
        dtime = np.diff(time_s)

        # Avoid division by zero
        safe_dtime = np.where(np.abs(dtime) < 1e-12, np.nan, dtime)
        temp_rate = dt_arr / safe_dtime  # °C/s
        temp_rate = np.nan_to_num(temp_rate, nan=0.0, posinf=0.0, neginf=0.0)

        # Temperature difference from initial (as proxy for ΔT)
        delta_t = t[1:] - t[0]

        # Avoid division by zero in diffusivity calculation
        safe_delta_t = np.where(np.abs(delta_t) < 1e-12, np.nan, delta_t)

        # Simplified thermal diffusivity: α ∝ (dT/dt) / ΔT
        # Using normalized characteristic length L=1
        thermal_diff = temp_rate / safe_delta_t
        thermal_diff = np.nan_to_num(thermal_diff, nan=0.0, posinf=0.0, neginf=0.0)

        max_temp_rate = float(np.max(np.abs(temp_rate))) if len(temp_rate) > 0 else 0.0
        mean_diff = float(np.mean(thermal_diff)) if len(thermal_diff) > 0 else 0.0

        return {
            "thermal_diffusivity": thermal_diff,
            "mean_diffusivity": mean_diff,
            "max_temp_rate": max_temp_rate,
            "temp_gradient": dt_arr,
            "time_gradient": dtime,
        }

    @staticmethod
    def compute_coulombic_efficiency(
        discharge_capacity: np.ndarray | float,
        charge_capacity: np.ndarray | float,
    ) -> dict[str, Any]:
        """Compute Coulombic efficiency (CE).

        Coulombic efficiency = discharge_capacity / charge_capacity.
        Ideal CE = 1.0 (100%). Values below 1.0 indicate irreversible
        capacity loss (e.g., SEI growth, lithium plating).

        Args:
            discharge_capacity: Discharge capacity (Ah).
            charge_capacity: Charge capacity (Ah).

        Returns:
            Dictionary with:
                - efficiency: CE value(s) (ratio, 0.0-1.0+)
                - efficiency_percent: CE as percentage
                - mean_efficiency: Mean CE (for arrays)
                - min_efficiency: Minimum CE
                - max_efficiency: Maximum CE
                - irreversible_loss: 1.0 - CE (ratio of lost capacity)
        """
        d_cap = np.asarray(discharge_capacity, dtype=float)
        c_cap = np.asarray(charge_capacity, dtype=float)

        # Avoid division by zero
        safe_c = np.where(np.abs(c_cap) < 1e-12, np.nan, c_cap)
        efficiency = d_cap / safe_c
        efficiency = np.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0)

        result: dict[str, Any] = {
            "efficiency": efficiency,
            "efficiency_percent": float(efficiency.flat[0] * 100.0)
            if efficiency.size == 1
            else (np.asarray(efficiency * 100.0) if efficiency.size > 1 else np.array([])),
            "irreversible_loss": np.asarray(1.0 - efficiency),
        }

        if efficiency.size > 1:
            result["mean_efficiency"] = float(np.mean(efficiency))
            result["min_efficiency"] = float(np.min(efficiency))
            result["max_efficiency"] = float(np.max(efficiency))
        else:
            val = float(efficiency.flat[0])
            result["mean_efficiency"] = val
            result["min_efficiency"] = val
            result["max_efficiency"] = val

        return result
