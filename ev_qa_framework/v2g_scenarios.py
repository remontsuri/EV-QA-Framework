"""
V2G-HealthNet: Vehicle-to-Grid scenario testing for EV-QA-Framework.

Generates V2G scenarios and analyzes their impact on battery health.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .battery_scoring import BatteryScorer
from .config import FrameworkConfig
from .physics_features import PhysicsFeatureExtractor


class V2GScenarioGenerator:
    """Generate V2G (Vehicle-to-Grid) scenarios."""

    def __init__(self, battery_capacity_ah: float = 100.0, nominal_voltage: float = 400.0):
        self.battery_capacity_ah = battery_capacity_ah
        self.nominal_voltage = nominal_voltage

    def generate_v2g_cycle(
        self,
        duration_hours: int = 24,
        grid_demand_profile: str = "typical",
    ) -> pd.DataFrame:
        """
        Generate a V2G cycle profile.

        Args:
            duration_hours: duration in hours
            grid_demand_profile: "typical", "peak_shaving", "frequency_regulation"

        Returns:
            DataFrame with 'current' and 'duration_h' columns
        """
        rng = np.random.default_rng(42)

        if grid_demand_profile == "typical":
            # Typical V2G: discharge during peak hours (17-21), charge at night
            hours = np.arange(duration_hours)
            current = np.zeros(duration_hours)

            for h in hours:
                if 17 <= h <= 21:  # peak discharge
                    current[h] = -rng.uniform(30, 80)
                elif 0 <= h <= 6:  # night charge
                    current[h] = rng.uniform(20, 50)
                else:  # idle / light use
                    current[h] = rng.uniform(-10, 10)

        elif grid_demand_profile == "peak_shaving":
            current = np.zeros(duration_hours)
            for h in range(duration_hours):
                if 16 <= h <= 20:
                    current[h] = -rng.uniform(50, 100)
                elif 22 <= h <= 24 or 0 <= h <= 5:
                    current[h] = rng.uniform(30, 60)

        elif grid_demand_profile == "frequency_regulation":
            # Rapid small charge/discharge cycles
            current = rng.uniform(-30, 30, duration_hours)

        else:
            raise ValueError(f"Unknown profile: {grid_demand_profile}")

        return pd.DataFrame(
            {
                "current": current,
                "duration_h": np.ones(duration_hours),
            }
        )

    def generate_peak_shaving_scenario(
        self,
        peak_hours: tuple[int, int] = (17, 21),
        battery_capacity: float = 100.0,
        discharge_power_kw: float = 50.0,
    ) -> pd.DataFrame:
        """Generate peak shaving scenario."""
        duration = 24
        current = np.zeros(duration)
        discharge_current = -(discharge_power_kw * 1000) / self.nominal_voltage

        for h in range(peak_hours[0], peak_hours[1] + 1):
            current[h] = discharge_current

        # Charge at night
        for h in range(0, 6):
            current[h] = abs(discharge_current) * 0.5

        return pd.DataFrame({"current": current, "duration_h": np.ones(duration)})

    def generate_frequency_regulation_scenario(
        self,
        signal_profile: str = "aggressive",
        duration_hours: int = 4,
    ) -> pd.DataFrame:
        """Generate frequency regulation scenario."""
        rng = np.random.default_rng(42)
        n = duration_hours * 60  # 1-minute resolution

        if signal_profile == "aggressive":
            current = rng.uniform(-80, 80, n)
        elif signal_profile == "moderate":
            current = rng.uniform(-40, 40, n)
        elif signal_profile == "conservative":
            current = rng.uniform(-20, 20, n)
        else:
            raise ValueError(f"Unknown signal profile: {signal_profile}")

        return pd.DataFrame(
            {
                "current": current,
                "duration_h": np.ones(n) / 60,  # minutes to hours
            }
        )


class V2GHealthAnalyzer:
    """Analyze impact of V2G on battery health."""

    def __init__(self, config: FrameworkConfig | None = None):
        self.config = config or FrameworkConfig()
        self.scorer = BatteryScorer()
        self.physics = PhysicsFeatureExtractor()

    def compute_v2g_impact(
        self,
        baseline_df: pd.DataFrame,
        v2g_df: pd.DataFrame,
    ) -> dict:
        """
        Compare baseline vs V2G battery data.

        Returns:
            dict with impact metrics
        """
        baseline_score = self.scorer.compute_score(baseline_df)
        v2g_score = self.scorer.compute_score(v2g_df)

        return {
            "baseline_score": baseline_score["score"],
            "v2g_score": v2g_score["score"],
            "score_delta": v2g_score["score"] - baseline_score["score"],
            "baseline_soh": baseline_score.get("soh_score", 0),
            "v2g_soh": v2g_score.get("soh_score", 0),
            "soh_delta": v2g_score.get("soh_score", 0) - baseline_score.get("soh_score", 0),
            "baseline_grade": baseline_score["grade"],
            "v2g_grade": v2g_score["grade"],
        }

    def estimate_cycle_life_impact(
        self,
        v2g_cycles_per_day: float = 1.0,
        avg_depth_of_discharge: float = 0.5,
    ) -> dict:
        """
        Estimate impact of V2G on cycle life.

        Args:
            v2g_cycles_per_day: equivalent full cycles per day from V2G
            avg_depth_of_discharge: average DoD per V2G cycle

        Returns:
            dict with estimated cycle life metrics
        """
        # Simplified: each V2G cycle causes wear proportional to DoD
        equivalent_full_cycles_per_day = v2g_cycles_per_day * avg_depth_of_discharge

        # Assume 2000 cycle life at 80% DoD
        base_cycle_life = 2000
        adjusted_cycle_life = base_cycle_life / max(equivalent_full_cycles_per_day * 0.8, 0.01)

        years_to_80 = adjusted_cycle_life / 365

        return {
            "equivalent_full_cycles_per_day": equivalent_full_cycles_per_day,
            "estimated_total_cycles": adjusted_cycle_life,
            "estimated_years_to_80_soh": years_to_80,
            "annual_degradation_pct": 100.0 / max(years_to_80, 0.1),
        }

    def get_v2g_recommendations(self, current_soh: float) -> list[str]:
        """Get V2G recommendations based on current SOH."""
        recommendations = []

        if current_soh >= 90:
            recommendations.append("Battery is healthy. V2G operations are safe.")
            recommendations.append("Recommended: up to 2 V2G cycles per day at 50% DoD.")
        elif current_soh >= 80:
            recommendations.append("Battery is in good condition. Moderate V2G recommended.")
            recommendations.append("Recommended: up to 1 V2G cycle per day at 30% DoD.")
            recommendations.append("Monitor SOH monthly.")
        elif current_soh >= 70:
            recommendations.append("Battery degradation detected. Limit V2G operations.")
            recommendations.append("Recommended: max 0.5 V2G cycles per day at 20% DoD.")
            recommendations.append("Monitor SOH weekly. Consider battery replacement soon.")
        else:
            recommendations.append("CRITICAL: Battery SOH below 70%. V2G NOT recommended.")
            recommendations.append("Battery replacement strongly recommended.")
            recommendations.append("If V2G is required, limit to emergency use only.")

        return recommendations


class V2SScenarioGenerator:
    """Generate Vehicle-to-Station (V2S) and charging station profiles."""

    def __init__(self, battery_capacity_ah: float = 100.0, nominal_voltage: float = 400.0):
        self.battery_capacity_ah = battery_capacity_ah
        self.nominal_voltage = nominal_voltage

    def generate_charging_station_profile(
        self, station_type: str, duration_hours: float = 4
    ) -> pd.DataFrame:
        """
        Generate a charging station power profile.

        Args:
            station_type: 'ac_slow' (7kW), 'dc_fast' (50kW), 'dc_ultra' (150kW+)
            duration_hours: duration of the charging session

        Returns:
            DataFrame with 'current', 'voltage', 'duration_h', 'soc' columns
        """
        if station_type == "ac_slow":
            power_kw = 7.0
        elif station_type == "dc_fast":
            power_kw = 50.0
        elif station_type == "dc_ultra":
            power_kw = 150.0
        else:
            raise ValueError(f"Unknown station type: {station_type}")

        current_a = (power_kw * 1000) / self.nominal_voltage
        n_samples = int(duration_hours * 60)  # 1-minute resolution
        time_step_h = 1.0 / 60

        soc = np.linspace(20.0, 95.0, n_samples)
        voltage = np.full(n_samples, self.nominal_voltage)
        current = np.full(n_samples, current_a)

        return pd.DataFrame(
            {
                "current": current,
                "voltage": voltage,
                "duration_h": np.full(n_samples, time_step_h),
                "soc": soc,
            }
        )

    def generate_v2s_dispatch(
        self, grid_signal: str, duration_hours: int = 24
    ) -> pd.DataFrame:
        """
        Generate a V2S dispatch profile based on grid signal.

        Args:
            grid_signal: 'peak_shaving', 'frequency_regulation', 'solar_buffering'
            duration_hours: duration of the dispatch

        Returns:
            DataFrame with 'current', 'voltage', 'duration_h', 'soc' columns
        """
        rng = np.random.default_rng(42)
        time_step_h = 1.0 / 60  # 1-minute resolution
        n_samples = int(duration_hours * 60)

        if grid_signal == "peak_shaving":
            current = np.zeros(n_samples)
            soc = np.linspace(80.0, 30.0, n_samples)  # discharge from 80% to 30%
            for i in range(n_samples):
                hour = (i * time_step_h) % 24
                if 16 <= hour <= 20:
                    current[i] = -rng.uniform(50, 100)
                elif 0 <= hour <= 5:
                    current[i] = rng.uniform(30, 60)
        elif grid_signal == "frequency_regulation":
            current = rng.uniform(-30, 30, n_samples)
            soc = np.full(n_samples, 50.0)  # maintain mid SOC
        elif grid_signal == "solar_buffering":
            current = np.zeros(n_samples)
            soc = np.linspace(40.0, 90.0, n_samples)  # charge from solar
            for i in range(n_samples):
                hour = (i * time_step_h) % 24
                if 8 <= hour <= 16:
                    current[i] = rng.uniform(20, 40)
        else:
            raise ValueError(f"Unknown grid signal: {grid_signal}")

        voltage = np.full(n_samples, self.nominal_voltage)

        return pd.DataFrame(
            {
                "current": current,
                "voltage": voltage,
                "duration_h": np.full(n_samples, time_step_h),
                "soc": soc,
            }
        )


class ChargingStationSimulator:
    """Simulate CC-CV Li-ion charging at a station."""

    def __init__(self, battery_capacity_ah: float = 100.0, nominal_voltage: float = 400.0):
        self.battery_capacity_ah = battery_capacity_ah
        self.nominal_voltage = nominal_voltage

    def simulate_charging_session(
        self,
        station_power_kw: float,
        initial_soc: float,
        target_soc: float = 95.0,
        cell_count: int = 96,
    ) -> pd.DataFrame:
        """
        Simulate a CC-CV charging session.

        Args:
            station_power_kw: charger power in kW
            initial_soc: starting SOC in percent
            target_soc: target SOC in percent
            cell_count: number of cells in series

        Returns:
            DataFrame with timestamps, current, voltage, soc, energy_delivered_kwh
        """
        if target_soc < initial_soc:
            raise ValueError("target_soc must be >= initial_soc")

        cell_nominal_voltage = self.nominal_voltage / cell_count
        cv_voltage_per_cell = cell_nominal_voltage * 1.05  # CV at 105% of nominal
        max_current_a = (station_power_kw * 1000) / self.nominal_voltage

        # CC-CV parameters
        cc_fraction = 0.8  # 80% of charge in CC phase
        soc_cc_end = initial_soc + (target_soc - initial_soc) * cc_fraction

        soc = initial_soc
        voltage = cell_nominal_voltage * cell_count * 0.9  # start at 90% of nominal
        current = max_current_a
        timestamp = 0.0
        dt = 0.01  # 1-minute time step in hours

        results = []

        while soc < target_soc and timestamp < 100:  # safety limit
            if soc < soc_cc_end:
                # Constant Current phase
                current = max_current_a
                voltage = min(voltage + 0.05 * dt, cv_voltage_per_cell * cell_count)
            else:
                # Constant Voltage phase
                voltage = cv_voltage_per_cell * cell_count
                current = max(current * 0.995, 0.1)  # taper current

            # Update SOC
            soc += (current * dt / self.battery_capacity_ah) * 100

            # Calculate energy delivered
            energy_kwh = voltage * current * dt / 1000

            results.append(
                {
                    "timestamp": timestamp,
                    "current": current,
                    "voltage": voltage,
                    "soc": soc,
                    "energy_delivered_kwh": energy_kwh,
                }
            )
            timestamp += dt

        return pd.DataFrame(results)


__all__ = [
    "V2GScenarioGenerator",
    "V2GHealthAnalyzer",
    "V2SScenarioGenerator",
    "ChargingStationSimulator",
]
