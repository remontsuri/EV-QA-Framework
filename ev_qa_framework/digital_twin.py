"""
Battery Digital Twin — virtual battery model for EV-QA-Framework.

Simulates battery behavior including:
- Charge/discharge cycles
- Capacity fade (linear + knee-point model)
- Resistance growth (exponential model)
- Thermal model (simplified)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import FrameworkConfig
from .physics_features import PhysicsFeatureExtractor


@dataclass
class BatteryState:
    """Current state of the battery digital twin."""
    voltage: float = 400.0
    current: float = 0.0
    temperature: float = 25.0
    soc: float = 80.0
    soh: float = 100.0
    cycle_count: float = 0.0
    capacity_ah: float = 100.0
    internal_resistance: float = 0.05  # Ohms
    ambient_temperature: float = 25.0

    def to_dict(self) -> dict:
        return {
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temperature,
            "soc": self.soc,
            "soh": self.soh,
            "cycle_count": self.cycle_count,
            "capacity_ah": self.capacity_ah,
            "internal_resistance": self.internal_resistance,
        }


class BatteryDigitalTwin:
    """
    Virtual battery model for simulation and prediction.

    Models:
    - Capacity fade: linear + knee-point degradation
    - Resistance growth: exponential with cycle count
    - Thermal: simplified dT = f(I²R, ambient)
    - SOC: Coulomb counting
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self.state = BatteryState()
        self.physics = PhysicsFeatureExtractor()
        self._history: list[dict] = []

        # Degradation model params
        self._fade_rate = 0.002  # % capacity loss per cycle
        self._knee_point = 80.0  # SOH where degradation accelerates
        self._knee_factor = 2.0  # multiplier after knee
        self._resistance_growth_rate = 0.001  # per cycle
        self._thermal_coeff = 0.5  # heating coefficient

    def reset(self):
        """Reset to initial state."""
        self.state = BatteryState()
        self._history = []

    def get_state(self) -> dict:
        """Return current state as dict."""
        return self.state.to_dict()

    def step(self, dt: float, current: float):
        """
        Simulate one time step.

        Args:
            dt: time step in hours
            current: current in Amps (positive=charge, negative=discharge)
        """
        s = self.state

        # SOC update (Coulomb counting)
        capacity_remaining = s.capacity_ah * s.soc / 100.0
        delta_ah = current * dt
        new_capacity = capacity_remaining + delta_ah
        s.soc = max(0.0, min(100.0, (new_capacity / s.capacity_ah) * 100.0))

        # Voltage (simplified OCV + IR drop)
        ocv = 300.0 + (s.soc / 100.0) * 100.0  # 300-400V range
        s.voltage = ocv - current * s.internal_resistance
        s.current = current

        # Thermal model: dT = I²R * coeff - cooling
        heat_generated = (current ** 2) * s.internal_resistance * self._thermal_coeff
        cooling = 0.1 * (s.temperature - s.ambient_temperature)
        s.temperature += (heat_generated - cooling) * dt

        # Cycle counting (equivalent full cycles)
        if abs(current) > 0:
            s.cycle_count += abs(current) * dt / s.capacity_ah / 2

        # Degradation
        self._update_degradation(dt)

        # Record history
        self._history.append(self.get_state())

    def _update_degradation(self, dt: float):
        """Update SOH and resistance based on degradation models."""
        s = self.state

        # Capacity fade
        if s.soh > self._knee_point:
            fade = self._fade_rate * dt
        else:
            fade = self._fade_rate * self._knee_factor * dt

        s.soh = max(0.0, s.soh - fade)
        s.capacity_ah = 100.0 * s.soh / 100.0

        # Resistance growth
        s.internal_resistance = 0.05 * math.exp(
            self._resistance_growth_rate * s.cycle_count
        )

    def simulate_drive_cycle(
        self,
        cycle_profile: pd.DataFrame,
        dt: float = 1.0,
    ) -> pd.DataFrame:
        """
        Simulate a drive cycle.

        Args:
            cycle_profile: DataFrame with 'current' column (Amps)
            dt: time step in hours

        Returns:
            DataFrame with state at each step
        """
        for _, row in cycle_profile.iterrows():
            self.step(dt, row["current"])

        return pd.DataFrame(self._history)

    def predict_soh(self, n_cycles: int, avg_current: float = 50.0) -> float:
        """
        Predict SOH after N equivalent full cycles.

        Args:
            n_cycles: number of equivalent full cycles
            avg_current: average current per cycle

        Returns:
            Predicted SOH
        """
        # Simulate without modifying state
        temp_twin = BatteryDigitalTwin(self.config)
        temp_twin.state = BatteryState(
            voltage=self.state.voltage,
            soc=self.state.soc,
            soh=self.state.soh,
            cycle_count=self.state.cycle_count,
            capacity_ah=self.state.capacity_ah,
            internal_resistance=self.state.internal_resistance,
            temperature=self.state.temperature,
        )

        dt = 1.0  # 1 hour steps
        for _ in range(n_cycles):
            # Simulate one full cycle: charge then discharge
            temp_twin.step(dt, avg_current)  # charge
            temp_twin.step(dt, -avg_current)  # discharge

        return temp_twin.state.soh

    def get_degradation_summary(self) -> dict:
        """Return degradation summary."""
        return {
            "current_soh": self.state.soh,
            "cycle_count": self.state.cycle_count,
            "capacity_remaining_ah": self.state.capacity_ah,
            "internal_resistance": self.state.internal_resistance,
            "estimated_cycles_to_80": self._estimate_cycles_to_soh(80.0),
            "estimated_cycles_to_70": self._estimate_cycles_to_soh(70.0),
        }

    def _estimate_cycles_to_soh(self, target_soh: float) -> Optional[float]:
        """Estimate remaining cycles to reach target SOH."""
        if self.state.soh <= target_soh:
            return 0.0

        predicted = self.predict_soh(1000)
        if predicted <= target_soh:
            # Binary search for exact cycle count
            lo, hi = 0, 10000
            while lo < hi:
                mid = (lo + hi) // 2
                p = self.predict_soh(mid)
                if p <= target_soh:
                    hi = mid
                else:
                    lo = mid + 1
            return float(lo)

        return None  # Won't reach target in 10000 cycles
