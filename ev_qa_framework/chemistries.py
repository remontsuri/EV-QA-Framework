"""
EV-QA-Framework Battery Chemistry Profiles Module

Defines per-chemistry battery parameters: voltage ranges, temperature limits,
SOH degradation curves, cell imbalance thresholds, and safety limits.

Chemistry-specific models:
  - OCV (Open Circuit Voltage) curves as a function of SOC
  - Aging models: calendar aging + cycle aging with temperature dependence
  - Thermal models: heat capacity, thermal conductivity, Arrhenius aging

Supports:
  - LFP (LiFePO₄) — BYD Blade, CATL cells
  - NMC (LiNiMnCoO₂) — standard Li-ion, pouch/prismatic
  - NCA (LiNiCoAlO₂) — Tesla-style cylindrical cells
  - Custom profiles via JSON
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Supported chemistry identifiers
# ---------------------------------------------------------------------------
ChemistryKey = Literal["lfp", "nmc", "nca"]

ALL_CHEMISTRIES: list[ChemistryKey] = ["lfp", "nmc", "nca"]


# ---------------------------------------------------------------------------
# SOH / degradation parameters
# ---------------------------------------------------------------------------
@dataclass
class SOHDegradationParams:
    """Parameters describing battery capacity fade over time and cycles.

    Attributes:
        cycle_life_min: Minimum expected full cycles to 80 % SOH.
        cycle_life_max: Expected maximum full cycles to 80 % SOH.
        deg_knee_point_soh: SOH % at which degradation accelerates noticeably.
        annual_fade_rate_pct: Calendar capacity fade per year at 25 °C / 50 % SOC.
        calendric_fade_pct_per_year: Additional calendar aging per year at
            elevated temperature (40 °C / 100 % SOC stress).
    """

    cycle_life_min: int = 1000
    cycle_life_max: int = 2000
    deg_knee_point_soh: float = 80.0
    annual_fade_rate_pct: float = 2.0
    calendric_fade_pct_per_year: float = 4.0


# ---------------------------------------------------------------------------
# Cell imbalance thresholds
# ---------------------------------------------------------------------------
@dataclass
class CellImbalanceThresholds:
    """Voltage imbalance limits for cell balancing decisions.

    Attributes:
        max_delta_mv: Maximum allowed voltage delta between any two cells (mV).
        warning_delta_mv: Delta above which a warning is raised (mV).
        recovery_rate_mv_per_h: Typical balancing recovery speed (mV/hour).
    """

    max_delta_mv: float = 100.0
    warning_delta_mv: float = 50.0
    recovery_rate_mv_per_h: float = 10.0


# ---------------------------------------------------------------------------
# Thermal parameters
# ---------------------------------------------------------------------------
@dataclass
class ThermalParams:
    """Chemistry-specific thermal characteristics.

    Attributes:
        specific_heat_capacity: Specific heat capacity in J/(kg·K).
        thermal_conductivity: Thermal conductivity in W/(m·K).
        density: Density in kg/m³.
        arrhenius_activation_energy: Activation energy for aging (J/mol).
        heat_generation_coeff: Coefficient for I²R heat generation (W/A²).
        max_heat_dissipation: Maximum heat dissipation rate in W.
        self_heating_rate: Self-heating rate at 1C discharge in K/min.
    """

    specific_heat_capacity: float = 1000.0
    thermal_conductivity: float = 1.5
    density: float = 2500.0
    arrhenius_activation_energy: float = 50000.0
    heat_generation_coeff: float = 0.5
    max_heat_dissipation: float = 50.0
    self_heating_rate: float = 0.5


# ---------------------------------------------------------------------------
# OCV curve
# ---------------------------------------------------------------------------
@dataclass
class OCVCurve:
    """Open Circuit Voltage curve parameterisation.

    Stores OCV as a function of SOC (0-100 %) using empirical data points
    and provides interpolation for arbitrary SOC values.

    Chemistry-specific shapes:
      - LFP: very flat plateau around 3.2V, steep drops at extremes
      - NMC: moderate slope, good SOC-OCV correlation
      - NCA: similar to NMC but slightly different shape

    Attributes:
        soc_points: SOC values (0-100) at which OCV is measured.
        ocv_points: Corresponding OCV values in Volts.
        name: Human-readable name for this curve.
    """

    soc_points: list[float] = field(default_factory=list)
    ocv_points: list[float] = field(default_factory=list)
    name: str = "Generic OCV"

    def get_ocv(self, soc: float) -> float:
        """Get OCV at a given SOC using linear interpolation.

        Args:
            soc: State of Charge (0-100 %).

        Returns:
            Open Circuit Voltage in Volts.
        """
        if not self.soc_points or not self.ocv_points:
            return 3.7
        soc_clamped = max(self.soc_points[0], min(self.soc_points[-1], soc))
        return float(np.interp(soc_clamped, self.soc_points, self.ocv_points))

    def get_ocv_array(self, soc_array: np.ndarray) -> np.ndarray:
        """Get OCV for an array of SOC values.

        Args:
            soc_array: Array of SOC values (0-100).

        Returns:
            Array of OCV values in Volts.
        """
        if not self.soc_points or not self.ocv_points:
            return np.full_like(soc_array, 3.7)
        return np.interp(soc_array, self.soc_points, self.ocv_points)

    def get_soc_from_ocv(self, ocv: float) -> float:
        """Estimate SOC from OCV (inverse lookup).

        Note: For LFP, this is unreliable in the flat plateau region
        (SOC 20-80 %) because OCV barely changes.

        Args:
            ocv: Open Circuit Voltage in Volts.

        Returns:
            Estimated SOC (0-100 %).
        """
        if not self.soc_points or not self.ocv_points:
            return 50.0
        ocv_clamped = max(self.ocv_points[0], min(self.ocv_points[-1], ocv))
        return float(np.interp(ocv_clamped, self.ocv_points, self.soc_points))

    def to_dict(self) -> dict:
        return {
            "soc_points": self.soc_points,
            "ocv_points": self.ocv_points,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OCVCurve:
        return cls(
            soc_points=data.get("soc_points", []),
            ocv_points=data.get("ocv_points", []),
            name=data.get("name", "Generic OCV"),
        )


# ---------------------------------------------------------------------------
# Aging model
# ---------------------------------------------------------------------------
@dataclass
class AgingModel:
    """Battery aging model combining calendar and cycle aging.

    Calendar aging: dQ_cal = A_cal * exp(-Ea_cal / (R * T)) * t^0.5 * SOC_stress
    Cycle aging:    dQ_cyc = A_cyc * exp(-Ea_cyc / (R * T)) * N^0.5 * stress

    References:
        - Schmalstieg et al. (2014) — NMC aging model
        - Wang et al. (2011) — LFP aging model
        - NCA: adapted from NMC with higher rate constants

    Attributes:
        arrhenius_preexponential_calendar: Pre-exponential factor for calendar aging.
        activation_energy_calendar: Activation energy for calendar aging (J/mol).
        arrhenius_preexponential_cycle: Pre-exponential factor for cycle aging.
        activation_energy_cycle: Activation energy for cycle aging (J/mol).
        knee_point_soh: SOH at which degradation rate increases.
        knee_factor: Multiplier for degradation rate below knee point.
        calendar_life_years: Expected calendar life at 25°C/50% SOC (years).
        cycle_life_80: Number of full cycles to reach 80% SOH.
    """

    arrhenius_preexponential_calendar: float = 1.0e6
    activation_energy_calendar: float = 50000.0
    arrhenius_preexponential_cycle: float = 500.0
    activation_energy_cycle: float = 30000.0
    knee_point_soh: float = 80.0
    knee_factor: float = 2.0
    calendar_life_years: float = 15.0
    cycle_life_80: int = 1500

    R: float = 8.314  # Gas constant J/(mol·K)

    def calendar_aging_rate(self, temperature_c: float, soc_pct: float = 50.0) -> float:
        """Compute calendar aging rate (% capacity loss per year).

        Uses Arrhenius equation with SOC stress factor.

        Args:
            temperature_c: Temperature in Celsius.
            soc_pct: State of Charge (0-100 %). Higher SOC = faster aging.

        Returns:
            Capacity loss per year in %.
        """
        T_kelvin = temperature_c + 273.15
        T_ref = 298.15  # 25°C reference
        arrhenius_factor = math.exp(
            -self.activation_energy_calendar / self.R * (1.0 / T_kelvin - 1.0 / T_ref)
        )
        soc_stress = 1.0 + 0.02 * (soc_pct - 50.0)
        base_rate = 20.0 / self.calendar_life_years
        return base_rate * arrhenius_factor * soc_stress

    def cycle_aging_rate(
        self,
        temperature_c: float,
        c_rate: float = 1.0,
        dod_pct: float = 80.0,
    ) -> float:
        """Compute cycle aging rate (% capacity loss per equivalent full cycle).

        Args:
            temperature_c: Temperature in Celsius.
            c_rate: Charge/discharge C-rate.
            dod_pct: Depth of Discharge (0-100 %).

        Returns:
            Capacity loss per equivalent full cycle in %.
        """
        T_kelvin = temperature_c + 273.15
        T_ref = 298.15
        arrhenius_factor = math.exp(
            -self.activation_energy_cycle / self.R * (1.0 / T_kelvin - 1.0 / T_ref)
        )
        c_stress = c_rate**1.5
        dod_stress = (dod_pct / 80.0) ** 1.3
        base_rate = 20.0 / self.cycle_life_80
        return base_rate * arrhenius_factor * c_stress * dod_stress

    def predict_soh(
        self,
        initial_soh: float = 100.0,
        years: float = 1.0,
        cycles_per_year: int = 300,
        temperature_c: float = 25.0,
        soc_pct: float = 50.0,
        c_rate: float = 1.0,
        dod_pct: float = 80.0,
    ) -> float:
        """Predict SOH after given time and cycling conditions.

        Args:
            initial_soh: Starting SOH (%).
            years: Number of years.
            cycles_per_year: Equivalent full cycles per year.
            temperature_c: Average temperature (°C).
            soc_pct: Average SOC during storage (%).
            c_rate: Average C-rate during cycling.
            dod_pct: Average depth of discharge (%).

        Returns:
            Predicted SOH (%).
        """
        soh = initial_soh
        cal_rate = self.calendar_aging_rate(temperature_c, soc_pct)
        cyc_rate = self.cycle_aging_rate(temperature_c, c_rate, dod_pct)

        for _ in range(int(years)):
            cal_loss = cal_rate
            cyc_loss = cyc_rate * cycles_per_year
            if soh < self.knee_point_soh:
                cal_loss *= self.knee_factor
                cyc_loss *= self.knee_factor
            soh -= cal_loss + cyc_loss

        frac = years - int(years)
        if frac > 0:
            cal_loss = cal_rate * frac
            cyc_loss = cyc_rate * cycles_per_year * frac
            if soh < self.knee_point_soh:
                cal_loss *= self.knee_factor
                cyc_loss *= self.knee_factor
            soh -= cal_loss + cyc_loss

        return max(0.0, soh)

    def to_dict(self) -> dict:
        return {
            "arrhenius_preexponential_calendar": self.arrhenius_preexponential_calendar,
            "activation_energy_calendar": self.activation_energy_calendar,
            "arrhenius_preexponential_cycle": self.arrhenius_preexponential_cycle,
            "activation_energy_cycle": self.activation_energy_cycle,
            "knee_point_soh": self.knee_point_soh,
            "knee_factor": self.knee_factor,
            "calendar_life_years": self.calendar_life_years,
            "cycle_life_80": self.cycle_life_80,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgingModel:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------
@dataclass
class ThermalModel:
    """Simplified thermal model for battery cells.

    Models temperature evolution:
        dT/dt = (I²R - h * (T - T_ambient)) / (m * Cp)

    Attributes:
        thermal_params: Thermal parameters for this chemistry.
        cell_mass_kg: Mass of a single cell in kg.
        heat_transfer_coeff: Convective heat transfer coefficient (W/K).
        reference_resistance: Internal resistance at 25°C (Ohm).
        resistance_temp_coeff: Temperature coefficient of resistance (1/K).
    """

    thermal_params: ThermalParams = field(default_factory=ThermalParams)
    cell_mass_kg: float = 0.045
    heat_transfer_coeff: float = 0.01
    reference_resistance: float = 0.05
    resistance_temp_coeff: float = -0.002

    def get_resistance(self, temperature_c: float) -> float:
        """Get temperature-dependent internal resistance.

        Args:
            temperature_c: Temperature in Celsius.

        Returns:
            Internal resistance in Ohms.
        """
        return self.reference_resistance * (
            1.0 + self.resistance_temp_coeff * (temperature_c - 25.0)
        )

    def compute_temperature(
        self,
        current: float,
        temperature: float,
        ambient_temperature: float,
        dt_seconds: float,
        internal_resistance: float | None = None,
    ) -> float:
        """Compute cell temperature after one time step.

        Args:
            current: Current in Amps (positive=charge, negative=discharge).
            temperature: Current cell temperature (°C).
            ambient_temperature: Ambient temperature (°C).
            dt_seconds: Time step in seconds.
            internal_resistance: Internal resistance (Ohm). If None, uses
                temperature-dependent model.

        Returns:
            New cell temperature (°C).
        """
        if internal_resistance is None:
            internal_resistance = self.get_resistance(temperature)

        heat_gen = current**2 * internal_resistance
        heat_diss = self.heat_transfer_coeff * (temperature - ambient_temperature)
        net_heat = heat_gen - heat_diss
        mass = self.cell_mass_kg
        cp = self.thermal_params.specific_heat_capacity
        delta_t = (net_heat * dt_seconds) / (mass * cp)
        return temperature + delta_t

    def simulate_thermal(
        self,
        current_profile: list[float],
        initial_temperature: float,
        ambient_temperature: float,
        dt_seconds: float = 1.0,
    ) -> list[float]:
        """Simulate temperature over a current profile.

        Args:
            current_profile: List of current values (Amps) at each time step.
            initial_temperature: Starting temperature (°C).
            ambient_temperature: Ambient temperature (°C).
            dt_seconds: Time step in seconds.

        Returns:
            List of temperatures at each time step.
        """
        temperatures = [initial_temperature]
        temp = initial_temperature
        for current in current_profile:
            temp = self.compute_temperature(current, temp, ambient_temperature, dt_seconds)
            temperatures.append(temp)
        return temperatures

    def to_dict(self) -> dict:
        return {
            "thermal_params": {
                "specific_heat_capacity": self.thermal_params.specific_heat_capacity,
                "thermal_conductivity": self.thermal_params.thermal_conductivity,
                "density": self.thermal_params.density,
                "arrhenius_activation_energy": self.thermal_params.arrhenius_activation_energy,
                "heat_generation_coeff": self.thermal_params.heat_generation_coeff,
                "max_heat_dissipation": self.thermal_params.max_heat_dissipation,
                "self_heating_rate": self.thermal_params.self_heating_rate,
            },
            "cell_mass_kg": self.cell_mass_kg,
            "heat_transfer_coeff": self.heat_transfer_coeff,
            "reference_resistance": self.reference_resistance,
            "resistance_temp_coeff": self.resistance_temp_coeff,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThermalModel:
        tp = data.get("thermal_params", {})
        thermal_params = ThermalParams(
            specific_heat_capacity=tp.get("specific_heat_capacity", 1000.0),
            thermal_conductivity=tp.get("thermal_conductivity", 1.5),
            density=tp.get("density", 2500.0),
            arrhenius_activation_energy=tp.get("arrhenius_activation_energy", 50000.0),
            heat_generation_coeff=tp.get("heat_generation_coeff", 0.5),
            max_heat_dissipation=tp.get("max_heat_dissipation", 50.0),
            self_heating_rate=tp.get("self_heating_rate", 0.5),
        )
        return cls(
            thermal_params=thermal_params,
            cell_mass_kg=data.get("cell_mass_kg", 0.045),
            heat_transfer_coeff=data.get("heat_transfer_coeff", 0.01),
            reference_resistance=data.get("reference_resistance", 0.05),
            resistance_temp_coeff=data.get("resistance_temp_coeff", -0.002),
        )


# ===================================================================
# Built-in OCV curves (published datasheet data)
# ===================================================================

# LFP OCV curve — very flat plateau around 3.2V
# Data from: Wang et al. (2011), CATL LFP datasheet
OCV_LFP = OCVCurve(
    name="LFP (LiFePO₄) OCV",
    soc_points=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100],
    ocv_points=[
        2.50,   # FIX: was 2.00 — LFP min OCV at 0% SOC is ~2.5V, not 2.0V
        2.80,
        3.00,
        3.10,
        3.15,
        3.20,
        3.22,
        3.23,
        3.24,
        3.25,
        3.26,
        3.28,
        3.32,
        3.45,   # FIX: was 3.40 — smoother transition to 3.65
        3.65,
    ],
)

# NMC OCV curve — moderate slope, good SOC correlation
# Data from: Schmalstieg et al. (2014), LG Chem NMC datasheet
OCV_NMC = OCVCurve(
    name="NMC (LiNiMnCoO₂) OCV",
    soc_points=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100],
    ocv_points=[
        3.00,   # FIX: was 2.70 — NMC min OCV at 0% SOC is ~3.0V
        3.20,
        3.35,
        3.45,
        3.55,
        3.62,
        3.68,
        3.74,
        3.80,
        3.88,
        3.92,
        3.98,
        4.08,
        4.15,   # FIX: added missing point — NMC OCV at 95% SOC
        4.20,
    ],
)

# NCA OCV curve — similar to NMC but slightly different
# Data from: Panasonic NCA datasheet, Tesla 2170 cell
OCV_NCA = OCVCurve(
    name="NCA (LiNiCoAlO₂) OCV",
    soc_points=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100],
    ocv_points=[
        3.00,   # FIX: was 2.50 — NCA min OCV at 0% SOC is ~3.0V
        3.20,   # FIX: was 2.90 — NCA OCV at 5% SOC is ~3.2-3.3V
        3.30,
        3.42,
        3.52,
        3.60,
        3.67,
        3.73,
        3.80,
        3.87,
        3.92,
        3.98,
        4.08,
        4.15,   # FIX: added missing point — NCA OCV at 95% SOC
        4.20,
    ],
)


# ===================================================================
# Built-in aging models
# ===================================================================

AGING_LFP = AgingModel(
    arrhenius_preexponential_calendar=5.0e5,
    activation_energy_calendar=45000.0,
    arrhenius_preexponential_cycle=300.0,
    activation_energy_cycle=25000.0,
    knee_point_soh=80.0,
    knee_factor=1.5,
    calendar_life_years=20.0,
    cycle_life_80=3000,
)

AGING_NMC = AgingModel(
    arrhenius_preexponential_calendar=1.0e6,
    activation_energy_calendar=50000.0,
    arrhenius_preexponential_cycle=500.0,
    activation_energy_cycle=30000.0,
    knee_point_soh=80.0,
    knee_factor=2.0,
    calendar_life_years=15.0,
    cycle_life_80=1500,
)

AGING_NCA = AgingModel(
    arrhenius_preexponential_calendar=2.0e6,
    activation_energy_calendar=55000.0,
    arrhenius_preexponential_cycle=800.0,
    activation_energy_cycle=35000.0,
    knee_point_soh=80.0,
    knee_factor=2.5,
    calendar_life_years=12.0,
    cycle_life_80=1000,
)


# ===================================================================
# Built-in thermal models
# ===================================================================

THERMAL_LFP = ThermalModel(
    thermal_params=ThermalParams(
        specific_heat_capacity=1100.0,
        thermal_conductivity=1.8,
        density=2300.0,
        arrhenius_activation_energy=45000.0,
        heat_generation_coeff=0.4,
        max_heat_dissipation=60.0,
        self_heating_rate=0.3,
    ),
    cell_mass_kg=0.048,
    heat_transfer_coeff=0.012,
    reference_resistance=0.06,
    resistance_temp_coeff=-0.0015,
)

THERMAL_NMC = ThermalModel(
    thermal_params=ThermalParams(
        specific_heat_capacity=1000.0,
        thermal_conductivity=1.5,
        density=2500.0,
        arrhenius_activation_energy=50000.0,
        heat_generation_coeff=0.5,
        max_heat_dissipation=50.0,
        self_heating_rate=0.5,
    ),
    cell_mass_kg=0.050,
    heat_transfer_coeff=0.011,
    reference_resistance=0.045,
    resistance_temp_coeff=-0.002,
)

THERMAL_NCA = ThermalModel(
    thermal_params=ThermalParams(
        specific_heat_capacity=950.0,
        thermal_conductivity=1.3,
        density=2600.0,
        arrhenius_activation_energy=55000.0,
        heat_generation_coeff=0.6,
        max_heat_dissipation=45.0,
        self_heating_rate=0.7,
    ),
    cell_mass_kg=0.047,
    heat_transfer_coeff=0.009,
    reference_resistance=0.055,
    resistance_temp_coeff=-0.0025,
)


# ===================================================================
# Main chemistry profile
# ===================================================================
@dataclass
class BatteryChemistryProfile:
    """Complete set of parameters characterising a specific battery chemistry.

    Per-cell values can be scaled to pack level via *cells_in_series*.
    Includes chemistry-specific OCV curves, aging models, and thermal models.
    """

    # --- Identity ---
    name: str = "Generic Li-ion"
    short_name: ChemistryKey = "nmc"
    manufacturer: str = ""
    description: str = ""

    # --- Per-cell electrical ---
    cell_nominal_voltage: float = 3.7
    cell_min_voltage: float = 3.0
    cell_max_voltage: float = 4.2
    cell_max_charge_current: float = 1.0
    cell_max_discharge_current: float = 3.0

    # --- Temperature limits ---
    temp_min_charge: float = 0.0
    temp_max_charge: float = 45.0
    temp_min_discharge: float = -20.0
    temp_max_discharge: float = 60.0
    temp_max_jump: float = 5.0
    temp_optimal_min: float = 15.0
    temp_optimal_max: float = 35.0

    # --- Safety limits (per-cell) ---
    overcharge_voltage: float = 4.25
    overdischarge_voltage: float = 2.7
    thermal_runaway_temp: float = 80.0
    gas_venting_temp: float = 70.0

    # --- SOH / degradation ---
    soh_params: SOHDegradationParams = field(default_factory=SOHDegradationParams)

    # --- Cell balance ---
    balance: CellImbalanceThresholds = field(default_factory=CellImbalanceThresholds)

    # --- SOC limits ---
    min_soc_pct: float = 10.0
    max_soc_pct: float = 95.0
    critical_soh_pct: float = 70.0

    # --- Chemistry-specific models ---
    ocv_curve: OCVCurve = field(default_factory=OCVCurve)
    aging_model: AgingModel = field(default_factory=AgingModel)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)

    # ------------------------------------------------------------------
    # Chemistry-aware methods
    # ------------------------------------------------------------------
    def get_ocv(self, soc: float) -> float:
        """Get Open Circuit Voltage at given SOC.

        Args:
            soc: State of Charge (0-100 %).

        Returns:
            Open Circuit Voltage in Volts.
        """
        return self.ocv_curve.get_ocv(soc)

    def get_soc_from_ocv(self, ocv: float) -> float:
        """Estimate SOC from OCV.

        Args:
            ocv: Open Circuit Voltage in Volts.

        Returns:
            Estimated SOC (0-100 %).
        """
        return self.ocv_curve.get_soc_from_ocv(ocv)

    def predict_soh(
        self,
        years: float = 1.0,
        cycles_per_year: int = 300,
        temperature_c: float = 25.0,
        soc_pct: float = 50.0,
        c_rate: float = 1.0,
        dod_pct: float = 80.0,
        initial_soh: float = 100.0,
    ) -> float:
        """Predict SOH after given time and cycling conditions.

        Args:
            years: Number of years.
            cycles_per_year: Equivalent full cycles per year.
            temperature_c: Average temperature (°C).
            soc_pct: Average SOC during storage (%).
            c_rate: Average C-rate during cycling.
            dod_pct: Average depth of discharge (%).
            initial_soh: Starting SOH (%).

        Returns:
            Predicted SOH (%).
        """
        return self.aging_model.predict_soh(
            initial_soh=initial_soh,
            years=years,
            cycles_per_year=cycles_per_year,
            temperature_c=temperature_c,
            soc_pct=soc_pct,
            c_rate=c_rate,
            dod_pct=dod_pct,
        )

    def compute_cell_temperature(
        self,
        current: float,
        temperature: float,
        ambient_temperature: float,
        dt_seconds: float,
    ) -> float:
        """Compute cell temperature after one time step.

        Args:
            current: Current in Amps.
            temperature: Current cell temperature (°C).
            ambient_temperature: Ambient temperature (°C).
            dt_seconds: Time step in seconds.

        Returns:
            New cell temperature (°C).
        """
        return self.thermal_model.compute_temperature(
            current, temperature, ambient_temperature, dt_seconds
        )

    # ------------------------------------------------------------------
    # Pack-level helpers
    # ------------------------------------------------------------------
    def pack_min_voltage(self, cells_in_series: int = 96) -> float:
        """Compute minimum pack voltage."""
        return self.cell_min_voltage * cells_in_series

    def pack_max_voltage(self, cells_in_series: int = 96) -> float:
        """Compute maximum pack voltage."""
        return self.cell_max_voltage * cells_in_series

    def pack_nominal_voltage(self, cells_in_series: int = 96) -> float:
        """Compute nominal pack voltage."""
        return self.cell_nominal_voltage * cells_in_series

    def pack_ocv(self, soc: float, cells_in_series: int = 96) -> float:
        """Compute pack OCV at given SOC.

        Args:
            soc: State of Charge (0-100 %).
            cells_in_series: Number of cells in series.

        Returns:
            Pack Open Circuit Voltage in Volts.
        """
        return self.get_ocv(soc) * cells_in_series

    def to_safety_thresholds_dict(
        self,
        cells_in_series: int = 96,
        cells_parallel: int = 1,
    ) -> dict:
        """Convert this chemistry profile into a dict compatible with
        ``SafetyThresholds``, scaled to *cells_in_series* and *cells_parallel*.

        Returns a flat dict with keys expected by SafetyThresholds:
            max_temperature, min_temperature, max_temperature_jump,
            min_voltage, max_voltage, min_soc, critical_soh, max_current
        """
        return {
            "max_temperature": self.temp_max_charge,
            "min_temperature": self.temp_min_discharge,
            "max_temperature_jump": self.temp_max_jump,
            "min_voltage": self.pack_min_voltage(cells_in_series),
            "max_voltage": self.pack_max_voltage(cells_in_series),
            "min_soc": self.min_soc_pct,
            "critical_soh": self.critical_soh_pct,
            "max_current": self.cell_max_discharge_current * cells_parallel,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["ocv_curve"] = self.ocv_curve.to_dict()
        d["aging_model"] = self.aging_model.to_dict()
        d["thermal_model"] = self.thermal_model.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> BatteryChemistryProfile:
        soh = SOHDegradationParams(**data.pop("soh_params", {}))
        bal = CellImbalanceThresholds(**data.pop("balance", {}))
        ocv_data = data.pop("ocv_curve", {})
        ocv = OCVCurve.from_dict(ocv_data) if ocv_data else OCVCurve()
        aging_data = data.pop("aging_model", {})
        aging = AgingModel.from_dict(aging_data) if aging_data else AgingModel()
        thermal_data = data.pop("thermal_model", {})
        thermal = ThermalModel.from_dict(thermal_data) if thermal_data else ThermalModel()
        return cls(
            soh_params=soh,
            balance=bal,
            ocv_curve=ocv,
            aging_model=aging,
            thermal_model=thermal,
            **data,
        )

    @classmethod
    def from_json(cls, text: str) -> BatteryChemistryProfile:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_from_file(cls, path: str) -> BatteryChemistryProfile:
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())


# ===================================================================
# Built-in profiles
# ===================================================================

PROFILE_LFP_CATL = BatteryChemistryProfile(
    name="LFP (CATL / BYD Blade)",
    short_name="lfp",
    manufacturer="CATL / BYD",
    description=(
        "Lithium Iron Phosphate — long cycle life, excellent thermal stability. "
        "Used in BYD Blade, CATL prismatic cells. Lower energy density but safer."
    ),
    cell_nominal_voltage=3.2,
    cell_min_voltage=2.5,
    cell_max_voltage=3.65,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    temp_min_charge=-10.0,
    temp_max_charge=55.0,
    temp_min_discharge=-30.0,
    temp_max_discharge=65.0,
    temp_max_jump=8.0,
    temp_optimal_min=10.0,
    temp_optimal_max=40.0,
    overcharge_voltage=3.8,
    overdischarge_voltage=2.3,
    thermal_runaway_temp=250.0,  # FIX: was 120.0 — LFP thermal runaway onset ~200-250°C
    gas_venting_temp=130.0,      # FIX: was 90.0 — LFP gas venting onset ~130°C
    soh_params=SOHDegradationParams(
        cycle_life_min=2000,
        cycle_life_max=5000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=1.5,
        calendric_fade_pct_per_year=2.5,
    ),
    balance=CellImbalanceThresholds(
        max_delta_mv=150.0,
        warning_delta_mv=80.0,
        recovery_rate_mv_per_h=8.0,
    ),
    min_soc_pct=10.0,
    max_soc_pct=95.0,
    critical_soh_pct=65.0,
    ocv_curve=OCV_LFP,
    aging_model=AGING_LFP,
    thermal_model=THERMAL_LFP,
)

PROFILE_NMC_STANDARD = BatteryChemistryProfile(
    name="NMC (Standard Li-ion)",
    short_name="nmc",
    manufacturer="LG / Samsung SDI / Panasonic",
    description=(
        "Lithium Nickel Manganese Cobalt Oxide — high energy density, "
        "good power. Most common EV chemistry. Used in LG Chem, Samsung SDI packs."
    ),
    cell_nominal_voltage=3.7,
    cell_min_voltage=2.5,       # FIX: was 3.0 — NMC/NCA min is 2.5V
    cell_max_voltage=4.2,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    temp_min_charge=0.0,
    temp_max_charge=45.0,
    temp_min_discharge=-20.0,
    temp_max_discharge=60.0,
    temp_max_jump=5.0,
    temp_optimal_min=15.0,
    temp_optimal_max=35.0,
    overcharge_voltage=4.25,
    overdischarge_voltage=2.5,       # FIX: was 2.7 — NMC overdischarge threshold is 2.5V
    thermal_runaway_temp=150.0,      # FIX: was 80.0 — NMC thermal runaway onset ~150°C
    gas_venting_temp=120.0,          # FIX: was 70.0 — NMC gas venting onset ~120°C
    soh_params=SOHDegradationParams(
        cycle_life_min=1000,
        cycle_life_max=2000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=2.0,
        calendric_fade_pct_per_year=4.0,
    ),
    balance=CellImbalanceThresholds(
        max_delta_mv=100.0,
        warning_delta_mv=50.0,
        recovery_rate_mv_per_h=15.0,
    ),
    min_soc_pct=10.0,
    max_soc_pct=95.0,
    critical_soh_pct=70.0,
    ocv_curve=OCV_NMC,
    aging_model=AGING_NMC,
    thermal_model=THERMAL_NMC,
)

PROFILE_NCA_TESLA = BatteryChemistryProfile(
    name="NCA (Tesla-style)",
    short_name="nca",
    manufacturer="Panasonic / Tesla (4680, 2170)",
    description=(
        "Lithium Nickel Cobalt Aluminium Oxide — highest energy density, "
        "used by Tesla in 18650, 2170, and 4680 form factors. "
        "Slightly shorter cycle life than NMC."
    ),
    cell_nominal_voltage=3.6,
    cell_min_voltage=2.5,       # FIX: was 3.0 — NMC/NCA min is 2.5V
    cell_max_voltage=4.2,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    temp_min_charge=0.0,
    temp_max_charge=45.0,
    temp_min_discharge=-20.0,
    temp_max_discharge=55.0,
    temp_max_jump=5.0,
    temp_optimal_min=15.0,
    temp_optimal_max=35.0,
    overcharge_voltage=4.25,
    overdischarge_voltage=2.5,       # FIX: was 2.7 — NCA overdischarge threshold is 2.5V
    thermal_runaway_temp=140.0,      # FIX: was 75.0 — NCA thermal runaway onset ~140°C (lower than NMC)
    gas_venting_temp=120.0,          # FIX: was 65.0 — NCA gas venting onset ~120°C
    soh_params=SOHDegradationParams(
        cycle_life_min=500,
        cycle_life_max=1000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=2.5,
        calendric_fade_pct_per_year=5.0,
    ),
    balance=CellImbalanceThresholds(
        max_delta_mv=80.0,
        warning_delta_mv=40.0,
        recovery_rate_mv_per_h=12.0,
    ),
    min_soc_pct=15.0,
    max_soc_pct=90.0,
    critical_soh_pct=75.0,
    ocv_curve=OCV_NCA,
    aging_model=AGING_NCA,
    thermal_model=THERMAL_NCA,
)

# Registry
BUILTIN_PROFILES: dict[ChemistryKey, BatteryChemistryProfile] = {
    "lfp": PROFILE_LFP_CATL,
    "nmc": PROFILE_NMC_STANDARD,
    "nca": PROFILE_NCA_TESLA,
}


# ===================================================================
# Convenience accessors
# ===================================================================
def get_profile(key: ChemistryKey) -> BatteryChemistryProfile:
    """Return the built-in profile for *key*.

    Raises ``KeyError`` if *key* is not a recognised chemistry.
    """
    if key not in BUILTIN_PROFILES:
        valid = ", ".join(sorted(BUILTIN_PROFILES))
        raise KeyError(f"Unknown chemistry '{key}'. Valid options: {valid}")
    return BUILTIN_PROFILES[key]


def list_profiles() -> list[dict]:
    """Return metadata for all built-in profiles."""
    return [
        {
            "short_name": p.short_name,
            "name": p.name,
            "manufacturer": p.manufacturer,
            "cell_nominal_voltage": p.cell_nominal_voltage,
            "cycle_life": f"{p.soh_params.cycle_life_min}–{p.soh_params.cycle_life_max}",
        }
        for p in BUILTIN_PROFILES.values()
    ]


def register_custom_profile(profile: BatteryChemistryProfile) -> None:
    """Add or override a profile in the built-in registry."""
    BUILTIN_PROFILES[profile.short_name] = profile


def load_custom_profile_from_file(path: str) -> BatteryChemistryProfile:
    """Load a profile from a JSON file and register it."""
    profile = BatteryChemistryProfile.load_from_file(path)
    register_custom_profile(profile)
    return profile
