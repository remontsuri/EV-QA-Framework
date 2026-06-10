"""
EV-QA-Framework Battery Chemistry Profiles Module

Defines per-chemistry battery parameters: voltage ranges, temperature limits,
SOH degradation curves, cell imbalance thresholds, and safety limits.

Supports:
  - LFP (LiFePO₄) — BYD Blade, CATL cells
  - NMC (LiNiMnCoO₂) — standard Li-ion, pouch/prismatic
  - NCA (LiNiCoAlO₂) — Tesla-style cylindrical cells
  - Custom profiles via JSON
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Literal

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
        deg_knee_point_soh: SOH % at which degradation accelerates noticeably
            (e.g., LFP has a "knee" near 80 % SOH).
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
            Above this → critical failure.
        warning_delta_mv: Delta above which a warning is raised (mV).
        recovery_rate_mv_per_h: Typical balancing recovery speed (mV/hour).
    """

    max_delta_mv: float = 100.0
    warning_delta_mv: float = 50.0
    recovery_rate_mv_per_h: float = 10.0


# ---------------------------------------------------------------------------
# Main chemistry profile
# ---------------------------------------------------------------------------
@dataclass
class BatteryChemistryProfile:
    """Complete set of parameters characterising a specific battery chemistry.

    Per-cell values can be scaled to pack level via *cells_in_series*.
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
    cell_max_charge_current: float = 1.0  # C-rate
    cell_max_discharge_current: float = 3.0  # C-rate

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

    # ------------------------------------------------------------------
    # Pack-level helpers
    # ------------------------------------------------------------------
    def pack_min_voltage(self, cells_in_series: int = 96) -> float:
        """Compute minimum pack voltage for a given number of cells in series."""
        return self.cell_min_voltage * cells_in_series

    def pack_max_voltage(self, cells_in_series: int = 96) -> float:
        """Compute maximum pack voltage for a given number of cells in series."""
        return self.cell_max_voltage * cells_in_series

    def pack_nominal_voltage(self, cells_in_series: int = 96) -> float:
        """Compute nominal pack voltage."""
        return self.cell_nominal_voltage * cells_in_series

    def to_safety_thresholds_dict(
        self,
        cells_in_series: int = 96,
    ) -> dict:
        """Convert this chemistry profile into a dict compatible with
        ``SafetyThresholds``, scaled to *cells_in_series*.

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
            "max_current": self.cell_max_discharge_current
            * cells_in_series
            * 2.5,  # ~A estimate from C-rate × typical cell Ah
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

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
        return cls(soh_params=soh, balance=bal, **data)

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
    # Per-cell: LFP nominal 3.2 V, range 2.5–3.65 V
    cell_nominal_voltage=3.2,
    cell_min_voltage=2.5,
    cell_max_voltage=3.65,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    # Temperature — wide range, tolerant
    temp_min_charge=-10.0,
    temp_max_charge=55.0,
    temp_min_discharge=-30.0,
    temp_max_discharge=65.0,
    temp_max_jump=8.0,
    temp_optimal_min=10.0,
    temp_optimal_max=40.0,
    # Safety — very stable cathode, higher thermal runway threshold
    overcharge_voltage=3.8,
    overdischarge_voltage=2.3,
    thermal_runaway_temp=120.0,
    gas_venting_temp=90.0,
    # SOH — LFP has very long cycle life, knee near 80 %
    soh_params=SOHDegradationParams(
        cycle_life_min=2000,
        cycle_life_max=5000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=1.5,
        calendric_fade_pct_per_year=2.5,
    ),
    # Cell balance — LFP has flat OCV curve, harder to balance
    balance=CellImbalanceThresholds(
        max_delta_mv=150.0,
        warning_delta_mv=80.0,
        recovery_rate_mv_per_h=8.0,
    ),
    min_soc_pct=10.0,
    max_soc_pct=95.0,
    critical_soh_pct=65.0,
)

PROFILE_NMC_STANDARD = BatteryChemistryProfile(
    name="NMC (Standard Li-ion)",
    short_name="nmc",
    manufacturer="LG / Samsung SDI / Panasonic",
    description=(
        "Lithium Nickel Manganese Cobalt Oxide — high energy density, "
        "good power. Most common EV chemistry. Used in LG Chem, Samsung SDI packs."
    ),
    # Per-cell: NMC nominal 3.6–3.7 V, range 3.0–4.2 V
    cell_nominal_voltage=3.7,
    cell_min_voltage=3.0,
    cell_max_voltage=4.2,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    # Temperature
    temp_min_charge=0.0,
    temp_max_charge=45.0,
    temp_min_discharge=-20.0,
    temp_max_discharge=60.0,
    temp_max_jump=5.0,
    temp_optimal_min=15.0,
    temp_optimal_max=35.0,
    # Safety — lower thermal runway threshold than LFP
    overcharge_voltage=4.25,
    overdischarge_voltage=2.7,
    thermal_runaway_temp=80.0,
    gas_venting_temp=70.0,
    # SOH — 1000–2000 cycles typical
    soh_params=SOHDegradationParams(
        cycle_life_min=1000,
        cycle_life_max=2000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=2.0,
        calendric_fade_pct_per_year=4.0,
    ),
    # Cell balance — steeper OCV curve, easier to balance
    balance=CellImbalanceThresholds(
        max_delta_mv=100.0,
        warning_delta_mv=50.0,
        recovery_rate_mv_per_h=15.0,
    ),
    min_soc_pct=10.0,
    max_soc_pct=95.0,
    critical_soh_pct=70.0,
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
    # Per-cell: NCA 3.6 V nominal, range 3.0–4.2 V
    cell_nominal_voltage=3.6,
    cell_min_voltage=3.0,
    cell_max_voltage=4.2,
    cell_max_charge_current=1.0,
    cell_max_discharge_current=3.0,
    # Temperature — similar to NMC but slightly narrower
    temp_min_charge=0.0,
    temp_max_charge=45.0,
    temp_min_discharge=-20.0,
    temp_max_discharge=55.0,
    temp_max_jump=5.0,
    temp_optimal_min=15.0,
    temp_optimal_max=35.0,
    # Safety — more reactive cathode
    overcharge_voltage=4.25,
    overdischarge_voltage=2.7,
    thermal_runaway_temp=75.0,
    gas_venting_temp=65.0,
    # SOH — shorter cycle life, faster fade
    soh_params=SOHDegradationParams(
        cycle_life_min=500,
        cycle_life_max=1000,
        deg_knee_point_soh=80.0,
        annual_fade_rate_pct=2.5,
        calendric_fade_pct_per_year=5.0,
    ),
    # Cell balance — sensitive to imbalance
    balance=CellImbalanceThresholds(
        max_delta_mv=80.0,
        warning_delta_mv=40.0,
        recovery_rate_mv_per_h=12.0,
    ),
    min_soc_pct=15.0,
    max_soc_pct=90.0,
    critical_soh_pct=75.0,
)

# Registry: maps short_name → profile
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
    """Return metadata for all built-in profiles (no deep data)."""
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
    """Add or override a profile in the built-in registry.

    The profile's *short_name* is used as the key.
    """
    BUILTIN_PROFILES[profile.short_name] = profile


def load_custom_profile_from_file(path: str) -> BatteryChemistryProfile:
    """Load a profile from a JSON file and register it."""
    profile = BatteryChemistryProfile.load_from_file(path)
    register_custom_profile(profile)
    return profile
