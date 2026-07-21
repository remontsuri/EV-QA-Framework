"""Shared utilities for EV-QA-Framework."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column name aliases: canonical_name -> list of aliases
COLUMN_ALIASES: dict[str, list[str]] = {
    "temp": ["temperature", "temp_c", "battery_temp", "battery_temperature"],
    "voltage": ["volt", "voltage_v", "battery_voltage"],
    "current": ["curr", "current_a", "battery_current"],
    "soc": ["state_of_charge", "soc_percent"],
    "soh": ["state_of_health", "soh_percent"],
}


def normalize_columns(
    df: pd.DataFrame,
    aliases: Optional[dict[str, list[str]]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Normalize DataFrame column names using known aliases.

    For each canonical name, if an alias is found in df.columns
    but the canonical name is not, rename the alias to canonical.

    Args:
        df: Input DataFrame.
        aliases: Override alias mapping. Defaults to COLUMN_ALIASES.
        inplace: If True, modify df in place.

    Returns:
        DataFrame with normalized column names.
    """
    if aliases is None:
        aliases = COLUMN_ALIASES
    if not inplace:
        df = df.copy()
    for canonical, alts in aliases.items():
        if canonical in df.columns:
            continue
        for alt in alts:
            if alt in df.columns:
                df.rename(columns={alt: canonical}, inplace=True)
                logger.debug("Renamed column '%s' -> '%s'", alt, canonical)
                break
    return df


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
