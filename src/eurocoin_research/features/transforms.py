"""Time series transforms: stationarity, growth rates, normalization.

Standard transformations applied to raw series before they enter the
factor model or transformer pipeline.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class TransformType(str, Enum):
    """Standard transformation types for macroeconomic series."""

    NONE = "none"               # No transformation (already stationary, e.g., surveys)
    LOG = "log"                 # Log transform (levels → log-levels)
    DIFF = "diff"               # First difference
    LOG_DIFF = "log_diff"       # Log difference (growth rate)
    PCH = "pch"                 # Percentage change
    PCH_YEAR = "pch_year"       # Year-on-year percentage change
    NORMALIZE = "normalize"     # Standardize (z-score)


# Default transformation mapping by series type
SERIES_TRANSFORMS = {
    # Levels → growth rates
    "GDP_LEVELS": TransformType.LOG_DIFF,
    "IP_TOTAL": TransformType.LOG_DIFF,
    "HICP_TOTAL": TransformType.PCH_YEAR,
    "HICP_ENERGY": TransformType.PCH_YEAR,
    "PPI_DOM": TransformType.PCH_YEAR,
    "UNEMP": TransformType.NONE,        # Already a rate
    "M3": TransformType.LOG_DIFF,
    "BOND_10Y": TransformType.NONE,     # Already a rate/yield
    # Surveys → no transform (indices, roughly stationary)
    "ESI": TransformType.NONE,
    "ICI": TransformType.NONE,
    "CCI": TransformType.NONE,
    "SCI": TransformType.NONE,
    "ConstCI": TransformType.NONE,
}


def apply_transform(
    series: pl.Series | np.ndarray,
    transform: TransformType = TransformType.NONE,
) -> np.ndarray:
    """Apply a transformation to a time series.

    Args:
        series: Input time series (Polars Series or numpy array).
        transform: Type of transformation.

    Returns:
        Transformed numpy array. May have one fewer observation than input
        (for difference-based transforms).
    """
    if isinstance(series, pl.Series):
        arr = series.to_numpy().astype(float)
    else:
        arr = np.asarray(series, dtype=float)

    if transform == TransformType.NONE:
        return arr

    elif transform == TransformType.LOG:
        return np.log(arr)

    elif transform == TransformType.DIFF:
        return np.diff(arr, prepend=np.nan)

    elif transform == TransformType.LOG_DIFF:
        log_arr = np.log(arr)
        return np.diff(log_arr, prepend=np.nan)

    elif transform == TransformType.PCH:
        return np.diff(arr, prepend=np.nan) / np.abs(arr)

    elif transform == TransformType.PCH_YEAR:
        # Year-on-year percentage change (12 months for monthly, 4 quarters for quarterly)
        # For monthly series: yoy_t = (x_t / x_{t-12}) - 1
        lag = 12  # Default: monthly YoY
        result = np.full_like(arr, np.nan)
        for i in range(lag, len(arr)):
            if arr[i - lag] != 0:
                result[i] = arr[i] / arr[i - lag] - 1
        return result

    elif transform == TransformType.NORMALIZE:
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std == 0:
            return arr - mean
        return (arr - mean) / std

    else:
        raise ValueError(f"Unknown transform type: {transform}")


def transform_panel(
    panel: pl.DataFrame,
    transform_map: dict[str, TransformType] | None = None,
) -> pl.DataFrame:
    """Apply transformations to each column in a panel.

    Args:
        panel: Wide-format panel with date column and series columns.
        transform_map: Mapping of column name to transform type.
                       If None, uses SERIES_TRANSFORMS defaults.

    Returns:
        Panel with transformed series (same shape).
    """
    if transform_map is None:
        transform_map = SERIES_TRANSFORMS

    result = panel.clone()
    for col in panel.columns:
        if col == "date":
            continue
        transform = transform_map.get(col, TransformType.NONE)
        if transform == TransformType.NONE:
            continue
        transformed = apply_transform(panel[col], transform)
        result = result.with_columns(pl.Series(col, transformed))

    logger.info("Panel transformed using %d transformations", len(transform_map))
    return result


def get_transform_for_series(series_id: str) -> TransformType:
    """Get the default transformation for a series."""
    return SERIES_TRANSFORMS.get(series_id, TransformType.NONE)
