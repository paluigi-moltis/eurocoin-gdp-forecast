"""MLRG (Medium- to Long-Run Growth) target construction.

Implements the band-pass filter that extracts the smooth medium-term
component from GDP quarter-on-quarter growth — the target of the
Eurocoin indicator.

The MLRG is defined as the component of GDP growth with all oscillations
of period ≤ 1 year (≈ 4 quarters) removed.

References:
- Altissimo et al. (2007), §3
- Christiano & Fitzgerald (2003)
- Baxter & King (1999)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Spectral threshold: π/6 radians corresponds to period = 2π/(π/6) = 12 months = 4 quarters
DEFAULT_THRESHOLD = np.pi / 6


def ideal_bandpass_weights(threshold: float, max_lag: int) -> np.ndarray:
    """Compute the ideal symmetric band-pass filter weights.

    The ideal filter removes all frequencies above `threshold`:
        b_k = sin(k * threshold) / (k * π)  for k != 0
        b_0 = threshold / π

    Args:
        threshold: Cutoff frequency in radians (π/6 for ~1 year period).
        max_lag: Maximum lag (truncation point). Larger = better approximation.

    Returns:
        Array of weights [-K, ..., -1, 0, 1, ..., K], length 2*max_lag+1.
    """
    weights = np.zeros(2 * max_lag + 1)
    center = max_lag
    weights[center] = threshold / np.pi  # b_0
    for k in range(1, max_lag + 1):
        b_k = np.sin(k * threshold) / (k * np.pi)
        weights[center + k] = b_k
        weights[center - k] = b_k
    return weights


def apply_ideal_bandpass(
    series: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    max_lag: int = 12,
) -> np.ndarray:
    """Apply the ideal band-pass filter to a time series.

    Uses symmetric (two-sided) weights with truncation at max_lag.
    The first and last max_lag observations are filled with the
    sample mean (approximation at endpoints).

    Args:
        series: 1D array of the time series.
        threshold: Cutoff frequency (default π/6).
        max_lag: Truncation point for the filter.

    Returns:
        Filtered series (same length as input).
    """
    n = len(series)
    if n < 2 * max_lag + 1:
        logger.warning(
            "Series too short (%d) for band-pass filter with max_lag=%d. "
            "Returning original series.",
            n,
            max_lag,
        )
        return series.copy()

    weights = ideal_bandpass_weights(threshold, max_lag)
    filtered = np.full(n, np.nan)

    # Convolution with symmetric weights
    for t in range(max_lag, n - max_lag):
        window = series[t - max_lag : t + max_lag + 1]
        filtered[t] = np.dot(weights, window)

    # Fill endpoints with sample mean of the filtered portion
    valid = filtered[~np.isnan(filtered)]
    if len(valid) > 0:
        endpoint_val = np.mean(valid)
        filtered[:max_lag] = endpoint_val
        filtered[n - max_lag:] = endpoint_val

    return filtered


def christianofitzgerald_bp(
    series: np.ndarray,
    min_period: int = 2,
    max_period: int | None = None,
) -> np.ndarray:
    """Apply the Christiano-Fitzgerald (2003) band-pass filter.

    The CF filter provides a better approximation than the truncated ideal
    filter, especially at the endpoints. It uses a data-dependent asymmetric
    weighting scheme.

    Args:
        series: 1D array.
        min_period: Minimum cycle period in units of the data frequency
                    (e.g., 2 quarters for quarterly data).
        max_period: Maximum cycle period. None = no upper bound (use series length).

    Returns:
        Filtered series (same length).
    """
    n = len(series)
    if max_period is None:
        max_period = n

    # Convert periods to frequencies
    omega_l = 2 * np.pi / max_period  # lower frequency bound
    omega_u = 2 * np.pi / min_period  # upper frequency bound

    filtered = np.zeros(n)

    for t in range(n):
        b_sum = 0.0
        b0 = (omega_u - omega_l) / np.pi
        b_sum += b0 * series[t]

        for k in range(1, n - 1):
            # CF asymmetric weights
            b_k = (np.sin(k * omega_u) - np.sin(k * omega_l)) / (k * np.pi)

            # Adjust for finite sample endpoints
            adj = 0.5 * (
                np.sin(k * omega_u) * np.cos((n - 1 - t) * omega_u)
                + np.sin(k * omega_l) * np.cos((n - 1 - t) * omega_l)
            ) / np.pi

            if t - k >= 0:
                b_sum += b_k * series[t - k]
            if t + k < n:
                b_sum += b_k * series[t + k]

        filtered[t] = b_sum

    return filtered


def compute_mlrg(
    gdp_growth: np.ndarray | pl.Series,
    method: Literal["ideal", "cf"] = "ideal",
    threshold: float = DEFAULT_THRESHOLD,
    max_lag: int = 12,
    min_period: int = 2,
    max_period: int | None = None,
) -> np.ndarray:
    """Compute the Medium- to Long-Run Growth (MLRG) component.

    The MLRG is the GDP q-o-q growth series with all oscillations of
    period ≤ 1 year removed.

    Args:
        gdp_growth: GDP quarter-on-quarter growth rates (as fraction, e.g., 0.005 = 0.5%).
        method: "ideal" for truncated ideal filter, "cf" for Christiano-Fitzgerald.
        threshold: Cutoff frequency for ideal filter (default π/6 ≈ 1-year period).
        max_lag: Truncation point for ideal filter.
        min_period: Minimum cycle period for CF filter.
        max_period: Maximum cycle period for CF filter.

    Returns:
        MLRG series (same length as input).
    """
    if isinstance(gdp_growth, pl.Series):
        arr = gdp_growth.to_numpy().astype(float)
    else:
        arr = np.asarray(gdp_growth, dtype=float)

    # Handle NaN
    valid_mask = ~np.isnan(arr)
    if not valid_mask.all():
        # Interpolate NaNs before filtering
        arr_interp = _interpolate_nans(arr)
    else:
        arr_interp = arr

    if method == "ideal":
        mlrg = apply_ideal_bandpass(arr_interp, threshold=threshold, max_lag=max_lag)
    elif method == "cf":
        mlrg = christianofitzgerald_bp(arr_interp, min_period=min_period, max_period=max_period)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ideal' or 'cf'.")

    logger.info(
        "MLRG computed via %s filter: %d observations. "
        "Variance ratio (MLRG/total): %.1f%%",
        method,
        len(mlrg),
        100 * np.nanvar(mlrg) / np.nanvar(arr_interp) if np.nanvar(arr_interp) > 0 else 0,
    )
    return mlrg


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in a 1D array."""
    result = arr.copy()
    n = len(result)
    for i in range(n):
        if np.isnan(result[i]):
            # Find nearest non-NaN before and after
            before = i - 1
            while before >= 0 and np.isnan(result[before]):
                before -= 1
            after = i + 1
            while after < n and np.isnan(result[after]):
                after += 1
            if before >= 0 and after < n:
                # Linear interpolation
                t = (i - before) / (after - before)
                result[i] = result[before] * (1 - t) + result[after] * t
            elif before >= 0:
                result[i] = result[before]
            elif after < n:
                result[i] = result[after]
    return result


def gdp_levels_to_qoq_growth(levels: np.ndarray | pl.Series) -> np.ndarray:
    """Convert GDP levels to quarter-on-quarter growth rates.

    growth_t = (GDP_t / GDP_{t-1}) - 1

    Or in log differences: growth_t = ln(GDP_t) - ln(GDP_{t-1})
    """
    if isinstance(levels, pl.Series):
        arr = levels.to_numpy().astype(float)
    else:
        arr = np.asarray(levels, dtype=float)

    log_levels = np.log(arr)
    growth = np.diff(log_levels, prepend=np.nan)
    return growth
