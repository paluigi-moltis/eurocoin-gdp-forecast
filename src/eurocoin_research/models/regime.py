"""Changepoint detection and regime utilities using River (online/streaming).

Pipeline:
1. Online Modeling: Fit HoltWinters or SNARIMAX to track trend + seasonality.
2. Residual Extraction: Compute prediction errors at each step.
3. Online Change Point Detection: Feed residuals into ADWIN or PageHinkley.

When the trend or seasonal cycle abruptly shifts, the time-series model
temporarily generates large errors, causing the detector to flag a changepoint.

This approach is more robust than raw-series changepoint detection because
it explicitly models and removes the trend/seasonal component before
searching for structural breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChangepointResult:
    """Result of changepoint detection."""

    changepoint_indices: list[int] = field(default_factory=list)
    residuals: np.ndarray | None = None
    predictions: np.ndarray | None = None
    method: str = ""
    detector_params: dict = field(default_factory=dict)


@dataclass
class Regime:
    """A detected economic regime."""

    start_idx: int
    end_idx: int
    start_date: str = ""
    end_date: str = ""
    label: str = ""
    mean_growth: float = 0.0
    std_growth: float = 0.0


# ---------------------------------------------------------------------------
# Core detection pipeline
# ---------------------------------------------------------------------------


def detect_changepoints(
    series: np.ndarray,
    *,
    ts_model: Literal["holtwinters", "snarimax"] = "holtwinters",
    detector: Literal["adwin", "page_hinkley"] = "adwin",
    seasonal_period: int = 12,
    # HoltWinters params
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    # SNARIMAX params
    p: int = 3,
    d: int = 1,
    q: int = 0,
    m: int | None = None,
    # Detector params
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50.0,
    page_hinkley_alpha: float = 0.9999,
) -> ChangepointResult:
    """Detect changepoints in a time series using the River pipeline.

    Steps:
      1. Stream the series through an online time-series model (HoltWinters or
         SNARIMAX) that tracks trend and seasonality incrementally.
      2. At each step compute the residual (actual - prediction).
      3. Feed the residuals into a drift detector (ADWIN or PageHinkley).
      4. When the detector fires, record the index as a changepoint.

    Args:
        series: 1D array of observations (levels or growth rates).
        ts_model: "holtwinters" or "snarimax".
        detector: "adwin" or "page_hinkley".
        seasonal_period: Number of observations per seasonal cycle
                         (12 for monthly, 4 for quarterly, 1 for no seasonality).
        alpha, beta, gamma: HoltWinters smoothing parameters. If None,
                             River uses sensible defaults.
        p, d, q, m: SNARIMAX order parameters (autoregressive, differencing,
                    moving-average, seasonal period).
        adwin_delta: ADWIN sensitivity (lower = more sensitive).
        page_hinkley_threshold: Page-Hinkley detection threshold
                                (lower = more sensitive).
        page_hinkley_alpha: Page-Hinkley forgetting factor.

    Returns:
        ChangepointResult with changepoint indices, residuals, predictions.
    """
    from river import drift, time_series

    y = np.asarray(series, dtype=float)

    # Handle NaNs — interpolate so the online models don't break
    nan_mask = np.isnan(y)
    if nan_mask.any():
        logger.info("Interpolating %d NaN values before changepoint detection", nan_mask.sum())
        y = _interpolate_nans(y)

    n = len(y)

    # --- 1. Build the online time-series model -------------------------------
    if ts_model == "holtwinters":
        kwargs: dict = {}
        if seasonal_period > 1:
            kwargs["seasonality"] = seasonal_period
        kwargs["alpha"] = alpha if alpha is not None else 0.3
        if beta is not None:
            kwargs["beta"] = beta
        if gamma is not None:
            kwargs["gamma"] = gamma
        model = time_series.HoltWinters(**kwargs)
    elif ts_model == "snarimax":
        # For non-seasonal data (seasonal_period=1), use m=1 (River default) with sp=0
        effective_m = m if m is not None else max(seasonal_period, 1)
        model = time_series.SNARIMAX(
            p=p, d=d, q=q,
            m=effective_m,
            sp=1 if effective_m > 1 else 0,
            sd=0,
            sq=0,
        )
    else:
        raise ValueError(f"Unknown ts_model: {ts_model}")

    # --- 2. Build the drift detector -----------------------------------------
    if detector == "adwin":
        det = drift.ADWIN(delta=adwin_delta)
    elif detector == "page_hinkley":
        det = drift.PageHinkley(
            threshold=page_hinkley_threshold,
            alpha=page_hinkley_alpha,
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")

    # --- 3. Stream the series -------------------------------------------------
    predictions = np.full(n, np.nan)
    residuals = np.full(n, np.nan)
    changepoints: list[int] = []

    # Warmup: skip detection for the first few observations while the model stabilizes
    warmup = max(seasonal_period, p + q + 1, 3) if ts_model == "snarimax" else 3

    for i in range(n):
        actual = y[i]

        # One-step-ahead prediction (model needs at least 2 observations to forecast)
        try:
            pred_list = model.forecast(horizon=1)
            pred = pred_list[0] if isinstance(pred_list, (list, tuple, np.ndarray)) else pred_list
        except (IndexError, Exception):
            pred = actual
        predictions[i] = pred
        residuals[i] = actual - pred

        # Update the time-series model with the actual observation
        model.learn_one(y=actual)

        # Skip detection during warmup period
        if i < warmup:
            continue

        # Feed the residual into the detector.
        # For PageHinkley, using |residual| detects changes in volatility/error magnitude.
        # For ADWIN, using raw residual detects changes in the mean of the residual stream.
        residual = actual - pred
        feed_value = abs(residual) if detector == "page_hinkley" else residual
        det.update(feed_value)

        # Check if the detector flagged a drift / changepoint
        if det.drift_detected:
            changepoints.append(i)
            logger.info(
                "Changepoint detected at index %d (residual=%.4f, value=%.4f, pred=%.4f)",
                i, residual, actual, pred,
            )
            # Reset the detector so it can detect subsequent changepoints
            if detector == "adwin":
                det = drift.ADWIN(delta=adwin_delta)
            else:
                det = drift.PageHinkley(
                    threshold=page_hinkley_threshold,
                    alpha=page_hinkley_alpha,
                )

    logger.info(
        "Changepoint detection complete: %d changepoints found using %s + %s",
        len(changepoints), ts_model, detector,
    )

    return ChangepointResult(
        changepoint_indices=changepoints,
        residuals=residuals,
        predictions=predictions,
        method=f"river-{ts_model}-{detector}",
        detector_params={
            "seasonal_period": seasonal_period,
            "adwin_delta": adwin_delta,
            "page_hinkley_threshold": page_hinkley_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Regime definition and labeling
# ---------------------------------------------------------------------------


def define_regimes(
    changepoint_indices: list[int],
    dates: list[str] | None = None,
    n_observations: int | None = None,
) -> list[Regime]:
    """Define regimes from detected changepoints."""
    if n_observations is None:
        if dates:
            n_observations = len(dates)
        else:
            n_observations = max(changepoint_indices) + 1 if changepoint_indices else 1

    boundaries = [0] + [cp + 1 for cp in changepoint_indices] + [n_observations]
    boundaries = sorted(set(boundaries))

    regimes: list[Regime] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        start_date = dates[start] if dates and start < len(dates) else ""
        end_date = dates[end] if dates and end < len(dates) else ""
        regimes.append(Regime(
            start_idx=start,
            end_idx=end,
            start_date=start_date,
            end_date=end_date,
        ))
    return regimes


def label_regimes_economic(regimes: list[Regime]) -> list[Regime]:
    """Label regimes based on known economic history of the Euro Area."""
    episode_labels = [
        (2000, 2007, "Pre-crisis expansion"),
        (2008, 2009, "Global Financial Crisis"),
        (2010, 2011, "Recovery"),
        (2011, 2013, "Sovereign debt crisis"),
        (2014, 2019, "Low-growth recovery"),
        (2020, 2020, "Pandemic shock"),
        (2021, 2021, "Post-pandemic rebound"),
        (2022, 2023, "Energy shock / inflation"),
        (2024, 2030, "Normalization"),
    ]
    for regime in regimes:
        try:
            start_year = int(regime.start_date[:4]) if regime.start_date else 0
        except (ValueError, IndexError):
            start_year = 0
        for (start_ep, end_ep, label) in episode_labels:
            if start_ep <= start_year <= end_ep:
                regime.label = label
                break
        else:
            regime.label = f"Period {regime.start_idx}-{regime.end_idx}"
    return regimes


# ---------------------------------------------------------------------------
# Regime descriptive statistics
# ---------------------------------------------------------------------------


def compute_regime_statistics(
    regimes: list[Regime],
    target: np.ndarray,
    covariates: np.ndarray | None = None,
    covariate_names: list[str] | None = None,
) -> list[dict]:
    """Compute descriptive statistics for each regime.

    For each regime computes:
    - Target mean and std
    - Correlation of each covariate with the target (within the regime)
    - Lead-lag cross-correlations (shift -3 to +3)

    Returns:
        List of dicts, one per regime.
    """
    results = []
    for regime in regimes:
        t0, t1 = regime.start_idx, regime.end_idx + 1
        target_seg = target[t0:t1]
        target_valid = target_seg[~np.isnan(target_seg)]

        stats: dict = {
            "regime_label": regime.label,
            "start_date": regime.start_date,
            "end_date": regime.end_date,
            "n_observations": len(target_valid),
            "target_mean": float(np.mean(target_valid)) if len(target_valid) > 0 else float("nan"),
            "target_std": float(np.std(target_valid)) if len(target_valid) > 0 else float("nan"),
        }

        # Correlations
        if covariates is not None and covariate_names:
            corrs: dict[str, float] = {}
            lead_lag: dict[str, dict] = {}
            for j, name in enumerate(covariate_names):
                if j >= covariates.shape[1]:
                    continue
                cov_seg = covariates[t0:t1, j]
                tgt = target_seg
                valid = ~(np.isnan(cov_seg) | np.isnan(tgt))
                if valid.sum() > 3:
                    corrs[name] = float(np.corrcoef(cov_seg[valid], tgt[valid])[0, 1])
                else:
                    corrs[name] = float("nan")

                # Lead-lag analysis (need full series, not just segment)
                ll: dict[int, float] = {}
                full_cov = covariates[:, j] if covariates is not None else None
                full_tgt = target
                for lag in range(-3, 4):
                    shifted = _shift(full_cov, lag)
                    valid2 = ~(np.isnan(shifted) | np.isnan(full_tgt))
                    # Only use observations within this regime
                    regime_mask = np.zeros(len(full_tgt), dtype=bool)
                    regime_mask[t0:t1] = True
                    use = valid2 & regime_mask
                    if use.sum() > 3:
                        ll[lag] = float(np.corrcoef(shifted[use], full_tgt[use])[0, 1])
                lead_lag[name] = ll

            stats["correlations"] = corrs
            stats["lead_lag"] = lead_lag

        results.append(stats)
        regime.mean_growth = stats["target_mean"]
        regime.std_growth = stats["target_std"]

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in a 1D array."""
    result = arr.copy()
    n = len(result)
    for i in range(n):
        if np.isnan(result[i]):
            before = i - 1
            while before >= 0 and np.isnan(result[before]):
                before -= 1
            after = i + 1
            while after < n and np.isnan(result[after]):
                after += 1
            if before >= 0 and after < n:
                t = (i - before) / (after - before)
                result[i] = result[before] * (1 - t) + result[after] * t
            elif before >= 0:
                result[i] = result[before]
            elif after < n:
                result[i] = result[after]
    return result


def _shift(arr: np.ndarray, lag: int) -> np.ndarray:
    """Shift a 1D array by `lag` positions (positive = forward/lagging)."""
    result = np.full_like(arr, np.nan)
    n = len(arr)
    if lag > 0:
        result[lag:] = arr[:n - lag]
    elif lag < 0:
        result[:n + lag] = arr[-lag:]
    else:
        result[:] = arr
    return result
