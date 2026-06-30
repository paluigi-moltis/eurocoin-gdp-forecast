"""Forecast evaluation metrics.

Implements the metrics used to compare model forecasts against
the true MLRG target and against published Eurocoin values.
"""

from __future__ import annotations

import numpy as np


def rmse(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((forecast[mask] - actual[mask]) ** 2)))


def mse(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Mean Squared Error."""
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if not mask.any():
        return float("nan")
    return float(np.mean((forecast[mask] - actual[mask]) ** 2))


def msd(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Mean Squared Deviation — same as MSE, matching Aprigliano et al. (2022) terminology."""
    return mse(forecast, actual)


def mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(forecast[mask] - actual[mask])))


def correlation(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Pearson correlation between forecast and actual."""
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if mask.sum() < 2:
        return float("nan")
    f, a = forecast[mask], actual[mask]
    if np.std(f) == 0 or np.std(a) == 0:
        return 0.0
    return float(np.corrcoef(f, a)[0, 1])


def directional_accuracy(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Proportion of periods where the forecast correctly predicts the direction of change."""
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if mask.sum() < 2:
        return float("nan")
    f, a = forecast[mask], actual[mask]
    f_diff = np.diff(f)
    a_diff = np.diff(a)
    correct = (np.sign(f_diff) == np.sign(a_diff)).sum()
    return float(correct / len(f_diff))


def turning_point_accuracy(
    forecast: np.ndarray,
    actual: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Detect turning points (sign changes) and measure accuracy.

    A turning point is defined as a period where the series changes direction
    (from positive to negative growth, or vice versa).

    Returns:
        Dictionary with precision, recall, and F1 for turning point detection.
    """
    mask = ~(np.isnan(forecast) | np.isnan(actual))
    if mask.sum() < 3:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    f, a = forecast[mask], actual[mask]

    # Identify turning points in actual
    actual_tp = np.zeros(len(a) - 2, dtype=bool)
    for i in range(1, len(a) - 1):
        if (a[i] > a[i - 1] and a[i] > a[i + 1]) or (a[i] < a[i - 1] and a[i] < a[i + 1]):
            actual_tp[i - 1] = True

    # Identify turning points in forecast
    forecast_tp = np.zeros(len(f) - 2, dtype=bool)
    for i in range(1, len(f) - 1):
        if (f[i] > f[i - 1] and f[i] > f[i + 1]) or (f[i] < f[i - 1] and f[i] < f[i + 1]):
            forecast_tp[i - 1] = True

    # Compute precision, recall
    tp = (actual_tp & forecast_tp).sum()
    fp = (~actual_tp & forecast_tp).sum()
    fn = (actual_tp & ~forecast_tp).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_forecast(
    forecast: np.ndarray,
    actual: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compute all evaluation metrics for a forecast series.

    Args:
        forecast: Forecasted values.
        actual: True (ex-post) values.
        label: Optional label for the model name.

    Returns:
        Dictionary of all metrics.
    """
    metrics = {
        "label": label,
        "rmse": rmse(forecast, actual),
        "mse": mse(forecast, actual),
        "msd": msd(forecast, actual),
        "mae": mae(forecast, actual),
        "correlation": correlation(forecast, actual),
        "directional_accuracy": directional_accuracy(forecast, actual),
    }
    tp = turning_point_accuracy(forecast, actual)
    metrics.update(tp)
    return metrics
