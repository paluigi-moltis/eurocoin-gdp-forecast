"""Sliding window and patch creation for Transformer models.

Converts panel data into supervised learning windows:
- Input: lookback window of covariates
- Target: MLRG value at the forecast date
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def create_windows(
    data: np.ndarray,
    lookback: int = 36,
    horizon: int = 0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from a time series panel.

    Args:
        data: 2D array (time_steps × n_features).
        lookback: Number of past time steps to use as input.
        horizon: Forecast horizon (0 = nowcast, 1 = 1-step ahead).
        stride: Step size between windows.

    Returns:
        Tuple of (X, y) where:
        - X: shape (n_windows, lookback, n_features)
        - y: shape (n_windows,) — target at t + horizon
    """
    n_steps, n_features = data.shape
    n_windows = max(0, n_steps - lookback - horizon)

    if n_windows == 0:
        logger.warning(
            "Not enough data for windowing: %d steps, lookback=%d, horizon=%d",
            n_steps, lookback, horizon,
        )
        return np.array([]), np.array([])

    X = np.zeros((n_windows, lookback, n_features))
    y = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * stride
        end = start + lookback
        X[i] = data[start:end]
        y[i] = data[end + horizon, 0]  # Target is first column (GDP/MLRG)

    logger.info(
        "Created %d windows: X shape=%s, y shape=%s",
        n_windows, X.shape, y.shape,
    )
    return X, y


def create_patches(
    data: np.ndarray,
    patch_size: int = 3,
    patch_stride: int | None = None,
) -> np.ndarray:
    """Create patches from a time series (PatchTST-style tokenization).

    Groups consecutive observations into patches to reduce sequence length.

    Args:
        data: 2D array (time_steps × n_features).
        patch_size: Number of consecutive time steps per patch.
        patch_stride: Stride between patches. Defaults to patch_size (non-overlapping).

    Returns:
        Patches: shape (n_patches, n_features, patch_size)
    """
    if patch_stride is None:
        patch_stride = patch_size

    n_steps, n_features = data.shape
    n_patches = (n_steps - patch_size) // patch_stride + 1

    if n_patches <= 0:
        logger.warning("Series too short for patching: %d steps, patch_size=%d", n_steps, patch_size)
        return np.array([])

    patches = np.zeros((n_patches, n_features, patch_size))

    for i in range(n_patches):
        start = i * patch_stride
        end = start + patch_size
        patches[i] = data[start:end].T  # Transpose: (features, patch_size)

    logger.debug(
        "Created %d patches (patch_size=%d, stride=%d) from %d steps",
        n_patches, patch_size, patch_stride, n_steps,
    )
    return patches


def normalize_windows(
    X: np.ndarray,
    method: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each window independently.

    Args:
        X: shape (n_windows, lookback, n_features)
        method: "standard" (z-score) or "minmax"

    Returns:
        Tuple of (normalized_X, means, stds) for inverse transform.
    """
    if method == "standard":
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        return (X - means) / stds, means, stds
    elif method == "minmax":
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        return (X - mins) / ranges, mins, ranges
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def panel_to_numpy(
    panel: pl.DataFrame,
    target_col: str = "GDP_LEVELS",
    feature_cols: list[str] | None = None,
    drop_na: bool = True,
) -> np.ndarray:
    """Convert a Polars panel to a numpy array for model input.

    Args:
        panel: Wide-format panel with date and series columns.
        target_col: Column to use as target (placed first in the array).
        feature_cols: Columns to use as features. If None, uses all non-date columns.
        drop_na: If True, drops rows with any null values.

    Returns:
        2D numpy array (time_steps × n_features), target in first column.
    """
    if feature_cols is None:
        feature_cols = [c for c in panel.columns if c != "date"]

    # Put target first
    if target_col in feature_cols:
        feature_cols = [target_col] + [c for c in feature_cols if c != target_col]

    # Select and convert
    df = panel.select(feature_cols)
    arr = df.to_numpy().astype(float)

    if drop_na:
        valid_mask = ~np.isnan(arr).any(axis=1)
        arr = arr[valid_mask]
        logger.info("Dropped %d rows with NaN", (~valid_mask).sum())

    return arr
