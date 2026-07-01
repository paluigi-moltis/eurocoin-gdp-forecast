"""Transformer-based forecasting models for MLRG prediction.

Uses the NeuralForecast library (Nixtla) which provides:
- TFT (Temporal Fusion Transformer)
- Informer
- PatchTST
- NBEATS
- DLinear (linear baseline)

All models configured for CPU-only training with small parameter counts
suitable for the ~330 monthly observation dataset.

Input: monthly covariate panel (16 features)
Target: quarterly MLRG (interpolated to monthly for training)
Output: MLRG forecast
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single transformer model."""

    name: str
    model_class: str  # "TFT", "Informer", "PatchTST", "NBEATS", "DLinear"
    h: int = 1  # forecast horizon (months)
    input_size: int = 36  # lookback window (months)
    max_steps: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 16
    hidden_size: int = 64
    n_head: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dropout: float = 0.1
    extra_kwargs: dict = field(default_factory=dict)


def create_model_configs() -> list[ModelConfig]:
    """Create configurations for all models to be tested.

    Returns configurations for:
    - DLinear (mandatory linear baseline)
    - NBEATS (deep learning baseline)
    - TFT (Temporal Fusion Transformer — top priority)
    - Informer (efficient long-sequence)
    - PatchTST (patching + channel independence)
    """
    return [
        ModelConfig(
            name="DLinear",
            model_class="DLinear",
            input_size=36,
            max_steps=300,
            learning_rate=1e-3,
        ),
        ModelConfig(
            name="NBEATS",
            model_class="NBEATS",
            input_size=36,
            max_steps=500,
            learning_rate=1e-3,
            hidden_size=64,
        ),
        ModelConfig(
            name="TFT",
            model_class="TFT",
            input_size=36,
            max_steps=500,
            learning_rate=1e-3,
            hidden_size=64,
            n_head=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            dropout=0.1,
        ),
        ModelConfig(
            name="Informer",
            model_class="Informer",
            input_size=36,
            max_steps=500,
            learning_rate=1e-3,
            hidden_size=64,
            n_head=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            dropout=0.1,
        ),
        ModelConfig(
            name="PatchTST",
            model_class="PatchTST",
            input_size=36,
            max_steps=500,
            learning_rate=1e-3,
            hidden_size=64,
            n_head=4,
            dropout=0.1,
            extra_kwargs={"patch_len": 3, "stride": 3},  # 3-month patches = quarterly tokens
        ),
    ]


def prepare_data_for_neuralforecast(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Convert a wide-format panel to NeuralForecast long format.

    NeuralForecast expects columns: ds (date), y (target), unique_id, and covariates.

    Args:
        panel: Wide-format DataFrame with date, target, and feature columns.
        target_col: Name of the target column.
        feature_cols: List of covariate column names.
        date_col: Name of the date column.

    Returns:
        DataFrame in NeuralForecast format.
    """
    df = panel.copy()
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df["unique_id"] = "ea_gdp"

    # Select relevant columns
    cols = ["unique_id", "ds", "y"] + feature_cols
    df = df[[c for c in cols if c in df.columns]]

    # Drop rows where target is NaN
    df = df.dropna(subset=["y"])

    logger.info(
        "Prepared data for NeuralForecast: %d rows, %d features, target=%s",
        len(df), len(feature_cols), target_col,
    )
    return df


def train_and_forecast(
    config: ModelConfig,
    train_df: pd.DataFrame,
    h: int = 1,
    hist_exog_list: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Train a single model and generate forecasts.

    Args:
        config: Model configuration.
        train_df: Training data in NeuralForecast format.
        h: Forecast horizon.
        hist_exog_list: Historical exogenous variables (covariates).

    Returns:
        Tuple of (forecasts, model_info).
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        DLinear,
        Informer,
        NBEATS,
        PatchTST,
        TFT,
    )

    model_map = {
        "DLinear": DLinear,
        "NBEATS": NBEATS,
        "TFT": TFT,
        "Informer": Informer,
        "PatchTST": PatchTST,
    }

    model_cls = model_map.get(config.model_class)
    if model_cls is None:
        raise ValueError(f"Unknown model class: {config.model_class}")

    # Common trainer kwargs
    trainer_kwargs = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "accelerator": "cpu",
    }

    # Build model kwargs — NeuralForecast models accept trainer params directly
    kwargs: dict[str, Any] = {
        "h": h,
        "input_size": config.input_size,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }

    # Remove trainer_kwargs (not a valid parameter for these models)
    kwargs.pop("trainer_kwargs", None)

    # Model-specific parameters
    if config.model_class == "TFT":
        kwargs["hidden_size"] = config.hidden_size
        kwargs["n_head"] = config.n_head
        kwargs["dropout"] = config.dropout
        if hist_exog_list:
            kwargs["hist_exog_list"] = hist_exog_list
    elif config.model_class == "Informer":
        kwargs["hidden_size"] = config.hidden_size
        kwargs["n_head"] = config.n_head
        kwargs["encoder_layers"] = config.n_encoder_layers
        kwargs["decoder_layers"] = config.n_decoder_layers
        kwargs["dropout"] = config.dropout
        kwargs["factor"] = 3
        if hist_exog_list:
            kwargs["hist_exog_list"] = hist_exog_list
    elif config.model_class == "PatchTST":
        kwargs["hidden_size"] = config.hidden_size
        kwargs["n_heads"] = config.n_head
        kwargs["encoder_layers"] = config.n_encoder_layers
        kwargs["dropout"] = config.dropout
        kwargs.update(config.extra_kwargs)
    elif config.model_class == "NBEATS":
        # Use only identity stacks (no trend/seasonality which need h>1)
        kwargs["stack_types"] = ["identity", "identity", "identity"]
        kwargs["mlp_units"] = [[config.hidden_size, config.hidden_size]] * 3
    elif config.model_class == "DLinear":
        pass  # DLinear is minimal

    logger.info("Training %s (h=%d, input=%d, steps=%d)...",
                config.name, h, config.input_size, config.max_steps)

    model = model_cls(**kwargs)

    # Create NeuralForecast pipeline
    nf = NeuralForecast(models=[model], freq="MS")  # MS = month start

    # Train
    nf.fit(df=train_df)

    # Forecast
    forecasts = nf.predict()

    # Extract forecast values
    fcst_col = [c for c in forecasts.columns if c != "unique_id" and c != "ds"]
    if fcst_col:
        forecast_values = forecasts[fcst_col[0]].values
    else:
        forecast_values = np.array([])

    model_info = {
        "name": config.name,
        "model_class": config.model_class,
        "n_train_obs": len(train_df),
        "forecast_horizon": h,
        "forecast_values": forecast_values,
    }

    logger.info(
        "%s trained: %d obs, forecast shape=%s",
        config.name, len(train_df), forecast_values.shape,
    )

    return forecast_values, model_info


def run_transformer_comparison(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_end_date: str | None = None,
    h: int = 1,
) -> pd.DataFrame:
    """Run all transformer models and return comparison results.

    Args:
        panel: Wide-format DataFrame.
        target_col: Target column name.
        feature_cols: Covariate column names.
        train_end_date: Date string for train/test split (exclusive).
        h: Forecast horizon.

    Returns:
        DataFrame with model comparison results.
    """
    # Split train/test
    if train_end_date:
        train_df = panel[panel["date"] < train_end_date].copy()
        test_df = panel[panel["date"] >= train_end_date].copy()
    else:
        # Use last 20% as test
        split_idx = int(len(panel) * 0.8)
        train_df = panel.iloc[:split_idx].copy()
        test_df = panel.iloc[split_idx:].copy()

    # Prepare NeuralForecast format
    nf_train = prepare_data_for_neuralforecast(
        train_df, target_col, feature_cols,
    )

    configs = create_model_configs()
    results = []

    for config in configs:
        try:
            # Only pass hist_exog_list to models that support it (TFT)
            hist_exog = feature_cols if config.model_class == "TFT" else None
            fcst, info = train_and_forecast(
                config, nf_train, h=h,
                hist_exog_list=hist_exog,
            )
            results.append({
                "model": config.name,
                "status": "OK",
                "n_forecasts": len(fcst),
                "first_forecast": float(fcst[0]) if len(fcst) > 0 else None,
                "forecasts": fcst.tolist() if len(fcst) > 0 else [],
            })
        except Exception as e:
            logger.error("Failed to train %s: %s", config.name, str(e)[:200])
            results.append({
                "model": config.name,
                "status": f"ERROR: {str(e)[:100]}",
                "n_forecasts": 0,
                "first_forecast": None,
                "forecasts": [],
            })

    return pd.DataFrame(results)
