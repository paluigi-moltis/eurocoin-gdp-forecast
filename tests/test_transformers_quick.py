"""Quick test: train each model individually with reduced steps."""
import sys
sys.path.insert(0, "src")
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import polars as pl
import pandas as pd
from eurocoin_research.data.target import compute_mlrg, gdp_levels_to_qoq_growth
from eurocoin_research.features.transforms import transform_panel
from eurocoin_research.models.transformers import prepare_data_for_neuralforecast, train_and_forecast, ModelConfig

panel = pl.read_parquet("data/processed/panel_monthly.parquet")
transformed = transform_panel(panel)

# Build monthly MLRG target
gdp_panel = panel.filter(pl.col("GDP_LEVELS").is_not_null()).select(["date", "GDP_LEVELS"])
gdp_levels = gdp_panel["GDP_LEVELS"].to_numpy()
gdp_growth = gdp_levels_to_qoq_growth(gdp_levels)
mlrg = compute_mlrg(gdp_growth, method="ideal", max_lag=8)

gdp_dates = gdp_panel["date"].to_list()
mlrg_monthly = {}
for d, val in zip(gdp_dates, mlrg):
    if not np.isnan(val):
        for month_offset in range(3):
            m = d.month + month_offset
            y = d.year
            if m > 12:
                m -= 12
                y += 1
            mlrg_monthly[(y, m)] = val

all_dates = transformed["date"].to_list()
mlrg_series = [mlrg_monthly.get((d.year, d.month), np.nan) for d in all_dates]
transformed = transformed.with_columns(pl.Series("MLRG", mlrg_series))
df_with_target = transformed.filter(pl.col("MLRG").is_not_null())

pdf = df_with_target.to_pandas()
pdf["date"] = pd.to_datetime(pdf["date"])
feature_cols = [c for c in pdf.columns if c not in ("date", "GDP_LEVELS", "MLRG")]
pdf[feature_cols] = pdf[feature_cols].ffill().bfill()

# Split: train up to 2020-01
train_df = pdf[pdf["date"] < "2020-01-01"].copy()
print(f"Train: {len(train_df)} rows, Features: {len(feature_cols)}")

nf_train = prepare_data_for_neuralforecast(train_df, "MLRG", feature_cols)

# Test each model individually with small config
models_to_test = [
    ("DLinear", "DLinear", 100),
    ("NBEATS", "NBEATS", 100),
    ("TFT", "TFT", 100),
    ("PatchTST", "PatchTST", 100),
    ("Informer", "Informer", 100),
]

results = []
for name, cls, steps in models_to_test:
    config = ModelConfig(
        name=name, model_class=cls,
        input_size=24, max_steps=steps,
        learning_rate=1e-3, batch_size=16, hidden_size=32,
    )
    hist_exog = feature_cols if cls == "TFT" else None
    try:
        fcst, info = train_and_forecast(config, nf_train, h=1, hist_exog_list=hist_exog)
        val = fcst[0] * 100 if len(fcst) > 0 else float('nan')
        print(f"  OK  {name:12s}: forecast={val:+.4f}%")
        results.append({"model": name, "status": "OK", "forecast": val})
    except Exception as e:
        print(f"  ERR {name:12s}: {str(e)[:120]}")
        results.append({"model": name, "status": f"ERR: {str(e)[:80]}", "forecast": None})

print("\nDone.")
