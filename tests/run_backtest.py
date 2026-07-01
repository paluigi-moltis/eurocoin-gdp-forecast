"""Run the full expanding-window backtest for all models."""
import sys
sys.path.insert(0, "src")
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import polars as pl
import pandas as pd

from eurocoin_research.data.target import compute_mlrg, gdp_levels_to_qoq_growth
from eurocoin_research.features.transforms import transform_panel

# Load and prepare data
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

# Rename for NeuralForecast format
pdf = pdf.rename(columns={"date": "ds", "MLRG": "y"})

print(f"Full dataset: {len(pdf)} rows")
print(f"Features: {len(feature_cols)}")
print(f"Date range: {pdf['ds'].min()} to {pdf['ds'].max()}")

# Run backtest (quarterly dates)
from eurocoin_research.evaluation.backtest import expanding_window_backtest, compute_metrics

backtest_dates = [
    "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01",
]

print(f"\nRunning backtest at {len(backtest_dates)} dates...")
results = expanding_window_backtest(
    pdf, target_col="y", feature_cols=feature_cols,
    backtest_dates=backtest_dates,
    h=1,
    transformer_steps=150,
    transformer_hidden=32,
    transformer_input=24,
)

# Compute metrics
metrics = compute_metrics(results)

print(f"\n{'='*70}")
print("=== BACKTEST RESULTS ===")
print(f"{'='*70}")
print(metrics.to_string(index=False))

print(f"\n{'='*70}")
print("=== INDIVIDUAL FORECASTS ===")
print(f"{'='*70}")
pivot = results.pivot(index="backtest_date", columns="model", values="forecast")
print(pivot.to_string())

# Save results
results.to_csv("results/backtest_results.csv", index=False)
metrics.to_csv("results/backtest_metrics.csv", index=False)
print(f"\nResults saved to results/")
