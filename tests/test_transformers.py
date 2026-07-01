"""Test transformer models on the Eurocoin dataset."""
import sys
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import polars as pl
import pandas as pd

from eurocoin_research.data.target import compute_mlrg, gdp_levels_to_qoq_growth
from eurocoin_research.features.transforms import transform_panel

# Load panel
panel = pl.read_parquet("data/processed/panel_monthly.parquet")
print(f"Panel: {len(panel)} months x {len(panel.columns)-1} series")

# Transform to stationary
transformed = transform_panel(panel)

# Build monthly target: interpolate quarterly MLRG to monthly
gdp_panel = panel.filter(pl.col("GDP_LEVELS").is_not_null()).select(["date", "GDP_LEVELS"])
gdp_levels = gdp_panel["GDP_LEVELS"].to_numpy()
gdp_growth = gdp_levels_to_qoq_growth(gdp_levels)
mlrg = compute_mlrg(gdp_growth, method="ideal", max_lag=8)

# Create monthly MLRG by repeating quarterly values
gdp_dates = gdp_panel["date"].to_list()
mlrg_monthly = {}
for d, val in zip(gdp_dates, mlrg):
    if not np.isnan(val):
        for month_offset in range(3):
            import datetime
            m = d.month + month_offset
            y = d.year
            if m > 12:
                m -= 12
                y += 1
            mlrg_monthly[datetime.date(y, m, 1)] = val

# Build a monthly DataFrame with MLRG as target
all_dates = transformed["date"].to_list()
mlrg_series = [mlrg_monthly.get(d, np.nan) for d in all_dates]
transformed = transformed.with_columns(pl.Series("MLRG", mlrg_series))

# Drop rows where MLRG is NaN
df_with_target = transformed.filter(pl.col("MLRG").is_not_null())
print(f"Rows with MLRG target: {len(df_with_target)}")

# Convert to pandas for NeuralForecast
pdf = df_with_target.to_pandas()
pdf["date"] = pd.to_datetime(pdf["date"])

# Prepare features (all monthly covariates except GDP and MLRG)
feature_cols = [c for c in pdf.columns if c not in ("date", "GDP_LEVELS", "MLRG")]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Fill NaN in features with forward-fill then backward-fill
pdf[feature_cols] = pdf[feature_cols].ffill().bfill()

# Run transformer comparison
from eurocoin_research.models.transformers import run_transformer_comparison

# Use data up to 2020-01 as train, 2020+ as test (includes pandemic)
results = run_transformer_comparison(
    pdf,
    target_col="MLRG",
    feature_cols=feature_cols,
    train_end_date="2020-01-01",
    h=1,
)

print(f"\n{'='*60}")
print("=== Transformer Model Comparison ===")
print(f"{'='*60}")
for _, row in results.iterrows():
    status = row["status"]
    name = row["model"]
    n = row["n_forecasts"]
    fcst = row["first_forecast"]
    fcst_str = f"{fcst*100:+.4f}%" if fcst is not None else "N/A"
    print(f"  {name:15s}: {status:15s} n_fcst={n:3d}  first_fcst={fcst_str}")

print("\nTransformer test complete.")
