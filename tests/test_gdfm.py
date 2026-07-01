"""Test the GDFM baseline with frequency-domain projection."""
import sys
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import polars as pl
import openpyxl
from datetime import datetime

from eurocoin_research.data.target import compute_mlrg, gdp_levels_to_qoq_growth
from eurocoin_research.features.transforms import transform_panel
from eurocoin_research.models.gdfm import GDFM

# Load panel
panel = pl.read_parquet("data/processed/panel_monthly.parquet")
print(f"Panel: {len(panel)} months x {len(panel.columns)-1} series")

# Transform to stationary
transformed = transform_panel(panel)

# Extract GDP for target
gdp_panel = panel.filter(pl.col("GDP_LEVELS").is_not_null()).select(["date", "GDP_LEVELS"])
gdp_levels = gdp_panel["GDP_LEVELS"].to_numpy()
gdp_growth = gdp_levels_to_qoq_growth(gdp_levels)
mlrg = compute_mlrg(gdp_growth, method="ideal", max_lag=8)

# Prepare monthly covariate panel
monthly_cols = [c for c in transformed.columns if c != "date" and c != "GDP_LEVELS"]
monthly_df = transformed.select(monthly_cols)
arr = monthly_df.to_numpy().astype(float)

# Drop rows with excessive NaN
nan_frac = np.isnan(arr).sum(axis=1) / arr.shape[1]
valid_rows = nan_frac < 0.5
arr_clean = arr[valid_rows]
print(f"Monthly panel: {arr.shape} -> {arr_clean.shape}")

# Impute remaining NaNs
col_means = np.nanmean(arr_clean, axis=0)
for j in range(arr_clean.shape[1]):
    mask = np.isnan(arr_clean[:, j])
    arr_clean[mask, j] = col_means[j]

# Estimate GDFM with different q values
for q in [3, 5, 8]:
    print(f"\n{'='*60}")
    print(f"=== GDFM with q={q} factors ===")
    gdfm = GDFM(n_factors=q, max_lag=12, freq_band=(0.0, np.pi/6))
    result = gdfm.fit_transform(arr_clean)

    print(f"Eigenvalues: {np.round(result.eigenvalues, 4)}")
    print(f"Variance ratio: {result.variance_ratio:.1%}")

    # Frequency-domain projection
    projected = gdfm._project_target(
        result.factors, mlrg,
        factor_freq="monthly", target_freq="quarterly",
    )

    # Metrics
    valid = ~np.isnan(projected) & ~np.isnan(mlrg[:len(projected)])
    if valid.sum() > 5:
        corr = np.corrcoef(projected[valid], mlrg[:len(projected)][valid])[0, 1]
        rmse = np.sqrt(np.mean((projected[valid] - mlrg[:len(projected)][valid])**2))
        print(f"Correlation with MLRG: {corr:.3f}")
        print(f"RMSE: {rmse*100:.4f}%")

        # Compare with Eurocoin for pandemic
        print(f"\n{'Quarter':<10} {'GDP QoQ':>10} {'MLRG':>10} {'GDFM':>10} {'Eurocoin':>10}")
        wb = openpyxl.load_workbook("data/Ecoin_realtime.xlsx", data_only=True)
        ws = wb["Ecoin_real_time"]
        ecoin = {}
        for row in ws.iter_rows(min_row=3, max_row=ws.max_row, values_only=True):
            dt = row[1]
            val = row[2]
            if dt and val and isinstance(dt, datetime):
                qtr = (dt.month - 1) // 3 + 1
                key = f"{dt.year}-Q{qtr}"
                if key not in ecoin:
                    ecoin[key] = []
                ecoin[key].append(float(val))

        gdp_dates = gdp_panel["date"].to_list()
        for i, d in enumerate(gdp_dates):
            if d.year in [2019, 2020, 2021, 2022] and i < len(projected):
                qtr = (d.month - 1) // 3 + 1
                key = f"{d.year}-Q{qtr}"
                gdp_str = f"{gdp_growth[i]*100:+.2f}%" if not np.isnan(gdp_growth[i]) else "N/A"
                mlrg_str = f"{mlrg[i]*100:+.2f}%" if not np.isnan(mlrg[i]) else "N/A"
                gdfm_str = f"{projected[i]*100:+.2f}%" if not np.isnan(projected[i]) else "N/A"
                ecoin_avg = np.mean(ecoin.get(key, [float('nan')]))
                ecoin_str = f"{ecoin_avg:+.2f}" if not np.isnan(ecoin_avg) else "N/A"
                print(f"  {key:<10} {gdp_str:>10} {mlrg_str:>10} {gdfm_str:>10} {ecoin_str:>10}")

print(f"\n{'='*60}")
print("GDFM baseline test complete.")
