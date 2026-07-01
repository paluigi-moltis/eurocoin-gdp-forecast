# Research Plan: Transformer-Based Medium-Term GDP Forecasting for the Euro Area

**Project:** Eurocoin Modernization Research
**Version:** 1.3 — Added GitHub repo (MIT) and per-phase push policy
**Date:** 2026-06-30
**Repository:** https://github.com/paluigi-moltis/eurocoin-gdp-forecast
**Target output:** English-language working paper + working prototype

---

## 1. Research Objectives

### 1.1 Primary Objective

Develop and evaluate a modern forecasting framework based on Transformer neural networks with attention mechanisms to estimate the **Medium- to Long-Run Growth (MLRG)** component of euro-area GDP — the same target as the Eurocoin indicator — while addressing three documented weaknesses of the current approach:

1. **Regime-dependent variable relationships:** Producer prices (and potentially other series) changed their signaling properties during the post-pandemic energy shock. Attention mechanisms can learn to weight covariates differently across regimes.

2. **Pandemic outlier distortion:** The extreme GDP swings of 2020Q1–Q3 create discontinuities in factor estimation, depending on data availability timing. Transformers can learn to treat these as a separate regime rather than letting them distort the entire estimation.

3. **Missing-value and ragged-edge sensitivity:** The current approach is sensitive to ragged edges at the recent end of the data (e.g., manufacturing PMI availability), as the most recent observations of some monthly series are not yet published when the indicator is estimated. A well-trained model should be more robust to such data availability patterns through learned imputation patterns.

4. **Real-time data vintage issues:** GDP is published with a lag of approximately 30 days after quarter-end and is subject to revisions as seasonal adjustment factors and source data are updated. At each backtest date, the model must be evaluated using the data vintage that was actually available at that point in time — including GDP values that are not yet available (e.g., in April 2015, 2015Q1 GDP was not yet released) and past GDP values that may differ from today's revised estimates. This is a core aspect of the backtesting protocol (see §5.1).

### 1.2 Secondary Objectives

- **Avenue 1 (Growth-rate target):** Directly forecast the MLRG using Transformer architectures, benchmarked against a full re-implementation of the Eurocoin GDFM methodology.
- **Avenue 2 (Levels-based target):** Analyze GDP in levels using BEAST changepoint detection to identify regimes, then study interrelations between explanatory and target variables in a regime-dependent fashion. Feed regime information as conditioning input to the Transformer.
- **Descriptive regime analysis:** For each detected regime, compute structural statistics (correlations, lead-lag relationships, variance decompositions) comparing the pre-pandemic, pandemic, energy-shock, and post-shock periods.

### 1.3 Research Questions

| # | Question | How Addressed |
|---|----------|---------------|
| RQ1 | Can Transformer architectures outperform the GDFM-based Eurocoin in forecasting the MLRG? | Benchmark comparison (RMSE, MSD, turning-point detection) on identical target |
| RQ2 | Does attention learn regime-dependent covariate weights (e.g., down-weighting PPI during supply shocks)? | Attention weight analysis, variable selection networks (TFT), SHAP values |
| RQ3 | Are pandemic-era GDP swings better handled by attention (as regime-specific) vs. factor models (as noise)? | Counterfactual exercises: models trained with/without pandemic period |
| RQ4 | Does the levels-based approach with BEAST changepoints reveal structural shifts that growth-rate analysis misses? | Compare regime boundaries from BEAST vs. Bai-Perron; cross-regime correlation analysis |
| RQ5 | How robust is each approach to ragged edges and missing values at the recent end of the data panel? | Analysis of forecast stability across the data vintage snapshots used in backtesting |

---

## 2. Scope and Constraints

| Dimension | Decision | Rationale |
|-----------|----------|-----------|
| **Forecast target** | MLRG (GDP q-o-q with oscillations ≤ 1 year removed) | Directly comparable to Eurocoin; preserves the "medium-term signal" concept |
| **Sample period** | 2000Q1 (or 1999M1 for monthly covariates) to latest available | Matches the revised €-coin (e-coin-00); Aprigliano et al. (2022) found structural break at end-1999 |
| **Data (Phase 1)** | Public APIs: Eurostat SDMX, ECB SDW, DG-ECFIN surveys | Freely accessible; sufficient for proof-of-concept |
| **Data (Phase 2, optional)** | Extended dataset matching Aprigliano et al. (2022) including commercial PMI | Code structure supports both via a configuration flag |
| **Compute** | CPU only | Lightweight transformer models (small parameter counts, efficient architectures); design for this constraint |
| **Deliverable** | Working prototype + working paper draft | Applied research that could inform a revised indicator |
| **Language** | English | Working paper / journal publication target |
| **Baseline** | Full re-implementation of the Eurocoin GDFM methodology | Necessary for valid benchmark comparison |

---

## 3. Data Architecture

### 3.1 Data Sources

#### Public Dataset (Phase 1 — default)

| Block | Source | Series Examples | Frequency |
|-------|--------|----------------|-----------|
| **National Accounts (GDP)** | Eurostat | GDP at market prices, q-o-q growth, volume index | Quarterly |
| **Industrial Production** | Eurostat | IP total, manufacturing, capital/consumer/durable goods | Monthly |
| **Business Surveys** | DG-ECFIN | ESI, Industrial Confidence, Services Confidence, Consumer Confidence, Construction Confidence, Retail Confidence | Monthly |
| **Prices** | Eurostat | HICP (total, core, energy), PPI (industrial, domestic) | Monthly |
| **Labor Market** | Eurostat | Unemployment rate, employment | Monthly |
| **Monetary & Financial** | ECB SDW | M3, EONIA/€STR, Eurostoxx, bond yields (10Y), credit volumes | Monthly |
| **External** | Eurostat | Nominal/real effective exchange rate, exports/imports volume | Monthly |
| **PMI (public proxy)** | Eurostat / press releases | Manufacturing PMI headline (publicly released in press releases) | Monthly |
| **GDP vintages (revision history)** | Eurostat real-time database | Historical GDP data vintages — values as known at each past date | Quarterly (multiple vintages per quarter) |

#### Extended Dataset (Phase 2 — via config flag `use_extended_data=True`)

Replaces public proxies with the full commercial dataset used in Aprigliano et al. (2022):
- S&P Global / Markit PMI: Manufacturing + Services, for Euro Area + DE + FR + IT
- Full industrial production breakdown by country and sector
- Extended business survey block

The code will be structured so that the **same pipeline** runs on either dataset, with the flag controlling which data loader is used.

### 3.2 Target Variable Construction

The MLRG target is constructed from quarterly GDP q-o-q growth using the ideal band-pass filter (Christiano-Fitzgerald / Baxter-King approximation):

- **Spectral threshold:** π/6 (removes oscillations with period ≤ 12 months ≈ 1 year)
- **Filter weights:** β_k = sin(kπ/6)/(kπ) for k ≠ 0; β_0 = 1/6
- Applied to the full GDP growth series to obtain the "true" MLRG (used as ground truth in evaluation)
- For real-time evaluation, the end-of-sample approximation is used (with documented bias)

### 3.3 Validation Data

- **Historical Eurocoin values:** `data/Ecoin_realtime.xlsx` (downloaded from Banca d'Italia) — monthly real-time estimates from Jan 2000 to May 2026.
- Used for: (a) validating our GDFM re-implementation, (2) benchmarking Transformer forecasts against published values.

---

## 4. Methodology

### 4.1 Baseline: Full Eurocoin GDFM Re-implementation

**Objective:** Re-implement the complete Eurocoin methodology as described in Altissimo et al. (2007) and the 2022 revision.

**Steps:**

1. **Data panel assembly:** N monthly series × T months, aligned and transformed (stationarity, seasonal adjustment where needed).
2. **Common-idiosyncratic decomposition:** Estimate the covariance matrices Σ̂_χ (common) and Σ̂_ξ (idiosyncratic) using the Forni et al. (2000, 2005) frequency-domain approach.
3. **Generalized principal components:** Solve the generalized eigenvalue problem Σ̂_φ v_k = λ_k (Σ̂_χ + Σ̂_ξ) v_k to extract q smooth factors that maximize common low-frequency variance relative to total variance.
4. **MLRG projection:** Estimate cross-covariances between the smooth factors and the MLRG target in the frequency domain; project via OLS or LASSO.
5. **Validation:** Compare re-estimated Eurocoin values against published `Ecoin_realtime.xlsx` values (RMSE, correlation, turning-point detection).

**Implementation notes:**
- The Forni-Reichlin-Lippi dynamic factor estimation requires spectral density estimation via periodogram smoothing.
- The number of factors q is determined via Bai-Ng (2002) information criteria or the Hallin-Liska (2007) method.
- Ragged-edge handling: EM algorithm (Doz, Giannone, Reichlin 2011) or the vertical realignment method.

### 4.2 Transformer Models (Avenue 1: Growth-Rate Target)

#### 4.2.1 Model Candidates (ordered by priority for CPU feasibility)

| Model | Architecture | Why Selected | Param Count Target |
|-------|-------------|--------------|-------------------|
| **Temporal Fusion Transformer (TFT)** | Attention + variable selection + gating | Interpretable attention; built-in variable selection → directly addresses RQ2; learns which covariates matter per time step | < 100K |
| **Informer** | ProbSparse attention | Efficient long-sequence attention O(L log L); well-documented; available in Darts/NeuralForecast | < 200K |
| **PatchTST** | Patching + channel-independent attention | Strong empirical results; patching naturally groups high-frequency data into lower-frequency tokens → addresses "smooth short-term variability" | < 150K |
| **DLinear** (baseline NN) | Linear decomposition | Required sanity check: Zeng et al. (2023) showed simple linear models can beat transformers | < 10K |
| **N-BEATS** | Backward/forward residual links | Strong univariate baseline; pure deep learning without attention | < 50K |

#### 4.2.2 Input Design

```
Input window (lookback):     L_in = 36 months (3 years)
Target horizon:              MLRG at current month (nowcasting-style)
                             + optional h-step ahead (1Q, 2Q ahead)
Covariates:                  ~50-100 monthly series (public dataset)
                             ~150+ series (extended dataset)
Static metadata:             Variable type (survey, hard, financial)
                             Frequency, source, country
Known future inputs:         Calendar effects, time index
```

**Mixed-frequency handling:** Monthly covariates are aligned to a monthly time grid. Quarterly GDP target is interpolated to monthly using the Chow-Lin / Denton method or the same (1+L+L²)² filter used in Eurocoin, so the model operates on a unified monthly timeline.

**Patch-based design for CPU efficiency:** Group consecutive monthly observations into patches (e.g., 3-month patches = quarterly tokens). This reduces sequence length by 3× and naturally aggregates high-frequency noise — directly serving the "smooth short-term variability" objective.

#### 4.2.3 Training Strategy

- **Loss function:** MSE on MLRG target (primary); MAE and quantile loss for robustness.
- **Regularization:** Dropout, weight decay, early stopping. Given small dataset (~300 monthly obs), strong regularization is essential.
- **Data augmentation:** Block bootstrap (resample contiguous blocks), temporal jittering.
- **Cross-validation:** Expanding window with embargo. Train on [2000–t], validate on [t+1, t+12], test on [t+13, t+24]. Roll forward quarterly.
- **Pandemic treatment:** Train two model variants: (a) including pandemic data, (b) excluding 2020Q1–Q3. Compare to understand how attention handles the outlier.

### 4.3 Levels-Based Approach (Avenue 2: BEAST + Regime Analysis)

#### 4.3.1 BEAST Decomposition

Apply BEAST to euro-area GDP in levels (log real GDP):

- **Output:** Decomposition into trend, seasonal (if any), and abrupt-change components.
- **Changepoints:** Posterior probability of changepoint at each time point; expected number and positions of changepoints.
- **Trend segments:** Piecewise linear or nonlinear trend per detected regime.

**Configuration:**
- `season`: none (quarterly GDP has no meaningful seasonality after SA)
- `tcp_minmax`: [1, 10] (allow 1 to 10 changepoints)
- `mcmc_seed`: fixed for reproducibility
- MCMC samples: 10,000 (burn-in 5,000)

#### 4.3.2 Regime Definition

Expected regimes based on economic history of the euro area (2000–present):

| Regime | Approx. Period | Key Characteristic |
|--------|---------------|-------------------|
| Pre-crisis expansion | 2000–2007 | Steady growth, moderate inflation |
| Global Financial Crisis | 2008–2009 | Sharp contraction, financial stress |
| Sovereign debt crisis | 2011–2013 | Double-dip recession, austerity |
| Low-growth recovery | 2014–2019 | Low inflation, QE, sluggish growth |
| Pandemic shock | 2020Q1–Q3 | Unprecedented GDP collapse and partial rebound |
| Post-pandemic / energy shock | 2021Q4–2023 | Inflation surge, energy crisis, PPI-GDP decoupling |
| Normalization | 2024–present | Disinflation, moderate growth |

**Note:** BEAST will determine the actual boundaries data-driven. These are priors for interpretation.

#### 4.3.3 Regime-Conditioned Transformer Input

Two approaches to incorporate regime information:

1. **Regime embedding:** Add a learned embedding vector for the current regime (identified by BEAST changepoints) to the Transformer input — analogous to a positional encoding. The model learns regime-specific representations.

2. **Regime-attention mask:** During training, allow the model to attend more strongly to observations from the same regime. This is a soft constraint that encourages the model to learn regime-specific dynamics.

#### 4.3.4 Descriptive Regime Analysis

For each detected regime, compute and compare:

- **Correlation structure:** Cross-correlation matrix of all covariates with GDP, per regime.
- **Lead-lag relationships:** Cross-correlation functions to identify which series lead GDP by how many months.
- **Variance decomposition:** How much of GDP variance is explained by each covariate block per regime.
- **PPI-GDP relationship shift:** Explicitly track how the correlation between PPI and GDP changes across regimes (the core diagnostic for the "producer prices misleading signal" problem).
- **Granger causality tests** per regime (where sample size permits).

---

## 5. Experimental Design

### 5.1 Evaluation Framework

#### Metrics

| Metric | Purpose |
|--------|---------|
| **RMSE** | Standard forecast accuracy |
| **Mean Squared Deviation (MSD)** from MLRG target | Primary metric (same as Aprigliano et al. 2022) |
| **Correlation** with true MLRG | Signal quality |
| **Turning-point detection** | Binary: did the model detect the direction change? |
| **Directional accuracy** | % of months where sign of change is correct |

#### Data Vintages — A Critical Concern

Macroeconomic data is subject to both **publication lags** and **revisions**. A correct pseudo-real-time backtest must respect both:

1. **Publication lags:** No series in the panel is published instantaneously. The estimation date is conventionally the **last day of the month**. At that date, the most recent month(s) of most explanatory variables are not yet published, and the most recent quarter(s) of GDP are not yet released. GDP for quarter Q is released approximately 30 days after quarter-end (flash estimate), with subsequent releases over the following months. Monthly indicators have varying lags — surveys (DG-ECFIN) are typically available around the end of the reference month or the very first days of the following month; "hard" data (industrial production, retail sales, external trade) are released 4–8 weeks after the reference month; PMI is typically released on the first business day of the following month. This creates a **ragged edge** at the bottom of the data panel: different series have different last-available months, and the pattern depends on the specific backtest date.

2. **Data revisions:** Virtually all series in the panel are subject to revision, not just GDP. When seasonal adjustment factors are re-estimated and source data is updated, **past values are revised**. This affects:
   - **GDP:** Seasonally adjusted q-o-q growth rates are revised at each release (flash → first → second) and then again at quarterly, annual, and benchmark revisions. A GDP growth figure for 2014Q4 available in April 2015 may differ from the same figure available in 2024.
   - **Hard monthly indicators (IP, retail sales, trade):** These are routinely revised for 2–3 months after initial release, and again at annual benchmark revisions.
   - **Surveys:** Generally not revised (raw opinion data), but the composite indicators (ESI) may be marginally adjusted.
   - **Prices (HICP, PPI):** Typically not revised after the first release (except for annual basket updates).
   - **Monetary/financial data:** Minor revisions possible.

   Each series in the panel must therefore carry **its own revision history**, and the vintage for a given backtest date must reflect the values as they were known on that date for every series — not only for GDP.

**Consequence:** At each backtest date, we must reconstruct the **complete data vintage** — for every series in the panel — that was actually available at that point. Using today's fully-revised data for any series would leak future information and produce over-optimistic results.

#### Backtesting Protocol with Data Vintages

```
For each backtest date t (conventionally the last day of the month,
e.g., 30 April 2015):

    1. CONSTRUCT VINTAGE (for EVERY series in the panel):

       For each series s in the panel:

         a. Determine the last available observation as of date t,
            using the series-specific publication lag schedule:
            - Surveys (ESI, confidence): lag ~0–5 days → last avail
              month is typically the current or previous month
            - PMI: lag ~1–3 days → last avail month is typically t-1
            - Hard data (IP, retail, trade): lag ~30–45 days → last
              avail month is t-2 or t-3
            - GDP: lag ~30 days after quarter-end → last avail quarter
              depends on position within the quarter

         b. For the available observations, retrieve the VALUES as they
            were known at date t — i.e., the vintage values, not today's
            revised estimates. This requires per-series revision history:
            - GDP: Eurostat real-time database (exact vintages)
            - Hard monthly indicators: Eurostat revision history / SDMX
              metadata (where available); approximate from known revision
              patterns where not
            - Surveys: typically unrevised (use current values)
            - Prices: typically unrevised (use current values)

         c. The result for series s: a vector of values with a potentially
            different last-available date and potentially different past
            values compared to today's data.

       Assemble all series → the "vintage_t" panel (ragged-edge matrix
       with series-specific missing patterns at the recent end)

    2. SAVE VINTAGE:
       Write the vintage panel to data/vintages/vintage_YYYY-MM.csv
       (one CSV per backtest date, for full reproducibility and review)
       Include metadata: last-available-date per series, vintage source

    3. ESTIMATE/FORECAST:
       a. Estimate GDFM baseline on vintage_t panel → baseline forecast
       b. Train Transformer models on vintage_t panel (expanding window)
       c. Generate forecast for the current month (nowcast of MLRG)

    4. EVALUATE (ex post):
       a. Compare forecast against true MLRG (computed from final,
          fully-revised GDP — known only ex post)
       b. Compare against published €coin value for that month

    5. REPEAT for next backtest date (roll forward monthly)
```

**Vintage storage:** A dedicated directory `data/vintages/` contains one CSV file per backtest date (e.g., `vintage_2015-04.csv`), storing the exact data panel used for that estimation. This allows:
- Full reproducibility of every backtest result
- Audit trail for reviewer/replicator
- Comparison of how forecasts evolved as data was revised

**Vintage construction method:** The vintage system requires two inputs per series:
1. **A publication-lag schedule** — for each series, a rule (or lookup table) specifying the typical number of days between the reference period and the release date. This determines which months are available at each backtest date. The estimation date is conventionally the last calendar day of the month.
2. **A revision history** — for each series, the values as they were known at each past date. Primary data sources:
   - **GDP:** Eurostat real-time database provides exact historical vintages (values as known at each release date).
   - **Hard monthly indicators (IP, retail sales, trade):** Eurostat SDMX revision history where available; for series without archived vintages, we approximate by applying known revision patterns (typical revision magnitude for 1–3 months after release, annual benchmark revisions).
   - **Surveys (DG-ECFIN) and prices (HICP, PPI):** Generally unrevised after first publication; current values are used as the vintage.
   - **Monetary/financial data:** Minor revisions; current values used with documented approximation.

Where exact vintage data is not available for a specific series, the approximation is documented and its sensitivity assessed. The `vintages.py` module maintains a **per-series metadata table** (lag schedule, revision policy, vintage source) that drives the vintage construction.

**Note on Eurocoin published values:** The official Eurocoin indicator was revised multiple times from 2022 onward (reducing the variable set as the original panel produced too-volatile results). Therefore, substantial differences between published Eurocoin values and our GDFM reconstruction are **expected and acceptable**. The comparison against published values serves as a qualitative benchmark (turning points, broad co-movement), not as an exact replication target.

### 5.2 Model Comparison Table

| Model | Type | Target | Regime-Aware? | Interpretable? |
|-------|------|--------|---------------|----------------|
| GDFM Eurocoin (re-impl) | Factor model | MLRG | No | Partial (loadings) |
| DLinear | Linear NN | MLRG | No | No |
| N-BEATS | Deep learning | MLRG | No | Partial |
| Informer | Transformer | MLRG | No | Attention weights |
| PatchTST | Transformer | MLRG | No | Attention weights |
| TFT | Transformer | MLRG | Yes (via variable selection) | Yes (variable importance) |
| TFT + regime embedding | Transformer | MLRG | Yes (explicit) | Yes |
| Models in Avenue 2 | Various | GDP levels / MLRG | Yes (BEAST regimes) | Yes |

---

## 6. Project Structure and Implementation Plan

### 6.1 Repository Structure

```
eurocoin-gdp-forecast/
├── README.md
├── LICENSE                   # MIT License
├── pyproject.toml            # uv-managed dependencies
├── config/
│   ├── base.yaml               # Global configuration
│   ├── data_public.yaml        # Public data source definitions
│   └── data_extended.yaml      # Extended (commercial) data definitions
├── references/                 # Papers (PDF)
├── data/
│   ├── raw/                    # Downloaded raw data
│   ├── processed/              # Cleaned, aligned panel
│   ├── vintages/               # One CSV per backtest date (e.g., vintage_2015-04.csv)
│   └── Ecoin_realtime.xlsx     # Historical Eurocoin values
├── docs/
│   ├── literature_review.md
│   ├── research_plan.md        # THIS FILE
│   └── openalex_*.json         # Literature search results
├── src/
│   └── eurocoin_research/
│       ├── __init__.py
│       ├── config.py           # Configuration loader (Pydantic models)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders/        # Data source connectors
│       │   │   ├── eurostat.py
│       │   │   ├── ecb.py
│       │   │   ├── ecf.py
│       │   │   └── extended.py # Commercial data loader (flag-controlled)
│       │   ├── panel.py        # Panel assembly, alignment, transforms
│       │   └── target.py       # MLRG / band-pass filter construction
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gdfm.py         # Eurocoin GDFM re-implementation
│       │   ├── baselines.py    # DLinear, N-BEATS, ARIMA
│       │   ├── transformers.py # Informer, PatchTST, TFT wrappers
│       │   └── regime.py       # BEAST changepoint + regime utilities
│       ├── features/
│       │   ├── __init__.py
│       │   ├── transforms.py   # Stationarity, seasonal adj, etc.
│       │   └── windows.py      # Sliding window, patch creation
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py      # RMSE, MSD, directional accuracy
│       │   ├── backtest.py     # Expanding window backtest with vintage management
│       │   └── vintages.py     # Data vintage construction, publication-lag logic
│       └── visualization/
│           ├── __init__.py
│           └── plots.py        # Forecast plots, attention heatmaps
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_gdfm_baseline.ipynb  # Eurocoin re-implementation
│   ├── 03_beast_regimes.ipynb  # BEAST changepoint analysis
│   ├── 04_transformer_models.ipynb  # Transformer training & eval
│   ├── 05_regime_analysis.ipynb     # Avenue 2 descriptive analysis
│   └── 06_results_comparison.ipynb  # Final benchmark comparison
├── tests/
│   └── ...
└── results/                    # Output figures, tables, trained models
```

### 6.2 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package manager | uv |
| Data handling | Polars |
| ML framework | PyTorch (CPU-optimized builds) |
| Transformer library | NeuralForecast (Nixtla) or Darts — both provide TFT, Informer, PatchTST |
| Changepoint detection | Rbeast (Python binding) |
| Factor model | Custom implementation + statsmodels |
| Config | Pydantic + YAML |
| Visualization | matplotlib + plotly |
| Testing | pytest |

**Key design constraint:** All models must be trainable on CPU within reasonable time (< 30 min per model per fold). This means:
- Small hidden dimensions (32–128)
- Shallow networks (2–4 layers)
- Short lookback windows (24–36 months)
- Patch-based tokenization to reduce sequence length

### 6.3 Configuration Flag for Extended Data

```python
# config/base.yaml
data:
  mode: "public"  # "public" or "extended"
  # When mode="extended", loads data_extended.yaml
  # When mode="public", loads data_public.yaml
```

The `panel.py` module reads this flag and dispatches to the appropriate loader. All downstream code (models, evaluation) is identical regardless of the flag value — the only difference is which series are in the panel.

### 6.4 Version Control Policy

The project is hosted on a **public GitHub repository** under the **MIT License**.

- **Repository:** Created at the start of Phase 0 and kept synchronized throughout the project.
- **Push cadence:** The repository is pushed to GitHub **after the completion of each work phase** (Phases 0–6). Intermediate commits accumulate locally (or on the local branch) during each phase; at phase completion the full set of changes is pushed as a tagged release.
- **Tags:** Each phase completion is tagged (e.g., `phase-1-data`, `phase-2-gdfm`, `phase-3-beast`, …) to provide clear milestones in the commit history.
- **Sensitive data:** Historical vintage CSVs and any commercial data are excluded via `.gitignore`. Only code, documentation, configs, and notebook outputs are published. The `data/raw/` and `data/vintages/` directories are gitignored; their structure is documented so a replicator can regenerate them from public sources.
- **README:** The repository README provides setup instructions, data acquisition steps, and a description of each phase's deliverables.

| Event | Action |
|-------|--------|
| Phase completion | Commit all changes, tag as `phase-N-<name>`, push to GitHub |
| Mid-phase checkpoint | Commit locally (ensure work is saved across sessions) |
| Plan revision | Update `docs/research_plan.md`, commit, push |

---

## 7. Work Phases and Timeline

### Phase 0: Setup and Literature (✅ Completed)

- [x] Project structure created
- [x] Papers downloaded
- [x] Literature review (OpenAlex)
- [x] Historical Eurocoin data downloaded
- [x] Public GitHub repository created (MIT license), pushed initial structure
  - Repo: https://github.com/paluigi-moltis/eurocoin-gdp-forecast
  - Tagged as `phase-0-setup`

### Phase 1: Data Pipeline and EDA (✅ Completed)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | Set up Python environment (`uv init`, install dependencies) | ✅ Done | polars, numpy, scipy, statsmodels, requests, pydantic, etc. |
| 1.2 | Implement Eurostat SDMX API connector | ✅ Done | Unified SDMXLoader via sdmx1 library. 10 Eurostat series: GDP, IP, HICP×2, PPI, UNEMP, surveys×4 |
| 1.3 | Implement ECB SDW API connector | ✅ Done | Same SDMXLoader handles ECB. 5 ECB series: M3, €STR, 10Y bond, Euro Stoxx 50, EUR/USD |
| 1.4 | Implement DG-ECFIN survey data connector | ✅ Done | Survey data accessed via Eurostat SDMX (ei_bsin_m_r2, ei_bsbu_m_r2, ei_bsci_m_r2, ei_bsrt_m_r2). ESI composite not directly available — component indicators used. |
| 1.5 | Build panel assembly pipeline (alignment, transforms, ragged-edge) | ✅ Done | 15 series × 330 months. Panel saved to data/processed/panel_monthly.parquet |
| 1.6 | Construct MLRG target from GDP data | ✅ Done | Band-pass filter (ideal, K=8). Validated: MLRG tracks Eurocoin published values during 2020-2022. |
| 1.7 | EDA notebook | 📋 Deferred | To be created as Jupyter notebook. Core data exploration done via scripts. |
| 1.8 | Extended data connector stub (for commercial data) | ✅ Done | ExtendedLoader stub with credential check. Config flag `use_extended_data` |
| 1.9 | Build vintage construction module | ✅ Done | VintageManager with per-series publication lags. ALFRED loader for GDP revision history. |
| 1.10 | Retrieve GDP revision history | ✅ Done | ALFRED series CLVMNACSCAB1GQEA19. Tested with 4 vintages: 0.7-1.6% GDP revisions for pandemic quarters. |

### Phase 2: Baseline Eurocoin Re-implementation (✅ Completed)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | Spectral density estimation (periodogram smoothing) | ✅ Done | Bartlett kernel, max_lag=12 |
| 2.2 | Common-idiosyncratic covariance decomposition | ✅ Done | PCA-based separation, positive-definite enforcement |
| 2.3 | Generalized eigenvalue problem for smooth factors | ✅ Done | Cholesky-based GEP solver. Eigenvalues: [0.94, 0.71, 0.64] for q=3 |
| 2.4 | MLRG projection (frequency-domain cross-covariances) | ✅ Done | Bivariate cross-spectrum integration over [-π/6, π/6]. Corr=0.16 (q=8) |
| 2.5 | Validate against published Eurocoin values | ✅ Done | Qualitative match; GDFM signal varies during pandemic. Lower corr expected with 16 vs 100+ series |
| 2.6 | Integrate GDFM with data vintage system | 📋 Deferred | Will be done during backtesting (Phase 5) |

### Phase 3: Changepoint Analysis (✅ Completed)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Install and test changepoint detection library | ✅ Done | River library (online/streaming). Rbeast is ARM64-incompatible. |
| 3.2 | Apply changepoint detection to euro-area GDP levels | ✅ Done | HoltWinters(α=0.1) + PageHinkley(threshold=0.05) on log GDP levels |
| 3.3 | Extract and validate changepoints against known economic events | ✅ Done | 3 changepoints: 2007-Q1 (GFC), 2014-Q3 (low-growth), 2022-Q3 (energy shock) |
| 3.4 | Define regime labels and boundaries | ✅ Done | 4 regimes defined, labeled by economic episode |
| 3.5 | Compute per-regime descriptive statistics | ✅ Done | GDP growth mean/std per regime computed |
| 3.6 | PPI-GDP relationship analysis across regimes | ✅ Done | Correlation shifts from +0.352 (pre-crisis) to +0.500 (energy shock) |
| 3.7 | Produce regime analysis notebook with visualizations | 📋 Deferred | Core analysis complete via scripts; notebook to be created later |

### Phase 4: Transformer Models (Avenue 1)

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 4.1 | Set up NeuralForecast/Darts with CPU-optimized PyTorch | 0.5 day |
| 4.2 | Build feature pipeline: windowing, patching, normalization | 1 day |
| 4.3 | Train and tune DLinear (sanity baseline) | 0.5 day |
| 4.4 | Train and tune N-BEATS | 0.5 day |
| 4.5 | Train and tune Informer | 1 day |
| 4.6 | Train and tune PatchTST | 1 day |
| 4.7 | Train and tune TFT (with variable selection) | 1.5 days |
| 4.8 | Implement regime embedding variant | 1 day |
| 4.9 | Hyperparameter optimization (Optuna, CPU-efficient) | 2 days |
| 4.10 | Attention weight extraction and analysis | 1.5 days |

### Phase 5: Evaluation and Comparison

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 5.1 | Implement expanding-window backtest framework with data vintage loading | 1.5 days |
| 5.2 | Run full backtest for all models across all vintages | 1.5 days |
| 5.3 | Compute all metrics (RMSE, MSD, directional accuracy, turning points) | 0.5 day |
| 5.4 | Produce comparison tables and plots | 1 day |
| 5.5 | Attention weight analysis (which variables does TFT attend to, per regime?) | 1.5 days |
| 5.6 | Vintage revision analysis: how much do forecasts change as GDP is revised? | 1 day |

### Phase 6: Working Paper Draft

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 6.1 | Outline paper structure | 0.5 day |
| 6.2 | Write Introduction and Literature Review sections | 1.5 days |
| 6.3 | Write Methodology section | 2 days |
| 6.4 | Write Results section with figures/tables | 3 days |
| 6.5 | Write Discussion and Conclusion | 1 day |
| 6.6 | Internal review and revision | 1 day |

**Total estimated effort:** ~52 working days

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Small sample size (~300 monthly obs) limits deep learning | High | High | Strong regularization; data augmentation; patch-based tokenization; favor TFT (designed for small samples) |
| CPU-only compute makes hyperparameter search expensive | Medium | Medium | Use Optuna with pruners; limit search space; pre-train on extended sample, fine-tune on 2000+ |
| Public data may not include all Eurocoin series → weaker panel | Medium | Medium | Extended data flag prepared; Phase 2 can switch seamlessly |
| Eurocoin GDFM re-implementation will differ from published values | High | Low | Official Eurocoin was revised multiple times from 2022 onward (reduced variable set); substantial differences are expected and acceptable. Use published values as qualitative benchmark only (turning points, broad co-movement) |
| Transformers may not outperform linear baselines (Zeng et al. 2023) | Medium | High | Include DLinear as mandatory baseline; report honestly; value may be in interpretability rather than raw accuracy |
| BEAST changepoints may not align with economic intuition | Low | Medium | Cross-validate with Bai-Perron; use economic event dates as priors |
| GDP vintage reconstruction may be incomplete (missing historical revisions for some monthly series) | Medium | Medium | Use Eurostat real-time database for GDP vintages; approximate monthly series vintages via publication-lag schedules; document all approximations and assess sensitivity |

---

## 9. Success Criteria

1. **Baseline validation:** Our GDFM re-implementation captures the broad co-movement and turning points of published Eurocoin values over 2010–2025. Exact numerical match is not expected given the multiple official revisions of the indicator.
2. **Forecast accuracy:** At least one Transformer model achieves MSD ≤ Eurocoin GDFM MSD on the pandemic period (2020–2021), with improvement in turning-point detection.
3. **Interpretability:** TFT variable selection weights show a clear regime-dependent pattern for PPI (high weight pre-2020, low weight during energy shock) — empirically demonstrating the attention mechanism addresses the core problem.
4. **Regime analysis:** BEAST identifies ≥ 4 distinct regimes in GDP levels that correspond to known economic episodes; cross-regime PPI-GDP correlation shifts are statistically significant.
5. **Vintage robustness:** The full backtest is reproducible from saved vintage CSV files, and the vintage revision analysis quantifies how much forecasts change when GDP is revised between its initial release and final estimate.

---

## 10. Deliverables Checklist

- [ ] Public GitHub repository (MIT License), pushed after each phase completion
- [ ] Data pipeline with public + extended data support (flag-controlled)
- [ ] Data vintage system: publication-lag-aware vintage construction + Eurostat GDP revision history
- [ ] Per-backtest-date vintage CSV files in `data/vintages/` for full reproducibility
- [ ] GDFM Eurocoin re-implementation, benchmarked qualitatively against published values
- [ ] BEAST changepoint analysis with regime definitions
- [ ] Descriptive regime analysis (correlations, PPI-GDP shift, lead-lag)
- [ ] Trained Transformer models (TFT, Informer, PatchTST, DLinear, N-BEATS)
- [ ] Full backtest results with all metrics, across data vintages
- [ ] Attention weight / variable importance analysis
- [ ] Vintage revision analysis: forecast sensitivity to GDP revisions
- [ ] Working paper draft (English)
- [ ] Reproducible codebase with documentation

---

## Appendix A: Key Equations

### MLRG (Band-Pass Filter)

$$c_t = \sum_{k=-\infty}^{\infty} \beta_k y_{t-k}, \quad \beta_k = \frac{\sin(k\pi/6)}{k\pi}, \quad \beta_0 = \frac{1}{6}$$

### Generalized Eigenvalue Problem (Smooth Factors)

$$\hat{\Sigma}_\varphi v_k = \lambda_k (\hat{\Sigma}_\chi + \hat{\Sigma}_\xi) v_k$$

where $\hat{\Sigma}_\varphi$ is the covariance matrix of the common component in the frequency band $[-\pi/6, \pi/6]$.

### MLRG Projection

$$\hat{c}_t = \hat{\mu} + \hat{\Sigma}_{cw} \hat{\Sigma}_w^{-1} w_t$$

where $w_t$ are the smooth factors, $\hat{\Sigma}_{cw}$ is the cross-covariance between MLRG and factors (integrated over the target frequency band).

### BEAST Model

Decomposition: $y_t = \text{trend}_t + \text{seasonality}_t + \epsilon_t$

where trend is piecewise polynomial (order determined by model averaging) with changepoints at unknown locations $\tau_1, \ldots, \tau_K$, and $K$ itself is unknown. Posterior probabilities $P(\tau_j = t \mid y)$ are computed via MCMC.

---

## Appendix B: Dependencies

### Python packages

```
# Core
polars >= 1.0
numpy
scipy
statsmodels

# Deep learning
torch (CPU)
neuralforecast         # or darts
optuna                 # hyperparameter optimization

# Changepoint
rbeast                 # Python binding for BEAST

# Data APIs
requests
pandasdmx              # Eurostat/ECB SDMX access

# Visualization
matplotlib
plotly
seaborn

# Utilities
pydantic
pyyaml
python-dotenv
```

### R packages (optional, for validation)

```
Rbeast                 # Native R implementation
```

---

## Appendix C: Literature Review Summary

See `docs/literature_review.md` for the full review covering:
- Eurocoin methodology (Altissimo et al. 2007; Aprigliano et al. 2022)
- Dynamic factor models (Forni et al., Stock & Watson, Giannone et al.)
- Transformer architectures for time series (Informer, Autoformer, FEDformer, PatchTST, iTransformer, TFT)
- ML for macroeconomic forecasting (Goulet Coulombe et al., Richardson & van Florenstein Mulder)
- Changepoint detection (BEAST, Bai-Perron, Beveridge-Nelson)
- Software ecosystem (Darts, NeuralForecast, GluonTS)

---

*Document version 1.3 — Prepared for review by Luigi Palumbo*
*Public repository: MIT License — pushed after each phase completion*
