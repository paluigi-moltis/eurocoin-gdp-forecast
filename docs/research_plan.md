# Research Plan: Transformer-Based Medium-Term GDP Forecasting for the Euro Area

**Project:** Eurocoin Modernization Research
**Version:** 1.0 — Draft for Review
**Date:** 2026-06-30
**Target output:** English-language working paper + working prototype

---

## 1. Research Objectives

### 1.1 Primary Objective

Develop and evaluate a modern forecasting framework based on Transformer neural networks with attention mechanisms to estimate the **Medium- to Long-Run Growth (MLRG)** component of euro-area GDP — the same target as the Eurocoin indicator — while addressing three documented weaknesses of the current approach:

1. **Regime-dependent variable relationships:** Producer prices (and potentially other series) changed their signaling properties during the post-pandemic energy shock. Attention mechanisms can learn to weight covariates differently across regimes.

2. **Pandemic outlier distortion:** The extreme GDP swings of 2020Q1–Q3 create discontinuities in factor estimation, depending on data availability timing. Transformers can learn to treat these as a separate regime rather than letting them distort the entire estimation.

3. **Missing-value sensitivity:** The current approach is sensitive to ragged edges at the recent end of the data (e.g., manufacturing PMI availability). A well-trained model should be more robust to missing values through learned imputation patterns.

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
| RQ5 | How robust is each approach to ragged-edge missing values at the recent end? | Simulated real-time experiment with staggered data availability |

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
| **Real-time tracking error** | MSD computed using only data available at each historical month |

#### Backtesting Protocol

```
For each quarter-end t from 2010Q1 to latest:
    1. Assemble data panel as of date t (respecting publication lags)
    2. Estimate baseline GDFM Eurocoin → baseline forecast
    3. Train Transformer models on data [2000, t] (expanding window)
    4. Generate forecast for month t (nowcast) and t+3, t+6
    5. Compare against true MLRG (known ex post)
    6. Compare against published €coin value for month t
```

**Ragged-edge simulation:** For each t, create multiple "vintages" within the month (e.g., day 5, day 15, day 25) by progressively revealing data. Test how each model's forecast changes as more data arrives — directly measuring sensitivity to missing values.

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
├── pyproject.toml              # uv-managed dependencies
├── config/
│   ├── base.yaml               # Global configuration
│   ├── data_public.yaml        # Public data source definitions
│   └── data_extended.yaml      # Extended (commercial) data definitions
├── references/                 # Papers (PDF)
├── data/
│   ├── raw/                    # Downloaded raw data
│   ├── processed/              # Cleaned, aligned panel
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
│       │   ├── backtest.py     # Expanding window backtest
│       │   └── realtime.py     # Ragged-edge simulation
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

---

## 7. Work Phases and Timeline

### Phase 0: Setup and Literature (✅ Completed)

- [x] Project structure created
- [x] Papers downloaded
- [x] Literature review (OpenAlex)
- [x] Historical Eurocoin data downloaded

### Phase 1: Data Pipeline and EDA

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 1.1 | Set up Python environment (`uv init`, install dependencies) | 0.5 day |
| 1.2 | Implement Eurostat SDMX API connector | 1 day |
| 1.3 | Implement ECB SDW API connector | 0.5 day |
| 1.4 | Implement DG-ECFIN survey data connector | 0.5 day |
| 1.5 | Build panel assembly pipeline (alignment, transforms, ragged-edge handling) | 1.5 days |
| 1.6 | Construct MLRG target from GDP data | 0.5 day |
| 1.7 | EDA notebook: data availability heatmap, descriptive stats, missing patterns | 1 day |
| 1.8 | Extended data connector stub (for commercial data) | 0.5 day |

### Phase 2: Baseline Eurocoin Re-implementation

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 2.1 | Implement spectral density estimation (periodogram smoothing) | 1 day |
| 2.2 | Implement common-idiosyncratic covariance decomposition (Forni et al.) | 2 days |
| 2.3 | Implement generalized eigenvalue problem for smooth factors | 1 day |
| 2.4 | Implement MLRG projection (frequency-domain cross-covariances) | 1 day |
| 2.5 | Validate against published Eurocoin values | 1 day |
| 2.6 | Implement ragged-edge handling (EM / vertical realignment) | 1 day |

### Phase 3: BEAST Changepoint Analysis (Avenue 2 Foundation)

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 3.1 | Install and test Rbeast Python binding | 0.5 day |
| 3.2 | Apply BEAST to euro-area GDP levels | 0.5 day |
| 3.3 | Extract and validate changepoints against known economic events | 0.5 day |
| 3.4 | Define regime labels and boundaries | 0.5 day |
| 3.5 | Compute per-regime descriptive statistics (correlations, lead-lag, variance decomposition) | 1.5 days |
| 3.6 | PPI-GDP relationship analysis across regimes (the key diagnostic) | 1 day |
| 3.7 | Produce regime analysis notebook with visualizations | 1 day |

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
| 5.1 | Implement expanding-window backtest framework | 1 day |
| 5.2 | Implement ragged-edge simulation (multiple vintages per month) | 1.5 days |
| 5.3 | Run full backtest for all models | 1 day |
| 5.4 | Compute all metrics (RMSE, MSD, directional accuracy, turning points) | 0.5 day |
| 5.5 | Produce comparison tables and plots | 1 day |
| 5.6 | Attention weight analysis (which variables does TFT attend to, per regime?) | 1.5 days |

### Phase 6: Working Paper Draft

| Task | Description | Est. Effort |
|------|-------------|-------------|
| 6.1 | Outline paper structure | 0.5 day |
| 6.2 | Write Introduction and Literature Review sections | 1.5 days |
| 6.3 | Write Methodology section | 2 days |
| 6.4 | Write Results section with figures/tables | 3 days |
| 6.5 | Write Discussion and Conclusion | 1 day |
| 6.6 | Internal review and revision | 1 day |

**Total estimated effort:** ~50 working days

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Small sample size (~300 monthly obs) limits deep learning | High | High | Strong regularization; data augmentation; patch-based tokenization; favor TFT (designed for small samples) |
| CPU-only compute makes hyperparameter search expensive | Medium | Medium | Use Optuna with pruners; limit search space; pre-train on extended sample, fine-tune on 2000+ |
| Public data may not include all Eurocoin series → weaker panel | Medium | Medium | Extended data flag prepared; Phase 2 can switch seamlessly |
| Eurocoin GDFM re-implementation may not perfectly match published values | Medium | Low | Focus on correlation and turning-point match, not exact replication; document discrepancies |
| Transformers may not outperform linear baselines (Zeng et al. 2023) | Medium | High | Include DLinear as mandatory baseline; report honestly; value may be in interpretability rather than raw accuracy |
| BEAST changepoints may not align with economic intuition | Low | Medium | Cross-validate with Bai-Perron; use economic event dates as priors |
| Ragged-edge simulation is complex to implement realistically | Medium | Medium | Start with simplified vintage structure; refine iteratively |

---

## 9. Success Criteria

1. **Baseline validation:** Our GDFM re-implementation correlates > 0.90 with published Eurocoin values over 2010–2025.
2. **Forecast accuracy:** At least one Transformer model achieves MSD ≤ Eurocoin GDFM MSD on the pandemic period (2020–2021), with improvement in turning-point detection.
3. **Interpretability:** TFT variable selection weights show a clear regime-dependent pattern for PPI (high weight pre-2020, low weight during energy shock) — empirically demonstrating the attention mechanism addresses the core problem.
4. **Regime analysis:** BEAST identifies ≥ 4 distinct regimes in GDP levels that correspond to known economic episodes; cross-regime PPI-GDP correlation shifts are statistically significant.
5. **Robustness:** Transformer forecasts are less sensitive to ragged-edge missing values than the factor model (lower variance across vintages within a month).

---

## 10. Deliverables Checklist

- [ ] Data pipeline with public + extended data support (flag-controlled)
- [ ] GDFM Eurocoin re-implementation, validated against published values
- [ ] BEAST changepoint analysis with regime definitions
- [ ] Descriptive regime analysis (correlations, PPI-GDP shift, lead-lag)
- [ ] Trained Transformer models (TFT, Informer, PatchTST, DLinear, N-BEATS)
- [ ] Full backtest results with all metrics
- [ ] Attention weight / variable importance analysis
- [ ] Ragged-edge robustness experiment
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

*Document version 1.0 — Prepared for review by Luigi Palumbo*
