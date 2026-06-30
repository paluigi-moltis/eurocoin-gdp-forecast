# Transformer-Based Medium-Term GDP Forecasting for the Euro Area

**Eurocoin Modernization Research**

This project explores modern Transformer neural network architectures with attention mechanisms to forecast the Medium- to Long-Run Growth (MLRG) component of euro-area GDP — the same target as the Bank of Italy / CEPR [€-coin indicator](https://www.bancaditalia.it/statistiche/tematiche/indicatori/indicatore-euro-coin/).

## Motivation

The €-coin indicator, based on a Generalized Dynamic Factor Model (GDFM), has served as a timely monthly tracker of euro-area economic growth since the early 2000s. However, recent challenges have reduced its reliability:

- **Regime-dependent variable relationships:** Producer prices (and other series) changed their signaling properties during the post-pandemic energy shock — what historically indicated demand growth became a signal of output contraction.
- **Pandemic outlier distortion:** The extreme GDP swings of 2020 created discontinuities in factor estimation, with results varying depending on data availability timing.
- **Missing-value sensitivity:** The indicator is sensitive to ragged edges at the recent end of the data panel (e.g., manufacturing PMI availability).

This research investigates whether attention-based architectures can address these issues by learning regime-dependent covariate weights, treating pandemic outliers as a separate regime, and being more robust to missing data.

## Approaches

| Avenue | Target | Method |
|--------|--------|--------|
| **1. Growth-Rate** | MLRG (smoothed GDP q-o-q growth) | Transformer models benchmarked against a full GDFM re-implementation |
| **2. Levels-Based** | GDP in levels | BEAST changepoint detection → regime identification → regime-conditioned Transformer |

## Project Structure

```
eurocoin-gdp-forecast/
├── config/           # YAML configuration (data sources, model params)
├── data/             # Raw, processed, and vintage data (gitignored)
├── docs/             # Research plan, literature review
├── references/       # Source papers (PDF)
├── notebooks/        # EDA, models, results
├── src/              # Python package (src/ layout)
└── results/          # Output figures, tables, trained models
```

See [`docs/research_plan.md`](docs/research_plan.md) for the full methodology, experimental design, and work phases.

## Data

The project supports two data modes via a configuration flag:

- **`public`** (default): Eurostat SDMX, ECB SDW, DG-ECFIN surveys — all freely accessible via public APIs.
- **`extended`**: Adds commercial data (S&P Global / Markit PMI) matching the Aprigliano et al. (2022) dataset.

### Data Vintages

Backtesting respects **real-time data availability** — each backtest date uses a reconstructed data vintage that accounts for:
- Per-series publication lags
- GDP and indicator revisions (values as known at the time, not today's revised estimates)

Vintage panels are stored as individual CSV files in `data/vintages/` for full reproducibility.

## Key References

- Altissimo, Cristadoro, Forni, Lippi, Veronese (2007). *New Eurocoin: Tracking Economic Growth in Real Time*. Bank of Italy TD 631.
- Aprigliano, Emiliozzi, Lippi (2022). *Tracking Economic Growth in Real Time During the Pandemic — A Revision of €-coin*. Bank of Italy QEF 703.
- Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting*. IJF.
- Zhao et al. (2019). *Detecting change-point, trend, and seasonality in satellite time series.* RSE. ([BEAST / Rbeast](https://github.com/zhaokg/Rbeast))

Full literature review: [`docs/literature_review.md`](docs/literature_review.md)

## License

MIT — see [LICENSE](LICENSE).
