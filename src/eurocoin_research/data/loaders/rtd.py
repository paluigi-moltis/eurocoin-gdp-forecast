"""ECB Real-Time Database (RTD) loader.

The ECB RTD provides time series with full revision history (vintages).
Each observation may have been revised multiple times; include_history=True
returns all vintage snapshots.

Key advantages over Eurostat SDMX for IP:
- Uses S0 = Euro Area moving concept (changing composition natively)
- Provides vintage data for pseudo-real-time backtesting
- Revision timestamps embedded in the response

When include_history=True, sdmx1 returns a LIST of pandas Series,
each representing a vintage snapshot. The last item is the latest (final) vintage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from eurocoin_research.config import SeriesSpec
from eurocoin_research.data.loaders.base import BaseLoader, parse_period

logger = logging.getLogger(__name__)


class RTDLoader(BaseLoader):
    """Load ECB Real-Time Database (RTD) series with optional vintage history.

    For standard panel assembly: fetches the latest vintage only.
    For vintage backtesting: fetches all vintages with revision timestamps.
    """

    def __init__(
        self,
        base_url: str = "",
        cache_dir: Path | None = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(base_url, cache_dir)
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import sdmx
            self._client = sdmx.Client("ECB")
            logger.info("Initialized sdmx1 Client for ECB RTD")
        return self._client

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch the latest vintage of an RTD series (standard panel use).

        For vintage data, use fetch_vintages() instead.
        """
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        df = self._fetch_latest(spec, start)
        self._save_to_cache(spec.id, df)
        return df

    def _fetch_latest(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch only the latest vintage (no history)."""
        import pandas as pd
        import sdmx

        params: dict[str, Any] = {}
        if start:
            params["startPeriod"] = start

        logger.info("Fetching %s from ECB RTD (latest): %s/%s", spec.id, "RTD", spec.filter)
        response = self.client.data(resource_id="RTD", key=spec.filter, params=params)
        pdf = self._response_to_pandas(response)

        df = pl.DataFrame({
            "date": pl.Series(pdf["TIME_PERIOD"]).cast(pl.Date, strict=False),
            "series_id": pl.Series([spec.id] * len(pdf)),
            "value": pl.Series(pdf["value"].astype(float)).cast(pl.Float64, strict=False),
        })
        df = df.filter(pl.col("date").is_not_null() & pl.col("value").is_not_null())
        df = df.sort("date")
        logger.info("Fetched %s: %d observations", spec.id, len(df))
        return df

    def fetch_vintages(
        self,
        spec: SeriesSpec,
        start: str | None = None,
    ) -> pl.DataFrame:
        """Fetch ALL vintages of an RTD series with revision history.

        Returns a DataFrame with columns:
            date, vintage_date, value

        Each row represents one observation as it was known at a specific vintage.
        Multiple rows per date = multiple revisions.
        """
        import pandas as pd

        params: dict[str, Any] = {}
        if start:
            params["startPeriod"] = start

        logger.info("Fetching %s vintages from ECB RTD (include_history=True)", spec.id)
        import sdmx as sdmx_mod
        response = self.client.get(
            "data",
            resource_id="RTD",
            key=spec.filter,
            params=params,
            include_history=True,
        )

        result_list = sdmx_mod.to_pandas(response)

        if not isinstance(result_list, list):
            result_list = [result_list]

        logger.info("RTD returned %d vintage snapshots", len(result_list))

        records: list[dict] = []
        for vintage_idx, series in enumerate(result_list):
            if not isinstance(series, pd.Series):
                continue

            # Extract observations from the MultiIndex Series
            for idx_tuple, value in series.items():
                if isinstance(idx_tuple, tuple):
                    time_period = idx_tuple[0]
                else:
                    time_period = idx_tuple

                if pd.isna(value) or value is None:
                    continue

                # Parse the date
                try:
                    if isinstance(time_period, str):
                        parsed = pd.Timestamp(time_period).date()
                    elif hasattr(time_period, "date"):
                        parsed = time_period.date()
                    else:
                        parsed = pd.Timestamp(str(time_period)).date()
                except Exception:
                    continue

                records.append({
                    "date": parsed,
                    "vintage_index": vintage_idx,
                    "value": float(value),
                })

        if not records:
            return pl.DataFrame(schema={"date": pl.Date, "vintage_index": pl.Int32, "value": pl.Float64})

        df = pl.DataFrame(records).sort(["date", "vintage_index"])
        logger.info(
            "RTD vintages: %d total observations across %d vintages",
            len(df), df["vintage_index"].n_unique(),
        )
        return df

    @staticmethod
    def _response_to_pandas(response) -> "pd.DataFrame":
        """Convert a single-vintage SDMX response to a flat DataFrame."""
        import pandas as pd
        import sdmx

        pdf = sdmx.to_pandas(response, datetime="TIME_PERIOD")
        pdf = pdf.reset_index()

        # Flatten MultiIndex columns
        if isinstance(pdf.columns, pd.MultiIndex):
            pdf.columns = [
                col[0] if isinstance(col, tuple) else col for col in pdf.columns
            ]

        # Ensure TIME_PERIOD and value columns exist
        date_col = pdf.columns[0]
        value_col = pdf.columns[-1]

        pdf = pdf.rename(columns={date_col: "TIME_PERIOD", value_col: "value"})
        return pdf[["TIME_PERIOD", "value"]]
