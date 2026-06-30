"""FRED (Federal Reserve Economic Data) CSV API connector.

Fetches time series from FRED via the simple CSV download API.
No API key required for basic CSV downloads.

Used for financial and monetary data that mirrors ECB/Eurostat series
but is more reliably accessible via FRED.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from io import StringIO

import polars as pl
import requests

from eurocoin_research.config import SeriesSpec
from eurocoin_research.data.loaders.base import BaseLoader, parse_period

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FREDLoader(BaseLoader):
    """Load data from FRED via the CSV download API."""

    def __init__(
        self,
        base_url: str = FRED_CSV_URL,
        cache_dir: pl.Path | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        super().__init__(base_url, cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single series from FRED."""
        # Check cache
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        # Build URL — FRED uses ?id= parameter
        params = {"id": spec.code}
        if start:
            params["cosd"] = start

        # Fetch with retries
        csv_text = self._fetch_with_retries(params, spec.id)

        # Parse CSV response
        df = self._parse_csv_response(csv_text, spec)

        # Cache
        self._save_to_cache(spec.id, df)
        return df

    def _fetch_with_retries(self, params: dict, series_id: str) -> str:
        """Fetch URL with retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("Fetching %s from FRED (attempt %d/%d)", series_id, attempt, self.max_retries)
                resp = requests.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "FRED fetch failed for %s (attempt %d): %s. Retrying in %.1fs...",
                        series_id, attempt, str(e)[:200], wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("All retries exhausted for FRED series %s", series_id)
        raise RuntimeError(
            f"Failed to fetch FRED series {series_id} after {self.max_retries} attempts"
        ) from last_error

    def _parse_csv_response(self, csv_text: str, spec: SeriesSpec) -> pl.DataFrame:
        """Parse FRED CSV response into a Polars DataFrame.

        FRED CSV format:
        observation_date,SERIES_ID
        2020-01-01,123.45
        """
        if not csv_text.strip():
            logger.warning("Empty FRED CSV response for %s", spec.id)
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        # FRED sometimes returns "." for missing values
        csv_text = csv_text.replace(",.", ",nan")

        try:
            raw = pl.read_csv(StringIO(csv_text), try_parse_dates=True)
        except Exception as e:
            logger.error("Failed to parse FRED CSV for %s: %s", spec.id, e)
            logger.debug("CSV content (first 500 chars): %s", csv_text[:500])
            raise

        # First column = date, second = value
        date_col = raw.columns[0]
        value_col = raw.columns[1]

        # Ensure date is Date type
        if raw[date_col].dtype != pl.Date:
            raw = raw.with_columns(pl.col(date_col).cast(pl.Date))

        df = raw.select([
            pl.col(date_col).alias("date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        ]).with_columns([
            pl.lit(spec.id).alias("series_id"),
        ])

        # Drop rows with null values
        df = df.filter(pl.col("date").is_not_null() & pl.col("value").is_not_null())
        df = df.select(["date", "series_id", "value"]).sort("date")

        logger.debug("Parsed FRED %s: %d observations", spec.id, len(df))
        return df
