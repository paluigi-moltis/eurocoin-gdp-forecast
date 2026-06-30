"""ECB Statistical Data Warehouse (SDW) SDMX connector.

Fetches time series from the ECB via the SDMX 2.1 REST API.
API docs: https://data.ecb.europa.eu/help/api/overview
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from io import StringIO
from typing import Any

import polars as pl
import requests

from eurocoin_research.config import SeriesSpec
from eurocoin_research.data.loaders.base import BaseLoader, parse_period

logger = logging.getLogger(__name__)

DATA_ENDPOINT = "{base}/data/{series_code}"


class ECBLoader(BaseLoader):
    """Load data from ECB SDW via SDMX 2.1 REST API."""

    def __init__(
        self,
        base_url: str = "https://data-api.ecb.europa.eu/service",
        cache_dir: pl.Path | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        super().__init__(base_url, cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single series from ECB SDW."""
        # Check cache
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        # ECB SDW uses the series code as the dataflow key
        url = DATA_ENDPOINT.format(base=self.base_url, series_code=spec.code)
        params: dict[str, Any] = {"format": "csvdata"}
        if start:
            params["startPeriod"] = start

        # Fetch with retries
        csv_text = self._fetch_with_retries(url, params, spec.id)

        # Parse CSV response
        df = self._parse_csv_response(csv_text, spec)

        # Cache
        self._save_to_cache(spec.id, df)

        return df

    def _fetch_with_retries(
        self, url: str, params: dict, series_id: str
    ) -> str:
        """Fetch URL with exponential backoff retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Fetching %s (attempt %d/%d): %s",
                    series_id,
                    attempt,
                    self.max_retries,
                    url,
                )
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Fetch failed for %s (attempt %d): %s. Retrying in %.1fs...",
                        series_id,
                        attempt,
                        str(e)[:200],
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("All retries exhausted for %s", series_id)
        raise RuntimeError(
            f"Failed to fetch {series_id} after {self.max_retries} attempts"
        ) from last_error

    def _parse_csv_response(
        self, csv_text: str, spec: SeriesSpec
    ) -> pl.DataFrame:
        """Parse ECB SDW CSV response into a Polars DataFrame.

        ECB CSV format columns typically include:
        - KEY: series key
        - TITLE: series title
        - FREQ: frequency
        - DATE: observation period (e.g., "2020-01", "2020-Q1")
        - VALUE: observation value
        """
        if not csv_text.strip():
            logger.warning("Empty CSV response for %s", spec.id)
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        try:
            raw = pl.read_csv(StringIO(csv_text))
        except Exception as e:
            logger.error("Failed to parse CSV for %s: %s", spec.id, e)
            logger.debug("CSV content (first 500 chars): %s", csv_text[:500])
            raise

        # Find the date and value columns (ECB uses various column names)
        date_col = None
        value_col = None
        for col in raw.columns:
            col_lower = col.lower()
            if col_lower in ("date", "time_period", "period"):
                date_col = col
            elif col_lower in ("value", "obs_value", "observation_value"):
                value_col = col

        if date_col is None or value_col is None:
            # Fall back to positional (DATE and VALUE are usually present)
            logger.warning(
                "Standard columns not found in %s response. Columns: %s. "
                "Trying positional.",
                spec.id,
                raw.columns,
            )
            if "DATE" in raw.columns:
                date_col = "DATE"
            elif "TIME PERIOD" in raw.columns:
                date_col = "TIME PERIOD"
            if "VALUE" in raw.columns:
                value_col = "VALUE"

        if date_col is None or value_col is None:
            logger.error(
                "Could not identify date/value columns for %s. Available: %s",
                spec.id,
                raw.columns,
            )
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        # Parse
        df = raw.select([
            pl.col(date_col).alias("raw_date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        ])

        # Parse dates
        df = df.with_columns([
            df["raw_date"].map_elements(
                lambda x: self._parse_ecb_date(x, spec.frequency),
                return_dtype=pl.Date,
            ).alias("date"),
            pl.lit(spec.id).alias("series_id"),
        ])

        # Drop rows with null dates or values
        df = df.filter(pl.col("date").is_not_null() & pl.col("value").is_not_null())
        df = df.select(["date", "series_id", "value"]).sort("date")

        logger.debug("Parsed %s: %d observations", spec.id, len(df))
        return df

    @staticmethod
    def _parse_ecb_date(date_str: str, frequency: str) -> datetime | None:
        """Parse ECB date format to datetime."""
        if not date_str:
            return None
        date_str = str(date_str).strip()
        try:
            if "Q" in date_str:
                year, q = date_str.split("Q")
                month = (int(q) - 1) * 3 + 1
                return datetime(int(year), month, 1)
            elif "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 2:
                    return datetime(int(parts[0]), int(parts[1]), 1)
                else:
                    return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
            elif len(date_str) == 4:
                return datetime(int(date_str), 1, 1)
        except (ValueError, IndexError):
            pass
        return None
