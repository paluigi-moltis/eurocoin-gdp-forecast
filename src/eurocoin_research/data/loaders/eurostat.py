"""Eurostat SDMX 2.1 REST API connector.

Fetches time series from Eurostat via the SDMX REST API.
API docs: https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1
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

# Eurostat SDMX REST API endpoints
DATA_ENDPOINT = "{base}/sdmx/2.1/data/{dataset}/{filter}"
META_ENDPOINT = "{base}/sdmx/2.1/dataflow/{agency}/{dataset}/{version}"
# Compact data format (faster, less metadata)
COMPACT_FORMAT = "compactdata"
JSON_FORMAT = "jsondata"


class EurostatLoader(BaseLoader):
    """Load data from Eurostat via SDMX 2.1 REST API."""

    def __init__(
        self,
        base_url: str = "https://ec.europa.eu/eurostat/api/dissemination",
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
        """Fetch a single series from Eurostat.

        Uses the SDMX 2.1 REST API with compact data format.
        """
        # Check cache
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        # Build URL — SDMX key uses dots as dimension separators in the URL path
        dataset = spec.code
        key = spec.filter if spec.filter else ""
        url = DATA_ENDPOINT.format(
            base=self.base_url, dataset=dataset, filter=key
        )
        params: dict[str, Any] = {}
        if start:
            params["startPeriod"] = start

        # Fetch with retries
        response = self._fetch_with_retries(url, params, spec.id)

        # Parse response
        df = self._parse_compact_response(response, spec)

        # Cache
        self._save_to_cache(spec.id, df)

        return df

    def _fetch_with_retries(
        self, url: str, params: dict, series_id: str
    ) -> requests.Response:
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
                resp = requests.get(url, params=params, timeout=self.timeout, headers={
                    "Accept": "application/vnd.sdmx.data+csv;version=1.0.0",
                })
                resp.raise_for_status()
                return resp
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
        raise RuntimeError(f"Failed to fetch {series_id} after {self.max_retries} attempts") from last_error

    def _parse_compact_response(
        self, response: requests.Response, spec: SeriesSpec
    ) -> pl.DataFrame:
        """Parse SDMX CSV response into a Polars DataFrame.

        Eurostat SDMX-CSV format:
        DATAFLOW,LAST UPDATE,freq,unit,s_adj,na_item,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS
        """
        csv_text = response.text
        if not csv_text.strip():
            logger.warning("Empty response for %s", spec.id)
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        try:
            raw = pl.read_csv(StringIO(csv_text))
        except Exception as e:
            logger.error("Failed to parse CSV for %s: %s", spec.id, e)
            logger.debug("CSV content (first 500 chars): %s", csv_text[:500])
            raise

        # Identify TIME_PERIOD and OBS_VALUE columns
        date_col = None
        value_col = None
        for col in raw.columns:
            col_upper = col.upper().strip()
            if col_upper == "TIME_PERIOD":
                date_col = col
            elif col_upper == "OBS_VALUE":
                value_col = col

        if date_col is None or value_col is None:
            logger.error(
                "Could not identify TIME_PERIOD/OBS_VALUE columns for %s. Columns: %s",
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
        ]).with_columns([
            pl.col("raw_date").map_elements(
                lambda x: self._parse_sdmx_period(x, spec.frequency),
                return_dtype=pl.Datetime,
            ).alias("date"),
            pl.lit(spec.id).alias("series_id"),
        ])

        # Drop rows with null dates or values, cast to Date
        df = df.filter(pl.col("date").is_not_null() & pl.col("value").is_not_null())
        df = df.with_columns(pl.col("date").cast(pl.Date))
        df = df.select(["date", "series_id", "value"]).sort("date")

        logger.debug("Parsed %s: %d observations", spec.id, len(df))
        return df

    @staticmethod
    def _parse_sdmx_period(period: str, frequency: str) -> datetime | None:
        """Parse an SDMX time period string to a datetime.

        Handles formats:
        - Monthly: "2020-01" or "2020M01"
        - Quarterly: "2020Q1"
        - Annual: "2020"
        """
        if not period:
            return None
        period = str(period).strip()
        try:
            # Quarterly: "2020-Q1" or "2020Q1"
            if "Q" in period and frequency == "quarterly":
                parts = period.split("Q")
                year = parts[0].rstrip("-")
                q = parts[1]
                month = (int(q) - 1) * 3 + 1
                return datetime(int(year), month, 1)
            # Monthly: "2020M01" or "2020-01"
            elif "M" in period and len(period) == 7:
                year, month = period.replace("M", "-").split("-")
                return datetime(int(year), int(month), 1)
            elif "-" in period and len(period) == 7:
                year, month = period.split("-")
                return datetime(int(year), int(month), 1)
            elif len(period) == 4:
                return datetime(int(period), 1, 1)
        except (ValueError, IndexError):
            pass
        logger.debug("Could not parse SDMX period: '%s' (frequency=%s)", period, frequency)
        return None
