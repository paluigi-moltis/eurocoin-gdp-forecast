"""FRED/ALFRED vintage data loader.

Fetches real-time GDP vintages from the Federal Reserve Bank of St. Louis's
ALFRED (Archival Federal Reserve Economic Data) database.

ALFRED provides historical data vintages — the values of a series as they
were known at different points in time. This is essential for correct
pseudo-real-time backtesting.

No API key required for CSV downloads (though one provides higher rate limits).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from io import StringIO

import polars as pl
import requests

logger = logging.getLogger(__name__)

ALFRED_CSV_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# Euro Area series available on FRED/ALFRED
EA_SERIES_MAP = {
    "EUNNGDP": {
        "description": "Euro Area (19) GDP at market prices, current prices",
        "frequency": "quarterly",
        "unit": "euros (millions)",
    },
    "NAEXKP01EZQ661S": {
        "description": "Euro Area Real GDP, SA, index",
        "frequency": "quarterly",
        "unit": "index",
    },
}


class AlfredVintageLoader:
    """Load real-time GDP vintages from ALFRED.

    For each backtest date, this loader retrieves the GDP values as they
    were known at that point — including revisions to past observations.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_vintage(
        self,
        series_id: str,
        vintage_date: str | date,
        start_period: str | None = None,
    ) -> pl.DataFrame:
        """Fetch a specific vintage of a series from ALFRED.

        Args:
            series_id: FRED series ID (e.g., "EUNNGDP").
            vintage_date: The vintage date (as-of date) as string "YYYY-MM-DD" or date object.
            start_period: Optional start period filter.

        Returns:
            DataFrame with columns: observation_date, value
        """
        if isinstance(vintage_date, date):
            vintage_date = vintage_date.strftime("%Y-%m-%d")

        params = {
            "id": series_id,
            "vintage_date": vintage_date,
        }
        if start_period:
            params["cosd"] = start_period

        csv_text = self._fetch_with_retries(ALFRED_CSV_URL, params, series_id, vintage_date)

        df = self._parse_csv(csv_text, series_id, vintage_date)
        return df

    def fetch_multiple_vintages(
        self,
        series_id: str,
        vintage_dates: list[str | date],
        start_period: str | None = None,
    ) -> pl.DataFrame:
        """Fetch multiple vintages of a series and return in long format.

        Returns:
            DataFrame with columns: observation_date, vintage_date, value
        """
        frames: list[pl.DataFrame] = []
        for vd in vintage_dates:
            try:
                df = self.fetch_vintage(series_id, vd, start_period)
                frames.append(df)
                logger.info(
                    "Fetched vintage %s for %s: %d observations",
                    vd, series_id, len(df),
                )
            except Exception:
                logger.exception("Failed to fetch vintage %s for %s", vd, series_id)

        if not frames:
            return pl.DataFrame(schema={
                "observation_date": pl.Date,
                "vintage_date": pl.Date,
                "value": pl.Float64,
            })

        return pl.concat(frames, how="vertical")

    def _fetch_with_retries(
        self, url: str, params: dict, series_id: str, vintage_date: str
    ) -> str:
        """Fetch URL with retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "ALFRED fetch failed (%s, vintage=%s, attempt %d): %s. Retrying in %.1fs...",
                        series_id, vintage_date, attempt, str(e)[:200], wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"Failed to fetch {series_id} vintage {vintage_date} after {self.max_retries} attempts"
        ) from last_error

    @staticmethod
    def _parse_csv(csv_text: str, series_id: str, vintage_date: str) -> pl.DataFrame:
        """Parse ALFRED CSV response.

        Format:
        observation_date,SERIES_ID
        2020-01-01,2915405.5
        """
        if not csv_text.strip():
            return pl.DataFrame()

        try:
            raw = pl.read_csv(StringIO(csv_text), try_parse_dates=True)
        except Exception as e:
            logger.error("Failed to parse ALFRED CSV for %s vintage %s: %s", series_id, vintage_date, e)
            raise

        # First column = observation_date, second = value
        date_col = raw.columns[0]
        value_col = raw.columns[1]

        df = raw.select([
            pl.col(date_col).alias("observation_date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        ]).with_columns([
            pl.lit(datetime.strptime(vintage_date, "%Y-%m-%d").date()).alias("vintage_date"),
        ])

        return df.filter(pl.col("observation_date").is_not_null() & pl.col("value").is_not_null())
