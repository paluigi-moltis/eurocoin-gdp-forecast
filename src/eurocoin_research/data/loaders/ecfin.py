"""DG-ECFIN Business and Consumer Surveys data connector.

Fetches survey data from the European Commission DG-ECFIN.
Data is distributed as downloadable CSV files from the Commission's website.
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

# DG-ECFIN provides data via downloadable files
# The main consumer survey data file:
ECFIN_DATA_URL = (
    "https://ec.europa.eu/economy_finance/db/api/file"
    "?ext=5b116610-3fa2-4e99-b879-76c9d483aa8a&ext=.xlsx"
)
# Alternative: individual series via the web interface
# We use the bulk download for reliability


class ECFINLoader(BaseLoader):
    """Load business and consumer survey data from DG-ECFIN.

    DG-ECFIN distributes data as CSV/Excel files. This loader downloads
    the main dataset file and extracts individual series by identifier.
    """

    def __init__(
        self,
        base_url: str = "https://ec.europa.eu/economy_finance/db",
        cache_dir: pl.Path | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        super().__init__(base_url, cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._bulk_data: pl.DataFrame | None = None

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single survey series from DG-ECFIN."""
        # Check cache
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        # Load bulk data (cached in memory)
        bulk = self._get_bulk_data()
        if bulk is None or len(bulk) == 0:
            logger.warning("Bulk data unavailable for %s", spec.id)
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        # Extract series by code
        df = bulk.filter(pl.col("series_id") == spec.code)
        if len(df) == 0:
            # Try partial match
            df = bulk.filter(pl.col("series_id").str.contains(spec.code))

        if len(df) == 0:
            logger.warning("Series %s (code=%s) not found in ECFIN bulk data", spec.id, spec.code)
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        # Standardize
        df = df.with_columns(pl.lit(spec.id).alias("series_id"))
        df = df.select(["date", "series_id", "value"]).sort("date")

        # Filter by start date
        if start:
            start_date = parse_period(start, spec.frequency)
            df = df.filter(pl.col("date") >= start_date)

        # Cache
        self._save_to_cache(spec.id, df)
        return df

    def _get_bulk_data(self) -> pl.DataFrame | None:
        """Download and parse the ECFIN bulk data file.

        The file is cached in memory after first download.
        """
        if self._bulk_data is not None:
            return self._bulk_data

        # Try to download the bulk file
        logger.info("Downloading DG-ECFIN bulk survey data...")
        resp = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    ECFIN_DATA_URL,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                resp = None
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "ECFIN download failed (attempt %d): %s. Retrying in %.1fs...",
                        attempt,
                        str(e)[:200],
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Failed to download ECFIN bulk data after %d attempts", self.max_retries)
                    return None

        if resp is None:
            return None

        # Parse the file (Excel format)
        try:
            self._bulk_data = self._parse_ecfin_xlsx(resp.content)
            logger.info("ECFIN bulk data loaded: %d rows", len(self._bulk_data))
            return self._bulk_data
        except Exception:
            logger.exception("Failed to parse ECFIN bulk data")
            return None

    @staticmethod
    def _parse_ecfin_xlsx(content: bytes) -> pl.DataFrame:
        """Parse the ECFIN Excel file into long-format DataFrame.

        The DG-ECFIN file typically has series IDs as columns and dates as rows.
        """
        import tempfile

        import openpyxl

        # Write to temp file and load (openpyxl doesn't have load_workbook_from_bytes)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        wb = openpyxl.load_workbook(tmp_path, data_only=True)

        ws = wb.active
        rows_data = list(ws.iter_rows(values_only=True))
        if not rows_data:
            return pl.DataFrame()

        # First row = headers (series codes), First column = dates
        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows_data[0])]
        date_col_idx = 0  # Assumption: first column is dates

        records: list[dict] = []
        for row in rows_data[1:]:
            date_val = row[date_col_idx]
            if date_val is None:
                continue
            parsed_date = ECFINLoader._parse_ecfin_date(date_val)
            if parsed_date is None:
                continue

            for col_idx in range(1, len(headers)):
                value = row[col_idx]
                if value is not None and isinstance(value, (int, float)):
                    records.append({
                        "date": parsed_date,
                        "series_id": headers[col_idx],
                        "value": float(value),
                    })

        if not records:
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )
        return pl.DataFrame(records)

    @staticmethod
    def _parse_ecfin_date(date_val) -> datetime | None:
        """Parse an ECFIN date value (can be datetime, string, or serial)."""
        if isinstance(date_val, datetime):
            return date_val
        if isinstance(date_val, str):
            date_str = date_val.strip()
            try:
                if "-" in date_str and len(date_str) == 7:
                    y, m = date_str.split("-")
                    return datetime(int(y), int(m), 1)
                elif "M" in date_str:
                    parts = date_str.split("M")
                    return datetime(int(parts[0]), int(parts[1]), 1)
                elif len(date_str) >= 4:
                    return datetime(int(date_str[:4]), 1, 1)
            except (ValueError, IndexError):
                pass
        return None
