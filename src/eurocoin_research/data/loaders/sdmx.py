"""Unified SDMX data loader using the sdmx1 library.

Replaces the separate Eurostat and ECB loaders with a single sdmx1-based
loader that handles all SDMX sources (ESTAT, ECB, etc.).

The sdmx1 library (https://github.com/khaeru/sdmx1) handles:
- Source discovery and connection management
- SDMX 2.1 and 3.0 REST API protocol
- Response parsing (structure-specific, generic, JSON, CSV)
- Conversion to pandas DataFrames
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl
import sdmx

from eurocoin_research.config import SeriesSpec
from eurocoin_research.data.loaders.base import BaseLoader, parse_period

logger = logging.getLogger(__name__)

# Mapping from config source names to sdmx1 source IDs
SOURCE_MAP = {
    "eurostat": "ESTAT",
    "ecb": "ECB",
    "oecd": "OECD",
    "imf": "IMF",
}


class SDMXLoader(BaseLoader):
    """Unified SDMX loader for all data sources (Eurostat, ECB, etc.).

    Uses the sdmx1 Client to connect to the appropriate SDMX endpoint.
    The SeriesSpec.code field contains the dataflow ID.
    The SeriesSpec.filter field contains the series key.
    """

    def __init__(
        self,
        source_name: str,
        base_url: str = "",
        cache_dir: Path | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(base_url, cache_dir)
        self.source_name = source_name
        self.timeout = timeout
        self._client: sdmx.Client | None = None

        # Resolve source name to sdmx1 source ID
        self.sdmx_source = SOURCE_MAP.get(source_name.lower(), source_name.upper())

    @property
    def client(self) -> sdmx.Client:
        """Lazy-initialize the sdmx1 client."""
        if self._client is None:
            self._client = sdmx.Client(self.sdmx_source)
            logger.info("Initialized sdmx1 Client for %s (%s)", self.source_name, self.sdmx_source)
        return self._client

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single series via SDMX.

        Args:
            spec: Series specification. code=dataflow ID, filter=series key.
            start: Start period (e.g., "2020-01", "2020-Q1").

        Returns:
            Polars DataFrame with columns: date, series_id, value
        """
        # Check cache
        cached = self._load_from_cache(spec.id)
        if cached is not None:
            if start:
                start_date = parse_period(start, spec.frequency)
                cached = cached.filter(pl.col("date") >= start_date)
            return cached

        dataflow_id = spec.code
        series_key = spec.filter

        params: dict[str, Any] = {}
        if start:
            params["startPeriod"] = start

        logger.info(
            "Fetching %s from %s: dataflow=%s, key=%s",
            spec.id, self.source_name, dataflow_id, series_key,
        )

        # Fetch via sdmx1
        response = self.client.data(
            resource_id=dataflow_id,
            key=series_key,
            params=params,
        )

        # Convert to pandas, then to polars
        import pandas as pd
        pdf = sdmx.to_pandas(response, datetime="TIME_PERIOD")
        pdf = pdf.reset_index()

        # The DataFrame has a MultiIndex column structure from sdmx1.
        # Extract TIME_PERIOD (first col) and the value (last col).
        if isinstance(pdf.columns, pd.MultiIndex):
            pdf.columns = [
                col[0] if isinstance(col, tuple) else col for col in pdf.columns
            ]

        # First column is TIME_PERIOD, last column is the value
        date_col = pdf.columns[0]
        value_col = pdf.columns[-1]

        df = pl.DataFrame({
            "date": pl.Series(pdf[date_col]).cast(pl.Date, strict=False),
            "series_id": pl.Series([spec.id] * len(pdf)),
            "value": pl.Series(pdf[value_col].astype(float)).cast(pl.Float64, strict=False),
        })

        # Drop nulls
        df = df.filter(pl.col("date").is_not_null() & pl.col("value").is_not_null())
        df = df.sort("date")

        logger.info(
            "Fetched %s: %d observations [%s to %s]",
            spec.id, len(df), df["date"].min(), df["date"].max(),
        )

        # Cache
        self._save_to_cache(spec.id, df)
        return df
