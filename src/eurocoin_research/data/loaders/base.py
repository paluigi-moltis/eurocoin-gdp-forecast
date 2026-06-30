"""Base classes for data loaders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from eurocoin_research.config import SeriesSpec

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for all data source loaders.

    Each loader connects to a specific data source (Eurostat, ECB, etc.)
    and fetches time series as Polars DataFrames.
    """

    def __init__(self, base_url: str, cache_dir: Path | None = None) -> None:
        """Initialize the loader.

        Args:
            base_url: Base URL of the data source API.
            cache_dir: Directory for caching raw responses. If None, no caching.
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single time series.

        Args:
            spec: Series specification (id, code, filter, etc.).
            start: Start period (e.g., "1999-01"). If None, fetches all available.

        Returns:
            Polars DataFrame with columns: date, value, and series metadata.
            Expected schema:
                - date: Date (monthly or quarterly)
                - value: Float64
                - series_id: String
        """
        ...

    def _cache_path(self, series_id: str) -> Path | None:
        """Get the cache file path for a series."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{series_id}.csv"

    def _load_from_cache(self, series_id: str) -> pl.DataFrame | None:
        """Load a series from cache if available."""
        path = self._cache_path(series_id)
        if path and path.exists():
            logger.debug("Loading %s from cache: %s", series_id, path)
            return pl.read_csv(path, try_parse_dates=True)
        return None

    def _save_to_cache(self, series_id: str, df: pl.DataFrame) -> None:
        """Save a series to cache."""
        path = self._cache_path(series_id)
        if path:
            df.write_csv(path)

    def fetch_multiple(
        self, specs: list[SeriesSpec], start: str | None = None
    ) -> pl.DataFrame:
        """Fetch multiple series and concatenate into a long-format DataFrame.

        Args:
            specs: List of series specifications to fetch.
            start: Start period for all series.

        Returns:
            Long-format DataFrame: date, series_id, value
        """
        frames: list[pl.DataFrame] = []
        for spec in specs:
            try:
                df = self.fetch_series(spec, start=start)
                frames.append(df)
                logger.info(
                    "Fetched %s: %d observations [%s to %s]",
                    spec.id,
                    len(df),
                    df["date"].min() if len(df) > 0 else "N/A",
                    df["date"].max() if len(df) > 0 else "N/A",
                )
            except Exception:
                logger.exception("Failed to fetch series %s", spec.id)
        if not frames:
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )
        return pl.concat(frames, how="vertical")


def parse_period(period_str: str, frequency: str) -> datetime:
    """Parse a period string into a datetime.

    Args:
        period_str: Period string (e.g., "2020-01", "2020Q1").
        frequency: "monthly" or "quarterly".

    Returns:
        datetime object (first day of the period).
    """
    if frequency == "quarterly":
        year, q = period_str.replace("Q", "-").split("-")
        month = (int(q) - 1) * 3 + 1
        return datetime(int(year), month, 1)
    else:
        year, month = period_str.split("-")
        return datetime(int(year), int(month), 1)
