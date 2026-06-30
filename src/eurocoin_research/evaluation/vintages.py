"""Data vintage construction for pseudo-real-time backtesting.

Handles publication lags and data revisions for all series in the panel.
Each backtest date uses a reconstructed data vintage that reflects what was
actually available at that point in time.

Key concepts:
- Estimation date: conventionally the last day of the month
- Publication lag: per-series delay between reference period and release
- Data revision: past values of a series may differ between vintages
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from eurocoin_research.config import FullConfig, SeriesSpec

logger = logging.getLogger(__name__)


class VintageManager:
    """Construct and manage data vintages for backtesting.

    For each backtest date, produces a data panel that respects:
    1. Publication lags (which months are available per series)
    2. Data revisions (which values were known at that date)
    """

    def __init__(self, config: FullConfig | None = None) -> None:
        self.config = config or FullConfig()
        self.vintage_dir = Path(self.config.paths.vintages)

    def get_last_available_period(
        self,
        spec: SeriesSpec,
        estimation_date: date,
    ) -> date | None:
        """Determine the last available observation period for a series.

        Based on the series-specific publication lag.

        Args:
            spec: Series specification (contains publication_lag_days).
            estimation_date: The date as of which data is being collected
                             (conventionally last day of month).

        Returns:
            The last period (month/quarter) for which data was available,
            or None if no data would be available.
        """
        # The release date = reference period end + publication lag
        release_date = estimation_date - timedelta(days=spec.publication_lag_days)

        if spec.frequency == "quarterly":
            # Find the last completed quarter whose release has occurred
            quarter_end = self._last_quarter_end(release_date)
            return quarter_end
        else:
            # Monthly: last month whose release has occurred
            month_end = self._last_month_end(release_date)
            return month_end

    @staticmethod
    def _last_month_end(d: date) -> date:
        """Get the last day of the month preceding the given date."""
        if d.month == 1:
            return date(d.year - 1, 12, 1)
        else:
            # First day of current month minus 1 day = last day of previous month
            first_of_month = date(d.year, d.month, 1)
            return first_of_month - timedelta(days=1)

    @staticmethod
    def _last_quarter_end(d: date) -> date:
        """Get the last completed quarter end before the given date."""
        q = (d.month - 1) // 3 + 1
        if q == 1:
            return date(d.year - 1, 12, 31)
        # Last day of the previous quarter
        prev_q = q - 1
        month = prev_q * 3
        if month == 12:
            return date(d.year, 12, 31)
        else:
            first_of_next = date(d.year, month + 1, 1)
            return first_of_next - timedelta(days=1)

    def construct_vintage(
        self,
        full_panel: pl.DataFrame,
        estimation_date: date,
        specs: list[SeriesSpec] | None = None,
    ) -> pl.DataFrame:
        """Construct a vintage panel for a given estimation date.

        Applies publication lags to truncate each series at the correct point.

        Args:
            full_panel: The complete (fully-revised) panel with all data.
            estimation_date: The backtest date (last day of month).
            specs: Series specifications with lag info. If None, derived from config.

        Returns:
            Vintage panel: same shape as full_panel, but with recent observations
            set to null based on per-series publication lags.
        """
        if specs is None:
            specs = self.config.get_all_series()

        spec_lookup = {s.id: s for s in specs}
        vintage_panel = full_panel.clone()

        date_col = "date"
        if date_col not in vintage_panel.columns:
            logger.error("Panel must have a 'date' column")
            return vintage_panel

        for col in vintage_panel.columns:
            if col == date_col:
                continue
            spec = spec_lookup.get(col)
            if spec is None:
                logger.debug("No spec for column %s, leaving as-is", col)
                continue

            last_period = self.get_last_available_period(spec, estimation_date)
            if last_period is None:
                continue

            # Set values after last_period to null
            vintage_panel = vintage_panel.with_columns(
                pl.when(pl.col(date_col) > last_period)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

        # Count how many values were truncated
        n_truncated = 0
        for col in vintage_panel.columns:
            if col == date_col:
                continue
            n_truncated += full_panel[col].is_not_null().sum() - vintage_panel[col].is_not_null().sum()

        logger.info(
            "Vintage for %s: %d series, %d observations truncated due to publication lags",
            estimation_date,
            len(vintage_panel.columns) - 1,
            n_truncated,
        )
        return vintage_panel

    def save_vintage(
        self, vintage_panel: pl.DataFrame, estimation_date: date
    ) -> Path:
        """Save a vintage panel to the vintages directory.

        File naming: vintage_YYYY-MM.csv
        """
        self.vintage_dir.mkdir(parents=True, exist_ok=True)
        filename = f"vintage_{estimation_date.year}-{estimation_date.month:02d}.csv"
        path = self.vintage_dir / filename
        vintage_panel.write_csv(path)
        logger.info("Vintage saved to %s", path)
        return path

    def load_vintage(self, estimation_date: date) -> pl.DataFrame | None:
        """Load a previously saved vintage panel."""
        filename = f"vintage_{estimation_date.year}-{estimation_date.month:02d}.csv"
        path = self.vintage_dir / filename
        if not path.exists():
            logger.warning("Vintage file not found: %s", path)
            return None
        return pl.read_csv(path, try_parse_dates=True)

    def list_vintages(self) -> list[date]:
        """List all saved vintages."""
        vintages: list[date] = []
        if not self.vintage_dir.exists():
            return vintages
        for f in self.vintage_dir.glob("vintage_*.csv"):
            # Parse date from filename
            try:
                name = f.stem  # vintage_2015-04
                parts = name.split("_")[1]
                y, m = parts.split("-")
                vintages.append(date(int(y), int(m), 1))
            except (IndexError, ValueError):
                continue
        return sorted(vintages)

    @staticmethod
    def last_day_of_month(year: int, month: int) -> date:
        """Get the last day of a given month."""
        if month == 12:
            return date(year, 12, 31)
        else:
            first_next = date(year, month + 1, 1)
            return first_next - timedelta(days=1)

    def generate_backtest_dates(
        self,
        start: str | None = None,
        end: str | None = None,
        frequency: str = "monthly",
    ) -> list[date]:
        """Generate the list of backtest estimation dates.

        Args:
            start: Start period (e.g., "2010-01"). Defaults to config.
            end: End period. Defaults to config (or latest available).
            frequency: "monthly" or "quarterly".

        Returns:
            List of dates (last day of each period).
        """
        start_str = start or self.config.backtest.start
        end_str = end or self.config.backtest.end

        start_y, start_m = map(int, start_str.split("-"))
        if end_str:
            end_y, end_m = map(int, end_str.split("-"))
        else:
            today = date.today()
            end_y, end_m = today.year, today.month

        dates: list[date] = []
        y, m = start_y, start_m
        while (y, m) <= (end_y, end_m):
            dates.append(self.last_day_of_month(y, m))
            if frequency == "monthly":
                m += 1
                if m > 12:
                    m = 1
                    y += 1
            elif frequency == "quarterly":
                m += 3
                if m > 12:
                    m = m - 12
                    y += 1
        return dates
