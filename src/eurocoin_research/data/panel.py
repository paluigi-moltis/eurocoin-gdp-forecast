"""Panel assembly pipeline.

Fetches all configured series, aligns them to a common monthly time grid,
applies transformations, and handles ragged edges.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

from eurocoin_research.config import FullConfig, SeriesSpec, load_config
from eurocoin_research.data.loaders.base import BaseLoader
from eurocoin_research.data.loaders.ecb import ECBLoader
from eurocoin_research.data.loaders.ecfin import ECFINLoader
from eurocoin_research.data.loaders.eurostat import EurostatLoader
from eurocoin_research.data.loaders.extended import ExtendedLoader
from eurocoin_research.data.loaders.fred import FREDLoader

logger = logging.getLogger(__name__)


class PanelAssembler:
    """Assemble the macroeconomic data panel from multiple sources.

    Workflow:
    1. Load configuration (public or extended mode)
    2. Create appropriate data loaders for each source
    3. Fetch all series
    4. Align to a monthly grid
    5. Handle missing values and ragged edges
    6. Save assembled panel
    """

    def __init__(self, config: FullConfig | None = None) -> None:
        self.config = config or load_config()
        self._raw_cache_dir = Path(self.config.paths.raw)

    def _create_loaders(self) -> dict[str, BaseLoader]:
        """Create data loaders based on configuration."""
        loaders: dict[str, BaseLoader] = {}

        for source_name, source_cfg in self.config.sources.items():
            if source_name == "eurostat":
                loaders[source_name] = EurostatLoader(
                    base_url=source_cfg.base_url
                    or "https://ec.europa.eu/eurostat/api/dissemination",
                    cache_dir=self._raw_cache_dir / "eurostat",
                )
            elif source_name == "ecb":
                loaders[source_name] = ECBLoader(
                    base_url=source_cfg.base_url
                    or "https://data-api.ecb.europa.eu/service",
                    cache_dir=self._raw_cache_dir / "ecb",
                )
            elif source_name == "ecfin":
                loaders[source_name] = ECFINLoader(
                    base_url=source_cfg.base_url
                    or "https://ec.europa.eu/economy_finance/db",
                    cache_dir=self._raw_cache_dir / "ecfin",
                )
            elif source_name == "fred":
                loaders[source_name] = FREDLoader(
                    base_url=source_cfg.base_url
                    or "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    cache_dir=self._raw_cache_dir / "fred",
                )
            elif source_name in ("sp_global", "datastream"):
                loaders[source_name] = ExtendedLoader(
                    base_url=source_cfg.base_url
                    or "https://api.marketplace.spglobal.com",
                    cache_dir=self._raw_cache_dir / source_name,
                )
            else:
                logger.warning("Unknown source '%s', skipping", source_name)

        return loaders

    def fetch_all(self, start: str | None = None) -> pl.DataFrame:
        """Fetch all configured series and return in long format.

        Args:
            start: Start period override. If None, uses config sample start.

        Returns:
            Long-format DataFrame: date, series_id, value
        """
        start_period = start or self.config.sample.start
        loaders = self._create_loaders()
        all_series = self.config.get_all_series()

        # Group series by source
        series_by_source: dict[str, list[SeriesSpec]] = {}
        for source_name, source_cfg in self.config.sources.items():
            if source_cfg.series:
                series_by_source[source_name] = source_cfg.series

        frames: list[pl.DataFrame] = []
        for source_name, specs in series_by_source.items():
            loader = loaders.get(source_name)
            if loader is None:
                logger.warning("No loader for source '%s', skipping %d series", source_name, len(specs))
                continue

            logger.info("Fetching %d series from %s...", len(specs), source_name)
            try:
                df = loader.fetch_multiple(specs, start=start_period)
                if len(df) > 0:
                    frames.append(df)
                    logger.info("  → %d observations from %s", len(df), source_name)
            except Exception:
                logger.exception("Failed to fetch from source %s", source_name)

        if not frames:
            logger.error("No data fetched from any source")
            return pl.DataFrame(
                schema={"date": pl.Date, "series_id": pl.Utf8, "value": pl.Float64}
            )

        combined = pl.concat(frames, how="vertical")
        logger.info(
            "Total fetched: %d observations across %d series",
            len(combined),
            combined["series_id"].n_unique(),
        )
        return combined

    def assemble_panel(
        self,
        raw_data: pl.DataFrame | None = None,
        frequency: str = "monthly",
    ) -> pl.DataFrame:
        """Assemble a wide-format panel aligned to a monthly time grid.

        Args:
            raw_data: Pre-fetched long-format data. If None, fetches fresh.
            frequency: Target frequency ("monthly" for covariates).

        Returns:
            Wide-format DataFrame: one column per series, indexed by date.
            Missing values are null (ragged edge preserved).
        """
        if raw_data is None:
            raw_data = self.fetch_all()

        if len(raw_data) == 0:
            return pl.DataFrame()

        # Determine date range
        min_date = raw_data["date"].min()
        max_date = raw_data["date"].max()
        logger.info(
            "Assembling panel: %s to %s, %d series",
            min_date,
            max_date,
            raw_data["series_id"].n_unique(),
        )

        # Pivot to wide format
        panel = raw_data.pivot(
            values="value",
            index="date",
            on="series_id",
            aggregate_function="first",
        )

        # Create a complete monthly grid and left-join
        panel = panel.sort("date")
        panel = self._fill_monthly_grid(panel, min_date, max_date)

        logger.info(
            "Panel assembled: %d months × %d series",
            len(panel),
            len(panel.columns) - 1,  # exclude date column
        )
        return panel

    @staticmethod
    def _fill_monthly_grid(
        df: pl.DataFrame, min_date, max_date
    ) -> pl.DataFrame:
        """Ensure the panel has a row for every month in the range.

        Inserts nulls for missing months (preserving ragged edges).
        """
        # Generate complete monthly sequence
        start = datetime(min_date.year, min_date.month, 1) if hasattr(min_date, "year") else min_date
        end = datetime(max_date.year, max_date.month, 1) if hasattr(max_date, "year") else max_date

        months = []
        current = start
        while current <= end:
            months.append(current.date() if hasattr(current, "date") else current)
            # Increment month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        grid = pl.DataFrame({"date": months})
        return grid.join(df, on="date", how="left")

    def save_panel(self, panel: pl.DataFrame, path: Path | None = None) -> Path:
        """Save the assembled panel to processed/ directory."""
        save_path = path or Path(self.config.paths.processed) / "panel_monthly.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(save_path)
        logger.info("Panel saved to %s", save_path)
        return save_path


def assemble_data_panel(
    data_mode: str | None = None,
    project_root: Path | None = None,
) -> pl.DataFrame:
    """Convenience function to fetch and assemble the full panel.

    Args:
        data_mode: Override data mode ("public" or "extended").
        project_root: Project root directory.

    Returns:
        Wide-format panel DataFrame.
    """
    config = load_config(project_root=project_root, data_mode=data_mode)
    assembler = PanelAssembler(config=config)
    raw = assembler.fetch_all()
    panel = assembler.assemble_panel(raw_data=raw)
    assembler.save_panel(panel)
    return panel
