"""Configuration loader for the eurocoin research project.

Loads YAML configuration files and provides typed access via Pydantic models.
Supports the public/extended data mode flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "eurocoin-gdp-forecast"
    description: str = ""


class DataConfig(BaseModel):
    mode: Literal["public", "extended"] = "public"


class SampleConfig(BaseModel):
    start: str = "1999-01"
    gdp_start: str = "2000Q1"
    end: str | None = None


class PathsConfig(BaseModel):
    raw: str = "data/raw"
    processed: str = "data/processed"
    vintages: str = "data/vintages"
    references: str = "references"
    results: str = "results"


class TargetConfig(BaseModel):
    frequency_threshold: float = 0.5236  # π/6
    filter_type: str = "christiano-fitzgerald"
    cf_min_period: int = 2
    cf_max_period: str = "inf"


class BacktestConfig(BaseModel):
    start: str = "2010-01"
    end: str | None = None
    frequency: str = "monthly"
    estimation_date: str = "end_of_month"


class SeriesSpec(BaseModel):
    """Specification of a single time series in the data panel."""

    id: str
    code: str = ""
    filter: str = ""
    frequency: str = "monthly"
    description: str = ""
    publication_lag_days: int = 0
    revised: bool = False


class SourceConfig(BaseModel):
    """A data source with its base URL and series definitions."""

    base_url: str = ""
    series: list[SeriesSpec] = Field(default_factory=list)


class FullConfig(BaseModel):
    """Complete configuration combining all sections."""

    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    sample: SampleConfig = SampleConfig()
    paths: PathsConfig = PathsConfig()
    target: TargetConfig = TargetConfig()
    backtest: BacktestConfig = BacktestConfig()
    # Data sources (loaded from data_public.yaml or data_extended.yaml)
    sources: dict[str, SourceConfig] = Field(default_factory=dict)

    def get_all_series(self) -> list[SeriesSpec]:
        """Return all series specifications across all sources."""
        all_series: list[SeriesSpec] = []
        for source in self.sources.values():
            all_series.extend(source.series)
        return all_series

    def get_series_by_id(self, series_id: str) -> SeriesSpec | None:
        """Look up a series specification by its ID."""
        for source in self.sources.values():
            for s in source.series:
                if s.id == series_id:
                    return s
        return None


def _resolve_project_root() -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(
    project_root: Path | None = None,
    data_mode: str | None = None,
) -> FullConfig:
    """Load configuration from YAML files.

    Args:
        project_root: Root directory containing config/. If None, auto-detected.
        data_mode: Override data mode ("public" or "extended"). If None, uses config value.

    Returns:
        FullConfig instance with all settings.
    """
    if project_root is None:
        project_root = _resolve_project_root()

    config_dir = project_root / "config"

    # Load base config
    base_path = config_dir / "base.yaml"
    with open(base_path) as f:
        base = yaml.safe_load(f)

    # Determine data mode
    mode = data_mode or base.get("data", {}).get("mode", "public")

    # Load data source config
    data_file = f"data_{mode}.yaml"
    data_path = config_dir / data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data config file not found: {data_path}. "
            f"Expected mode '{mode}' in config directory."
        )

    with open(data_path) as f:
        data_sources = yaml.safe_load(f) or {}

    # Merge base + data sources
    merged = {**base, "sources": data_sources.get("sources", {})}
    if data_mode:
        merged["data"] = {**merged.get("data", {}), "mode": data_mode}

    return FullConfig(**merged)
