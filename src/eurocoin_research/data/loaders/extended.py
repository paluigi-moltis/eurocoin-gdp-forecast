"""Extended (commercial) data connector stub.

This module provides the interface for loading commercial data (S&P Global / Markit PMI)
when data mode is set to "extended".

The actual implementation requires:
1. API credentials stored in environment variables (SP_GLOBAL_API_KEY, SP_GLOBAL_API_SECRET)
2. A server with access to the commercial data sources

This stub documents the expected interface so the pipeline can be tested
in public mode and seamlessly switched to extended mode on a production server.
"""

from __future__ import annotations

import logging
import os

import polars as pl

from eurocoin_research.config import SeriesSpec
from eurocoin_research.data.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

# Required environment variables for commercial data access
REQUIRED_ENV_VARS = ["SP_GLOBAL_API_KEY", "SP_GLOBAL_API_SECRET"]


class ExtendedLoader(BaseLoader):
    """Stub loader for commercial data sources.

    When activated, this loader:
    1. Verifies that API credentials are available
    2. Connects to S&P Global / Markit API
    3. Downloads PMI and additional commercial series
    4. Returns them in the same format as public loaders

    In public mode, this loader is never instantiated.
    """

    def __init__(
        self,
        base_url: str = "https://api.marketplace.spglobal.com",
        cache_dir: pl.Path | None = None,
    ) -> None:
        super().__init__(base_url, cache_dir)

        # Check credentials
        self._credentials_available = self._check_credentials()
        if not self._credentials_available:
            logger.warning(
                "ExtendedLoader initialized without commercial data credentials. "
                "Set %s environment variables to enable.",
                ", ".join(REQUIRED_ENV_VARS),
            )

    @staticmethod
    def _check_credentials() -> bool:
        """Check if all required environment variables are set."""
        return all(os.getenv(var) for var in REQUIRED_ENV_VARS)

    def fetch_series(self, spec: SeriesSpec, start: str | None = None) -> pl.DataFrame:
        """Fetch a single commercial series.

        This is a stub. The actual implementation will:
        1. Authenticate with S&P Global API using OAuth2
        2. Query the series by code
        3. Return observations as a Polars DataFrame

        Raises:
            NotImplementedError: Always, until the production implementation is added.
        """
        if not self._credentials_available:
            raise RuntimeError(
                f"Cannot fetch commercial series {spec.id}: credentials not available. "
                f"Ensure {', '.join(REQUIRED_ENV_VARS)} are set."
            )
        raise NotImplementedError(
            f"ExtendedLoader.fetch_series() not yet implemented for {spec.id}. "
            "This stub must be replaced with the actual S&P Global API integration "
            "before running in extended mode."
        )
