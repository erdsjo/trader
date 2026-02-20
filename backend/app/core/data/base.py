from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    async def fetch_historical(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data. Returns DataFrame with columns:
        open, high, low, close, volume, timestamp."""

    @abstractmethod
    async def fetch_latest(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch the most recent OHLCV data points."""

    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return list of supported interval strings (e.g. '1m', '5m', '1d')."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this data source (e.g. 'yahoo')."""
