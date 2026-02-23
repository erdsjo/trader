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

    async def fetch_historical_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols. Returns {symbol: DataFrame}.
        Default implementation fetches sequentially; subclasses may optimize."""
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = await self.fetch_historical(symbol, start, end, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception:
                pass  # Skip failed symbols
        return results

    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return list of supported interval strings (e.g. '1m', '5m', '1d')."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this data source (e.g. 'yahoo')."""
