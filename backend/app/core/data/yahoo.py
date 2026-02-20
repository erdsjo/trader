import asyncio
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import yfinance as yf

from app.core.data.base import DataSource


class YahooDataSource(DataSource):
    INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo"]

    @property
    def source_name(self) -> str:
        return "yahoo"

    def supported_intervals(self) -> list[str]:
        return self.INTERVALS

    async def fetch_historical(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None, partial(self._download, symbol, start, end, interval)
        )
        return self._normalize(df)

    async def fetch_latest(self, symbol: str, interval: str) -> pd.DataFrame:
        end = datetime.now()
        if interval in ("1m", "2m", "5m", "15m", "30m"):
            start = end - timedelta(days=1)
        else:
            start = end - timedelta(days=5)

        return await self.fetch_historical(symbol, start, end, interval)

    def _download(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        return ticker.history(start=start, end=end, interval=interval)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "timestamp"]
            )
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else "Datetime"
        df = df.rename(
            columns={
                date_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(
            None
        )
        return df[["open", "high", "low", "close", "volume", "timestamp"]]
