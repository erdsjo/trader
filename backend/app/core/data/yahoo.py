import asyncio
import logging
import time
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import yfinance as yf

from app.core.data.base import DataSource

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


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

    async def fetch_historical_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
        batch_size: int = 50,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            try:
                batch_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda b=batch: yf.download(
                            " ".join(b),
                            start=start,
                            end=end,
                            interval=interval,
                            group_by="ticker",
                            threads=True,
                        ),
                    ),
                    timeout=120,
                )
                if len(batch) == 1:
                    symbol = batch[0]
                    df = self._normalize(batch_result)
                    if not df.empty:
                        results[symbol] = df
                else:
                    for symbol in batch:
                        try:
                            if symbol in batch_result.columns.get_level_values(0):
                                df = batch_result[symbol].dropna(how="all")
                                df = self._normalize(df)
                                if not df.empty:
                                    results[symbol] = df
                        except Exception:
                            pass
            except Exception:
                # Fallback: fetch individually
                for symbol in batch:
                    try:
                        df = await self.fetch_historical(symbol, start, end, interval)
                        if not df.empty:
                            results[symbol] = df
                    except Exception:
                        pass
            if i + batch_size < len(symbols):
                await asyncio.sleep(2)  # Rate limit between batches
        return results

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
        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval)
                if not df.empty:
                    return df
                logger.warning(
                    "Yahoo returned empty DataFrame for %s (attempt %d/%d)",
                    symbol, attempt, _MAX_RETRIES,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "Yahoo download failed for %s (attempt %d/%d): %s",
                    symbol, attempt, _MAX_RETRIES, e,
                )
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)

        if last_err:
            logger.error("All %d attempts failed for %s: %s", _MAX_RETRIES, symbol, last_err)
        return pd.DataFrame()

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
