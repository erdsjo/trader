from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



# Approximate number of bars per trading day for each interval.
# Used to convert intraday bar-level metrics to daily equivalents.
_BARS_PER_DAY: dict[str, int] = {
    "1m": 390,   # 6.5 hours * 60
    "2m": 195,
    "5m": 78,    # 6.5 hours * 12
    "15m": 26,
    "30m": 13,
    "1h": 7,     # ~6.5 rounded
    "1d": 1,
    "5d": 1,
    "1wk": 1,
    "1mo": 1,
}


class StockScreener:
    def __init__(
        self,
        min_volume: float = 1_000_000,
        min_volatility: float = 0.15,
        top_n_per_sector: int = 5,
        interval: str = "1d",
    ):
        self.min_volume = min_volume
        self.min_volatility = min_volatility
        self.top_n_per_sector = top_n_per_sector
        self.bars_per_day = _BARS_PER_DAY.get(interval, 1)

    def _compute_avg_daily_volume(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Compute average *daily* volume, aggregating intraday bars if needed."""
        recent = df.tail(lookback * self.bars_per_day)
        avg_bar_volume = float(recent["volume"].mean())
        return avg_bar_volume * self.bars_per_day

    def _compute_volatility(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Compute annualized volatility from bar returns, adjusted for bar frequency."""
        recent = df.tail(lookback * self.bars_per_day)
        returns = recent["close"].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        bar_std = float(returns.std())
        # Annualize: bars_per_day bars/day * 252 trading days/year
        annualized = bar_std * math.sqrt(self.bars_per_day * 252)
        return annualized

    def filter_candidates(
        self,
        price_data: dict[str, pd.DataFrame],
        universe: dict[str, str],
    ) -> dict[str, dict]:
        candidates: dict[str, dict] = {}
        for symbol, df in price_data.items():
            if symbol not in universe:
                continue
            if len(df) < 5:
                continue

            vol_avg = self._compute_avg_daily_volume(df)
            volatility = self._compute_volatility(df)

            if vol_avg < self.min_volume:
                continue
            if volatility < self.min_volatility:
                continue

            candidates[symbol] = {
                "sector": universe[symbol],
                "volume_avg": vol_avg,
                "volatility": volatility,
            }

        logger.info(f"Screener: {len(candidates)}/{len(price_data)} stocks passed filters")
        return candidates

    def select_top_n(
        self,
        candidates: dict[str, dict],
    ) -> dict[str, dict]:
        by_sector: dict[str, list[tuple[str, dict]]] = {}
        for symbol, info in candidates.items():
            sector = info["sector"]
            by_sector.setdefault(sector, []).append((symbol, info))

        selected: dict[str, dict] = {}
        for sector, stocks in by_sector.items():
            stocks.sort(
                key=lambda x: x[1].get("opportunity_score", x[1]["volatility"]),
                reverse=True,
            )
            for i, (symbol, info) in enumerate(stocks):
                info["selected"] = i < self.top_n_per_sector
                selected[symbol] = info

        n_selected = sum(1 for v in selected.values() if v["selected"])
        logger.info(f"Screener: selected {n_selected} stocks across {len(by_sector)} sectors")
        return selected
