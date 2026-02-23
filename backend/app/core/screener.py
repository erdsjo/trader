from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StockScreener:
    def __init__(
        self,
        min_volume: float = 1_000_000,
        min_volatility: float = 0.15,
        top_n_per_sector: int = 5,
    ):
        self.min_volume = min_volume
        self.min_volatility = min_volatility
        self.top_n_per_sector = top_n_per_sector

    def _compute_avg_volume(self, df: pd.DataFrame, lookback: int = 20) -> float:
        recent = df.tail(lookback)
        return float(recent["volume"].mean())

    def _compute_volatility(self, df: pd.DataFrame, lookback: int = 20) -> float:
        recent = df.tail(lookback)
        returns = recent["close"].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        daily_std = float(returns.std())
        annualized = daily_std * math.sqrt(252)
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

            vol_avg = self._compute_avg_volume(df)
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
