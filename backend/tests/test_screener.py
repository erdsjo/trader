import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.screener import StockScreener


def _make_price_data(symbol: str, days: int = 30) -> pd.DataFrame:
    np.random.seed(hash(symbol) % 2**31)
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(days)]
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close - np.random.rand(days),
        "high": close + np.random.rand(days) * 2,
        "low": close - np.random.rand(days) * 2,
        "close": close,
        "volume": np.random.randint(500_000, 5_000_000, days).astype(float),
    })


def test_compute_volume_avg():
    df = _make_price_data("AAPL")
    screener = StockScreener(min_volume=1_000_000, min_volatility=0.0)
    vol_avg = screener._compute_avg_volume(df, lookback=20)
    assert isinstance(vol_avg, float)
    assert vol_avg > 0


def test_compute_volatility():
    df = _make_price_data("AAPL")
    screener = StockScreener(min_volume=0, min_volatility=0.0)
    volatility = screener._compute_volatility(df, lookback=20)
    assert isinstance(volatility, float)
    assert volatility > 0


def test_filter_by_volume_and_volatility():
    data = {
        "HIGH_VOL": _make_price_data("HIGH_VOL"),
        "LOW_VOL": _make_price_data("LOW_VOL"),
    }
    data["LOW_VOL"]["volume"] = 100.0

    screener = StockScreener(min_volume=1_000_000, min_volatility=0.0)
    universe = {"HIGH_VOL": "Tech", "LOW_VOL": "Tech"}
    candidates = screener.filter_candidates(data, universe)

    assert "HIGH_VOL" in candidates
    assert "LOW_VOL" not in candidates


def test_filter_returns_sector_info():
    data = {"AAPL": _make_price_data("AAPL")}
    screener = StockScreener(min_volume=0, min_volatility=0.0)
    universe = {"AAPL": "Information Technology"}
    candidates = screener.filter_candidates(data, universe)

    assert "AAPL" in candidates
    assert candidates["AAPL"]["sector"] == "Information Technology"
    assert "volume_avg" in candidates["AAPL"]
    assert "volatility" in candidates["AAPL"]
