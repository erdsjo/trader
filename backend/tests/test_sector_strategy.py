import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.strategy.ml_strategy import MLStrategy


def _make_sector_data(symbols: list[str], days: int = 100) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**31)
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
        close = 100 + np.cumsum(np.random.randn(days) * 2)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close - np.random.rand(days),
            "high": close + np.random.rand(days) * 2,
            "low": close - np.random.rand(days) * 2,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, days).astype(float),
            "symbol": symbol,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def strategy(tmp_path):
    return MLStrategy(model_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_train_sector_model(strategy):
    data = _make_sector_data(["AAPL", "MSFT", "GOOGL", "META", "NVDA"])
    metrics = await strategy.train(data, sector="Information Technology")
    assert metrics is not None
    assert metrics.accuracy >= 0


@pytest.mark.asyncio
async def test_analyze_uses_sector_model(strategy):
    data = _make_sector_data(["AAPL", "MSFT", "GOOGL", "META", "NVDA"])
    await strategy.train(data, sector="Information Technology")

    single = data[data["symbol"] == "AAPL"].tail(30).copy()
    signal = await strategy.analyze("AAPL", single, sector="Information Technology")
    assert signal.action in ("buy", "sell", "hold")
    assert 0 <= signal.confidence <= 1


@pytest.mark.asyncio
async def test_multiple_sector_models(strategy):
    tech_data = _make_sector_data(["AAPL", "MSFT", "GOOGL", "META", "NVDA"])
    health_data = _make_sector_data(["JNJ", "PFE", "UNH", "MRK", "ABT"])

    await strategy.train(tech_data, sector="Information Technology")
    await strategy.train(health_data, sector="Health Care")

    assert strategy.is_trained(sector="Information Technology")
    assert strategy.is_trained(sector="Health Care")
    assert not strategy.is_trained(sector="Energy")


@pytest.mark.asyncio
async def test_backward_compat_no_sector(strategy):
    """Training without sector should still work (universal model)."""
    data = _make_sector_data(["AAPL", "MSFT"])
    data = data.drop(columns=["symbol"])
    metrics = await strategy.train(data)
    assert metrics is not None
