import numpy as np
import pandas as pd
import pytest

from app.core.strategy.base import Signal
from app.core.strategy.ml_strategy import MLStrategy


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 500
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


@pytest.fixture
def strategy(tmp_path):
    return MLStrategy(model_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_strategy_not_trained_initially(strategy):
    assert strategy.is_trained() is False


@pytest.mark.asyncio
async def test_train_returns_metrics(strategy, training_data):
    metrics = await strategy.train(training_data)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert strategy.is_trained() is True


@pytest.mark.asyncio
async def test_analyze_returns_signal(strategy, training_data):
    await strategy.train(training_data)
    signal = await strategy.analyze("AAPL", training_data.tail(50))
    assert isinstance(signal, Signal)
    assert signal.action in ("buy", "sell", "hold")
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.symbol == "AAPL"


@pytest.mark.asyncio
async def test_analyze_without_training_returns_hold(strategy, training_data):
    signal = await strategy.analyze("AAPL", training_data.tail(50))
    assert signal.action == "hold"
    assert signal.confidence == 0.0
