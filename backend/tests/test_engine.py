from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from app.core.broker.simulator import SimulatorBroker
from app.core.engine import TradingEngine
from app.core.strategy.base import Signal


def make_sample_df(n=50):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
        "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="h"),
    })


@pytest.fixture
def mock_data_source():
    source = AsyncMock()
    source.fetch_latest = AsyncMock(return_value=make_sample_df())
    source.source_name = "mock"
    return source


@pytest.fixture
def mock_strategy():
    strategy = AsyncMock()
    strategy.is_trained.return_value = True
    strategy.analyze = AsyncMock(return_value=Signal(
        action="buy", confidence=0.8, symbol="AAPL",
        suggested_quantity=5, reasoning="test signal",
    ))
    return strategy


@pytest.fixture
def broker():
    return SimulatorBroker(initial_cash=10000.0)


@pytest.fixture
def engine(mock_data_source, mock_strategy, broker):
    return TradingEngine(
        data_source=mock_data_source,
        strategy=mock_strategy,
        broker=broker,
        symbols=["AAPL"],
        min_confidence=0.6,
    )


@pytest.mark.asyncio
async def test_engine_tick_executes_buy(engine, broker):
    await engine.tick()
    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_engine_tick_hold_does_nothing(engine, mock_strategy, broker):
    mock_strategy.analyze.return_value = Signal(
        action="hold", confidence=0.9, symbol="AAPL",
        suggested_quantity=0, reasoning="hold",
    )
    await engine.tick()
    positions = await broker.get_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_engine_tick_low_confidence_skips(engine, mock_strategy, broker):
    mock_strategy.analyze.return_value = Signal(
        action="buy", confidence=0.3, symbol="AAPL",
        suggested_quantity=5, reasoning="low confidence",
    )
    await engine.tick()
    positions = await broker.get_positions()
    assert len(positions) == 0
