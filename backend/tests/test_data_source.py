from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.data.yahoo import YahooDataSource


@pytest.fixture
def yahoo():
    return YahooDataSource()


def _make_mock_history(n=20):
    """Create a mock yfinance-style DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D", tz="US/Eastern")
    close = 150 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.5),
            "Low": close - abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n),
        },
        index=pd.Index(dates, name="Date"),
    )


def test_source_name(yahoo):
    assert yahoo.source_name == "yahoo"


def test_supported_intervals(yahoo):
    intervals = yahoo.supported_intervals()
    assert "1d" in intervals
    assert "1h" in intervals
    assert isinstance(intervals, list)


@pytest.mark.asyncio
async def test_fetch_historical(yahoo):
    mock_df = _make_mock_history()
    with patch.object(yahoo, "_download", return_value=mock_df):
        end = datetime.now()
        start = end - timedelta(days=30)
        df = await yahoo.fetch_historical("AAPL", start, end, "1d")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ["open", "high", "low", "close", "volume", "timestamp"]:
            assert col in df.columns


@pytest.mark.asyncio
async def test_fetch_latest(yahoo):
    mock_df = _make_mock_history(5)
    with patch.object(yahoo, "_download", return_value=mock_df):
        df = await yahoo.fetch_latest("AAPL", "1d")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "close" in df.columns


@pytest.mark.asyncio
async def test_fetch_historical_empty(yahoo):
    with patch.object(yahoo, "_download", return_value=pd.DataFrame()):
        end = datetime.now()
        start = end - timedelta(days=30)
        df = await yahoo.fetch_historical("INVALID", start, end, "1d")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "close" in df.columns
