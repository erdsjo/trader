import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta
from app.core.data.yahoo import YahooDataSource


@pytest.fixture
def data_source():
    return YahooDataSource()


@pytest.mark.asyncio
async def test_fetch_historical_batch_returns_dict(data_source):
    symbols = ["AAPL", "MSFT"]
    end = datetime.utcnow()
    start = end - timedelta(days=5)

    with patch.object(data_source, "fetch_historical", new_callable=AsyncMock) as mock:
        mock.return_value = pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5],
            "close": [1.5], "volume": [100], "timestamp": [end],
        })
        result = await data_source.fetch_historical_batch(symbols, start, end, "1d")

    assert isinstance(result, dict)
    assert "AAPL" in result
    assert "MSFT" in result
    assert isinstance(result["AAPL"], pd.DataFrame)


@pytest.mark.asyncio
async def test_fetch_historical_batch_handles_failure(data_source):
    symbols = ["AAPL", "INVALID"]
    end = datetime.utcnow()
    start = end - timedelta(days=5)

    async def side_effect(symbol, start, end, interval):
        if symbol == "INVALID":
            raise Exception("fetch failed")
        return pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5],
            "close": [1.5], "volume": [100], "timestamp": [end],
        })

    with patch.object(data_source, "fetch_historical", side_effect=side_effect):
        result = await data_source.fetch_historical_batch(symbols, start, end, "1d")

    assert "AAPL" in result
    assert "INVALID" not in result  # Failed symbols excluded
