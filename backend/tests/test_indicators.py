import numpy as np
import pandas as pd
import pytest

from app.core.strategy.indicators import compute_indicators


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 100
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


def test_compute_indicators_adds_columns(sample_ohlcv):
    result = compute_indicators(sample_ohlcv)
    expected_cols = ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "sma_20", "ema_12"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_indicators_no_nans_after_warmup(sample_ohlcv):
    result = compute_indicators(sample_ohlcv)
    # After warmup period (first 33 rows for longest indicator), no NaNs
    trimmed = result.iloc[33:]
    indicator_cols = ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "sma_20", "ema_12"]
    for col in indicator_cols:
        assert not trimmed[col].isna().any(), f"NaN found in {col} after warmup"


def test_compute_indicators_preserves_original_columns(sample_ohlcv):
    result = compute_indicators(sample_ohlcv)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns
