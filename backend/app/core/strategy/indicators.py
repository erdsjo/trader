import pandas as pd
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # RSI
    result["rsi"] = ta.momentum.rsi(result["close"], window=14)

    # MACD
    macd = ta.trend.MACD(result["close"])
    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(result["close"], window=20)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_lower"] = bb.bollinger_lband()

    # Moving averages
    result["sma_20"] = ta.trend.sma_indicator(result["close"], window=20)
    result["ema_12"] = ta.trend.ema_indicator(result["close"], window=12)

    return result
