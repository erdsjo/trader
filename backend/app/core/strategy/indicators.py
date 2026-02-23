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


def compute_cross_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector-relative features. Expects DataFrame with 'symbol' column."""
    df = df.copy()
    if "symbol" not in df.columns:
        df["sector_avg_return"] = 0.0
        df["sector_rsi_mean"] = 0.0
        df["relative_volume"] = 1.0
        return df

    df["return"] = df.groupby("symbol")["close"].pct_change()

    sector_avg = df.groupby("timestamp")["return"].mean().rename("sector_avg_return")
    df = df.merge(sector_avg, on="timestamp", how="left")

    if "rsi" in df.columns:
        sector_rsi = df.groupby("timestamp")["rsi"].mean().rename("sector_rsi_mean")
        df = df.merge(sector_rsi, on="timestamp", how="left")
    else:
        df["sector_rsi_mean"] = 0.0

    sector_vol = df.groupby("timestamp")["volume"].mean().rename("sector_avg_vol")
    df = df.merge(sector_vol, on="timestamp", how="left")
    df["relative_volume"] = df["volume"] / df["sector_avg_vol"].replace(0, 1)

    df.drop(columns=["return", "sector_avg_vol"], inplace=True, errors="ignore")
    return df
