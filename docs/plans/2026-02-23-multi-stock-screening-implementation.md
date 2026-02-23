# Multi-Stock Screening & Sector-Based Trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scale the trading app from single-stock to dynamic S&P 500 multi-stock trading with daily screening and sector-specific ML models.

**Architecture:** A `StockScreener` batch pipeline fetches data for all S&P 500 stocks, filters by volume/volatility, trains 11 sector-specific XGBoost models, and ranks stocks by predicted opportunity. The engine trades the top-N per sector, re-screening daily.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0, XGBoost, yfinance, PostgreSQL, React 18 + TypeScript

---

### Task 1: Add Stock Universe DB Model & Migration

**Files:**
- Modify: `backend/app/models/db.py` (after line 102)
- Create: `backend/alembic/versions/<auto>_add_universe_and_screening.py` (via alembic)

**Step 1: Add StockUniverse and ScreeningResult models to db.py**

Add after the `MLModel` class (line 102):

```python
class StockUniverse(Base):
    __tablename__ = "stock_universe"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    sector: Mapped[str] = mapped_column(String, nullable=False, index=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SectorModel(Base):
    __tablename__ = "sector_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(ForeignKey("simulations.id"))
    sector: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    file_path: Mapped[str] = mapped_column(String, nullable=False)


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(
        ForeignKey("simulations.id"), index=True
    )
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    sector: Mapped[str] = mapped_column(String, nullable=False)
    volume_avg: Mapped[float] = mapped_column(Float, nullable=False)
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    opportunity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    screened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

**Step 2: Add `fetched_at` column to PriceData model**

In the `PriceData` class (line 49-65), add after the `source` field:

```python
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=True
    )
```

**Step 3: Generate and apply migration**

Run:
```bash
cd /Users/sjoerdniesink/trader/backend
alembic revision --autogenerate -m "add universe screening and sector models"
alembic upgrade head
```

**Step 4: Commit**

```bash
git add backend/app/models/db.py backend/alembic/versions/
git commit -m "feat: add stock universe, sector model, and screening result DB models"
```

---

### Task 2: S&P 500 CSV Data & Universe Loader

**Files:**
- Create: `backend/data/sp500.csv`
- Create: `backend/app/core/data/universe.py`
- Create: `backend/tests/test_universe.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_universe.py
import pytest
from pathlib import Path
from app.core.data.universe import StockUniverseLoader

SP500_CSV = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"


def test_csv_file_exists():
    assert SP500_CSV.exists(), "sp500.csv must exist in backend/data/"


def test_load_csv_returns_records():
    loader = StockUniverseLoader(SP500_CSV)
    records = loader.load_from_csv()
    assert len(records) > 400  # S&P 500 should have ~500 entries
    first = records[0]
    assert "symbol" in first
    assert "name" in first
    assert "sector" in first


def test_get_sectors_returns_eleven():
    loader = StockUniverseLoader(SP500_CSV)
    records = loader.load_from_csv()
    sectors = set(r["sector"] for r in records)
    assert len(sectors) == 11


def test_get_symbols_by_sector():
    loader = StockUniverseLoader(SP500_CSV)
    records = loader.load_from_csv()
    tech = [r for r in records if r["sector"] == "Information Technology"]
    assert len(tech) > 30  # IT sector should have plenty
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_universe.py -v`
Expected: FAIL — module not found, CSV not found

**Step 3: Create the S&P 500 CSV**

Fetch the current S&P 500 list from Wikipedia and save as `backend/data/sp500.csv`. The CSV must have columns: `symbol,name,sector`. You can use this Python snippet to generate it:

```python
# One-time script to generate sp500.csv — run manually
import pandas as pd

table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
df = table[["Symbol", "Security", "GICS Sector"]].rename(
    columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"}
)
df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B → BRK-B (Yahoo format)
df.to_csv("backend/data/sp500.csv", index=False)
```

If Wikipedia fetching is unreliable, manually create a CSV with at least the major stocks across all 11 sectors.

**Step 4: Implement the universe loader**

```python
# backend/app/core/data/universe.py
import csv
from pathlib import Path


class StockUniverseLoader:
    def __init__(self, csv_path: Path | None = None):
        if csv_path is None:
            csv_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sp500.csv"
        self.csv_path = csv_path

    def load_from_csv(self) -> list[dict]:
        records = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "symbol": row["symbol"].strip(),
                    "name": row["name"].strip(),
                    "sector": row["sector"].strip(),
                })
        return records

    def get_sectors(self) -> list[str]:
        records = self.load_from_csv()
        return sorted(set(r["sector"] for r in records))

    def get_symbols_by_sector(self, sector: str) -> list[str]:
        records = self.load_from_csv()
        return [r["symbol"] for r in records if r["sector"] == sector]
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_universe.py -v`
Expected: All 4 PASS

**Step 6: Commit**

```bash
git add backend/data/sp500.csv backend/app/core/data/universe.py backend/tests/test_universe.py
git commit -m "feat: add S&P 500 CSV data and universe loader"
```

---

### Task 3: Batch Data Fetching on YahooDataSource

**Files:**
- Modify: `backend/app/core/data/base.py` (add abstract method)
- Modify: `backend/app/core/data/yahoo.py` (add batch method)
- Create: `backend/tests/test_batch_fetch.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_batch_fetch.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_batch_fetch.py -v`
Expected: FAIL — `fetch_historical_batch` not found

**Step 3: Add `fetch_historical_batch` to DataSource base**

In `backend/app/core/data/base.py`, add after the `fetch_latest` method (after line 17):

```python
    async def fetch_historical_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols. Returns {symbol: DataFrame}.
        Default implementation fetches sequentially; subclasses may optimize."""
        results = {}
        for symbol in symbols:
            try:
                df = await self.fetch_historical(symbol, start, end, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception:
                pass  # Skip failed symbols
        return results
```

**Step 4: Add optimized batch method to YahooDataSource**

In `backend/app/core/data/yahoo.py`, add a `fetch_historical_batch` method that uses `yfinance.download` for batch fetching. Add after the `fetch_latest` method (after line 52):

```python
    async def fetch_historical_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
        batch_size: int = 50,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            try:
                batch_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda b=batch: yf.download(
                            " ".join(b),
                            start=start,
                            end=end,
                            interval=interval,
                            group_by="ticker",
                            threads=True,
                        ),
                    ),
                    timeout=120,
                )
                if len(batch) == 1:
                    # yfinance returns flat columns for single ticker
                    symbol = batch[0]
                    df = self._normalize(batch_result, symbol)
                    if not df.empty:
                        results[symbol] = df
                else:
                    for symbol in batch:
                        try:
                            if symbol in batch_result.columns.get_level_values(0):
                                df = batch_result[symbol].dropna(how="all")
                                df = self._normalize(df, symbol)
                                if not df.empty:
                                    results[symbol] = df
                        except Exception:
                            pass
            except Exception:
                # Fallback: fetch individually
                for symbol in batch:
                    try:
                        df = await self.fetch_historical(symbol, start, end, interval)
                        if not df.empty:
                            results[symbol] = df
                    except Exception:
                        pass
            if i + batch_size < len(symbols):
                await asyncio.sleep(2)  # Rate limit between batches
        return results
```

Also add `import asyncio` at the top of `yahoo.py` if not already present.

**Step 5: Run tests to verify they pass**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_batch_fetch.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add backend/app/core/data/base.py backend/app/core/data/yahoo.py backend/tests/test_batch_fetch.py
git commit -m "feat: add batch historical data fetching for multi-stock support"
```

---

### Task 4: Stock Screener — Volume/Volatility Filter

**Files:**
- Create: `backend/app/core/screener.py`
- Create: `backend/tests/test_screener.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_screener.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.screener import StockScreener


def _make_price_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate synthetic OHLCV data for a stock."""
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
    # Force LOW_VOL to have very low volume
    data["LOW_VOL"]["volume"] = 100.0

    screener = StockScreener(min_volume=1_000_000, min_volatility=0.0)
    universe = {"HIGH_VOL": "Tech", "LOW_VOL": "Tech"}
    candidates = screener.filter_candidates(data, universe)

    # LOW_VOL should be filtered out (volume too low)
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_screener.py -v`
Expected: FAIL — module `app.core.screener` not found

**Step 3: Implement the screener**

```python
# backend/app/core/screener.py
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
        universe: dict[str, str],  # {symbol: sector}
    ) -> dict[str, dict]:
        """Filter stocks by volume and volatility. Returns {symbol: {sector, volume_avg, volatility}}."""
        candidates: dict[str, dict] = {}
        for symbol, df in price_data.items():
            if symbol not in universe:
                continue
            if len(df) < 5:
                continue

            vol_avg = self._compute_avg_volume(df)
            volatility = self._compute_volatility(df)

            if vol_avg < self.min_volume:
                logger.debug(f"{symbol}: volume {vol_avg:.0f} below min {self.min_volume}")
                continue
            if volatility < self.min_volatility:
                logger.debug(f"{symbol}: volatility {volatility:.2%} below min {self.min_volatility:.2%}")
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
        """Select top N stocks per sector by opportunity_score. Marks selected=True."""
        by_sector: dict[str, list[tuple[str, dict]]] = {}
        for symbol, info in candidates.items():
            sector = info["sector"]
            by_sector.setdefault(sector, []).append((symbol, info))

        selected: dict[str, dict] = {}
        for sector, stocks in by_sector.items():
            # Sort by opportunity_score desc (falls back to volatility if no score)
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_screener.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add backend/app/core/screener.py backend/tests/test_screener.py
git commit -m "feat: add stock screener with volume/volatility filtering"
```

---

### Task 5: Sector-Aware ML Strategy

**Files:**
- Modify: `backend/app/core/strategy/ml_strategy.py`
- Modify: `backend/app/core/strategy/indicators.py`
- Create: `backend/tests/test_sector_strategy.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_sector_strategy.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.strategy.ml_strategy import MLStrategy


def _make_sector_data(symbols: list[str], days: int = 100) -> pd.DataFrame:
    """Generate multi-stock training data with symbol column."""
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

    # Analyze a single stock — should use the IT sector model
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
    # Remove symbol column to simulate old-style single-symbol data
    data = data.drop(columns=["symbol"])
    metrics = await strategy.train(data)
    assert metrics is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_sector_strategy.py -v`
Expected: FAIL — `train()` doesn't accept `sector` param

**Step 3: Add cross-stock features to indicators.py**

In `backend/app/core/strategy/indicators.py`, add a new function after `compute_indicators` (after line 25):

```python
def compute_cross_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector-relative features. Expects a DataFrame with a 'symbol' column
    containing data for multiple stocks in the same sector."""
    df = df.copy()
    if "symbol" not in df.columns:
        df["sector_avg_return"] = 0.0
        df["sector_rsi_mean"] = 0.0
        df["relative_volume"] = 1.0
        return df

    df["return"] = df.groupby("symbol")["close"].pct_change()

    # Sector average return per timestamp
    sector_avg = df.groupby("timestamp")["return"].mean().rename("sector_avg_return")
    df = df.merge(sector_avg, on="timestamp", how="left")

    # Compute RSI per symbol first (requires indicators already computed)
    if "rsi" in df.columns:
        sector_rsi = df.groupby("timestamp")["rsi"].mean().rename("sector_rsi_mean")
        df = df.merge(sector_rsi, on="timestamp", how="left")
    else:
        df["sector_rsi_mean"] = 0.0

    # Relative volume (stock volume / sector average volume per timestamp)
    sector_vol = df.groupby("timestamp")["volume"].mean().rename("sector_avg_vol")
    df = df.merge(sector_vol, on="timestamp", how="left")
    df["relative_volume"] = df["volume"] / df["sector_avg_vol"].replace(0, 1)

    df.drop(columns=["return", "sector_avg_vol"], inplace=True, errors="ignore")
    return df
```

**Step 4: Modify MLStrategy to support sector models**

Refactor `backend/app/core/strategy/ml_strategy.py`. The key changes:
- `self._models` becomes a dict: `{sector_name: model}` instead of a single `self._model`
- `train()` and `analyze()` accept an optional `sector` parameter
- `_prepare_features()` adds cross-stock features when `symbol` column is present
- Model files saved as `{model_dir}/{sector_name}.joblib`
- Backward compatible: `sector=None` uses key `"__universal__"`

```python
# backend/app/core/strategy/ml_strategy.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .base import ModelMetrics, Signal, Strategy
from .indicators import compute_cross_stock_features, compute_indicators

logger = logging.getLogger(__name__)

UNIVERSAL_KEY = "__universal__"


class MLStrategy(Strategy):
    def __init__(self, model_dir: str = "models", threshold: float = 0.005):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, XGBClassifier] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._threshold = threshold

    def _sector_key(self, sector: str | None) -> str:
        return sector if sector else UNIVERSAL_KEY

    def is_trained(self, sector: str | None = None) -> bool:
        return self._sector_key(sector) in self._models

    async def train(
        self, training_data: pd.DataFrame, sector: str | None = None
    ) -> ModelMetrics:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._train_sync, training_data, sector
        )

    async def analyze(
        self,
        symbol: str,
        data: pd.DataFrame,
        sector: str | None = None,
    ) -> Signal:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._analyze_sync, symbol, data, sector
        )

    def _feature_columns(self, has_symbol: bool) -> list[str]:
        base = ["close", "volume", "rsi", "macd", "macd_signal",
                "bb_upper", "bb_lower", "sma_20", "ema_12"]
        cross = ["sector_avg_return", "sector_rsi_mean", "relative_volume"]
        if has_symbol:
            return base + cross + ["symbol_encoded"]
        return base

    def _prepare_features(
        self, df: pd.DataFrame, sector: str | None = None
    ) -> pd.DataFrame:
        df = compute_indicators(df)
        has_symbol = "symbol" in df.columns
        if has_symbol:
            df = compute_cross_stock_features(df)
            key = self._sector_key(sector)
            if key not in self._label_encoders:
                self._label_encoders[key] = LabelEncoder()
                df["symbol_encoded"] = self._label_encoders[key].fit_transform(
                    df["symbol"]
                )
            else:
                # Handle unseen symbols gracefully
                le = self._label_encoders[key]
                known = set(le.classes_)
                df["symbol_encoded"] = df["symbol"].apply(
                    lambda s: le.transform([s])[0] if s in known else -1
                )
        cols = self._feature_columns(has_symbol)
        return df.dropna(subset=[c for c in cols if c in df.columns])

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        if "symbol" in df.columns:
            future_return = df.groupby("symbol")["close"].pct_change(5).shift(-5)
        else:
            future_return = df["close"].pct_change(5).shift(-5)
        labels = pd.Series(1, index=df.index)  # default hold
        labels[future_return > self._threshold] = 2   # buy
        labels[future_return < -self._threshold] = 0  # sell
        return labels

    def _train_sync(
        self, data: pd.DataFrame, sector: str | None = None
    ) -> ModelMetrics:
        key = self._sector_key(sector)
        df = self._prepare_features(data, sector)
        has_symbol = "symbol" in data.columns
        feature_cols = self._feature_columns(has_symbol)
        feature_cols = [c for c in feature_cols if c in df.columns]

        df["label"] = self._create_labels(df)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = float(np.mean(preds == y_test))

        self._models[key] = model
        model_path = self._model_dir / f"{key.replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        logger.info(f"Trained sector model '{key}': accuracy={accuracy:.3f}")

        return ModelMetrics(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            feature_importance=importance,
        )

    def _analyze_sync(
        self,
        symbol: str,
        data: pd.DataFrame,
        sector: str | None = None,
    ) -> Signal:
        key = self._sector_key(sector)
        if key not in self._models:
            return Signal("hold", 0.0, symbol, 0, "no model for sector")

        model = self._models[key]
        df = self._prepare_features(data.copy(), sector)
        has_symbol = "symbol" in data.columns
        feature_cols = self._feature_columns(has_symbol)
        feature_cols = [c for c in feature_cols if c in df.columns]

        if df.empty:
            return Signal("hold", 0.0, symbol, 0, "insufficient data")

        latest = df.iloc[[-1]][feature_cols]
        proba = model.predict_proba(latest)[0]

        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action_idx = int(np.argmax(proba))
        confidence = float(proba[action_idx])

        return Signal(
            action=action_map[action_idx],
            confidence=confidence,
            symbol=symbol,
            suggested_quantity=0,
            reasoning=f"sector={key} conf={confidence:.2f}",
        )
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_sector_strategy.py tests/test_strategy.py -v`
Expected: All PASS (both new sector tests and existing backward-compat tests)

**Step 6: Commit**

```bash
git add backend/app/core/strategy/ml_strategy.py backend/app/core/strategy/indicators.py backend/tests/test_sector_strategy.py
git commit -m "feat: sector-aware ML strategy with cross-stock features"
```

---

### Task 6: Integrate Screener into Simulation Routes

**Files:**
- Modify: `backend/app/api/routes/simulation.py`
- Modify: `backend/app/models/schemas.py`

**Step 1: Add new Pydantic schemas**

In `backend/app/models/schemas.py`, add after the existing schemas (after line 71):

```python
class ScreeningConfig(BaseModel):
    min_volume: float = 1_000_000
    min_volatility: float = 0.15
    top_n_per_sector: int = 5
    rescreen_hour_utc: int = 6


class ScreeningResultResponse(BaseModel):
    symbol: str
    sector: str
    volume_avg: float
    volatility: float
    opportunity_score: float | None
    selected: bool
    screened_at: datetime

    class Config:
        from_attributes = True
```

Add `from datetime import datetime` to the imports if not present.

**Step 2: Modify `_train_and_run` in simulation.py to support screener mode**

This is the largest change. In `backend/app/api/routes/simulation.py`, the `_train_and_run` function (line 155-197) needs to:

1. Check if `config` has `universe` key (screener mode) vs `symbols` key (legacy mode)
2. In screener mode: load universe → batch fetch → screen → train sector models → run
3. Support daily re-screening in the tick loop

Add new imports at the top of `simulation.py`:

```python
from app.core.screener import StockScreener
from app.core.data.universe import StockUniverseLoader
```

Add a new helper function after `_get_training_data` (after line 152):

```python
async def _run_screening(
    config: dict,
    data_source,
    event_log,
    db,
    simulation_id: int,
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    """Run the stock screening pipeline. Returns (universe_map, price_data)."""
    screening_cfg = config.get("screening", {})
    min_volume = screening_cfg.get("min_volume", 1_000_000)
    min_volatility = screening_cfg.get("min_volatility", 0.15)
    top_n = screening_cfg.get("top_n_per_sector", 5)

    # Load S&P 500 universe
    loader = StockUniverseLoader()
    records = loader.load_from_csv()
    universe_map = {r["symbol"]: r["sector"] for r in records}
    all_symbols = list(universe_map.keys())

    event_log.info(f"Screening {len(all_symbols)} stocks from S&P 500")

    # Batch fetch historical data
    interval = config.get("interval", "1d")
    from datetime import datetime, timedelta
    end = datetime.utcnow()
    max_days = _INTERVAL_MAX_DAYS.get(interval, 60)
    lookback = min(60, max_days)
    start = end - timedelta(days=lookback)

    event_log.info(f"Fetching {lookback} days of data for {len(all_symbols)} stocks...")
    price_data = await data_source.fetch_historical_batch(all_symbols, start, end, interval)
    event_log.info(f"Fetched data for {len(price_data)}/{len(all_symbols)} stocks")

    # Store fetched data in cache
    for symbol, df in price_data.items():
        await _store_price_data(db, df, symbol, interval)

    # Screen candidates
    screener = StockScreener(
        min_volume=min_volume,
        min_volatility=min_volatility,
        top_n_per_sector=top_n,
    )
    candidates = screener.filter_candidates(price_data, universe_map)
    event_log.info(f"Screening: {len(candidates)} candidates passed volume/volatility filters")

    selected = screener.select_top_n(candidates)

    # Save screening results to DB
    for symbol, info in selected.items():
        result = ScreeningResult(
            simulation_id=simulation_id,
            symbol=symbol,
            sector=info["sector"],
            volume_avg=info["volume_avg"],
            volatility=info["volatility"],
            opportunity_score=info.get("opportunity_score"),
            selected=info["selected"],
        )
        db.add(result)
    await db.commit()

    active_symbols = {s: info["sector"] for s, info in selected.items() if info["selected"]}
    event_log.info(f"Selected {len(active_symbols)} stocks for trading")

    # Return only data for candidates (for training)
    candidate_data = {s: price_data[s] for s in candidates if s in price_data}
    return active_symbols, candidate_data
```

Add the `ScreeningResult` import from `app.models.db`.

Modify `_train_and_run` to branch on `universe` vs `symbols`:

```python
async def _train_and_run(sim_id, config, strategy, broker, data_source, engine, event_log, db):
    try:
        interval = config.get("interval", "1d")

        if "universe" in config:
            # === SCREENER MODE ===
            active_symbols, candidate_data = await _run_screening(
                config, data_source, event_log, db, sim_id
            )

            # Train sector models
            from app.core.data.universe import StockUniverseLoader
            loader = StockUniverseLoader()
            universe_map = {r["symbol"]: r["sector"] for r in loader.load_from_csv()}

            sectors_to_train = set(active_symbols.values())
            for sector in sectors_to_train:
                sector_symbols = [s for s, sec in universe_map.items()
                                  if sec == sector and s in candidate_data]
                if len(sector_symbols) < 5:
                    event_log.warning(f"Sector '{sector}': only {len(sector_symbols)} stocks, skipping")
                    continue

                frames = []
                for s in sector_symbols:
                    df = candidate_data[s].copy()
                    df["symbol"] = s
                    frames.append(df)
                sector_data = pd.concat(frames, ignore_index=True)
                sector_data.sort_values("timestamp", inplace=True)

                event_log.info(f"Training model for sector: {sector} ({len(sector_symbols)} stocks)")
                metrics = await strategy.train(sector_data, sector=sector)
                event_log.info(f"Sector '{sector}' trained: accuracy={metrics.accuracy:.3f}")

            symbols = list(active_symbols.keys())
            engine.symbols = symbols
            engine._sector_map = active_symbols  # symbol → sector mapping for analyze()

        else:
            # === LEGACY MODE (fixed symbols) ===
            symbols = config.get("symbols", [])
            all_data = []
            for symbol in symbols:
                df = await _get_training_data(symbol, interval, data_source, event_log)
                if not df.empty:
                    all_data.append(df)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                event_log.info(f"Training on {len(combined)} rows for {len(symbols)} symbols")
                metrics = await strategy.train(combined)
                event_log.info(f"Training complete: accuracy={metrics.accuracy:.3f}")

        # Run tick loop
        event_log.info("Starting tick loop")
        await engine.run()

    except Exception as e:
        event_log.error(f"Engine crashed: {e}")
        logger.exception(f"Simulation {sim_id} crashed")
```

**Step 3: Modify `start_simulation` endpoint to pass sector map to engine**

In the `start_simulation` endpoint (line 231-321), modify the `TradingEngine` creation to support sector-aware analysis. The engine's `tick()` method needs to pass the sector to `strategy.analyze()`.

**Step 4: Add screening results endpoint**

Add a new endpoint in `simulation.py`:

```python
@router.get("/simulations/{sim_id}/screening")
async def get_screening_results(
    sim_id: int, db: AsyncSession = Depends(get_db)
):
    results = await db.execute(
        select(ScreeningResult)
        .where(ScreeningResult.simulation_id == sim_id)
        .order_by(ScreeningResult.screened_at.desc())
        .limit(500)
    )
    rows = results.scalars().all()
    return [
        {
            "symbol": r.symbol,
            "sector": r.sector,
            "volume_avg": r.volume_avg,
            "volatility": r.volatility,
            "opportunity_score": r.opportunity_score,
            "selected": r.selected,
            "screened_at": r.screened_at.isoformat(),
        }
        for r in rows
    ]
```

**Step 5: Run full test suite**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/ -v`
Expected: All existing tests PASS, no regressions

**Step 6: Commit**

```bash
git add backend/app/api/routes/simulation.py backend/app/models/schemas.py
git commit -m "feat: integrate stock screener into simulation pipeline"
```

---

### Task 7: Modify TradingEngine for Sector-Aware Analysis

**Files:**
- Modify: `backend/app/core/engine.py`
- Modify: `backend/tests/test_engine.py`

**Step 1: Write the failing test**

Add to `backend/tests/test_engine.py`:

```python
@pytest.mark.asyncio
async def test_engine_tick_with_sector_map(
    mock_data_source, mock_strategy, broker
):
    engine = TradingEngine(
        data_source=mock_data_source,
        strategy=mock_strategy,
        broker=broker,
        symbols=["AAPL", "JNJ"],
        sector_map={"AAPL": "Information Technology", "JNJ": "Health Care"},
    )
    mock_strategy.analyze.return_value = Signal("buy", 0.8, "AAPL", 10, "test")
    await engine.tick()
    # Verify analyze was called with sector parameter
    calls = mock_strategy.analyze.call_args_list
    assert len(calls) == 2
    # Check sector kwarg was passed
    for call in calls:
        assert "sector" in call.kwargs or len(call.args) >= 3
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_engine.py::test_engine_tick_with_sector_map -v`
Expected: FAIL — `sector_map` not accepted

**Step 3: Modify TradingEngine to accept and use sector_map**

In `backend/app/core/engine.py`, modify `__init__` (line 15-38) to accept `sector_map`:

```python
    def __init__(
        self,
        data_source,
        strategy,
        broker,
        symbols: list[str],
        min_confidence: float = 0.6,
        interval: str = "1d",
        on_trade=None,
        on_tick_complete=None,
        event_log=None,
        sector_map: dict[str, str] | None = None,
    ):
        # ... existing assignments ...
        self._sector_map = sector_map or {}
```

In the `tick()` method (line 55-104), modify the `strategy.analyze` call to pass sector:

```python
            sector = self._sector_map.get(symbol)
            signal = await self.strategy.analyze(symbol, enriched_data, sector=sector)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/core/engine.py backend/tests/test_engine.py
git commit -m "feat: add sector-aware analysis to trading engine"
```

---

### Task 8: Universe Refresh API Endpoint

**Files:**
- Create: `backend/app/api/routes/universe.py`
- Modify: `backend/app/main.py` (register route)

**Step 1: Create the universe route**

```python
# backend/app/api/routes/universe.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.api.deps import get_db
from app.core.data.universe import StockUniverseLoader
from app.models.db import StockUniverse

router = APIRouter()


@router.get("/universe")
async def get_universe(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StockUniverse).order_by(StockUniverse.sector, StockUniverse.symbol)
    )
    rows = result.scalars().all()
    if not rows:
        # Load from CSV if DB is empty
        await refresh_universe(db)
        result = await db.execute(
            select(StockUniverse).order_by(StockUniverse.sector, StockUniverse.symbol)
        )
        rows = result.scalars().all()
    return [
        {"symbol": r.symbol, "name": r.name, "sector": r.sector}
        for r in rows
    ]


@router.get("/universe/sectors")
async def get_sectors(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StockUniverse.sector).distinct().order_by(StockUniverse.sector)
    )
    return [row[0] for row in result.all()]


@router.post("/universe/refresh")
async def refresh_universe(db: AsyncSession = Depends(get_db)):
    loader = StockUniverseLoader()
    records = loader.load_from_csv()

    await db.execute(delete(StockUniverse))
    for r in records:
        db.add(StockUniverse(
            symbol=r["symbol"],
            name=r["name"],
            sector=r["sector"],
        ))
    await db.commit()

    return {"loaded": len(records)}
```

**Step 2: Register the route in main.py**

In `backend/app/main.py`, add:

```python
from app.api.routes.universe import router as universe_router
```

And in the router includes section (line 18-29), add:

```python
app.include_router(universe_router, prefix="/api", dependencies=[Depends(require_auth)])
```

**Step 3: Run existing tests to verify no regressions**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add backend/app/api/routes/universe.py backend/app/main.py
git commit -m "feat: add universe management API endpoints"
```

---

### Task 9: Frontend — SimulationControl with Universe Mode

**Files:**
- Modify: `frontend/src/components/SimulationControl.tsx`
- Modify: `frontend/src/api/client.ts`

**Step 1: Add new API methods to client.ts**

In `frontend/src/api/client.ts`, add new interfaces and methods:

```typescript
export interface ScreeningResultItem {
  symbol: string;
  sector: string;
  volume_avg: number;
  volatility: number;
  opportunity_score: number | null;
  selected: boolean;
  screened_at: string;
}

export const getScreeningResults = (simId: number) =>
  api.get<ScreeningResultItem[]>(`/simulations/${simId}/screening`).then(r => r.data);

export const getUniverse = () =>
  api.get<{ symbol: string; name: string; sector: string }[]>('/universe').then(r => r.data);
```

**Step 2: Modify SimulationControl to support universe mode**

In `frontend/src/components/SimulationControl.tsx`, add a toggle between "Custom Symbols" and "S&P 500 Universe" mode. When universe mode is selected:
- Hide the symbols text input
- Show screening config fields (min volume, min volatility, top N per sector)
- Set `config.universe = "sp500"` and `config.screening = {...}` instead of `config.symbols`

Add new state variables:

```typescript
const [mode, setMode] = useState<'custom' | 'universe'>('universe');
const [minVolume, setMinVolume] = useState('1000000');
const [minVolatility, setMinVolatility] = useState('0.15');
const [topNPerSector, setTopNPerSector] = useState('5');
```

Modify `handleCreate` to build the config based on mode:

```typescript
const config = mode === 'universe'
  ? {
      universe: 'sp500',
      interval,
      tick_seconds: parseFloat(tickSeconds),
      screening: {
        min_volume: parseFloat(minVolume),
        min_volatility: parseFloat(minVolatility),
        top_n_per_sector: parseInt(topNPerSector),
        rescreen_hour_utc: 6,
      },
      max_positions: 20,
      position_size_pct: 5.0,
    }
  : {
      symbols: symbols.split(',').map(s => s.trim()),
      interval,
      tick_seconds: parseFloat(tickSeconds),
    };
```

Add a mode toggle and conditional form fields in the JSX.

**Step 3: Commit**

```bash
git add frontend/src/components/SimulationControl.tsx frontend/src/api/client.ts
git commit -m "feat: add universe mode to simulation creation UI"
```

---

### Task 10: Frontend — Screener Results Panel

**Files:**
- Create: `frontend/src/components/ScreenerResults.tsx`
- Modify: `frontend/src/components/Dashboard.tsx`

**Step 1: Create ScreenerResults component**

```typescript
// frontend/src/components/ScreenerResults.tsx
import { usePolling } from '../hooks/usePolling';
import { getScreeningResults, ScreeningResultItem } from '../api/client';

interface Props {
  simulationId: number | null;
}

export default function ScreenerResults({ simulationId }: Props) {
  const { data: results } = usePolling<ScreeningResultItem[]>(
    () => simulationId ? getScreeningResults(simulationId) : Promise.resolve([]),
    10000,
    !!simulationId,
  );

  if (!results || results.length === 0) {
    return <div className="bg-white rounded-lg shadow p-4"><p className="text-gray-500">No screening results yet</p></div>;
  }

  // Group by sector
  const bySector: Record<string, ScreeningResultItem[]> = {};
  for (const r of results) {
    (bySector[r.sector] ??= []).push(r);
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Stock Screener Results</h2>
      {Object.entries(bySector).sort().map(([sector, stocks]) => (
        <div key={sector} className="mb-4">
          <h3 className="font-medium text-sm text-gray-600 mb-1">{sector}</h3>
          <div className="flex flex-wrap gap-1">
            {stocks
              .sort((a, b) => (b.opportunity_score ?? 0) - (a.opportunity_score ?? 0))
              .map(s => (
                <span
                  key={s.symbol}
                  className={`text-xs px-2 py-1 rounded ${
                    s.selected
                      ? 'bg-green-100 text-green-800 font-medium'
                      : 'bg-gray-100 text-gray-500'
                  }`}
                  title={`Vol: ${(s.volume_avg / 1e6).toFixed(1)}M | Volatility: ${(s.volatility * 100).toFixed(1)}%`}
                >
                  {s.symbol}
                </span>
              ))}
          </div>
        </div>
      ))}
      <p className="text-xs text-gray-400 mt-2">
        {results.filter(r => r.selected).length} selected / {results.length} candidates
      </p>
    </div>
  );
}
```

**Step 2: Add to Dashboard**

In `frontend/src/components/Dashboard.tsx`, import and add the new component:

```typescript
import ScreenerResults from './ScreenerResults';
```

Add it in the layout between SimulationControl and Portfolio.

**Step 3: Commit**

```bash
git add frontend/src/components/ScreenerResults.tsx frontend/src/components/Dashboard.tsx
git commit -m "feat: add screener results panel to dashboard"
```

---

### Task 11: Frontend — Per-Sector P/L in ProfitLoss

**Files:**
- Modify: `frontend/src/components/ProfitLoss.tsx`
- Modify: `frontend/src/api/client.ts`

**Step 1: Extend the performance API response**

In `backend/app/models/schemas.py`, add to `PerformanceResponse`:

```python
    sector_pnl: dict[str, float] | None = None  # {sector: pnl_amount}
```

In `backend/app/api/routes/portfolio.py`, modify the performance endpoint to compute sector P/L from trades grouped by sector (joining with `stock_universe` table).

**Step 2: Add sector P/L display to ProfitLoss.tsx**

Add a simple bar or list below the existing chart showing P/L per sector:

```typescript
{performance?.sector_pnl && (
  <div className="mt-4">
    <h3 className="text-sm font-medium text-gray-600 mb-2">P/L by Sector</h3>
    {Object.entries(performance.sector_pnl).sort((a, b) => b[1] - a[1]).map(([sector, pnl]) => (
      <div key={sector} className="flex justify-between text-sm py-1">
        <span className="text-gray-700">{sector}</span>
        <span className={pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
          ${pnl.toFixed(2)}
        </span>
      </div>
    ))}
  </div>
)}
```

**Step 3: Commit**

```bash
git add frontend/src/components/ProfitLoss.tsx frontend/src/api/client.ts backend/app/api/routes/portfolio.py backend/app/models/schemas.py
git commit -m "feat: add per-sector P/L breakdown to performance view"
```

---

### Task 12: Integration Test — Full Screener Pipeline

**Files:**
- Create: `backend/tests/test_screening_integration.py`

**Step 1: Write integration test**

```python
# backend/tests/test_screening_integration.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.core.screener import StockScreener
from app.core.strategy.ml_strategy import MLStrategy


def _make_sector_data(symbols, days=60):
    frames = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**31)
        dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(days)]
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
    return {s: frames[i] for i, s in enumerate(symbols)}


@pytest.mark.asyncio
async def test_full_screening_pipeline(tmp_path):
    """Test: screen → filter → train sector model → analyze."""
    tech_symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE"]
    universe = {s: "Information Technology" for s in tech_symbols}
    price_data = _make_sector_data(tech_symbols)

    # Screen
    screener = StockScreener(min_volume=500_000, min_volatility=0.01, top_n_per_sector=3)
    candidates = screener.filter_candidates(price_data, universe)
    assert len(candidates) > 0

    selected = screener.select_top_n(candidates)
    active = [s for s, info in selected.items() if info["selected"]]
    assert len(active) == 3  # top_n_per_sector = 3

    # Train sector model on candidate data
    strategy = MLStrategy(model_dir=str(tmp_path))
    frames = []
    for s in candidates:
        df = price_data[s].copy()
        df["symbol"] = s
        frames.append(df)
    sector_data = pd.concat(frames, ignore_index=True)
    sector_data.sort_values("timestamp", inplace=True)

    metrics = await strategy.train(sector_data, sector="Information Technology")
    assert metrics.accuracy >= 0

    # Analyze one of the active stocks
    signal = await strategy.analyze(
        active[0],
        price_data[active[0]].tail(30).assign(symbol=active[0]),
        sector="Information Technology",
    )
    assert signal.action in ("buy", "sell", "hold")
```

**Step 2: Run integration test**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/test_screening_integration.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add backend/tests/test_screening_integration.py
git commit -m "test: add full screening pipeline integration test"
```

---

### Task 13: Final Verification & Cleanup

**Step 1: Run the full backend test suite**

```bash
cd /Users/sjoerdniesink/trader/backend && python -m pytest tests/ -v --tb=short
```

**Step 2: Run the frontend build**

```bash
cd /Users/sjoerdniesink/trader/frontend && npm run build
```

**Step 3: Verify Docker build**

```bash
cd /Users/sjoerdniesink/trader && docker compose build
```

**Step 4: Verify alembic migrations apply cleanly**

```bash
cd /Users/sjoerdniesink/trader/backend && alembic upgrade head
```

**Step 5: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "chore: cleanup and verify multi-stock screening feature"
```
