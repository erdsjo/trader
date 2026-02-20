# Trading Bot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-learning stock trading bot with simulation mode, pluggable architecture, and a React dashboard.

**Architecture:** Monolithic FastAPI backend serving API and orchestrating the trading engine. React frontend with polling-based dashboard. PostgreSQL for persistence. Pluggable interfaces for data sources, strategies, and brokers.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, yfinance, XGBoost, pandas, React 18, TypeScript, Tailwind CSS, Recharts, Docker Compose, PostgreSQL 16.

**Design doc:** `docs/plans/2026-02-20-trading-bot-design.md`

---

## Task 1: Backend Project Scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes/__init__.py`
- Create: `backend/app/api/deps.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "trader"
version = "0.1.0"
description = "Self-learning stock trading bot"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy[asyncio]==2.0.36
alembic==1.14.1
asyncpg==0.30.0
psycopg2-binary==2.9.10
yfinance==0.2.51
pandas==2.2.3
numpy==2.2.1
scikit-learn==1.6.1
xgboost==2.1.3
ta==0.11.0
pydantic==2.10.4
pydantic-settings==2.7.1
joblib==1.4.2
pytest==8.3.4
pytest-asyncio==0.25.0
httpx==0.28.1
```

**Step 3: Create app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://trader:trader@localhost:5432/trader"
    database_url_sync: str = "postgresql+psycopg2://trader:trader@localhost:5432/trader"
    api_prefix: str = "/api"
    polling_interval_seconds: int = 10
    model_storage_path: str = "./model_artifacts"
    default_slippage: float = 0.001
    min_confidence_threshold: float = 0.6

    class Config:
        env_file = ".env"


settings = Settings()
```

**Step 4: Create app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(title="Trader Bot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

**Step 5: Create empty __init__.py files and deps.py**

All `__init__.py` files are empty. `deps.py`:

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
```

**Step 6: Create tests/conftest.py**

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

**Step 7: Write test for health endpoint**

Create `backend/tests/test_health.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

**Step 8: Install dependencies and run test**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/test_health.py -v
```

Expected: PASS

**Step 9: Commit**

```bash
git add backend/
git commit -m "feat: scaffold backend project with FastAPI, config, and health endpoint"
```

---

## Task 2: Database Models & Migrations

**Files:**
- Create: `backend/app/models/db.py`
- Create: `backend/app/models/schemas.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/tests/test_models.py`

**Step 1: Write the database models**

Create `backend/app/models/db.py`:

```python
import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SimulationStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"


class TradeSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    initial_capital = Column(Float, nullable=False)
    current_cash = Column(Float, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(SimulationStatus), default=SimulationStatus.CREATED)
    config = Column(JSON, default=dict)

    trades = relationship("Trade", back_populates="simulation")
    snapshots = relationship("PortfolioSnapshot", back_populates="simulation")


class PriceData(Base):
    __tablename__ = "price_data"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "interval", name="uq_price_data"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    interval = Column(String, nullable=False)
    source = Column(String, nullable=False)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    side = Column(Enum(TradeSide), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String, nullable=True)

    simulation = relationship("Simulation", back_populates="trades")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)

    simulation = relationship("Simulation", back_populates="snapshots")


class MLModel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    metrics = Column(JSON, default=dict)
    trained_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=False)
```

**Step 2: Write Pydantic schemas**

Create `backend/app/models/schemas.py`:

```python
from datetime import datetime

from pydantic import BaseModel


class SimulationCreate(BaseModel):
    name: str
    initial_capital: float
    config: dict = {}


class SimulationResponse(BaseModel):
    id: int
    name: str
    initial_capital: float
    current_cash: float
    start_time: datetime
    status: str
    config: dict

    class Config:
        from_attributes = True


class TradeResponse(BaseModel):
    id: int
    simulation_id: int
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    strategy: str | None

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    total_value: float
    cash: float
    positions: list[dict]
    total_pnl: float
    total_pnl_pct: float


class PerformancePoint(BaseModel):
    timestamp: datetime
    total_value: float


class PerformanceResponse(BaseModel):
    points: list[PerformancePoint]
    daily_pnl: float
    total_pnl: float
    total_pnl_pct: float


class SignalResponse(BaseModel):
    action: str
    confidence: float
    symbol: str
    reasoning: str


class StrategyStatusResponse(BaseModel):
    model_name: str
    model_version: int | None
    last_trained: datetime | None
    metrics: dict
    active_signals: list[SignalResponse]
```

**Step 3: Write model tests**

Create `backend/tests/test_models.py`:

```python
from app.models.db import (
    Base,
    MLModel,
    PortfolioSnapshot,
    PriceData,
    Simulation,
    SimulationStatus,
    Trade,
    TradeSide,
)
from app.models.schemas import PortfolioResponse, SimulationCreate, SimulationResponse


def test_simulation_model_defaults():
    sim = Simulation(name="test", initial_capital=10000.0, current_cash=10000.0)
    assert sim.name == "test"
    assert sim.initial_capital == 10000.0
    assert sim.status is None  # default applied by DB, not Python


def test_trade_side_enum():
    assert TradeSide.BUY == "buy"
    assert TradeSide.SELL == "sell"


def test_simulation_status_enum():
    assert SimulationStatus.RUNNING == "running"
    assert SimulationStatus.STOPPED == "stopped"


def test_simulation_create_schema():
    data = SimulationCreate(name="test", initial_capital=10000.0)
    assert data.name == "test"
    assert data.config == {}


def test_portfolio_response_schema():
    data = PortfolioResponse(
        total_value=10500.0,
        cash=5000.0,
        positions=[{"symbol": "AAPL", "quantity": 10, "current_price": 550.0}],
        total_pnl=500.0,
        total_pnl_pct=5.0,
    )
    assert data.total_pnl_pct == 5.0


def test_all_tables_defined():
    table_names = {t.name for t in Base.metadata.sorted_tables}
    assert "simulations" in table_names
    assert "price_data" in table_names
    assert "trades" in table_names
    assert "portfolio_snapshots" in table_names
    assert "models" in table_names
```

**Step 4: Run model tests**

```bash
cd backend
pytest tests/test_models.py -v
```

Expected: all PASS

**Step 5: Set up Alembic**

```bash
cd backend
alembic init alembic
```

Then edit `backend/alembic.ini` — set `sqlalchemy.url`:

```ini
sqlalchemy.url = postgresql+psycopg2://trader:trader@localhost:5432/trader
```

Edit `backend/alembic/env.py` — add target_metadata:

```python
from app.models.db import Base
target_metadata = Base.metadata
```

**Step 6: Generate initial migration**

```bash
cd backend
alembic revision --autogenerate -m "initial tables"
```

**Step 7: Commit**

```bash
git add backend/app/models/ backend/alembic/ backend/alembic.ini backend/tests/test_models.py
git commit -m "feat: add database models, schemas, and Alembic migrations"
```

---

## Task 3: Data Source Interface & Yahoo Finance

**Files:**
- Create: `backend/app/core/data/__init__.py`
- Create: `backend/app/core/data/base.py`
- Create: `backend/app/core/data/yahoo.py`
- Create: `backend/tests/test_data_source.py`

**Step 1: Write the abstract DataSource interface**

Create `backend/app/core/data/base.py`:

```python
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    async def fetch_historical(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data. Returns DataFrame with columns:
        open, high, low, close, volume, timestamp."""

    @abstractmethod
    async def fetch_latest(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch the most recent OHLCV data points."""

    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return list of supported interval strings (e.g. '1m', '5m', '1d')."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this data source (e.g. 'yahoo')."""
```

**Step 2: Write failing test for Yahoo implementation**

Create `backend/tests/test_data_source.py`:

```python
import pandas as pd
import pytest
from datetime import datetime, timedelta

from app.core.data.yahoo import YahooDataSource


@pytest.fixture
def yahoo():
    return YahooDataSource()


def test_source_name(yahoo):
    assert yahoo.source_name == "yahoo"


def test_supported_intervals(yahoo):
    intervals = yahoo.supported_intervals()
    assert "1d" in intervals
    assert "1h" in intervals
    assert isinstance(intervals, list)


@pytest.mark.asyncio
async def test_fetch_historical(yahoo):
    end = datetime.now()
    start = end - timedelta(days=30)
    df = await yahoo.fetch_historical("AAPL", start, end, "1d")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ["open", "high", "low", "close", "volume", "timestamp"]:
        assert col in df.columns


@pytest.mark.asyncio
async def test_fetch_latest(yahoo):
    df = await yahoo.fetch_latest("AAPL", "1d")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "close" in df.columns
```

**Step 3: Run test to verify it fails**

```bash
cd backend
pytest tests/test_data_source.py -v
```

Expected: FAIL — `YahooDataSource` doesn't exist yet

**Step 4: Implement YahooDataSource**

Create `backend/app/core/data/yahoo.py`:

```python
import asyncio
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import yfinance as yf

from app.core.data.base import DataSource


class YahooDataSource(DataSource):
    INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo"]

    @property
    def source_name(self) -> str:
        return "yahoo"

    def supported_intervals(self) -> list[str]:
        return self.INTERVALS

    async def fetch_historical(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, partial(self._download, symbol, start, end, interval)
        )
        return self._normalize(df)

    async def fetch_latest(self, symbol: str, interval: str) -> pd.DataFrame:
        end = datetime.now()
        if interval in ("1m", "2m", "5m", "15m", "30m"):
            start = end - timedelta(days=1)
        else:
            start = end - timedelta(days=5)

        return await self.fetch_historical(symbol, start, end, interval)

    def _download(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        return ticker.history(start=start, end=end, interval=interval)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "timestamp"]
            )
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else "Datetime"
        df = df.rename(
            columns={
                date_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(
            None
        )
        return df[["open", "high", "low", "close", "volume", "timestamp"]]
```

**Step 5: Run tests**

```bash
cd backend
pytest tests/test_data_source.py -v
```

Expected: all PASS (requires internet connection)

**Step 6: Commit**

```bash
git add backend/app/core/data/ backend/tests/test_data_source.py
git commit -m "feat: add pluggable DataSource interface and Yahoo Finance implementation"
```

---

## Task 4: Technical Indicators

**Files:**
- Create: `backend/app/core/strategy/__init__.py`
- Create: `backend/app/core/strategy/indicators.py`
- Create: `backend/tests/test_indicators.py`

**Step 1: Write failing test**

Create `backend/tests/test_indicators.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd backend
pytest tests/test_indicators.py -v
```

Expected: FAIL — `compute_indicators` doesn't exist

**Step 3: Implement indicators**

Create `backend/app/core/strategy/indicators.py`:

```python
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
```

**Step 4: Run tests**

```bash
cd backend
pytest tests/test_indicators.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add backend/app/core/strategy/ backend/tests/test_indicators.py
git commit -m "feat: add technical indicators (RSI, MACD, Bollinger, SMA, EMA)"
```

---

## Task 5: Strategy Interface & ML Strategy

**Files:**
- Create: `backend/app/core/strategy/base.py`
- Create: `backend/app/core/strategy/ml_strategy.py`
- Create: `backend/tests/test_strategy.py`

**Step 1: Write the abstract Strategy interface**

Create `backend/app/core/strategy/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    symbol: str
    suggested_quantity: int
    reasoning: str


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    feature_importance: dict


class Strategy(ABC):
    @abstractmethod
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Analyze data and return a trading signal."""

    @abstractmethod
    async def train(self, training_data: pd.DataFrame) -> ModelMetrics:
        """Train/retrain the strategy model on historical data."""

    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the strategy has a trained model ready."""
```

**Step 2: Write failing test for ML strategy**

Create `backend/tests/test_strategy.py`:

```python
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
```

**Step 3: Run test to verify it fails**

```bash
cd backend
pytest tests/test_strategy.py -v
```

Expected: FAIL — `MLStrategy` doesn't exist

**Step 4: Implement MLStrategy**

Create `backend/app/core/strategy/ml_strategy.py`:

```python
import asyncio
import os
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.core.strategy.base import ModelMetrics, Signal, Strategy
from app.core.strategy.indicators import compute_indicators


class MLStrategy(Strategy):
    def __init__(self, model_dir: str = "./model_artifacts", threshold: float = 0.005):
        self.model_dir = model_dir
        self.threshold = threshold  # price change threshold for labeling
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []
        os.makedirs(model_dir, exist_ok=True)

    def is_trained(self) -> bool:
        return self.model is not None

    async def train(self, training_data: pd.DataFrame) -> ModelMetrics:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._train_sync, training_data))

    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if not self.is_trained():
            return Signal(
                action="hold", confidence=0.0, symbol=symbol,
                suggested_quantity=0, reasoning="Model not trained"
            )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._analyze_sync, symbol, data))

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = compute_indicators(df)
        self.feature_columns = [
            "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
            "sma_20", "ema_12", "close", "volume",
        ]
        return enriched.dropna()

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        future_return = df["close"].shift(-1) / df["close"] - 1
        labels = pd.Series(np.where(
            future_return > self.threshold, 2,  # buy
            np.where(future_return < -self.threshold, 0, 1)  # sell / hold
        ), index=df.index)
        return labels

    def _train_sync(self, training_data: pd.DataFrame) -> ModelMetrics:
        df = self._prepare_features(training_data)
        labels = self._create_labels(df)

        # Drop last row (no future label)
        df = df.iloc[:-1]
        labels = labels.iloc[:-1]

        X = df[self.feature_columns]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))

        from sklearn.metrics import precision_score, recall_score
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        importance = dict(zip(
            self.feature_columns,
            [float(x) for x in self.model.feature_importances_],
        ))

        model_path = os.path.join(self.model_dir, "xgb_model.joblib")
        joblib.dump(self.model, model_path)

        return ModelMetrics(
            accuracy=accuracy, precision=precision,
            recall=recall, feature_importance=importance,
        )

    def _analyze_sync(self, symbol: str, data: pd.DataFrame) -> Signal:
        df = self._prepare_features(data)
        if df.empty:
            return Signal(
                action="hold", confidence=0.0, symbol=symbol,
                suggested_quantity=0, reasoning="Insufficient data"
            )

        latest = df[self.feature_columns].iloc[[-1]]
        proba = self.model.predict_proba(latest)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])

        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action = action_map[pred]

        return Signal(
            action=action,
            confidence=confidence,
            symbol=symbol,
            suggested_quantity=1,
            reasoning=f"XGBoost prediction: {action} with {confidence:.2%} confidence",
        )
```

**Step 5: Run tests**

```bash
cd backend
pytest tests/test_strategy.py -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add backend/app/core/strategy/ backend/tests/test_strategy.py
git commit -m "feat: add Strategy interface and XGBoost ML strategy"
```

---

## Task 6: Broker Interface & Simulator

**Files:**
- Create: `backend/app/core/broker/__init__.py`
- Create: `backend/app/core/broker/base.py`
- Create: `backend/app/core/broker/simulator.py`
- Create: `backend/tests/test_broker.py`

**Step 1: Write the abstract Broker interface**

Create `backend/app/core/broker/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return ((self.current_price / self.avg_price) - 1) * 100


class Broker(ABC):
    @abstractmethod
    async def buy(self, symbol: str, quantity: float, price: float) -> Order:
        """Execute a buy order."""

    @abstractmethod
    async def sell(self, symbol: str, quantity: float, price: float) -> Order:
        """Execute a sell order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""

    @abstractmethod
    async def get_balance(self) -> float:
        """Get current cash balance."""

    @abstractmethod
    async def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Get total portfolio value (cash + positions) given current prices."""
```

**Step 2: Write failing test for simulator**

Create `backend/tests/test_broker.py`:

```python
import pytest

from app.core.broker.simulator import SimulatorBroker


@pytest.fixture
def broker():
    return SimulatorBroker(initial_cash=10000.0, slippage=0.001)


@pytest.mark.asyncio
async def test_initial_balance(broker):
    balance = await broker.get_balance()
    assert balance == 10000.0


@pytest.mark.asyncio
async def test_buy_reduces_cash(broker):
    order = await broker.buy("AAPL", 10, 150.0)
    balance = await broker.get_balance()
    expected_cost = 10 * 150.0 * 1.001  # with slippage
    assert abs(balance - (10000.0 - expected_cost)) < 0.01
    assert order.side == "buy"
    assert order.symbol == "AAPL"


@pytest.mark.asyncio
async def test_buy_creates_position(broker):
    await broker.buy("AAPL", 10, 150.0)
    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 10


@pytest.mark.asyncio
async def test_sell_increases_cash(broker):
    await broker.buy("AAPL", 10, 150.0)
    cash_after_buy = await broker.get_balance()
    await broker.sell("AAPL", 10, 160.0)
    cash_after_sell = await broker.get_balance()
    assert cash_after_sell > cash_after_buy


@pytest.mark.asyncio
async def test_sell_removes_position(broker):
    await broker.buy("AAPL", 10, 150.0)
    await broker.sell("AAPL", 10, 160.0)
    positions = await broker.get_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_insufficient_funds_raises(broker):
    with pytest.raises(ValueError, match="Insufficient funds"):
        await broker.buy("AAPL", 1000, 150.0)


@pytest.mark.asyncio
async def test_insufficient_shares_raises(broker):
    with pytest.raises(ValueError, match="Insufficient shares"):
        await broker.sell("AAPL", 10, 150.0)


@pytest.mark.asyncio
async def test_portfolio_value(broker):
    await broker.buy("AAPL", 10, 150.0)
    value = await broker.get_portfolio_value({"AAPL": 160.0})
    cash = await broker.get_balance()
    assert abs(value - (cash + 10 * 160.0)) < 0.01
```

**Step 3: Run test to verify it fails**

```bash
cd backend
pytest tests/test_broker.py -v
```

Expected: FAIL — `SimulatorBroker` doesn't exist

**Step 4: Implement SimulatorBroker**

Create `backend/app/core/broker/simulator.py`:

```python
from datetime import datetime

from app.core.broker.base import Broker, Order, Position


class SimulatorBroker(Broker):
    def __init__(self, initial_cash: float, slippage: float = 0.001):
        self.cash = initial_cash
        self.slippage = slippage
        self.positions: dict[str, dict] = {}  # symbol -> {quantity, avg_price}
        self.trade_log: list[Order] = []

    async def buy(self, symbol: str, quantity: float, price: float) -> Order:
        fill_price = price * (1 + self.slippage)
        cost = quantity * fill_price

        if cost > self.cash:
            raise ValueError(f"Insufficient funds: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos["quantity"] + quantity
            pos["avg_price"] = (
                (pos["avg_price"] * pos["quantity"]) + (fill_price * quantity)
            ) / total_qty
            pos["quantity"] = total_qty
        else:
            self.positions[symbol] = {"quantity": quantity, "avg_price": fill_price}

        order = Order(
            symbol=symbol, side="buy", quantity=quantity,
            price=fill_price, timestamp=datetime.utcnow(),
        )
        self.trade_log.append(order)
        return order

    async def sell(self, symbol: str, quantity: float, price: float) -> Order:
        if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
            raise ValueError(f"Insufficient shares of {symbol}")

        fill_price = price * (1 - self.slippage)
        proceeds = quantity * fill_price
        self.cash += proceeds

        self.positions[symbol]["quantity"] -= quantity
        if self.positions[symbol]["quantity"] <= 0:
            del self.positions[symbol]

        order = Order(
            symbol=symbol, side="sell", quantity=quantity,
            price=fill_price, timestamp=datetime.utcnow(),
        )
        self.trade_log.append(order)
        return order

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=symbol,
                quantity=data["quantity"],
                avg_price=data["avg_price"],
                current_price=data["avg_price"],  # updated externally
            )
            for symbol, data in self.positions.items()
        ]

    async def get_balance(self) -> float:
        return self.cash

    async def get_portfolio_value(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            data["quantity"] * prices.get(symbol, data["avg_price"])
            for symbol, data in self.positions.items()
        )
        return self.cash + positions_value
```

**Step 5: Run tests**

```bash
cd backend
pytest tests/test_broker.py -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add backend/app/core/broker/ backend/tests/test_broker.py
git commit -m "feat: add Broker interface and paper trading simulator"
```

---

## Task 7: Trading Engine

**Files:**
- Create: `backend/app/core/engine.py`
- Create: `backend/tests/test_engine.py`

**Step 1: Write failing test**

Create `backend/tests/test_engine.py`:

```python
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

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
```

**Step 2: Run test to verify it fails**

```bash
cd backend
pytest tests/test_engine.py -v
```

Expected: FAIL — `TradingEngine` doesn't exist

**Step 3: Implement TradingEngine**

Create `backend/app/core/engine.py`:

```python
import asyncio
import logging
from datetime import datetime

from app.core.broker.base import Broker
from app.core.data.base import DataSource
from app.core.strategy.base import Strategy
from app.core.strategy.indicators import compute_indicators

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        data_source: DataSource,
        strategy: Strategy,
        broker: Broker,
        symbols: list[str],
        min_confidence: float = 0.6,
        interval: str = "1d",
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.min_confidence = min_confidence
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def tick(self):
        for symbol in self.symbols:
            try:
                data = await self.data_source.fetch_latest(symbol, self.interval)
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                enriched = compute_indicators(data)
                signal = await self.strategy.analyze(symbol, enriched)

                if signal.confidence < self.min_confidence:
                    logger.info(
                        f"{symbol}: {signal.action} skipped (confidence {signal.confidence:.2%} < {self.min_confidence:.2%})"
                    )
                    continue

                current_price = float(data["close"].iloc[-1])

                if signal.action == "buy" and signal.suggested_quantity > 0:
                    await self.broker.buy(symbol, signal.suggested_quantity, current_price)
                    logger.info(
                        f"BUY {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )
                elif signal.action == "sell" and signal.suggested_quantity > 0:
                    await self.broker.sell(symbol, signal.suggested_quantity, current_price)
                    logger.info(
                        f"SELL {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )

            except ValueError as e:
                logger.warning(f"Trade failed for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def run(self, tick_seconds: float = 60.0):
        self._running = True
        logger.info(f"Engine started: symbols={self.symbols}, interval={self.interval}")
        while self._running:
            await self.tick()
            await asyncio.sleep(tick_seconds)

    def start(self, tick_seconds: float = 60.0):
        self._task = asyncio.create_task(self.run(tick_seconds))

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Engine stopped")
```

**Step 4: Run tests**

```bash
cd backend
pytest tests/test_engine.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add backend/app/core/engine.py backend/tests/test_engine.py
git commit -m "feat: add TradingEngine orchestrator with tick-based execution"
```

---

## Task 8: Model Trainer Pipeline

**Files:**
- Create: `backend/app/core/trainer.py`
- Create: `backend/tests/test_trainer.py`

**Step 1: Write failing test**

Create `backend/tests/test_trainer.py`:

```python
import numpy as np
import pandas as pd
import pytest

from app.core.strategy.ml_strategy import MLStrategy
from app.core.trainer import Trainer


def make_training_data(n=500):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


@pytest.fixture
def strategy(tmp_path):
    return MLStrategy(model_dir=str(tmp_path))


@pytest.fixture
def trainer(strategy):
    return Trainer(strategy=strategy)


@pytest.mark.asyncio
async def test_trainer_train_returns_metrics(trainer):
    data = make_training_data()
    metrics = await trainer.train(data)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert trainer.strategy.is_trained()


@pytest.mark.asyncio
async def test_trainer_evaluate(trainer):
    data = make_training_data()
    await trainer.train(data)
    eval_metrics = await trainer.evaluate(data.tail(100))
    assert "accuracy" in eval_metrics
```

**Step 2: Run test to verify it fails**

```bash
cd backend
pytest tests/test_trainer.py -v
```

Expected: FAIL — `Trainer` doesn't exist

**Step 3: Implement Trainer**

Create `backend/app/core/trainer.py`:

```python
import logging

import pandas as pd

from app.core.strategy.base import ModelMetrics, Strategy
from app.core.strategy.indicators import compute_indicators

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    async def train(self, data: pd.DataFrame) -> ModelMetrics:
        logger.info(f"Training strategy on {len(data)} rows")
        metrics = await self.strategy.train(data)
        logger.info(
            f"Training complete: accuracy={metrics.accuracy:.4f}, "
            f"precision={metrics.precision:.4f}, recall={metrics.recall:.4f}"
        )
        return metrics

    async def evaluate(self, data: pd.DataFrame) -> dict:
        enriched = compute_indicators(data)
        enriched = enriched.dropna()

        if not self.strategy.is_trained():
            return {"error": "Strategy not trained"}

        correct = 0
        total = 0
        for i in range(len(enriched) - 1):
            window = enriched.iloc[: i + 1]
            signal = await self.strategy.analyze("EVAL", window)
            actual_return = (
                enriched["close"].iloc[i + 1] / enriched["close"].iloc[i]
            ) - 1

            if signal.action == "buy" and actual_return > 0:
                correct += 1
            elif signal.action == "sell" and actual_return < 0:
                correct += 1
            elif signal.action == "hold":
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "total_signals": total}
```

**Step 4: Run tests**

```bash
cd backend
pytest tests/test_trainer.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add backend/app/core/trainer.py backend/tests/test_trainer.py
git commit -m "feat: add model Trainer pipeline with train and evaluate"
```

---

## Task 9: API Routes

**Files:**
- Create: `backend/app/api/routes/simulation.py`
- Create: `backend/app/api/routes/portfolio.py`
- Create: `backend/app/api/routes/trades.py`
- Create: `backend/app/api/routes/strategy.py`
- Modify: `backend/app/main.py` — register routers
- Create: `backend/tests/test_api.py`

**Step 1: Write failing API tests**

Create `backend/tests/test_api.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_db():
    """Mock the database session for API tests."""
    session = AsyncMock()
    return session


@pytest.fixture
async def client():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_create_simulation(client):
    response = await client.post("/api/simulations", json={
        "name": "Test Sim",
        "initial_capital": 10000.0,
        "config": {"symbols": ["AAPL"], "interval": "1d"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Sim"
    assert data["initial_capital"] == 10000.0
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_list_simulations(client):
    # Create one first
    await client.post("/api/simulations", json={
        "name": "Test", "initial_capital": 5000.0,
    })
    response = await client.get("/api/simulations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**Step 2: Run test to verify it fails**

```bash
cd backend
pytest tests/test_api.py -v
```

Expected: FAIL — routes don't exist

**Step 3: Implement simulation routes**

Create `backend/app/api/routes/simulation.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.db import Simulation, SimulationStatus
from app.models.schemas import SimulationCreate, SimulationResponse

router = APIRouter(prefix="/simulations", tags=["simulations"])

# In-memory engine registry (per simulation)
_engines: dict[int, object] = {}


@router.post("", response_model=SimulationResponse)
async def create_simulation(
    data: SimulationCreate, db: AsyncSession = Depends(get_db)
):
    sim = Simulation(
        name=data.name,
        initial_capital=data.initial_capital,
        current_cash=data.initial_capital,
        status=SimulationStatus.CREATED,
        config=data.config,
    )
    db.add(sim)
    await db.commit()
    await db.refresh(sim)
    return sim


@router.get("", response_model=list[SimulationResponse])
async def list_simulations(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Simulation))
    return result.scalars().all()


@router.get("/{sim_id}", response_model=SimulationResponse)
async def get_simulation(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return sim


@router.post("/{sim_id}/start")
async def start_simulation(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if sim.status == SimulationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Already running")

    # Build engine from config
    from app.core.broker.simulator import SimulatorBroker
    from app.core.data.yahoo import YahooDataSource
    from app.core.engine import TradingEngine
    from app.core.strategy.ml_strategy import MLStrategy

    symbols = sim.config.get("symbols", ["AAPL"])
    interval = sim.config.get("interval", "1d")
    tick_seconds = sim.config.get("tick_seconds", 60.0)

    broker = SimulatorBroker(initial_cash=sim.current_cash)
    data_source = YahooDataSource()
    strategy = MLStrategy()

    engine = TradingEngine(
        data_source=data_source, strategy=strategy, broker=broker,
        symbols=symbols, interval=interval,
    )
    engine.start(tick_seconds=tick_seconds)

    _engines[sim_id] = {"engine": engine, "broker": broker, "strategy": strategy}

    sim.status = SimulationStatus.RUNNING
    await db.commit()

    return {"status": "started", "simulation_id": sim_id}


@router.post("/{sim_id}/stop")
async def stop_simulation(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if sim_id in _engines:
        _engines[sim_id]["engine"].stop()
        # Persist final cash
        broker = _engines[sim_id]["broker"]
        sim.current_cash = await broker.get_balance()
        del _engines[sim_id]

    sim.status = SimulationStatus.STOPPED
    await db.commit()

    return {"status": "stopped", "simulation_id": sim_id}
```

**Step 4: Implement portfolio routes**

Create `backend/app/api/routes/portfolio.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.routes.simulation import _engines
from app.models.db import PortfolioSnapshot, Simulation
from app.models.schemas import PerformancePoint, PerformanceResponse, PortfolioResponse

router = APIRouter(prefix="/simulations/{sim_id}", tags=["portfolio"])


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if sim_id in _engines:
        broker = _engines[sim_id]["broker"]
        positions = await broker.get_positions()
        cash = await broker.get_balance()
        positions_data = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_price": p.avg_price,
                "current_price": p.current_price,
                "pnl": p.pnl,
                "pnl_pct": p.pnl_pct,
            }
            for p in positions
        ]
        total_value = cash + sum(p.market_value for p in positions)
    else:
        cash = sim.current_cash
        positions_data = []
        total_value = cash

    total_pnl = total_value - sim.initial_capital
    total_pnl_pct = (total_pnl / sim.initial_capital) * 100 if sim.initial_capital else 0

    return PortfolioResponse(
        total_value=total_value,
        cash=cash,
        positions=positions_data,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
    )


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    result = await db.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.simulation_id == sim_id)
        .order_by(PortfolioSnapshot.timestamp)
    )
    snapshots = result.scalars().all()

    points = [
        PerformancePoint(timestamp=s.timestamp, total_value=s.total_value)
        for s in snapshots
    ]

    if len(points) >= 2:
        daily_pnl = points[-1].total_value - points[-2].total_value
    else:
        daily_pnl = 0.0

    current_value = points[-1].total_value if points else sim.initial_capital
    total_pnl = current_value - sim.initial_capital
    total_pnl_pct = (total_pnl / sim.initial_capital) * 100 if sim.initial_capital else 0

    return PerformanceResponse(
        points=points, daily_pnl=daily_pnl,
        total_pnl=total_pnl, total_pnl_pct=total_pnl_pct,
    )
```

**Step 5: Implement trades routes**

Create `backend/app/api/routes/trades.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.db import Simulation, Trade
from app.models.schemas import TradeResponse

router = APIRouter(prefix="/simulations/{sim_id}", tags=["trades"])


@router.get("/trades", response_model=list[TradeResponse])
async def get_trades(
    sim_id: int, limit: int = 50, db: AsyncSession = Depends(get_db)
):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    result = await db.execute(
        select(Trade)
        .where(Trade.simulation_id == sim_id)
        .order_by(desc(Trade.timestamp))
        .limit(limit)
    )
    return result.scalars().all()
```

**Step 6: Implement strategy routes**

Create `backend/app/api/routes/strategy.py`:

```python
from fastapi import APIRouter, HTTPException

from app.api.routes.simulation import _engines
from app.models.schemas import StrategyStatusResponse

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.get("/{sim_id}", response_model=StrategyStatusResponse)
async def get_strategy_status(sim_id: int):
    if sim_id not in _engines:
        return StrategyStatusResponse(
            model_name="xgboost",
            model_version=None,
            last_trained=None,
            metrics={},
            active_signals=[],
        )

    strategy = _engines[sim_id]["strategy"]
    return StrategyStatusResponse(
        model_name="xgboost",
        model_version=1 if strategy.is_trained() else None,
        last_trained=None,
        metrics={},
        active_signals=[],
    )


@router.post("/train")
async def trigger_training(sim_id: int):
    if sim_id not in _engines:
        raise HTTPException(status_code=404, detail="No active engine for simulation")

    from app.core.data.yahoo import YahooDataSource
    from app.core.trainer import Trainer
    from datetime import datetime, timedelta

    strategy = _engines[sim_id]["strategy"]
    trainer = Trainer(strategy=strategy)

    engine = _engines[sim_id]["engine"]
    data_source = YahooDataSource()

    # Fetch training data for each symbol
    results = {}
    for symbol in engine.symbols:
        end = datetime.now()
        start = end - timedelta(days=365)
        data = await data_source.fetch_historical(symbol, start, end, engine.interval)
        metrics = await trainer.train(data)
        results[symbol] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }

    return {"status": "trained", "results": results}
```

**Step 7: Register routers in main.py**

Update `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import portfolio, simulation, strategy, trades
from app.config import settings

app = FastAPI(title="Trader Bot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router, prefix=settings.api_prefix)
app.include_router(portfolio.router, prefix=settings.api_prefix)
app.include_router(trades.router, prefix=settings.api_prefix)
app.include_router(strategy.router, prefix=settings.api_prefix)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

**Step 8: Run API tests**

```bash
cd backend
pytest tests/test_api.py -v
```

Note: These tests require a running PostgreSQL. For CI/testing without DB, the tests should be adapted to use a test database or SQLite. For now, skip if no DB is available and test manually.

**Step 9: Commit**

```bash
git add backend/app/api/ backend/app/main.py backend/tests/test_api.py
git commit -m "feat: add API routes for simulations, portfolio, trades, and strategy"
```

---

## Task 10: Frontend Setup & Dashboard

**Files:**
- Create: `frontend/` — scaffolded via Vite
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/usePolling.ts`
- Create: `frontend/src/components/Dashboard.tsx`
- Create: `frontend/src/components/Portfolio.tsx`
- Create: `frontend/src/components/ProfitLoss.tsx`
- Create: `frontend/src/components/TradeHistory.tsx`
- Create: `frontend/src/components/SimulationControl.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Scaffold React project**

```bash
cd /path/to/trader
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install recharts axios tailwindcss @tailwindcss/vite
```

**Step 2: Configure Tailwind**

Update `frontend/vite.config.ts`:

```typescript
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

Add to top of `frontend/src/index.css`:

```css
@import "tailwindcss";
```

**Step 3: Create API client**

Create `frontend/src/api/client.ts`:

```typescript
import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export interface Simulation {
  id: number;
  name: string;
  initial_capital: number;
  current_cash: number;
  start_time: string;
  status: string;
  config: Record<string, unknown>;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
}

export interface Portfolio {
  total_value: number;
  cash: number;
  positions: Position[];
  total_pnl: number;
  total_pnl_pct: number;
}

export interface Trade {
  id: number;
  simulation_id: number;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  strategy: string | null;
}

export interface PerformancePoint {
  timestamp: string;
  total_value: number;
}

export interface Performance {
  points: PerformancePoint[];
  daily_pnl: number;
  total_pnl: number;
  total_pnl_pct: number;
}

export const getSimulations = () => api.get<Simulation[]>("/simulations");
export const createSimulation = (data: {
  name: string;
  initial_capital: number;
  config?: Record<string, unknown>;
}) => api.post<Simulation>("/simulations", data);
export const startSimulation = (id: number) =>
  api.post(`/simulations/${id}/start`);
export const stopSimulation = (id: number) =>
  api.post(`/simulations/${id}/stop`);
export const getPortfolio = (id: number) =>
  api.get<Portfolio>(`/simulations/${id}/portfolio`);
export const getTrades = (id: number) =>
  api.get<Trade[]>(`/simulations/${id}/trades`);
export const getPerformance = (id: number) =>
  api.get<Performance>(`/simulations/${id}/performance`);
```

**Step 4: Create polling hook**

Create `frontend/src/hooks/usePolling.ts`:

```typescript
import { useEffect, useRef, useState } from "react";

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 10000,
  enabled: boolean = true
): { data: T | null; loading: boolean; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const result = await fetcherRef.current();
        if (!cancelled) {
          setData(result);
          setLoading(false);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
          setLoading(false);
        }
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs, enabled]);

  return { data, loading, error };
}
```

**Step 5: Create SimulationControl component**

Create `frontend/src/components/SimulationControl.tsx`:

```tsx
import { useState } from "react";
import {
  createSimulation,
  getSimulations,
  startSimulation,
  stopSimulation,
  type Simulation,
} from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  selected: Simulation | null;
  onSelect: (sim: Simulation) => void;
}

export function SimulationControl({ selected, onSelect }: Props) {
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [capital, setCapital] = useState("10000");
  const [symbols, setSymbols] = useState("AAPL");

  const { data: simulations } = usePolling(
    async () => (await getSimulations()).data,
    5000
  );

  const handleCreate = async () => {
    const sim = await createSimulation({
      name,
      initial_capital: parseFloat(capital),
      config: {
        symbols: symbols.split(",").map((s) => s.trim()),
        interval: "1d",
      },
    });
    onSelect(sim.data);
    setShowCreate(false);
    setName("");
  };

  const handleStart = async () => {
    if (selected) {
      await startSimulation(selected.id);
    }
  };

  const handleStop = async () => {
    if (selected) {
      await stopSimulation(selected.id);
    }
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg flex items-center gap-4 flex-wrap">
      <select
        className="bg-gray-700 text-white px-3 py-2 rounded"
        value={selected?.id ?? ""}
        onChange={(e) => {
          const sim = simulations?.find((s) => s.id === Number(e.target.value));
          if (sim) onSelect(sim);
        }}
      >
        <option value="">Select simulation...</option>
        {simulations?.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} (${s.initial_capital.toLocaleString()})
          </option>
        ))}
      </select>

      {selected && (
        <span
          className={`px-2 py-1 rounded text-sm ${
            selected.status === "running"
              ? "bg-green-600"
              : selected.status === "stopped"
                ? "bg-red-600"
                : "bg-yellow-600"
          }`}
        >
          {selected.status}
        </span>
      )}

      {selected && selected.status !== "running" && (
        <button
          onClick={handleStart}
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
        >
          Start
        </button>
      )}
      {selected && selected.status === "running" && (
        <button
          onClick={handleStop}
          className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded"
        >
          Stop
        </button>
      )}

      <button
        onClick={() => setShowCreate(!showCreate)}
        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded ml-auto"
      >
        New Simulation
      </button>

      {showCreate && (
        <div className="w-full flex gap-2 mt-2">
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded flex-1"
          />
          <input
            placeholder="Capital"
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded w-32"
          />
          <input
            placeholder="Symbols (comma sep)"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded flex-1"
          />
          <button
            onClick={handleCreate}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
          >
            Create
          </button>
        </div>
      )}
    </div>
  );
}
```

**Step 6: Create Portfolio component**

Create `frontend/src/components/Portfolio.tsx`:

```tsx
import { getPortfolio, type Portfolio as PortfolioType } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function Portfolio({ simulationId }: Props) {
  const { data: portfolio, loading } = usePolling(
    async () =>
      simulationId ? (await getPortfolio(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return <div className="bg-gray-800 p-4 rounded-lg">Select a simulation</div>;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading...</div>;
  if (!portfolio) return null;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Portfolio</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-gray-400 text-sm">Total Value</div>
          <div className="text-xl font-bold">${portfolio.total_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        </div>
        <div>
          <div className="text-gray-400 text-sm">Cash</div>
          <div className="text-xl">${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        </div>
        <div>
          <div className="text-gray-400 text-sm">Total P/L</div>
          <div className={`text-xl font-bold ${portfolio.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
            ${portfolio.total_pnl.toFixed(2)} ({portfolio.total_pnl_pct.toFixed(2)}%)
          </div>
        </div>
      </div>

      {portfolio.positions.length > 0 && (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Symbol</th>
              <th className="text-right">Qty</th>
              <th className="text-right">Avg Price</th>
              <th className="text-right">Current</th>
              <th className="text-right">P/L</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.positions.map((p) => (
              <tr key={p.symbol} className="border-b border-gray-700">
                <td className="py-2 font-medium">{p.symbol}</td>
                <td className="text-right">{p.quantity}</td>
                <td className="text-right">${p.avg_price.toFixed(2)}</td>
                <td className="text-right">${p.current_price.toFixed(2)}</td>
                <td className={`text-right ${p.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                  ${p.pnl.toFixed(2)} ({p.pnl_pct.toFixed(1)}%)
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
```

**Step 7: Create ProfitLoss chart component**

Create `frontend/src/components/ProfitLoss.tsx`:

```tsx
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getPerformance } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function ProfitLoss({ simulationId }: Props) {
  const { data: performance, loading } = usePolling(
    async () =>
      simulationId ? (await getPerformance(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return null;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading chart...</div>;
  if (!performance || performance.points.length === 0) {
    return <div className="bg-gray-800 p-4 rounded-lg">No performance data yet</div>;
  }

  const chartData = performance.points.map((p) => ({
    time: new Date(p.timestamp).toLocaleDateString(),
    value: p.total_value,
  }));

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Performance</h2>
      <div className="flex gap-4 mb-4 text-sm">
        <div>
          <span className="text-gray-400">Daily P/L: </span>
          <span className={performance.daily_pnl >= 0 ? "text-green-400" : "text-red-400"}>
            ${performance.daily_pnl.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-400">Total P/L: </span>
          <span className={performance.total_pnl >= 0 ? "text-green-400" : "text-red-400"}>
            ${performance.total_pnl.toFixed(2)} ({performance.total_pnl_pct.toFixed(2)}%)
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9CA3AF" />
          <YAxis stroke="#9CA3AF" />
          <Tooltip
            contentStyle={{ backgroundColor: "#1F2937", border: "none" }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#10B981"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

**Step 8: Create TradeHistory component**

Create `frontend/src/components/TradeHistory.tsx`:

```tsx
import { getTrades } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function TradeHistory({ simulationId }: Props) {
  const { data: trades, loading } = usePolling(
    async () =>
      simulationId ? (await getTrades(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return null;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading trades...</div>;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
      {!trades || trades.length === 0 ? (
        <p className="text-gray-400">No trades yet</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Time</th>
              <th className="text-left">Symbol</th>
              <th className="text-left">Side</th>
              <th className="text-right">Qty</th>
              <th className="text-right">Price</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id} className="border-b border-gray-700">
                <td className="py-2">{new Date(t.timestamp).toLocaleString()}</td>
                <td>{t.symbol}</td>
                <td className={t.side === "buy" ? "text-green-400" : "text-red-400"}>
                  {t.side.toUpperCase()}
                </td>
                <td className="text-right">{t.quantity}</td>
                <td className="text-right">${t.price.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
```

**Step 9: Create Dashboard layout**

Create `frontend/src/components/Dashboard.tsx`:

```tsx
import { useState } from "react";
import type { Simulation } from "../api/client";
import { Portfolio } from "./Portfolio";
import { ProfitLoss } from "./ProfitLoss";
import { SimulationControl } from "./SimulationControl";
import { TradeHistory } from "./TradeHistory";

export function Dashboard() {
  const [selected, setSelected] = useState<Simulation | null>(null);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-2xl font-bold mb-6">Trader Bot</h1>

      <div className="space-y-4">
        <SimulationControl selected={selected} onSelect={setSelected} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Portfolio simulationId={selected?.id ?? null} />
          <ProfitLoss simulationId={selected?.id ?? null} />
        </div>

        <TradeHistory simulationId={selected?.id ?? null} />
      </div>
    </div>
  );
}
```

**Step 10: Update App.tsx**

Replace `frontend/src/App.tsx`:

```tsx
import { Dashboard } from "./components/Dashboard";

function App() {
  return <Dashboard />;
}

export default App;
```

**Step 11: Verify frontend builds**

```bash
cd frontend
npm run build
```

Expected: Build succeeds with no errors

**Step 12: Commit**

```bash
git add frontend/
git commit -m "feat: add React dashboard with portfolio, P/L chart, trades, and simulation controls"
```

---

## Task 11: Docker Compose & Deployment Config

**Files:**
- Create: `docker-compose.yml`
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Create: `.env.example`

**Step 1: Create backend Dockerfile**

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create frontend Dockerfile + nginx config**

Create `frontend/nginx.conf`:

```nginx
server {
    listen 80;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Create `frontend/Dockerfile`:

```dockerfile
FROM node:20-slim AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

**Step 3: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD:-trader}
      POSTGRES_DB: trader
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql+asyncpg://trader:${DB_PASSWORD:-trader}@db:5432/trader
      DATABASE_URL_SYNC: postgresql+psycopg2://trader:${DB_PASSWORD:-trader}@db:5432/trader
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  pgdata:
```

**Step 4: Create .env.example**

Create `.env.example`:

```
DB_PASSWORD=change_me_in_production
```

**Step 5: Create .gitignore**

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
venv/
.env
model_artifacts/
*.joblib

# Node
node_modules/
dist/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
```

**Step 6: Test Docker Compose builds**

```bash
docker compose build
```

Expected: All three services build successfully

**Step 7: Test Docker Compose starts**

```bash
docker compose up -d
docker compose ps
```

Expected: All three services running and healthy

**Step 8: Run Alembic migration inside container**

```bash
docker compose exec backend alembic upgrade head
```

Expected: Migration applied successfully

**Step 9: Verify health endpoint**

```bash
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok"}`

**Step 10: Commit**

```bash
git add docker-compose.yml backend/Dockerfile frontend/Dockerfile frontend/nginx.conf .env.example .gitignore
git commit -m "feat: add Docker Compose deployment with PostgreSQL, backend, and frontend"
```

---

## Summary

| Task | Description | Estimated steps |
|------|-------------|-----------------|
| 1 | Backend scaffolding | 9 |
| 2 | Database models & migrations | 7 |
| 3 | Data source interface + Yahoo | 6 |
| 4 | Technical indicators | 5 |
| 5 | Strategy interface + ML | 6 |
| 6 | Broker interface + simulator | 6 |
| 7 | Trading engine | 5 |
| 8 | Model trainer | 5 |
| 9 | API routes | 9 |
| 10 | Frontend dashboard | 12 |
| 11 | Docker Compose deployment | 10 |
| **Total** | | **80 steps** |

Dependencies: Tasks 1-2 must come first. Tasks 3-8 can be done in any order after Task 2. Task 9 depends on Tasks 2-8. Task 10 depends on Task 9. Task 11 can be done after Tasks 9-10.
