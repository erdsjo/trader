# Trading Bot Design

## Overview

A self-learning stock trading bot with simulation mode, a React dashboard, and a pluggable architecture for data sources, strategies, and brokers.

## Decisions

- **Stack**: Python (FastAPI) backend + React (TypeScript) frontend
- **Architecture**: Monolithic FastAPI — single backend serving API and orchestrating the trading engine
- **Data source**: Yahoo Finance (yfinance) first, pluggable interface for adding more later
- **ML approach**: Phase 1 = technical indicators + supervised ML (XGBoost). Phase 2 = reinforcement learning
- **Timeframes**: Configurable — minutes (day trading) and daily (swing trading)
- **Dashboard updates**: Polling (auto-refresh every 10s)
- **Execution**: Simulation/paper trading first, pluggable broker interface for real trading later
- **Storage**: PostgreSQL
- **Infrastructure**: Hetzner CPX42 VPS (8 vCPU AMD EPYC, 16 GB RAM, 320 GB SSD, Ubuntu 24.04)
- **Deployment**: Docker Compose (PostgreSQL + backend + frontend served via Nginx)

## Project Structure

```
trader/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── config.py                # Settings
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── portfolio.py     # Portfolio & P/L endpoints
│   │   │   │   ├── trades.py        # Trade history endpoints
│   │   │   │   ├── simulation.py    # Simulation management
│   │   │   │   └── strategy.py      # Strategy config & model status
│   │   │   └── deps.py              # Shared dependencies
│   │   ├── core/
│   │   │   ├── data/
│   │   │   │   ├── base.py          # Abstract DataSource interface
│   │   │   │   └── yahoo.py         # Yahoo Finance implementation
│   │   │   ├── strategy/
│   │   │   │   ├── base.py          # Abstract Strategy interface
│   │   │   │   ├── indicators.py    # Technical indicators
│   │   │   │   └── ml_strategy.py   # ML-based strategy
│   │   │   ├── broker/
│   │   │   │   ├── base.py          # Abstract Broker interface
│   │   │   │   └── simulator.py     # Paper trading broker
│   │   │   ├── engine.py            # Trading engine orchestrator
│   │   │   └── trainer.py           # Model training pipeline
│   │   └── models/
│   │       ├── db.py                # SQLAlchemy models
│   │       └── schemas.py           # Pydantic schemas
│   ├── alembic/                     # DB migrations
│   ├── tests/
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx        # Main dashboard layout
│   │   │   ├── Portfolio.tsx        # Holdings & allocation
│   │   │   ├── ProfitLoss.tsx       # P/L chart & summary
│   │   │   ├── TradeHistory.tsx     # Recent trades table
│   │   │   └── SimulationControl.tsx
│   │   ├── hooks/
│   │   │   └── usePolling.ts        # Auto-refresh hook
│   │   ├── api/
│   │   │   └── client.ts            # API client
│   │   └── App.tsx
│   ├── package.json
│   └── tsconfig.json
└── docs/
    └── plans/
```

## Data Layer

### Abstract interface

```python
class DataSource(ABC):
    async def fetch_historical(self, symbol, start, end, interval) -> pd.DataFrame
    async def fetch_latest(self, symbol, interval) -> pd.DataFrame
    def supported_intervals(self) -> list[str]
```

Yahoo Finance is the first implementation. New sources are added by implementing `DataSource`.

### Database tables

- **price_data** — symbol, timestamp, open, high, low, close, volume, interval, source
- **trades** — id, symbol, side (buy/sell), quantity, price, timestamp, strategy, simulation_id
- **portfolio_snapshots** — timestamp, total_value, cash, simulation_id
- **simulations** — id, name, initial_capital, start_time, status, config (JSON)
- **models** — id, strategy_name, version, metrics (JSON), trained_at, file_path

### Data flow

1. Engine requests data via DataSource interface
2. Raw OHLCV data stored in PostgreSQL
3. Technical indicators computed on the fly from stored data
4. Strategy receives enriched data (prices + indicators)

## Strategy Engine & ML Pipeline

### Phase 1: Technical Indicators + Supervised ML

**Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, volume-weighted metrics.

**ML pipeline**:
1. Feature engineering: indicators + price patterns -> feature matrix
2. Labeling: future price movement (up/down/hold) based on configurable thresholds
3. Model: XGBoost (fast, interpretable)
4. Prediction: signal (buy/sell/hold) with confidence score
5. Retraining: manual or scheduled, with validation comparison

**Strategy interface**:

```python
class Strategy(ABC):
    async def analyze(self, symbol, data: pd.DataFrame) -> Signal
    async def train(self, training_data: pd.DataFrame) -> ModelMetrics
```

Signal contains: action, confidence (0-1), suggested quantity, reasoning.

### Phase 2 (future): Reinforcement Learning

- RL agent via Stable-Baselines3
- State: portfolio + indicators. Action: buy/sell/hold. Reward: realized P/L.
- Same Strategy interface — engine unchanged.

## Simulation & Broker

### Broker interface

```python
class Broker(ABC):
    async def buy(self, symbol, quantity, order_type) -> Order
    async def sell(self, symbol, quantity, order_type) -> Order
    async def get_positions(self) -> list[Position]
    async def get_balance(self) -> float
```

### Simulator

- Implements Broker with in-memory + DB-backed virtual portfolio
- Configurable starting capital
- Simulates order fills at market price (with configurable slippage)
- All trades linked to a simulation_id

### Simulation modes

1. **Live simulation**: real-time data, virtual money
2. **Backtesting**: historical data, fast execution

### Engine loop

```
loop (configurable interval):
  1. Fetch latest data via DataSource
  2. Compute indicators
  3. Strategy.analyze() -> Signal
  4. If signal confidence sufficient -> Broker.buy/sell()
  5. Log trade, update portfolio snapshot
  6. Periodically trigger retraining if enabled
```

## Dashboard

### Layout

**Top bar**: Simulation selector + status + controls (start/stop/new)

**Panels**:
- Portfolio Summary: total value, cash, total P/L ($, %), daily P/L
- Holdings Table: symbol, qty, avg buy price, current price, P/L, % of portfolio
- P/L Chart: line chart of portfolio value over time
- Recent Trades: time, symbol, side, qty, price, P/L
- Strategy Status: model version, last trained, accuracy, active signals

### API Endpoints

- `GET /api/simulations` — list simulations
- `POST /api/simulations` — create new simulation
- `POST /api/simulations/{id}/start` — start trading loop
- `POST /api/simulations/{id}/stop` — stop trading loop
- `GET /api/simulations/{id}/portfolio` — current holdings + value
- `GET /api/simulations/{id}/trades` — trade history
- `GET /api/simulations/{id}/performance` — P/L time series
- `GET /api/simulations/{id}/strategy` — model status + signals
- `POST /api/strategy/train` — trigger retraining

### Polling

Dashboard polls portfolio, trades, and performance endpoints every 10 seconds.

## Key Python Dependencies

- fastapi, uvicorn — API server
- sqlalchemy, alembic — ORM + migrations
- asyncpg — async PostgreSQL driver
- yfinance — Yahoo Finance data
- pandas, numpy — data processing
- scikit-learn, xgboost — ML
- ta (technical analysis library) — indicators
- pydantic — data validation
- pytest — testing

## Key Frontend Dependencies

- react, react-dom — UI
- typescript — type safety
- recharts or chart.js — charting
- axios or fetch — API calls
- tailwindcss — styling

## Infrastructure — Hetzner CPX42

### Server specs

- 8 vCPU (AMD EPYC)
- 16 GB RAM
- 320 GB SSD
- Ubuntu 24.04 LTS
- Region: Nuremberg or Falkenstein (DE) for low latency to EU markets

### Deployment (Docker Compose)

```yaml
services:
  db:        # PostgreSQL 16
  backend:   # FastAPI + Uvicorn
  frontend:  # Nginx serving React build + reverse proxy to backend
```

### Resource budget on CPX42

| Component | CPU | RAM |
|-----------|-----|-----|
| PostgreSQL | 1-2 cores | 4 GB |
| FastAPI + trading engine | 2-3 cores | 3 GB |
| XGBoost training (burst) | 4-6 cores | 4 GB |
| Nginx + React static | negligible | 128 MB |
| OS + overhead | 1 core | 1 GB |
| **Headroom** | ~2 cores idle | ~4 GB free |

### Phase 2 note

For RL training, consider upgrading to CPX52 (16 vCPU, 32 GB RAM) or running training jobs on a separate dedicated GPU server while the bot continues trading on the CPX42.
