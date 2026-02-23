# Multi-Stock Screening & Sector-Based Trading Design

**Date**: 2026-02-23
**Status**: Approved

## Overview

Scale the trading application from single-stock simulations to dynamic multi-stock trading across the S&P 500 universe. Stocks are screened daily by volume/volatility, ranked by ML-predicted opportunity, and traded using sector-specific XGBoost models.

## Requirements

- **Universe**: S&P 500 (~500 stocks, 11 GICS sectors)
- **Screening**: Volume/volatility filter followed by ML-driven ranking
- **Models**: One XGBoost model per GICS sector (11 models)
- **Rebalance**: Daily re-screening to rotate active stocks
- **Trading**: Short-term, both long and short positions
- **Backward compatibility**: Existing simulations with fixed symbol lists continue to work

## Architecture: Batch Pipeline (Approach A)

A daily `StockScreener` pipeline runs as a scheduled job:

1. Fetch/cache data for all S&P 500 stocks (batched via yfinance)
2. Filter by volume + volatility thresholds
3. Train/update 11 sector-specific XGBoost models
4. Rank candidates by predicted opportunity score
5. Publish top-N stocks per sector to the simulation engine

## Section 1: Stock Universe & Sector Management

**New component**: `StockUniverse` (`backend/app/core/data/universe.py`)

Maintains the S&P 500 constituent list with GICS sector mappings.

**Data source**: Static CSV file (`backend/data/sp500.csv`) containing `symbol, name, sector`. Sourced from Wikipedia's S&P 500 list. Avoids runtime API dependencies.

**New DB table** `stock_universe`:

| Column     | Type      | Notes                         |
|------------|-----------|-------------------------------|
| symbol     | VARCHAR   | PK (e.g., "AAPL")            |
| name       | VARCHAR   | (e.g., "Apple Inc.")          |
| sector     | VARCHAR   | (e.g., "Information Technology") |
| market_cap | FLOAT     | Optional, for filtering       |
| updated_at | TIMESTAMP |                               |

**11 GICS sectors**: Information Technology, Health Care, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials.

**Update mechanism**: Manual refresh via CLI command or API endpoint (`POST /api/universe/refresh`). S&P 500 composition changes ~20-30 times/year.

## Section 2: Stock Screener Pipeline

**New component**: `StockScreener` (`backend/app/core/screener.py`)

Runs daily. Three-stage pipeline:

### Stage 1 -- Data Fetch (batch)

- Pull last 60 days of daily OHLCV for all ~500 stocks
- Use existing `YahooDataSource` + DB price cache
- Batch in groups of 50 with 1-2s delays for Yahoo rate limits
- Estimated time: ~5-10 minutes for full universe (mostly cached after first run)

### Stage 2 -- Volume/Volatility Filter

Calculate over the last 20 trading days:

- **Average daily volume**: must exceed threshold (default >1M shares/day)
- **Volatility**: annualized std dev of daily returns (default >15%)

Stocks passing both filters become candidates (~100-200 stocks). Thresholds configurable in simulation config.

### Stage 3 -- ML Ranking

- Compute full feature set for each candidate (RSI, MACD, Bollinger, SMA, EMA)
- Run through sector model to get predicted opportunity score
- Rank by score within each sector
- Select top-N stocks per sector for active trading (default N=5)

**New DB table** `screening_results`:

| Column            | Type      | Notes                    |
|-------------------|-----------|--------------------------|
| id                | SERIAL    | PK                       |
| simulation_id     | INT       | FK -> simulations        |
| symbol            | VARCHAR   |                          |
| sector            | VARCHAR   |                          |
| volume_avg        | FLOAT     |                          |
| volatility        | FLOAT     |                          |
| opportunity_score | FLOAT     |                          |
| selected          | BOOLEAN   | Top-N flag               |
| screened_at       | TIMESTAMP |                          |

**Scheduling**: Async background task runs daily at configurable time (default 06:00 UTC, before US market open).

## Section 3: Sector-Aware ML Strategy

**Modified component**: `MLStrategy` (`backend/app/core/strategy/ml_strategy.py`)

Currently: one global XGBoost model trained on all data concatenated. New: 11 sector-specific models.

**New DB table** `sector_models`:

| Column        | Type      | Notes                                  |
|---------------|-----------|----------------------------------------|
| id            | SERIAL    | PK                                     |
| simulation_id | INT       | FK -> simulations                      |
| sector        | VARCHAR   | (e.g., "Information Technology")       |
| version       | INT       |                                        |
| metrics       | JSON      | {accuracy, precision, recall, feature_importance} |
| trained_at    | TIMESTAMP |                                        |
| file_path     | VARCHAR   | Path to .joblib file                   |

### Training pipeline

1. Group training data by sector (using `stock_universe` table)
2. For each sector with enough candidates (minimum 5 stocks):
   - Gather historical data for all stocks in that sector
   - Add `symbol_encoded` feature (label-encoded within sector)
   - Add cross-stock features: sector average return, sector RSI mean
   - Train dedicated XGBoost model
   - Save model + metrics to DB/disk
3. Sectors with too few candidates fall back to a universal model

### Prediction flow

```
analyze(symbol, data):
  1. Look up symbol's sector
  2. Load the sector model
  3. Compute features (existing + new cross-stock features)
  4. Return signal with confidence
```

### New features (on top of existing 9)

- `sector_avg_return` -- mean return of all stocks in sector that day
- `sector_rsi_mean` -- average RSI across sector peers
- `relative_volume` -- stock's volume / sector average volume

Total features: 12 (up from 9).

## Section 4: Engine & Simulation Orchestration

**Modified component**: `TradingEngine` (`backend/app/core/engine.py`)

### New flow

```
Simulation start:
  1. StockScreener runs full pipeline (fetch -> filter -> rank)
  2. Engine receives selected stocks grouped by sector
  3. Load/train sector models for relevant sectors
  4. Begin tick loop with selected stocks

Daily (at configurable time):
  1. StockScreener re-runs
  2. Compare new selected stocks vs current active stocks
  3. New stocks added -> start tracking
  4. Removed stocks -> close open positions, stop tracking
  5. Sector models retrained if new data available

Per tick (unchanged concept, bigger scope):
  For each active stock:
    fetch latest -> compute indicators -> analyze -> trade if confident
```

### Simulation config changes

```json
{
    "universe": "sp500",
    "interval": "1d",
    "tick_seconds": 60.0,
    "screening": {
        "min_volume": 1000000,
        "min_volatility": 0.15,
        "top_n_per_sector": 5,
        "rescreen_hour_utc": 6
    },
    "max_positions": 20,
    "position_size_pct": 5.0
}
```

- `universe`: replaces `symbols` for screener-based simulations
- `screening`: screener configuration block
- `max_positions`: cap total open positions (default 20)
- `position_size_pct`: % of capital per position (default 5%)

**Backward compatibility**: If `symbols` is provided in config (old-style), skip screener and use fixed list.

## Section 5: Data Layer & Performance

### Batch fetching

- New method: `YahooDataSource.fetch_historical_batch(symbols, start, end, interval)`
- Uses `yfinance.download()` with multiple tickers (up to 50 per call)
- Falls back to individual fetches on failure
- Rate limiting: max 5 batch requests per minute

### DB cache improvements

- Add `fetched_at` column to `price_data` table
- Screening considers data "fresh" if `fetched_at` < 24 hours ago
- Fresh data skips Yahoo fetch entirely

### Estimated load

- Daily: ~500 new rows (1 bar per stock)
- Initial fetch: ~30,000 rows (60 days x 500 stocks) -- one-time
- 11 sector models in memory: ~50MB total
- Price data loaded per-sector during training, not all at once

## Section 6: Frontend Changes

Minimal UI changes to surface multi-stock behavior.

### SimulationControl.tsx -- Creation form

- Replace "Symbols" input with "Universe" dropdown (S&P 500)
- Add screening config fields: min volume, min volatility, top N per sector
- Keep "Custom symbols" option for manual stock lists (backward compat)

### Dashboard.tsx -- New "Screener" panel

- Shows latest screening results grouped by sector
- Per stock: symbol, sector, volume, volatility, opportunity score
- Visual indicator for newly added / removed stocks
- 10s polling cycle

### ProfitLoss.tsx -- Minor enhancement

- Add per-sector P/L breakdown (group trades by sector)
- Keep existing total portfolio chart

### No changes needed

- **Portfolio.tsx**: already shows positions by symbol
- **EngineActivity.tsx**: already shows timestamped events
