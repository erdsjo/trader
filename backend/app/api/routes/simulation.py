import logging
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import async_session, get_db
from app.core.event_log import EventLog
from app.core.data.universe import StockUniverseLoader
from app.core.screener import StockScreener
from app.models.db import (
    PortfolioSnapshot,
    PriceData,
    ScreeningResult,
    SectorModel,
    Simulation,
    SimulationStatus,
    Trade,
    TradeSide,
)
from app.models.schemas import SimulationCreate, SimulationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulations", tags=["simulations"])

# In-memory engine registry (per simulation)
_engines: dict[int, dict] = {}


async def _load_cached_data(
    symbol: str, interval: str, start: datetime, end: datetime,
) -> pd.DataFrame:
    """Load price data from the DB cache."""
    async with async_session() as session:
        result = await session.execute(
            select(PriceData)
            .where(
                PriceData.symbol == symbol,
                PriceData.interval == interval,
                PriceData.timestamp >= start,
                PriceData.timestamp <= end,
            )
            .order_by(PriceData.timestamp)
        )
        rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]
    )


async def _store_price_data(
    df: pd.DataFrame, symbol: str, interval: str, source: str,
) -> int:
    """Store price rows in DB, skipping duplicates. Returns count of new rows."""
    if df.empty:
        return 0
    stored = 0
    async with async_session() as session:
        for _, row in df.iterrows():
            ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
            # Use merge-style upsert: try insert, skip on conflict
            existing = await session.execute(
                select(PriceData.id).where(
                    PriceData.symbol == symbol,
                    PriceData.timestamp == ts,
                    PriceData.interval == interval,
                )
            )
            if existing.scalar() is not None:
                continue
            session.add(
                PriceData(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    interval=interval,
                    source=source,
                )
            )
            stored += 1
        await session.commit()
    return stored


# Yahoo imposes max lookback per interval
_INTERVAL_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": 365 * 10, "5d": 365 * 10, "1wk": 365 * 10, "1mo": 365 * 10,
}


async def _get_training_data(
    symbol: str,
    interval: str,
    data_source,
    event_log: EventLog,
) -> pd.DataFrame:
    """Get training data for a symbol: use DB cache, backfill from Yahoo if needed."""
    max_days = _INTERVAL_MAX_DAYS.get(interval, 365)
    end = datetime.now()
    start = end - timedelta(days=max_days)

    cached = await _load_cached_data(symbol, interval, start, end)
    if len(cached) >= 50:
        event_log.info(f"{symbol}: loaded {len(cached)} cached rows from DB")
        return cached

    # Not enough cached data — fetch from Yahoo
    if not cached.empty:
        event_log.info(
            f"{symbol}: only {len(cached)} cached rows, fetching more from Yahoo"
        )
    else:
        event_log.info(f"{symbol}: no cached data, fetching from Yahoo")

    try:
        fresh = await data_source.fetch_historical(symbol, start, end, interval)
    except Exception as e:
        event_log.error(f"{symbol}: Yahoo fetch error: {e}")
        fresh = pd.DataFrame()

    if not fresh.empty:
        stored = await _store_price_data(fresh, symbol, interval, "yahoo")
        event_log.info(f"{symbol}: fetched {len(fresh)} rows, stored {stored} new in DB cache")
        return fresh

    event_log.warning(f"{symbol}: Yahoo returned no data (interval={interval}, lookback={max_days}d)")

    # Yahoo failed but we have some cached data
    if not cached.empty:
        event_log.warning(
            f"{symbol}: Yahoo unavailable, using {len(cached)} cached rows"
        )
        return cached

    event_log.warning(f"{symbol}: no data available")
    return pd.DataFrame()


async def _run_screening(
    config: dict,
    data_source,
    event_log: EventLog,
    simulation_id: int,
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    """Run stock screening: load universe, fetch data, filter, select top-N.

    Returns:
        (active_symbols_map {symbol: sector}, candidate_price_data {symbol: DataFrame})
    """
    screening_cfg = config.get("screening", {})
    min_volume = screening_cfg.get("min_volume", 1_000_000)
    min_volatility = screening_cfg.get("min_volatility", 0.15)
    top_n = screening_cfg.get("top_n_per_sector", 5)

    loader = StockUniverseLoader()
    records = loader.load_from_csv()
    universe_map = {r["symbol"]: r["sector"] for r in records}
    all_symbols = list(universe_map.keys())

    event_log.info(f"Screening {len(all_symbols)} stocks from S&P 500")

    interval = config.get("interval", "1d")
    end = datetime.now()
    max_days = _INTERVAL_MAX_DAYS.get(interval, 60)
    lookback = min(60, max_days)
    start = end - timedelta(days=lookback)

    event_log.info(f"Fetching {lookback} days of data for {len(all_symbols)} stocks...")
    price_data = await data_source.fetch_historical_batch(all_symbols, start, end, interval)
    event_log.info(f"Fetched data for {len(price_data)}/{len(all_symbols)} stocks")

    # Store fetched data in cache
    for symbol, df in price_data.items():
        await _store_price_data(df, symbol, interval, "yahoo")

    screener = StockScreener(
        min_volume=min_volume,
        min_volatility=min_volatility,
        top_n_per_sector=top_n,
        interval=interval,
    )
    candidates = screener.filter_candidates(price_data, universe_map)
    event_log.info(f"Screening: {len(candidates)} candidates passed filters")

    selected = screener.select_top_n(candidates)

    # Save screening results to DB
    async with async_session() as session:
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
            session.add(result)
        await session.commit()

    active_symbols = {s: info["sector"] for s, info in selected.items() if info["selected"]}
    event_log.info(f"Selected {len(active_symbols)} stocks for trading")

    candidate_data = {s: price_data[s] for s in candidates if s in price_data}
    return active_symbols, candidate_data


async def _train_and_run(
    engine,
    strategy,
    data_source,
    config: dict,
    sim_id: int,
    event_log: EventLog,
):
    """Load/fetch training data, train the model, then start the tick loop."""
    try:
        interval = config.get("interval", "1d")
        tick_seconds = config.get("tick_seconds", 60.0)

        # --- Phase 1: Training ---
        try:
            if "universe" in config:
                # === SCREENER MODE ===
                active_symbols, candidate_data = await _run_screening(
                    config, data_source, event_log, sim_id
                )

                # Train sector models
                loader = StockUniverseLoader()
                universe_map = {r["symbol"]: r["sector"] for r in loader.load_from_csv()}

                sectors_to_train = set(active_symbols.values())
                for sector in sectors_to_train:
                    sector_symbols = [
                        s for s, sec in universe_map.items()
                        if sec == sector and s in candidate_data
                    ]
                    if len(sector_symbols) < 5:
                        event_log.warning(
                            f"Sector '{sector}': only {len(sector_symbols)} stocks, skipping"
                        )
                        continue

                    frames = []
                    for s in sector_symbols:
                        df = candidate_data[s].copy()
                        df["symbol"] = s
                        frames.append(df)
                    sector_data = pd.concat(frames, ignore_index=True)
                    sector_data.sort_values("timestamp", inplace=True)

                    event_log.info(
                        f"Training model for sector: {sector} ({len(sector_symbols)} stocks)"
                    )
                    metrics = await strategy.train(sector_data, sector=sector)
                    event_log.info(
                        f"Sector '{sector}' trained: accuracy={metrics.accuracy:.2%}"
                    )

                engine.symbols = list(active_symbols.keys())
                engine._sector_map = active_symbols
            else:
                # === LEGACY MODE (fixed symbols) ===
                event_log.info("Preparing training data...")
                symbols = config.get("symbols", [])
                all_data = []
                for symbol in symbols:
                    df = await _get_training_data(symbol, interval, data_source, event_log)
                    if not df.empty:
                        all_data.append(df)

                if all_data:
                    combined = pd.concat(all_data, ignore_index=True)
                    event_log.info(f"Training model on {len(combined)} rows...")
                    metrics = await strategy.train(combined)
                    event_log.info(
                        f"Model trained — accuracy: {metrics.accuracy:.2%}, "
                        f"precision: {metrics.precision:.2%}, recall: {metrics.recall:.2%}"
                    )
                else:
                    event_log.warning("No training data available — running in degraded mode")
        except Exception as e:
            event_log.error(f"Training failed: {e} — running in degraded mode")
            logger.exception("Model training failed")

        # --- Phase 2: Tick loop ---
        event_log.info(
            f"Starting tick loop: every {tick_seconds}s, "
            f"fetching latest {interval} data and evaluating trades"
        )
        await engine.run(tick_seconds=tick_seconds)
    except Exception as e:
        event_log.error(f"Engine crashed: {e}")
        logger.exception("Engine background task crashed")


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

    from app.core.broker.simulator import SimulatorBroker
    from app.core.data.yahoo import YahooDataSource
    from app.core.engine import TradingEngine
    from app.core.strategy.ml_strategy import MLStrategy

    # For screener mode, start with empty symbols (screener will populate them)
    if "universe" in sim.config:
        symbols = []
    else:
        symbols = sim.config.get("symbols", ["AAPL"])

    interval = sim.config.get("interval", "1d")

    broker = SimulatorBroker(initial_cash=sim.current_cash)
    data_source = YahooDataSource()
    strategy = MLStrategy()
    event_log = EventLog()

    # --- Callbacks ---
    captured_sim_id = sim_id

    async def on_trade(order):
        try:
            async with async_session() as session:
                trade = Trade(
                    simulation_id=captured_sim_id,
                    symbol=order.symbol,
                    side=TradeSide.BUY if order.side == "buy" else TradeSide.SELL,
                    quantity=order.quantity,
                    price=order.price,
                    timestamp=order.timestamp.replace(tzinfo=None),
                    strategy="ml_strategy",
                )
                session.add(trade)
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist trade for sim %d: %s", captured_sim_id, e)

    async def on_tick_complete():
        try:
            async with async_session() as session:
                total_value = await broker.get_portfolio_value(engine.latest_prices)
                cash = await broker.get_balance()
                snapshot = PortfolioSnapshot(
                    simulation_id=captured_sim_id,
                    timestamp=datetime.utcnow(),
                    total_value=total_value,
                    cash=cash,
                )
                session.add(snapshot)
                await session.commit()
        except Exception as e:
            logger.error("Failed to persist snapshot for sim %d: %s", captured_sim_id, e)

    engine = TradingEngine(
        data_source=data_source,
        strategy=strategy,
        broker=broker,
        symbols=symbols,
        interval=interval,
        on_trade=on_trade,
        on_tick_complete=on_tick_complete,
        event_log=event_log,
    )

    # Launch training + tick loop as a background task (non-blocking)
    import asyncio

    engine._running = True
    engine._task = asyncio.create_task(
        _train_and_run(
            engine, strategy, data_source, sim.config, sim_id, event_log,
        )
    )

    _engines[sim_id] = {
        "engine": engine,
        "broker": broker,
        "strategy": strategy,
        "event_log": event_log,
    }

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
        broker = _engines[sim_id]["broker"]
        sim.current_cash = await broker.get_balance()
        del _engines[sim_id]

    sim.status = SimulationStatus.STOPPED
    await db.commit()

    return {"status": "stopped", "simulation_id": sim_id}


@router.delete("/{sim_id}")
async def delete_simulation(sim_id: int, db: AsyncSession = Depends(get_db)):
    sim = await db.get(Simulation, sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Stop engine if running
    if sim_id in _engines:
        _engines[sim_id]["engine"].stop()
        del _engines[sim_id]

    # Delete related records, then the simulation
    from sqlalchemy import delete as sql_delete

    await db.execute(sql_delete(ScreeningResult).where(ScreeningResult.simulation_id == sim_id))
    await db.execute(sql_delete(SectorModel).where(SectorModel.simulation_id == sim_id))
    await db.execute(sql_delete(PortfolioSnapshot).where(PortfolioSnapshot.simulation_id == sim_id))
    await db.execute(sql_delete(Trade).where(Trade.simulation_id == sim_id))
    await db.delete(sim)
    await db.commit()

    return {"status": "deleted", "simulation_id": sim_id}


@router.get("/{sim_id}/logs")
async def get_simulation_logs(
    sim_id: int,
    limit: int = Query(default=50, ge=1, le=200),
):
    if sim_id not in _engines:
        raise HTTPException(
            status_code=404,
            detail="Simulation not running or not found",
        )
    event_log: EventLog = _engines[sim_id]["event_log"]
    return {"simulation_id": sim_id, "events": event_log.get_events(limit)}


@router.get("/{sim_id}/screening")
async def get_screening_results(sim_id: int):
    async with async_session() as session:
        result = await session.execute(
            select(ScreeningResult)
            .where(ScreeningResult.simulation_id == sim_id)
            .order_by(ScreeningResult.screened_at.desc())
            .limit(500)
        )
        rows = result.scalars().all()
    return [
        {
            "symbol": r.symbol,
            "sector": r.sector,
            "volume_avg": r.volume_avg,
            "volatility": r.volatility,
            "opportunity_score": r.opportunity_score,
            "selected": r.selected,
            "screened_at": r.screened_at.isoformat() if r.screened_at else None,
        }
        for r in rows
    ]
