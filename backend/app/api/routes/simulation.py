import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import async_session, get_db
from app.core.event_log import EventLog
from app.models.db import (
    PortfolioSnapshot,
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


async def _train_and_run(
    engine,
    strategy,
    data_source,
    symbols: list[str],
    interval: str,
    tick_seconds: float,
    event_log: EventLog,
):
    """Fetch historical data, train the model, then start the tick loop."""
    try:
        event_log.info("Fetching historical data for training...")
        end = datetime.now()
        start = end - timedelta(days=365)
        all_data = []
        for symbol in symbols:
            df = await data_source.fetch_historical(symbol, start, end, interval)
            if not df.empty:
                all_data.append(df)
                event_log.info(f"Fetched {len(df)} rows for {symbol}")
            else:
                event_log.warning(f"No historical data for {symbol}")

        if all_data:
            import pandas as pd

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

    # Start the tick loop (runs until stopped)
    await engine.run(tick_seconds=tick_seconds)


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

    symbols = sim.config.get("symbols", ["AAPL"])
    interval = sim.config.get("interval", "1d")
    tick_seconds = sim.config.get("tick_seconds", 60.0)

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
            engine, strategy, data_source, symbols, interval, tick_seconds, event_log,
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
