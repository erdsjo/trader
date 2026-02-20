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
