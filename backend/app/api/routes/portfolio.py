from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
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
