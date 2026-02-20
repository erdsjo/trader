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
