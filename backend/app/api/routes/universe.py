from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.data.universe import StockUniverseLoader
from app.models.db import StockUniverse

router = APIRouter(prefix="/universe", tags=["universe"])


@router.get("")
async def get_universe(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StockUniverse).order_by(StockUniverse.sector, StockUniverse.symbol)
    )
    rows = result.scalars().all()
    if not rows:
        await _load_from_csv(db)
        result = await db.execute(
            select(StockUniverse).order_by(StockUniverse.sector, StockUniverse.symbol)
        )
        rows = result.scalars().all()
    return [{"symbol": r.symbol, "name": r.name, "sector": r.sector} for r in rows]


@router.get("/sectors")
async def get_sectors(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(StockUniverse.sector).distinct().order_by(StockUniverse.sector)
    )
    return [row[0] for row in result.all()]


@router.post("/refresh")
async def refresh_universe(db: AsyncSession = Depends(get_db)):
    count = await _load_from_csv(db)
    return {"loaded": count}


async def _load_from_csv(db: AsyncSession) -> int:
    loader = StockUniverseLoader()
    records = loader.load_from_csv()

    await db.execute(delete(StockUniverse))
    for r in records:
        db.add(
            StockUniverse(
                symbol=r["symbol"],
                name=r["name"],
                sector=r["sector"],
            )
        )
    await db.commit()
    return len(records)
