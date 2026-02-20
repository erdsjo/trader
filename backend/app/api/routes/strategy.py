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
