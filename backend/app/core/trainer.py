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
