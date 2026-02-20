from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    symbol: str
    suggested_quantity: int
    reasoning: str


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    feature_importance: dict


class Strategy(ABC):
    @abstractmethod
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Analyze data and return a trading signal."""

    @abstractmethod
    async def train(self, training_data: pd.DataFrame) -> ModelMetrics:
        """Train/retrain the strategy model on historical data."""

    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the strategy has a trained model ready."""
