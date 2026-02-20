import numpy as np
import pandas as pd
import pytest

from app.core.strategy.ml_strategy import MLStrategy
from app.core.trainer import Trainer


def make_training_data(n=500):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


@pytest.fixture
def strategy(tmp_path):
    return MLStrategy(model_dir=str(tmp_path))


@pytest.fixture
def trainer(strategy):
    return Trainer(strategy=strategy)


@pytest.mark.asyncio
async def test_trainer_train_returns_metrics(trainer):
    data = make_training_data()
    metrics = await trainer.train(data)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert trainer.strategy.is_trained()


@pytest.mark.asyncio
async def test_trainer_evaluate(trainer):
    data = make_training_data()
    await trainer.train(data)
    eval_metrics = await trainer.evaluate(data.tail(100))
    assert "accuracy" in eval_metrics
