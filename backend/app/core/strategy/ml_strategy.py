import asyncio
import os
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.core.strategy.base import ModelMetrics, Signal, Strategy
from app.core.strategy.indicators import compute_indicators


class MLStrategy(Strategy):
    def __init__(self, model_dir: str = "./model_artifacts", threshold: float = 0.005):
        self.model_dir = model_dir
        self.threshold = threshold  # price change threshold for labeling
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []
        os.makedirs(model_dir, exist_ok=True)

    def is_trained(self) -> bool:
        return self.model is not None

    async def train(self, training_data: pd.DataFrame) -> ModelMetrics:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._train_sync, training_data))

    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        if not self.is_trained():
            return Signal(
                action="hold", confidence=0.0, symbol=symbol,
                suggested_quantity=0, reasoning="Model not trained"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._analyze_sync, symbol, data))

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = compute_indicators(df)
        self.feature_columns = [
            "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
            "sma_20", "ema_12", "close", "volume",
        ]
        return enriched.dropna()

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        future_return = df["close"].shift(-1) / df["close"] - 1
        labels = pd.Series(np.where(
            future_return > self.threshold, 2,  # buy
            np.where(future_return < -self.threshold, 0, 1)  # sell / hold
        ), index=df.index)
        return labels

    def _train_sync(self, training_data: pd.DataFrame) -> ModelMetrics:
        df = self._prepare_features(training_data)
        labels = self._create_labels(df)

        # Drop last row (no future label)
        df = df.iloc[:-1]
        labels = labels.iloc[:-1]

        X = df[self.feature_columns]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="mlogloss",
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))

        from sklearn.metrics import precision_score, recall_score
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        importance = dict(zip(
            self.feature_columns,
            [float(x) for x in self.model.feature_importances_],
        ))

        model_path = os.path.join(self.model_dir, "xgb_model.joblib")
        joblib.dump(self.model, model_path)

        return ModelMetrics(
            accuracy=accuracy, precision=precision,
            recall=recall, feature_importance=importance,
        )

    def _analyze_sync(self, symbol: str, data: pd.DataFrame) -> Signal:
        df = self._prepare_features(data)
        if df.empty:
            return Signal(
                action="hold", confidence=0.0, symbol=symbol,
                suggested_quantity=0, reasoning="Insufficient data"
            )

        latest = df[self.feature_columns].iloc[[-1]]
        proba = self.model.predict_proba(latest)[0]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])

        action_map = {0: "sell", 1: "hold", 2: "buy"}
        action = action_map[pred]

        return Signal(
            action=action,
            confidence=confidence,
            symbol=symbol,
            suggested_quantity=1,
            reasoning=f"XGBoost prediction: {action} with {confidence:.2%} confidence",
        )
