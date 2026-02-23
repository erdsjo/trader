import asyncio
import os
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.core.strategy.base import ModelMetrics, Signal, Strategy
from app.core.strategy.indicators import compute_cross_stock_features, compute_indicators

_UNIVERSAL_KEY = "__universal__"

# Base features used by all models (same as original)
_BASE_FEATURES = [
    "rsi", "macd", "macd_signal", "bb_upper", "bb_lower",
    "sma_20", "ema_12", "close", "volume",
]

# Additional features available when sector data (multi-symbol) is provided
_SECTOR_EXTRA_FEATURES = [
    "symbol_encoded", "sector_avg_return", "sector_rsi_mean", "relative_volume",
]


class MLStrategy(Strategy):
    def __init__(self, model_dir: str = "./model_artifacts", threshold: float = 0.005):
        self.model_dir = model_dir
        self.threshold = threshold  # price change threshold for labeling
        self._models: dict[str, XGBClassifier] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._feature_columns: dict[str, list[str]] = {}
        os.makedirs(model_dir, exist_ok=True)

        # Keep backward-compatible attribute: expose feature_columns for the
        # universal model so any code reading strategy.feature_columns still works.
        self.feature_columns: list[str] = []

    # -- backward-compatible property for self.model ---------------------
    @property
    def model(self) -> XGBClassifier | None:
        return self._models.get(_UNIVERSAL_KEY)

    @model.setter
    def model(self, value: XGBClassifier | None):
        if value is None:
            self._models.pop(_UNIVERSAL_KEY, None)
        else:
            self._models[_UNIVERSAL_KEY] = value

    # -- public API -------------------------------------------------------

    def is_trained(self, sector: str | None = None) -> bool:
        key = self._sector_key(sector)
        return key in self._models

    async def train(self, training_data: pd.DataFrame, sector: str | None = None) -> ModelMetrics:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._train_sync, training_data, sector)
        )

    async def analyze(self, symbol: str, data: pd.DataFrame, sector: str | None = None) -> Signal:
        key = self._sector_key(sector)
        if key not in self._models:
            # Fall back to universal model if sector model missing
            if sector is not None and _UNIVERSAL_KEY in self._models:
                key = _UNIVERSAL_KEY
            else:
                return Signal(
                    action="hold", confidence=0.0, symbol=symbol,
                    suggested_quantity=0, reasoning="Model not trained",
                )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._analyze_sync, symbol, data, key)
        )

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _sector_key(sector: str | None) -> str:
        if sector is None:
            return _UNIVERSAL_KEY
        return sector.replace(" ", "_")

    def _has_multi_symbol(self, df: pd.DataFrame) -> bool:
        return "symbol" in df.columns and df["symbol"].nunique() > 1

    def _prepare_features(self, df: pd.DataFrame, sector_key: str) -> pd.DataFrame:
        """Compute indicators and optionally cross-stock features.

        For multi-symbol data (sector training), we compute per-symbol
        indicators then merge cross-stock features. For single-symbol /
        universal data, we keep original behaviour.
        """
        has_multi = self._has_multi_symbol(df)

        if has_multi:
            # Compute per-symbol indicators
            parts = []
            for sym, grp in df.groupby("symbol"):
                enriched = compute_indicators(grp)
                parts.append(enriched)
            enriched = pd.concat(parts, ignore_index=True)

            # Cross-stock features need a timestamp column
            enriched = compute_cross_stock_features(enriched)

            # Encode symbol as numeric
            le = LabelEncoder()
            enriched["symbol_encoded"] = le.fit_transform(enriched["symbol"].astype(str))
            self._label_encoders[sector_key] = le

            feature_cols = _BASE_FEATURES + _SECTOR_EXTRA_FEATURES
        else:
            enriched = compute_indicators(df)
            feature_cols = list(_BASE_FEATURES)

        self._feature_columns[sector_key] = feature_cols

        # Expose on instance for backward compat
        if sector_key == _UNIVERSAL_KEY:
            self.feature_columns = feature_cols

        return enriched.dropna(subset=[c for c in feature_cols if c in enriched.columns])

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        if self._has_multi_symbol(df):
            # Per-symbol future return
            future_return = df.groupby("symbol")["close"].transform(
                lambda s: s.shift(-1) / s - 1
            )
        else:
            future_return = df["close"].shift(-1) / df["close"] - 1

        labels = pd.Series(np.where(
            future_return > self.threshold, 2,  # buy
            np.where(future_return < -self.threshold, 0, 1)  # sell / hold
        ), index=df.index)
        return labels

    def _train_sync(self, training_data: pd.DataFrame, sector: str | None = None) -> ModelMetrics:
        key = self._sector_key(sector)
        df = self._prepare_features(training_data, key)
        labels = self._create_labels(df)
        feature_cols = self._feature_columns[key]

        # Drop rows with NaN labels (last row per symbol, or last row overall)
        valid = labels.notna() & ~labels.isna()
        # Also drop last row per group if multi-symbol, else last row
        if self._has_multi_symbol(df):
            last_idx = df.groupby("symbol").tail(1).index
            valid.loc[last_idx] = False
        else:
            valid.iloc[-1] = False

        df = df.loc[valid]
        labels = labels.loc[valid]

        X = df[feature_cols]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        self._models[key] = model

        y_pred = model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))

        from sklearn.metrics import precision_score, recall_score
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        importance = dict(zip(
            feature_cols,
            [float(x) for x in model.feature_importances_],
        ))

        # Save model to disk
        safe_key = key.replace(" ", "_")
        model_path = os.path.join(self.model_dir, f"{safe_key}.joblib")
        joblib.dump(model, model_path)

        # Backward compat: also save as xgb_model.joblib for universal
        if key == _UNIVERSAL_KEY:
            compat_path = os.path.join(self.model_dir, "xgb_model.joblib")
            joblib.dump(model, compat_path)

        return ModelMetrics(
            accuracy=accuracy, precision=precision,
            recall=recall, feature_importance=importance,
        )

    def _analyze_sync(self, symbol: str, data: pd.DataFrame, model_key: str) -> Signal:
        model = self._models[model_key]
        feature_cols = self._feature_columns.get(model_key, _BASE_FEATURES)

        # Determine if this model was trained with multi-symbol features
        uses_sector_features = "symbol_encoded" in feature_cols

        if uses_sector_features:
            enriched = compute_indicators(data)
            enriched = compute_cross_stock_features(enriched)

            # Encode symbol
            le = self._label_encoders.get(model_key)
            if le is not None and symbol in le.classes_:
                enriched["symbol_encoded"] = le.transform(
                    [symbol] * len(enriched)
                )
            else:
                # Unknown symbol: use 0 as fallback
                enriched["symbol_encoded"] = 0

            df = enriched.dropna(subset=[c for c in feature_cols if c in enriched.columns])
        else:
            df = compute_indicators(data).dropna()

        if df.empty:
            return Signal(
                action="hold", confidence=0.0, symbol=symbol,
                suggested_quantity=0, reasoning="Insufficient data",
            )

        latest = df[feature_cols].iloc[[-1]]
        proba = model.predict_proba(latest)[0]
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
