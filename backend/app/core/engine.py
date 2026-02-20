import asyncio
import logging

from app.core.broker.base import Broker
from app.core.data.base import DataSource
from app.core.strategy.base import Strategy
from app.core.strategy.indicators import compute_indicators

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        data_source: DataSource,
        strategy: Strategy,
        broker: Broker,
        symbols: list[str],
        min_confidence: float = 0.6,
        interval: str = "1d",
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.min_confidence = min_confidence
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def tick(self):
        for symbol in self.symbols:
            try:
                data = await self.data_source.fetch_latest(symbol, self.interval)
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                enriched = compute_indicators(data)
                signal = await self.strategy.analyze(symbol, enriched)

                if signal.confidence < self.min_confidence:
                    logger.info(
                        f"{symbol}: {signal.action} skipped (confidence {signal.confidence:.2%} < {self.min_confidence:.2%})"
                    )
                    continue

                current_price = float(data["close"].iloc[-1])

                if signal.action == "buy" and signal.suggested_quantity > 0:
                    await self.broker.buy(symbol, signal.suggested_quantity, current_price)
                    logger.info(
                        f"BUY {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )
                elif signal.action == "sell" and signal.suggested_quantity > 0:
                    await self.broker.sell(symbol, signal.suggested_quantity, current_price)
                    logger.info(
                        f"SELL {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )

            except ValueError as e:
                logger.warning(f"Trade failed for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def run(self, tick_seconds: float = 60.0):
        self._running = True
        logger.info(f"Engine started: symbols={self.symbols}, interval={self.interval}")
        while self._running:
            await self.tick()
            await asyncio.sleep(tick_seconds)

    def start(self, tick_seconds: float = 60.0):
        self._task = asyncio.create_task(self.run(tick_seconds))

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Engine stopped")
