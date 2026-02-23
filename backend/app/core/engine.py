import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.core.broker.base import Broker, Order
from app.core.data.base import DataSource
from app.core.event_log import EventLog
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
        on_trade: Callable[[Order], Awaitable[None]] | None = None,
        on_tick_complete: Callable[[], Awaitable[None]] | None = None,
        event_log: EventLog | None = None,
        sector_map: dict[str, str] | None = None,
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.min_confidence = min_confidence
        self.interval = interval
        self.on_trade = on_trade
        self.on_tick_complete = on_tick_complete
        self.event_log = event_log
        self._sector_map = sector_map or {}
        self.latest_prices: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def _log_info(self, msg: str) -> None:
        logger.info(msg)
        if self.event_log:
            self.event_log.info(msg)

    def _log_warning(self, msg: str) -> None:
        logger.warning(msg)
        if self.event_log:
            self.event_log.warning(msg)

    def _log_error(self, msg: str) -> None:
        logger.error(msg)
        if self.event_log:
            self.event_log.error(msg)

    async def tick(self):
        for symbol in self.symbols:
            try:
                data = await self.data_source.fetch_latest(symbol, self.interval)
                if data.empty:
                    self._log_warning(f"No data for {symbol}")
                    continue

                current_price = float(data["close"].iloc[-1])
                self.latest_prices[symbol] = current_price

                enriched = compute_indicators(data)
                sector = self._sector_map.get(symbol)
                signal = await self.strategy.analyze(symbol, enriched, sector=sector)

                if signal.confidence < self.min_confidence:
                    self._log_info(
                        f"{symbol}: {signal.action} skipped "
                        f"(confidence {signal.confidence:.2%} < {self.min_confidence:.2%})"
                    )
                    continue

                if signal.action == "buy" and signal.suggested_quantity > 0:
                    order = await self.broker.buy(
                        symbol, signal.suggested_quantity, current_price
                    )
                    self._log_info(
                        f"BUY {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )
                    if self.on_trade:
                        await self.on_trade(order)
                elif signal.action == "sell" and signal.suggested_quantity > 0:
                    order = await self.broker.sell(
                        symbol, signal.suggested_quantity, current_price
                    )
                    self._log_info(
                        f"SELL {signal.suggested_quantity} {symbol} @ {current_price:.2f}"
                    )
                    if self.on_trade:
                        await self.on_trade(order)

            except ValueError as e:
                self._log_warning(f"Trade failed for {symbol}: {e}")
            except Exception as e:
                self._log_error(f"Error processing {symbol}: {e}")

        if self.on_tick_complete:
            try:
                await self.on_tick_complete()
            except Exception as e:
                self._log_error(f"on_tick_complete callback failed: {e}")

    async def run(self, tick_seconds: float = 60.0):
        self._running = True
        self._log_info(f"Engine started: symbols={self.symbols}, interval={self.interval}")
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
        self._log_info("Engine stopped")
