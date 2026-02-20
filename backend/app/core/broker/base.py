from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return ((self.current_price / self.avg_price) - 1) * 100


class Broker(ABC):
    @abstractmethod
    async def buy(self, symbol: str, quantity: float, price: float) -> Order:
        """Execute a buy order."""

    @abstractmethod
    async def sell(self, symbol: str, quantity: float, price: float) -> Order:
        """Execute a sell order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all current positions."""

    @abstractmethod
    async def get_balance(self) -> float:
        """Get current cash balance."""

    @abstractmethod
    async def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Get total portfolio value (cash + positions) given current prices."""
