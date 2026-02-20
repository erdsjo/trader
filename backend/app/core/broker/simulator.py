from datetime import datetime, timezone

from app.core.broker.base import Broker, Order, Position


class SimulatorBroker(Broker):
    def __init__(self, initial_cash: float, slippage: float = 0.001):
        self.cash = initial_cash
        self.slippage = slippage
        self.positions: dict[str, dict] = {}  # symbol -> {quantity, avg_price}
        self.trade_log: list[Order] = []

    async def buy(self, symbol: str, quantity: float, price: float) -> Order:
        fill_price = price * (1 + self.slippage)
        cost = quantity * fill_price

        if cost > self.cash:
            raise ValueError(f"Insufficient funds: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos["quantity"] + quantity
            pos["avg_price"] = (
                (pos["avg_price"] * pos["quantity"]) + (fill_price * quantity)
            ) / total_qty
            pos["quantity"] = total_qty
        else:
            self.positions[symbol] = {"quantity": quantity, "avg_price": fill_price}

        order = Order(
            symbol=symbol, side="buy", quantity=quantity,
            price=fill_price, timestamp=datetime.now(timezone.utc),
        )
        self.trade_log.append(order)
        return order

    async def sell(self, symbol: str, quantity: float, price: float) -> Order:
        if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
            raise ValueError(f"Insufficient shares of {symbol}")

        fill_price = price * (1 - self.slippage)
        proceeds = quantity * fill_price
        self.cash += proceeds

        self.positions[symbol]["quantity"] -= quantity
        if self.positions[symbol]["quantity"] <= 0:
            del self.positions[symbol]

        order = Order(
            symbol=symbol, side="sell", quantity=quantity,
            price=fill_price, timestamp=datetime.now(timezone.utc),
        )
        self.trade_log.append(order)
        return order

    async def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=symbol,
                quantity=data["quantity"],
                avg_price=data["avg_price"],
                current_price=data["avg_price"],  # updated externally
            )
            for symbol, data in self.positions.items()
        ]

    async def get_balance(self) -> float:
        return self.cash

    async def get_portfolio_value(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            data["quantity"] * prices.get(symbol, data["avg_price"])
            for symbol, data in self.positions.items()
        )
        return self.cash + positions_value
