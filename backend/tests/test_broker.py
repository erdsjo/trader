import pytest

from app.core.broker.simulator import SimulatorBroker


@pytest.fixture
def broker():
    return SimulatorBroker(initial_cash=10000.0, slippage=0.001)


@pytest.mark.asyncio
async def test_initial_balance(broker):
    balance = await broker.get_balance()
    assert balance == 10000.0


@pytest.mark.asyncio
async def test_buy_reduces_cash(broker):
    order = await broker.buy("AAPL", 10, 150.0)
    balance = await broker.get_balance()
    expected_cost = 10 * 150.0 * 1.001  # with slippage
    assert abs(balance - (10000.0 - expected_cost)) < 0.01
    assert order.side == "buy"
    assert order.symbol == "AAPL"


@pytest.mark.asyncio
async def test_buy_creates_position(broker):
    await broker.buy("AAPL", 10, 150.0)
    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == 10


@pytest.mark.asyncio
async def test_sell_increases_cash(broker):
    await broker.buy("AAPL", 10, 150.0)
    cash_after_buy = await broker.get_balance()
    await broker.sell("AAPL", 10, 160.0)
    cash_after_sell = await broker.get_balance()
    assert cash_after_sell > cash_after_buy


@pytest.mark.asyncio
async def test_sell_removes_position(broker):
    await broker.buy("AAPL", 10, 150.0)
    await broker.sell("AAPL", 10, 160.0)
    positions = await broker.get_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_insufficient_funds_raises(broker):
    with pytest.raises(ValueError, match="Insufficient funds"):
        await broker.buy("AAPL", 1000, 150.0)


@pytest.mark.asyncio
async def test_insufficient_shares_raises(broker):
    with pytest.raises(ValueError, match="Insufficient shares"):
        await broker.sell("AAPL", 10, 150.0)


@pytest.mark.asyncio
async def test_portfolio_value(broker):
    await broker.buy("AAPL", 10, 150.0)
    value = await broker.get_portfolio_value({"AAPL": 160.0})
    cash = await broker.get_balance()
    assert abs(value - (cash + 10 * 160.0)) < 0.01
