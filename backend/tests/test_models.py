from app.models.db import (
    Base,
    MLModel,
    PortfolioSnapshot,
    PriceData,
    Simulation,
    SimulationStatus,
    Trade,
    TradeSide,
)
from app.models.schemas import PortfolioResponse, SimulationCreate, SimulationResponse


def test_simulation_model_defaults():
    sim = Simulation(name="test", initial_capital=10000.0, current_cash=10000.0)
    assert sim.name == "test"
    assert sim.initial_capital == 10000.0
    assert sim.status is None  # default applied by DB, not Python


def test_trade_side_enum():
    assert TradeSide.BUY == "buy"
    assert TradeSide.SELL == "sell"


def test_simulation_status_enum():
    assert SimulationStatus.RUNNING == "running"
    assert SimulationStatus.STOPPED == "stopped"


def test_simulation_create_schema():
    data = SimulationCreate(name="test", initial_capital=10000.0)
    assert data.name == "test"
    assert data.config == {}


def test_portfolio_response_schema():
    data = PortfolioResponse(
        total_value=10500.0,
        cash=5000.0,
        positions=[{"symbol": "AAPL", "quantity": 10, "current_price": 550.0}],
        total_pnl=500.0,
        total_pnl_pct=5.0,
    )
    assert data.total_pnl_pct == 5.0


def test_all_tables_defined():
    table_names = {t.name for t in Base.metadata.sorted_tables}
    assert "simulations" in table_names
    assert "price_data" in table_names
    assert "trades" in table_names
    assert "portfolio_snapshots" in table_names
    assert "models" in table_names
