from datetime import datetime

from pydantic import BaseModel


class SimulationCreate(BaseModel):
    name: str
    initial_capital: float
    config: dict = {}


class SimulationResponse(BaseModel):
    id: int
    name: str
    initial_capital: float
    current_cash: float
    start_time: datetime
    status: str
    config: dict

    class Config:
        from_attributes = True


class TradeResponse(BaseModel):
    id: int
    simulation_id: int
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    strategy: str | None

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    total_value: float
    cash: float
    positions: list[dict]
    total_pnl: float
    total_pnl_pct: float


class PerformancePoint(BaseModel):
    timestamp: datetime
    total_value: float


class PerformanceResponse(BaseModel):
    points: list[PerformancePoint]
    daily_pnl: float
    total_pnl: float
    total_pnl_pct: float


class SignalResponse(BaseModel):
    action: str
    confidence: float
    symbol: str
    reasoning: str


class StrategyStatusResponse(BaseModel):
    model_name: str
    model_version: int | None
    last_trained: datetime | None
    metrics: dict
    active_signals: list[SignalResponse]


class ScreeningConfig(BaseModel):
    min_volume: float = 1_000_000
    min_volatility: float = 0.15
    top_n_per_sector: int = 5
    rescreen_hour_utc: int = 6


class ScreeningResultResponse(BaseModel):
    symbol: str
    sector: str
    volume_avg: float
    volatility: float
    opportunity_score: float | None
    selected: bool
    screened_at: datetime

    class Config:
        from_attributes = True
