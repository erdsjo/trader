import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SimulationStatus(str, enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"


class TradeSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    initial_capital = Column(Float, nullable=False)
    current_cash = Column(Float, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(SimulationStatus, native_enum=False), default=SimulationStatus.CREATED)
    config = Column(JSON, default=dict)

    trades = relationship("Trade", back_populates="simulation")
    snapshots = relationship("PortfolioSnapshot", back_populates="simulation")


class PriceData(Base):
    __tablename__ = "price_data"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "interval", name="uq_price_data"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    interval = Column(String, nullable=False)
    source = Column(String, nullable=False)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    side = Column(Enum(TradeSide, native_enum=False), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String, nullable=True)

    simulation = relationship("Simulation", back_populates="trades")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)

    simulation = relationship("Simulation", back_populates="snapshots")


class MLModel(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    metrics = Column(JSON, default=dict)
    trained_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=False)
