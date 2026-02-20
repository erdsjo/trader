"""initial tables

Revision ID: 82dd4da97f84
Revises: 
Create Date: 2026-02-20 15:09:39.555305

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '82dd4da97f84'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "simulations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("current_cash", sa.Float(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("created", "running", "stopped", "completed", name="simulationstatus"),
            nullable=True,
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "price_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("interval", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "timestamp", "interval", name="uq_price_data"),
    )
    op.create_index(op.f("ix_price_data_symbol"), "price_data", ["symbol"])
    op.create_index(op.f("ix_price_data_timestamp"), "price_data", ["timestamp"])

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column(
            "side",
            sa.Enum("buy", "sell", name="tradeside"),
            nullable=False,
        ),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["simulation_id"], ["simulations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_trades_symbol"), "trades", ["symbol"])

    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("total_value", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["simulation_id"], ["simulations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("strategy_name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("trained_at", sa.DateTime(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("models")
    op.drop_table("portfolio_snapshots")
    op.drop_index(op.f("ix_trades_symbol"), table_name="trades")
    op.drop_table("trades")
    op.drop_index(op.f("ix_price_data_timestamp"), table_name="price_data")
    op.drop_index(op.f("ix_price_data_symbol"), table_name="price_data")
    op.drop_table("price_data")
    op.drop_table("simulations")
    sa.Enum("created", "running", "stopped", "completed", name="simulationstatus").drop(op.get_bind())
    sa.Enum("buy", "sell", name="tradeside").drop(op.get_bind())
