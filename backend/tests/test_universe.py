import pytest
from pathlib import Path
from app.core.data.universe import StockUniverseLoader

SP500_CSV = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"


def test_csv_file_exists():
    assert SP500_CSV.exists(), "sp500.csv must exist in backend/data/"


def test_load_csv_returns_records():
    loader = StockUniverseLoader(SP500_CSV)
    records = loader.load_from_csv()
    assert len(records) > 400  # S&P 500 should have ~500 entries
    first = records[0]
    assert "symbol" in first
    assert "name" in first
    assert "sector" in first


def test_get_sectors_returns_eleven():
    loader = StockUniverseLoader(SP500_CSV)
    sectors = loader.get_sectors()
    assert len(sectors) == 11


def test_get_symbols_by_sector():
    loader = StockUniverseLoader(SP500_CSV)
    tech = loader.get_symbols_by_sector("Information Technology")
    assert len(tech) > 30  # IT sector should have plenty
