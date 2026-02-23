import csv
from pathlib import Path


class StockUniverseLoader:
    def __init__(self, csv_path: Path | None = None):
        if csv_path is None:
            csv_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sp500.csv"
        self.csv_path = csv_path

    def load_from_csv(self) -> list[dict]:
        records = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "symbol": row["symbol"].strip(),
                    "name": row["name"].strip(),
                    "sector": row["sector"].strip(),
                })
        return records

    def get_sectors(self) -> list[str]:
        records = self.load_from_csv()
        return sorted(set(r["sector"] for r in records))

    def get_symbols_by_sector(self, sector: str) -> list[str]:
        records = self.load_from_csv()
        return [r["symbol"] for r in records if r["sector"] == sector]
