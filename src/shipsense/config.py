"""Project paths and constants. Every other module imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CSV_NAME = "DataCoSupplyChainDataset.csv"
# Kaggle's download unzips to archive/ (in this folder or its parent); data/raw/ also works.
_CANDIDATES = (ROOT / "data" / "raw" / _CSV_NAME, ROOT / "archive" / _CSV_NAME,
               ROOT.parent / "archive" / _CSV_NAME)
RAW_CSV = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
DB_PATH = ROOT / "data" / "processed" / "shipsense.db"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Time-based split boundaries (orders run Jan 2015 - Jan 2018).
VAL_START = "2017-01-01"
TEST_START = "2017-07-01"

TARGET = "late"
CATEGORICAL = [
    "shipping_mode", "payment_type", "market", "order_region",
    "order_country", "customer_segment", "main_department",
]
NUMERIC = [
    "days_scheduled", "n_items", "total_quantity", "total_sales",
    "avg_discount_rate", "n_departments", "order_hour", "order_dow", "order_month",
]

# In DataCo the scheduled days are fixed per shipping mode (verified: no exceptions),
# so the API derives days_scheduled instead of trusting the client to send it.
DAYS_SCHEDULED_BY_MODE = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}

SEED = 42
