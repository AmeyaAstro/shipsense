"""Load the raw DataCo CSV into SQLite and build the order-level feature table.

Pipeline:  raw CSV  ->  order_items (SQLite)  ->  order_features (SQL)  ->  pandas
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from shipsense.config import DB_PATH, RAW_CSV, SQL_DIR

# Raw column name -> clean snake_case name. Only the columns we need are loaded.
RAW_COLUMNS = {
    "Order Id": "order_id",
    "order date (DateOrders)": "order_ts",
    "Shipping Mode": "shipping_mode",
    "Days for shipment (scheduled)": "days_scheduled",
    "Type": "payment_type",
    "Market": "market",
    "Order Region": "order_region",
    "Order Country": "order_country",
    "Customer Segment": "customer_segment",
    "Order Item Quantity": "quantity",
    "Sales": "sales",
    "Order Item Discount Rate": "discount_rate",
    "Department Name": "department_name",
    "Order Status": "order_status",
    "Late_delivery_risk": "late_delivery_risk",
}


def load_raw(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw item-level CSV (it is Latin-1 encoded, not UTF-8)."""
    df = pd.read_csv(csv_path, encoding="latin1", usecols=list(RAW_COLUMNS))
    df = df.rename(columns=RAW_COLUMNS)
    df["order_ts"] = pd.to_datetime(df["order_ts"], format="%m/%d/%Y %H:%M")
    # Store timestamps as ISO strings so SQLite sorts them correctly.
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def build_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Write item rows to SQLite and run the feature SQL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql("order_items", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)")
        con.executescript((SQL_DIR / "order_features.sql").read_text())


def load_order_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read the order-level table and add simple calendar features."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM order_features ORDER BY order_ts", con)
    ts = pd.to_datetime(df["order_ts"])
    df["order_ts"] = ts
    df["order_hour"] = ts.dt.hour
    df["order_dow"] = ts.dt.day_of_week
    df["order_month"] = ts.dt.month
    return df


def time_split(df: pd.DataFrame, val_start: str, test_start: str):
    """Split by order date: train < val_start <= val < test_start <= test.

    A time split mimics production (train on the past, predict the future)
    and is stricter than a random split.
    """
    train = df[df["order_ts"] < val_start]
    val = df[(df["order_ts"] >= val_start) & (df["order_ts"] < test_start)]
    test = df[df["order_ts"] >= test_start]
    return train.copy(), val.copy(), test.copy()


if __name__ == "__main__":
    raw = load_raw()
    build_database(raw)
    feats = load_order_features()
    print(f"items: {len(raw):,}  orders: {len(feats):,}  late rate: {feats['late'].mean():.3f}")
