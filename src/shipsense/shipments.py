"""Order-level shipment records for the copilot's lookup_shipment tool.

Why a separate table: this holds what ACTUALLY happened (real shipping days, delivery status,
ship date). Those fields are leakage for the model, which is why data.py never loads them; the
model code never reads this table. Looking up a past order's outcome is still a fair ops question.
No customer personal data (names, emails, streets, passwords) is copied: the copilot doesn't need it.

Run:  python -m shipsense.shipments   (part of `make copilot-data`)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from shipsense.config import DB_PATH, RAW_CSV

SHIPMENT_COLUMNS = {
    "Order Id": "order_id",
    "order date (DateOrders)": "order_ts",
    "shipping date (DateOrders)": "shipping_ts",
    "Shipping Mode": "shipping_mode",
    "Days for shipment (scheduled)": "days_scheduled",
    "Days for shipping (real)": "days_real",
    "Delivery Status": "delivery_status",
    "Order Status": "order_status",
    "Order City": "order_city",
    "Order Country": "order_country",
    "Market": "market",
    "Customer Segment": "customer_segment",
    "Sales": "sales",
}


def load_shipment_items(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1", usecols=list(SHIPMENT_COLUMNS)).rename(columns=SHIPMENT_COLUMNS)
    for col in ("order_ts", "shipping_ts"):
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %H:%M").dt.strftime("%Y-%m-%d %H:%M")
    return df


def build_shipments_table(items: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Collapse item rows to one row per order (all statuses, including canceled) and write `shipments`."""
    orders = items.groupby("order_id", as_index=False).agg(
        order_ts=("order_ts", "min"), shipping_ts=("shipping_ts", "min"),
        shipping_mode=("shipping_mode", "first"), days_scheduled=("days_scheduled", "first"),
        days_real=("days_real", "first"), delivery_status=("delivery_status", "first"),
        order_status=("order_status", "first"), order_city=("order_city", "first"),
        order_country=("order_country", "first"), market=("market", "first"),
        customer_segment=("customer_segment", "first"),
        n_items=("order_id", "size"), order_total=("sales", "sum"),
    )
    orders["order_total"] = orders["order_total"].round(2)
    orders["days_late"] = (orders["days_real"] - orders["days_scheduled"]).clip(lower=0)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        orders.to_sql("shipments", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_shipments_order ON shipments(order_id)")
    return len(orders)


if __name__ == "__main__":
    n = build_shipments_table(load_shipment_items())
    print(f"shipments table: {n:,} orders -> {DB_PATH.name}")
