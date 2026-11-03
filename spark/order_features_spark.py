"""PySpark version of sql/order_features.sql: same order-level features, distributed engine.

Why: SQLite is fine for 180k rows on one machine; Spark runs the same logic on a
cluster when data outgrows one machine. Two implementations of one feature
pipeline can silently drift, so tests/test_spark_features.py proves they match.

Run:  make spark-features            (writes data/processed/order_features_spark.parquet)
      make spark-features ARGS=--compare   (also checks every row against the SQLite table)
Needs Java 17+ (JAVA_HOME); Spark runs in local mode, using this Mac's cores.
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from shipsense.config import DB_PATH, RAW_CSV, ROOT
from shipsense.data import RAW_COLUMNS

OUT_PATH = ROOT / "data" / "processed" / "order_features_spark.parquet"
EXCLUDED_STATUSES = ["CANCELED", "SUSPECTED_FRAUD"]  # never ship -> not in the model (same as the SQL)


def get_spark(app_name: str = "shipsense") -> SparkSession:
    return (SparkSession.builder.master("local[*]").appName(app_name)
            .config("spark.sql.session.timeZone", "UTC")    # parse timestamps without local DST shifts
            .config("spark.sql.shuffle.partitions", "8")    # default 200 is overkill for one machine
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def load_items(spark: SparkSession, csv_path=RAW_CSV) -> DataFrame:
    """Read the raw CSV (Latin-1) and rename columns exactly like data.load_raw does."""
    raw = spark.read.csv(str(csv_path), header=True, inferSchema=True, encoding="ISO-8859-1")
    items = raw.select([F.col(f"`{src}`").alias(dst) for src, dst in RAW_COLUMNS.items()])
    # Raw dates look like "2/21/2015 0:06"; store as ISO text, matching what SQLite holds.
    ts = F.to_timestamp("order_ts", "M/d/yyyy H:mm")
    return items.withColumn("order_ts", F.date_format(ts, "yyyy-MM-dd HH:mm:ss"))


def build_order_features(items: DataFrame) -> DataFrame:
    """Item rows -> one row per order. Mirrors sql/order_features.sql line by line."""
    shipped = items.filter(~F.col("order_status").isin(EXCLUDED_STATUSES))

    # main_department = department of the most expensive line item (ties broken alphabetically).
    by_sales = Window.partitionBy("order_id").orderBy(F.col("sales").desc(), F.col("department_name").asc())
    main_dept = (shipped.withColumn("rank", F.row_number().over(by_sales))
                 .filter(F.col("rank") == 1)
                 .select("order_id", F.col("department_name").alias("main_department")))

    orders = shipped.groupBy("order_id").agg(
        F.min("order_ts").alias("order_ts"),
        F.min("shipping_mode").alias("shipping_mode"),
        F.min("days_scheduled").alias("days_scheduled"),
        F.min("payment_type").alias("payment_type"),
        F.min("market").alias("market"),
        F.min("order_region").alias("order_region"),
        F.min("order_country").alias("order_country"),
        F.min("customer_segment").alias("customer_segment"),
        F.count("*").alias("n_items"),
        F.sum("quantity").alias("total_quantity"),
        F.sum("sales").alias("total_sales"),
        F.avg("discount_rate").alias("avg_discount_rate"),
        F.countDistinct("department_name").alias("n_departments"),
        F.max("late_delivery_risk").alias("late"),
    )
    return orders.join(main_dept, on="order_id", how="left")


def compare_with_sql(spark_df: pd.DataFrame, sql_df: pd.DataFrame) -> list[str]:
    """Return a list of mismatches (empty = identical). Floats compared with a small tolerance,
    because a distributed sum adds numbers in a different order than SQLite does."""
    problems = []
    if len(spark_df) != len(sql_df):
        problems.append(f"row count: spark {len(spark_df)} vs sql {len(sql_df)}")
    if set(spark_df.columns) != set(sql_df.columns):
        problems.append(f"columns differ: {set(spark_df.columns) ^ set(sql_df.columns)}")
        return problems
    a = spark_df.sort_values("order_id").reset_index(drop=True)
    b = sql_df.sort_values("order_id").reset_index(drop=True)[a.columns]
    for col in a.columns:
        if pd.api.types.is_float_dtype(a[col]) or pd.api.types.is_float_dtype(b[col]):
            bad = ~((a[col].astype(float) - b[col].astype(float)).abs() <= 1e-6 * (1 + b[col].abs()))
        else:
            bad = a[col].astype(str) != b[col].astype(str)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} rows differ, e.g. order_id {int(a.loc[bad, 'order_id'].iloc[0])}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", action="store_true", help="check every row against the SQLite table")
    args = parser.parse_args()

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    features = build_order_features(load_items(spark))
    features.write.mode("overwrite").parquet(str(OUT_PATH))
    print(f"wrote {features.count():,} orders to {OUT_PATH.relative_to(ROOT)}")

    if args.compare:
        with sqlite3.connect(DB_PATH) as con:
            sql_df = pd.read_sql("SELECT * FROM order_features", con)
        problems = compare_with_sql(features.toPandas(), sql_df)
        if problems:
            raise SystemExit("MISMATCH:\n  " + "\n  ".join(problems))
        print(f"MATCH: Spark == SQL on all {len(sql_df):,} rows and every column")
    spark.stop()


if __name__ == "__main__":
    main()
