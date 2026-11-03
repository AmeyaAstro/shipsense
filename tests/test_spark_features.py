"""Spark/SQL parity: both feature pipelines must produce identical order-level rows."""
import sqlite3

import pandas as pd
import pytest
from test_pipeline import make_items

pytest.importorskip("pyspark")
from order_features_spark import build_order_features, compare_with_sql, get_spark  # noqa: E402

from shipsense.data import build_database  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = get_spark("shipsense-tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def both_outputs(spark, tmp_path_factory):
    """Run the same synthetic items through SQLite (SQL) and Spark."""
    items = make_items(n_orders=200, seed=1)
    db = tmp_path_factory.mktemp("spark") / "test.db"
    build_database(items, db_path=db)
    with sqlite3.connect(db) as con:
        sql_df = pd.read_sql("SELECT * FROM order_features", con)
    spark_df = build_order_features(spark.createDataFrame(items)).toPandas()
    return spark_df, sql_df


def test_spark_matches_sql(both_outputs):
    spark_df, sql_df = both_outputs
    assert len(sql_df) > 0
    assert compare_with_sql(spark_df, sql_df) == []


def test_comparison_detects_a_difference(both_outputs):
    """Guard against a comparison that can never fail."""
    spark_df, sql_df = both_outputs
    tampered = spark_df.copy()
    tampered.loc[0, "total_sales"] += 1.0
    tampered.loc[0, "main_department"] = "Nonexistent"
    problems = compare_with_sql(tampered, sql_df)
    assert any(p.startswith("total_sales") for p in problems)
    assert any(p.startswith("main_department") for p in problems)
