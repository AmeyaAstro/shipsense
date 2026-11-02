"""Fast unit tests on small synthetic data (no 96 MB CSV needed, so CI stays quick)."""
import numpy as np
import pandas as pd
import pytest
import torch

from shipsense.baselines import ShippingModeRule, gradient_boosting, logistic_regression
from shipsense.config import CATEGORICAL, NUMERIC
from shipsense.data import RAW_COLUMNS, build_database, load_order_features, time_split
from shipsense.evaluate import best_threshold, score
from shipsense.torch_model import LateDeliveryNet, TabularEncoder, TorchClassifier

LEAKY = {"Days for shipping (real)", "Delivery Status", "shipping date (DateOrders)",
         "Benefit per order", "Order Profit Per Order"}


def make_items(n_orders: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic item-level rows shaped like the cleaned DataCo data."""
    rng = np.random.default_rng(seed)
    rows = []
    for oid in range(1, n_orders + 1):
        mode = rng.choice(["Standard Class", "First Class", "Second Class", "Same Day"])
        late = int(rng.random() < (0.9 if mode == "First Class" else 0.4))
        ts = pd.Timestamp("2015-01-01") + pd.Timedelta(hours=int(oid * 90))
        status = rng.choice(["COMPLETE", "PENDING", "CANCELED"], p=[0.6, 0.35, 0.05])
        for _ in range(int(rng.integers(1, 4))):
            rows.append({
                "order_id": oid, "order_ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "shipping_mode": mode,
                "days_scheduled": {"Standard Class": 4, "First Class": 1, "Second Class": 2, "Same Day": 0}[mode],
                "payment_type": rng.choice(["DEBIT", "TRANSFER", "CASH"]),
                "market": rng.choice(["LATAM", "Europe"]), "order_region": rng.choice(["A", "B"]),
                "order_country": rng.choice(["X", "Y", "Z"]), "customer_segment": "Consumer",
                "quantity": int(rng.integers(1, 5)), "sales": float(rng.uniform(10, 500)),
                "discount_rate": float(rng.uniform(0, 0.25)),
                "department_name": rng.choice(["Fitness", "Apparel"]),
                "order_status": status, "late_delivery_risk": 0 if status == "CANCELED" else late,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    build_database(make_items(), db_path=db)
    return load_order_features(db_path=db)


def test_no_leaky_columns_are_loaded():
    assert not LEAKY & set(RAW_COLUMNS), "post-delivery columns must never be features"


def test_one_row_per_order_and_cancellations_dropped(orders):
    assert orders["order_id"].is_unique
    items = make_items()
    canceled = set(items.loc[items["order_status"] == "CANCELED", "order_id"])
    assert canceled.isdisjoint(set(orders["order_id"]))


def test_feature_columns_present(orders):
    assert set(CATEGORICAL + NUMERIC + ["late"]) <= set(orders.columns)


def test_time_split_has_no_overlap(orders):
    tr, va, te = time_split(orders, "2015-01-08", "2015-01-11")
    assert tr["order_ts"].max() < va["order_ts"].min()
    assert va["order_ts"].max() < te["order_ts"].min()
    assert len(tr) + len(va) + len(te) == len(orders)


def test_shipping_mode_rule_learns_rates(orders):
    rule = ShippingModeRule().fit(orders, orders["late"])
    p = rule.predict_proba(orders)[:, 1]
    first = orders["shipping_mode"] == "First Class"
    assert p[first].mean() > p[~first].mean()


@pytest.mark.parametrize("factory", [logistic_regression, gradient_boosting])
def test_sklearn_models_output_probabilities(orders, factory):
    X = orders[CATEGORICAL + NUMERIC]
    p = factory().fit(X, orders["late"]).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_encoder_handles_unseen_category(orders):
    enc = TabularEncoder().fit(orders)
    new = orders.head(2).copy()
    new["order_country"] = "Atlantis"
    cats, nums = enc.transform(new)
    assert (cats[:, CATEGORICAL.index("order_country")] == 0).all()
    assert nums.shape == (2, len(NUMERIC))


def test_network_forward_shape():
    net = LateDeliveryNet([5] * len(CATEGORICAL), len(NUMERIC))
    net.eval()
    out = net(torch.zeros(8, len(CATEGORICAL), dtype=torch.long), torch.zeros(8, len(NUMERIC)))
    assert out.shape == (8,)


def test_torch_classifier_trains(orders):
    X, y = orders[CATEGORICAL + NUMERIC], orders["late"]
    clf = TorchClassifier(epochs=3).fit(X, y, X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(1), 1)
    assert len(clf.history) >= 1


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    t = best_threshold(y, p)
    m = score(y, p, t)
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0
