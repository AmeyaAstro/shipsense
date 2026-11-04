"""API tests on a tiny model trained from synthetic data (no CSV or real model needed in CI)."""
import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from test_pipeline import make_items

from shipsense.api import MODEL_NAME, Order, create_app, to_features
from shipsense.baselines import gradient_boosting
from shipsense.config import CATEGORICAL, NUMERIC, TARGET
from shipsense.data import build_database, load_order_features

THRESHOLD = 0.5

VALID = {
    "order_ts": "2017-09-14T10:30:00", "shipping_mode": "First Class", "payment_type": "DEBIT",
    "market": "LATAM", "order_region": "A", "order_country": "X", "customer_segment": "Consumer",
    "main_department": "Fitness", "n_items": 2, "total_quantity": 3, "total_sales": 120.0,
    "avg_discount_rate": 0.1, "n_departments": 1,
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Train a small GBM on synthetic orders and write model + metrics files like `make train` does."""
    tmp = tmp_path_factory.mktemp("api")
    build_database(make_items(), db_path=tmp / "test.db")
    df = load_order_features(db_path=tmp / "test.db")
    model = gradient_boosting().fit(df[CATEGORICAL + NUMERIC], df[TARGET])
    joblib.dump(model, tmp / "model.joblib")
    (tmp / "metrics.json").write_text(json.dumps({"models": {MODEL_NAME: {"val": {"threshold": THRESHOLD}}}}))
    return model, tmp / "model.joblib", tmp / "metrics.json"


@pytest.fixture(scope="module")
def client(artifacts):
    _, model_path, metrics_path = artifacts
    with TestClient(create_app(model_path, metrics_path)) as c:  # `with` runs startup (model loading)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith(f"{MODEL_NAME}-")


def test_predict_valid_order(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["late_probability"] <= 1.0
    assert body["threshold"] == THRESHOLD
    assert body["is_late_risk"] == (body["late_probability"] >= THRESHOLD)


def test_api_matches_model_called_directly(client, artifacts):
    """Train/serve consistency: the API must give the same score as the pipeline itself."""
    model = artifacts[0]
    expected = model.predict_proba(to_features([Order(**VALID)]))[0, 1]
    assert client.post("/predict", json=VALID).json()["late_probability"] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("change", [
    {"shipping_mode": "Rocket"},                 # not an allowed category
    {"n_items": 0},                              # out of range
    {"total_sales": "lots"},                     # wrong type
    {"order_ts": "yesterday"},                   # unparseable date
    {"n_items": 3, "total_quantity": 2},         # fewer units than items
    {"n_items": 1, "n_departments": 2},          # more departments than items
])
def test_invalid_input_returns_422(client, change):
    assert client.post("/predict", json={**VALID, **change}).status_code == 422


def test_missing_field_returns_422(client):
    body = {k: v for k, v in VALID.items() if k != "shipping_mode"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch_matches_single_predictions(client):
    orders = [VALID, {**VALID, "shipping_mode": "Standard Class"}, {**VALID, "order_country": "Unseen"}]
    r = client.post("/predict/batch", json={"orders": orders})
    assert r.status_code == 200
    batch = [p["late_probability"] for p in r.json()["predictions"]]
    singles = [client.post("/predict", json=o).json()["late_probability"] for o in orders]
    assert batch == singles


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"orders": []}).status_code == 422


def test_model_can_come_from_s3(artifacts, monkeypatch):
    """With SHIPSENSE_MODEL_S3_URI set, the app loads the downloaded file (S3 faked: no AWS needed)."""
    import shipsense.api as api

    _, model_path, metrics_path = artifacts
    asked = []

    def fake_download(uri):
        asked.append(uri)
        return model_path

    monkeypatch.setenv("SHIPSENSE_MODEL_S3_URI", "s3://bucket/models/gb.joblib")
    monkeypatch.setattr(api, "download_from_s3", fake_download)
    with TestClient(create_app(Path("/nonexistent.joblib"), metrics_path)) as c:
        assert c.get("/health").status_code == 200
    assert asked == ["s3://bucket/models/gb.joblib"]


def test_bad_s3_uri_is_rejected():
    from shipsense.api import download_from_s3

    with pytest.raises(ValueError):
        download_from_s3("s3://bucket-only")
